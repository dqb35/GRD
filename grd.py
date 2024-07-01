import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LSTM, LSTMCell
import math

# from ..utils.device import get_device

from torch_geometric.nn import ARMAConv
from torch_geometric.nn import Sequential

_device = None

def get_device():
    ''' Returns the currently used computing device.'''
    if _device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_device(device)
    return _device

def set_device(device):
    ''' Sets the computing device. '''
    global _device
    _device = device



class GRD(torch.nn.Module):
    '''
    Anomaly detection neural network model for multivariate sensor time series.
    Graph structure is randomly initialized and learned during training.
    Uses an attention layer that scores the attention weights for the input
    time series window and the sensor embedding vector.

    Args:
        args (dict): Argparser with config information.
    '''

    def __init__(self, n_features,window_size,out_dim,lr,batch,device):
        super().__init__()

        self.device = device
        self.lr = lr
        self.batch = batch

        self.num_nodes = n_features
        # self.horizon = args.horizon
        self.topk = 10 # SMD
        self.embed_dim = 16
        self.lags = window_size

        # model parameters
        channels = 32  # channel == node embedding size because they are added # SMD
        hidden_size = 512

        # learned graph embeddings
        self.graph_embedding = SingleEmbedding(self.num_nodes, channels, topk=self.topk, warmup_epochs=10)

        # encoder
        self.tgconv = TGConv(1, channels)  # 时间图卷积层，用于编码器部分
        self.lstm = LSTM(channels * self.num_nodes, hidden_size, 2, batch_first=True,dropout=0.20)  # 长短期记忆网络（LSTM），用于编码器部分
        self.gru = nn.GRU(channels * self.num_nodes, hidden_size, 1, batch_first=True)# GRU模型 如果num_layers=1,则dropout为0.0

        # GNN模块
        self.gnn = ARMAConv(in_channels=1, out_channels=channels, num_stacks=1, num_layers=1, act=nn.GELU(), dropout=0.2)  # 图神经网络层，用于解码器部分
        # 重建模块
        self.recon_model = ReconstructionModel(window_size, hidden_size, 150, out_dim, 1,0.2)
        # self.cell1 = LSTMCell(self.num_nodes * channels, hidden_size)  # LSTM单元，用于解码器部分
        # self.cell2 = LSTMCell(hidden_size, hidden_size)

        # linear prediction layer
        # self.pred = nn.Linear(hidden_size, self.num_nodes)

        # cached offsets for batch stacking for each batch_size and number of edges 用于批处理堆叠的缓存
        self.batch_edge_offset_cache = {}

        # initial graph
        self._edge_index, self.edge_attr, self.A = self.graph_embedding()

    def get_graph(self):
        """返回学习到的图的邻接矩阵"""
        return self.graph_embedding.get_A()

    def get_embedding(self):
        """返回图嵌入的边索引列表和节点嵌入的权重"""
        return self.graph_embedding.get_E()

    def forward(self, window):
        # 这里传入的window的size是(batch_size,window_size,num_nodes)
        batch_size, window_size, num_nodes = window.shape
        window = window.permute(0, 2, 1).reshape(batch_size * num_nodes, window_size) # 此时window.shape为(160,100)即(batch_size * num_nodes, window_size)
        # batch stacked window; input shape: [num_nodes*batch_size, lags]
        N = self.num_nodes  # number of nodes
        T = self.lags  # number of input time steps
        B = window.size(0) // N  # batch size

        # get learned graph representation
        edge_index, edge_attr, _ = self.graph_embedding() #  在数据通道变量数为5的情况下：edge_index.shape(2,25),edge_attr.shape=(25,) (双向图)
        _, W = self.get_embedding()
        W = W.pop()  # w.shape = (通道变量数,embed_dim)

        # batching works by stacking graphs; creates a mega graph with disjointed subgraphs
        # for each input sample. E.g. for a batch of B inputs with 51 nodes each;
        # samples i in {0, ..., B} => node indices [0...50], [51...101], [102...152], ... ,[51*B...50*51*B]
        # => node indices for sample i = [0, ..., num_nodes-1] + (i*num_nodes)
        num_edges = len(edge_attr)
        try:
            batch_offset = self.batch_edge_offset_cache[(B, num_edges)]
        except:
            batch_offset = torch.arange(0, N * B, N).view(1, B, 1).expand(2, B, num_edges).flatten(1, -1).to(
                self.device)
            self.batch_edge_offset_cache[(B, num_edges)] = batch_offset
        # repeat edge indices B times and add i*num_nodes where i is the input index
        batched_edge_index = edge_index.unsqueeze(1).expand(2, B, -1).flatten(1,
                                                                              -1) + batch_offset  # batched_edge_index:(2,2592) 81*32=2592 即(2,batch_size*num_edges)
        # repeat edge weights B times
        batched_edge_attr = edge_attr.unsqueeze(0).expand(B, -1).flatten()

        # add node feature dimension to input
        x = window.unsqueeze(-1)  # (B*N, T, 1)

        ### ENCODER
        # GNN layer; batch stacked output with C feature channels for each time step
        x = self.tgconv(x, batched_edge_index, batched_edge_attr)  # (B*N, T, C) (160,100,32)
        x = x.view(B, N, T, -1).permute(0, 2, 1, 3).contiguous()  # -> (B, T, N, C) (32,100,5,32)即(batch_size,window_size,num_nodes,32)
        # add node embeddings to feature vector as node positional embeddings
        x = x + W  # (B, T, N, C) + (N, C)
        # concatenate node features for LSTM input
        x = x.view(B, T, -1)  # -> (B, T, N*C) (32,100,160)
        # GRU layer
        output,h_end = self.gru(x) # h_end.shape = (b,hidden_size = 512) # out_put.shape = (32,100,512)即(batch_size,window_size,hidden_size = 512)
        h_end = h_end.view(x.shape[0], -1) # h_end.shape=(32,512)
        recons = self.recon_model(h_end)  # recons.shape (b,n,out_dim)  # 注：这里的 out_dim 应该就是多元时序数据的通道变量数

        return recons

        # # get hidden and cell states for each layer
        # h1 = h_n[0, ...].squeeze(0)  # (32,512)
        # h2 = h_n[1, ...].squeeze(0)  # (32,512)
        # c1 = h_n[0, ...].squeeze(0)  # (32,512)
        # c2 = h_n[1, ...].squeeze(0)  # (32,512)
        #
        # # TODO: try attention on h
        #
        # ### DECODER
        # predictions = []
        # # if prediction horizon > 1, iterate through decoder LSTM step by step
        # for _ in range(self.horizon - 1):
        #     # single decoder step per loop iteration
        #     pred = self.pred(h2).view(-1, 1)
        #     predictions.append(pred)
        #
        #     # GNN layer analogous to encoder without time dimension
        #     x = self.gnn(pred, batched_edge_index, batched_edge_attr)
        #     x = x.view(B, N, -1) + W
        #     x = x.view(B, -1)
        #     # LSTM layer 1
        #     h1, c1 = self.cell1(x, (h1, c1))
        #     h1 = F.dropout(h1, 0.2)
        #     c1 = F.dropout(c1, 0.2)
        #     # LSTM layer 2
        #     h2, c2 = self.cell2(h1, (h2, c2))
        # # final prediction
        # pred = self.pred(h2).view(-1, 1)  # (864,1)即(32*27,1)
        # predictions.append(pred)

        # return torch.cat(predictions, dim=1)


class SingleEmbedding(nn.Module):
    r''' Layer for graph representation learning
    using a linear embedding layer and cosine similarity
    to produce an index list of edges for a fixed number of
    neighbors for each node.

    Args:
        num_nodes (int): Number of nodes.
        embed_dim (int): Dimension of embedding.
        topk (int, optional): Number of neighbors per node.
        warmup_epochs : 预热期的轮数,默认为20
    '''

    def __init__(self, num_nodes, embed_dim, topk=15, warmup_epochs=20):
        super().__init__()

        self.device = get_device()

        self.topk = topk
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes

        self.embedding = nn.Embedding(num_nodes, embed_dim)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

        self._A = None  # 邻接矩阵，初始时设为None
        self._edges = None  # 边的索引列表，初始时设为None

        ### pre-computed index matrices  用于索引和操作邻接矩阵的预计算矩阵
        # square matrix for adjacency matrix indexing
        self._edge_indices = torch.arange(num_nodes).to(self.device).expand(num_nodes,
                                                                            num_nodes)  # self._edge_indices.shape = (27,27) [[0,1,2,...,26],[0,1,2,...,26]....]
        # matrix containing column indices for the right side of a matrix - will be used to remove all but topk entries
        self._i = torch.arange(self.num_nodes).unsqueeze(1).expand(self.num_nodes, self.num_nodes - self.topk).flatten()

        # fully connected graph 全连接图的边索引
        self._fc_edge_indices = torch.stack([self._edge_indices.T.flatten(), self._edge_indices.flatten()],
                                            dim=0)  # tensor([[ 0,  0,  0,  ..., 26, 26, 26], [ 0,  1,  2,  ..., 24, 25, 26]])

        self.warmup_counter = 0
        self.warmup_durantion = warmup_epochs

    def get_A(self):
        """返回邻接矩阵 self._A，如果尚未计算，则先调用 forward 方法进行计算"""
        if self._A is None:
            self.forward()
        return self._A

    def get_E(self):
        """返回边的索引列表 self._edges 和节点嵌入的权重，如果尚未计算，则先调用 forward 方法进行计算"""
        if self._edges is None:
            self.forward()
        return self._edges, [self.embedding.weight.clone()]

    def forward(self):
        W = self.embedding.weight.clone()  # row vector represents sensor embedding # W.shape:(num_nodes,embed_dim)

        eps = 1e-8  # avoid division by 0 避免被0整除
        W_norm = W / torch.clamp(W.norm(dim=1)[:, None], min=eps)  # None 用于增加一个新的轴 ; 归一化 W 以得到单位长度的嵌入向量 W_norm
        A = W_norm @ W_norm.t()  # 矩阵乘法 A.shape (通道变量数,通道变量数)即(num_nodes,num_nodes) # 计算归一化嵌入向量之间的余弦相似度矩阵 A

        # remove self loops
        A.fill_diagonal_(0)  # 将对角线上的元素填充为0

        # remove negative scores
        A = A.clamp(0)  # 小于0的值设置为0
        # edge_attr 表示边的属性，具体来说是边的权重。这些权重是根据节点嵌入之间的余弦相似度计算得到的
        if self.warmup_counter < self.warmup_durantion:
            edge_indices = self._fc_edge_indices
            edge_attr = A.flatten()  # 在预热期内,edge_attr 是邻接矩阵 A 扁平化后的所有值，表示完全连接图中所有边的权重

            self.warmup_counter += 1  # 2024/1/12 看到这里l
        else:
            '''在预热期之后，只保留每个节点的 topk 个最大相似度的邻居，并相应地更新邻接矩阵 A 和边索引 edge_indices。'''
            # topk entries
            _, topk_idx = A.sort(descending=True)

            j = topk_idx[:, self.topk:].flatten()
            A[self._i, j] = 0

            # # row degree
            # row_degree = A.sum(1).view(-1, 1) + 1e-8 # column vector
            # col_degree = A.sum(0) + 1e-8 # row vector

            # # normalized adjacency matrix
            # A /= torch.sqrt(row_degree)
            # A /= torch.sqrt(col_degree)

            msk = A > 0  # boolean mask

            edge_idx_src = self._edge_indices.T[msk]  # source edge indices
            edge_idx_dst = self._edge_indices[msk]  # target edge indices
            edge_attr = A[
                msk].flatten()  # edge weights 在预热期之后，edge_attr 只包含每个节点的 topk 个最相似邻居的边权重，这些权重是从修剪后的邻接矩阵 A 中提取的，其中只保留了每个节点的 topk 个最大相似度值

            # shape [2, topk*num_nodes] tensor holding topk edge-index-pairs for each node
            edge_indices = torch.stack([edge_idx_src, edge_idx_dst], dim=0)

        # save for later
        self._A = A
        self._edges = edge_indices

        return edge_indices, edge_attr, A


class TGConv(nn.Module):
    r'''
    Parallel graph convolution for multiple time steps.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        p (float): Dropout value between 0 and 1
    '''

    def __init__(self, in_channels: int, out_channels: int, p: float = 0.0):
        super(TGConv, self).__init__()

        self.device = get_device()

        self.graph_conv = ARMAConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_stacks=1,
            num_layers=1,
            act=nn.GELU(),
            dropout=p,
        )

        # cached offsets for temporal batch stacking for each batch_size and number of edges
        self.batch_edge_offset_cache = {}

    def forward(self, x: torch.FloatTensor, edge_index: torch.FloatTensor,
                edge_attr: torch.FloatTensor = None) -> torch.FloatTensor:
        '''
        Forward pass through temporal convolution block.

        Input data of shape: (batch, time_steps, in_channels).
        Output data of shape: (batch, time_steps, out_channels).
        '''

        # input dims
        BN, T, C = x.shape  # (batch*nodes, time, in_channels)
        N = edge_index.max().item() + 1  # number of nodes in the batch stack

        # batch stacking the temporal dimension to create a mega giga graph consisting of batched temporally-stacked graphs
        # analogous to batch stacking in main GNN, see description there.
        x = x.contiguous().view(-1, C)  # (B*N*T, C)

        # create temporal batch edge and weight lists
        num_edges = len(edge_attr)
        try:
            batch_offset = self.batch_edge_offset_cache[(BN, num_edges)]
        except:
            batch_offset = torch.arange(0, BN * T, N).view(1, T, 1).expand(2, T, num_edges).flatten(1, -1).to(x.device)
            self.batch_edge_offset_cache[(BN, num_edges)] = batch_offset
        # repeat edge indices T times and add offset for the edge indices
        temporal_batched_edge_index = edge_index.unsqueeze(1).expand(2, T, -1).flatten(1, -1) + batch_offset
        # repeat edge weights T times
        temporal_batched_edge_attr = edge_attr.unsqueeze(0).expand(T, -1).flatten()

        # GNN with C output channels
        x = self.graph_conv(x, temporal_batched_edge_index, temporal_batched_edge_attr)  # (B*N*T, C)
        x = x.view(BN, T, -1)  # -> (B*N, T, C)

        return x

class ReconstructionModel(nn.Module):
    """重建模块
    :param window_size: length of the input sequence
    :param in_dim: number of input features  就是 gru_hid_dim
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN    就是 recon_hid_dim=150
    :param out_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x   # h_end.shape (b,gru_hid_dim)
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)  # h_end_rep.shape (b,n,gru_hid_dim)

        decoder_out = self.decoder(h_end_rep)  # decoder.shape (b,n,recon_hid_dim=150)
        out = self.fc(decoder_out)   # out.shape (b,n,out_dim)
        return out

class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features 输入特征的数量
    :param n_layers: number of layers in
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out