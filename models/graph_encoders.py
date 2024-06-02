import torch
from torch import nn
import numpy as np
from torch_geometric.nn.conv.transformer_conv import TransformerConv
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, PNAConv, global_mean_pool
from torch_geometric.nn.models import GAT, GIN, PNA, GraphSAGE, MLP
from torch_geometric.nn.models.graph_unet import GraphUNet as GUNet

class GCN(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GCN, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class GraphResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(GraphResBlock, self).__init__()
        self.conv1 = GCNConv(channel_in, channel_in + (channel_out - channel_in)//2)
        self.conv2 = GCNConv(channel_in + (channel_out - channel_in)//2, channel_out)
        self.conv3 = GCNConv(channel_in, channel_out)
        self.bn1 = nn.BatchNorm1d(channel_in + (channel_out - channel_in)//2)
        self.bn2 = nn.BatchNorm1d(channel_out)
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = h.relu()
        x = self.conv3(x, edge_index)
        return h.relu() + x
    
class GraphResNet(nn.Module):
    def __init__(self, num_node_features, num_blocks, block_channels, nout, nhid):
        super(GraphResNet, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, block_channels[0])
        self.res_blocks = nn.ModuleList([GraphResBlock(block_channels[i], block_channels[i+1]) for i in range(num_blocks)])
        self.mol_hidden1 = nn.Linear(block_channels[-1], nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
    
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        for block in self.res_blocks:
            x = block(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class GINEncoder(nn.Module):
    def __init__(self, num_node_features, num_blocks, hidden_channels, nout, nhid, dropout):
        super(GINEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.gin = GIN(in_channels = num_node_features, hidden_channels=hidden_channels, out_channels=nhid, num_layers=num_blocks, dropout=dropout)
        self.mol_hidden1 = nn.Linear(nhid, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.gin(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class ResGINBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(ResGINBlock, self).__init__()
        self.ginconv1 = self.init_conv(in_channels, out_channels)
        self.ginconv2 = self.init_conv(out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, edge_index):
        h = self.ginconv1(x, edge_index)
        h = self.bn1(h)
        h = h.relu()
        h = self.dropout(h)
        h = self.ginconv2(h, edge_index)
        h = self.bn2(h)
        h = h.relu()
        return (h + x)/np.sqrt(2) # Residual connection with normalization (better for stability?)
    def init_conv(self, in_channels, out_channels):
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act='relu',
            act_first=False,
            norm=None,
            norm_kwargs=None,
        )
        return GINConv(mlp)


class GINEncoderV2(nn.Module):
    def __init__(self, num_node_features, num_blocks, hidden_channels, nout, nhid, dropout=0.0):
        super(GINEncoderV2, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        if type(hidden_channels)==int:
            # First block careful about input channels
            self.ginres_blocks = torch.nn.ModuleList([ResGINBlock(num_node_features, hidden_channels, dropout)])
            # Other blocks
            for _ in range(num_blocks-1):
                self.ginres_blocks.append(ResGINBlock(hidden_channels, hidden_channels, dropout))
            self.mol_hidden1 = nn.Linear(hidden_channels, nhid)
        else:
            # First block careful about input channels
            self.ginres_blocks = torch.nn.ModuleList([ResGINBlock(num_node_features, hidden_channels[0], dropout)])
            # Other blocks
            for i in range(num_blocks-1):
                self.ginres_blocks.append(ResGINBlock(hidden_channels[i], hidden_channels[i+1], dropout))
            self.mol_hidden1 = nn.Linear(hidden_channels[-1], nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        for block in self.ginres_blocks:
            x = block(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

class GraphSAGEEncoder(nn.Module):
    def __init__(self, num_node_features, num_blocks, hidden_channels, nout, nhid):
        super(GraphSAGEEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.sage = GraphSAGE(in_channels = num_node_features, hidden_channels=hidden_channels, out_channels=nhid, num_layers=num_blocks)
        self.mol_hidden1 = nn.Linear(nhid, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.sage(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

class GraphPNAEncoder(nn.Module):
    def __init__(self, num_node_features, num_blocks, hidden_channels, nout, nhid, deg=None):
        super(GraphPNAEncoder, self).__init__()
        if deg is None:
            raise ValueError('Degree histogram is required for PNA')
        self.nhid = nhid
        self.nout = nout
        self.deg = deg
        self.deg.requires_grad = False
        self.aggregators = ['mean', 'min', 'max', 'std']
        self.scalers = ['identity', 'amplification', 'attenuation']
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.relu = nn.ReLU()
        self.pna = PNA(in_channels = num_node_features, hidden_channels=hidden_channels, out_channels=nhid, num_layers=num_blocks, aggregators=aggregators, scalers=scalers, deg=deg)
        self.mol_hidden1 = nn.Linear(nhid, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.pna(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

class ResPNABlock(nn.Module):
    def __init__(self, in_channels, out_channels, aggregators, scalers, deg):
        super(ResPNABlock, self).__init__()
        self.pnaconv1 = PNAConv(in_channels, out_channels, aggregators, scalers, deg)
        self.pnaconv2 = PNAConv(out_channels, out_channels, aggregators, scalers, deg)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    def forward(self, x, edge_index):
        h = self.pnaconv1(x, edge_index)
        h = self.bn1(h)
        h = h.relu()
        h = self.pnaconv2(h, edge_index)
        h = self.bn2(h)
        h = h.relu()
        return (h + x)/np.sqrt(2) # Residual connection with normalization (better for stability?)
    
class GraphPNAEncoderV2(nn.Module):
    def __init__(self, num_node_features, num_blocks, hidden_channels, nout, nhid, deg=None):
        super(GraphPNAEncoderV2, self).__init__()
        if deg is None:
            raise ValueError('Degree histogram is required for PNA')
        self.nhid = nhid
        self.nout = nout
        self.deg = deg
        self.deg.requires_grad = False
        self.aggregators = ['mean', 'min', 'max', 'std']
        self.scalers = ['identity', 'amplification', 'attenuation']
        
        self.relu = nn.ReLU()

        if type(hidden_channels)==int:
            # First block careful about input channels
            self.pnares_blocks = torch.nn.ModuleList([ResPNABlock(num_node_features, hidden_channels, self.aggregators, self.scalers, self.deg)])
            # Other blocks
            for _ in range(num_blocks-1):
                self.pnares_blocks.append(ResPNABlock(hidden_channels, hidden_channels, self.aggregators, self.scalers, self.deg))
            self.mol_hidden1 = nn.Linear(hidden_channels, nhid)
        else:
            # First block careful about input channels
            self.pnares_blocks = torch.nn.ModuleList([ResPNABlock(num_node_features, hidden_channels[0], self.aggregators, self.scalers, self.deg)])
            # Other blocks
            for i in range(num_blocks-1):
                self.pnares_blocks.append(ResPNABlock(hidden_channels[i], hidden_channels[i+1], self.aggregators, self.scalers, self.deg))
            self.mol_hidden1 = nn.Linear(hidden_channels[-1], nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        for block in self.pnares_blocks:
            x = block(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x


class GraphTransformer(nn.Module):
    def __init__(self, num_node_features, num_blocks, hidden_dim, nout, nhid, dropout=0.0, v2=False):
        super(GraphTransformer, self).__init__()
        self.transformer = GAT(in_channels = num_node_features, hidden_channels=hidden_dim, out_channels=hidden_dim, num_layers=num_blocks, dropout=dropout, v2=v2)
        self.mol_hidden1 = nn.Linear(hidden_dim, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.transformer(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class ResGATBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, heads, v2):
        super(ResGATBlock, self).__init__()
        self.heads = heads
        self.gatconv1 = self.init_conv(in_channels, out_channels, heads, v2, dropout)
        self.gatconv2 = self.init_conv(heads*out_channels, heads*out_channels, heads=1, v2=v2, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(heads*out_channels)
        self.bn2 = nn.BatchNorm1d(heads*out_channels)
        if heads > 1:
            self.residual_projection = nn.Linear(in_channels, heads*out_channels)
            self.final_projection =nn.Linear(heads*out_channels, out_channels)
    def forward(self, x, edge_index):
        h = self.gatconv1(x, edge_index)
        h = self.bn1(h)
        h = h.relu()
        h = self.gatconv2(h, edge_index)
        h = self.bn2(h)
        h = h.relu()
        if self.heads > 1:
            residual = self.residual_projection(x)
            return self.final_projection((h + residual)/np.sqrt(2))
        else:
            return (h + x)/np.sqrt(2) # Residual connection with normalization (better for stability?)
    def init_conv(self, in_channels, out_channels, heads, v2, dropout):
        if v2:
            conv = GATv2Conv(in_channels=in_channels, out_channels=out_channels, heads=heads, dropout=dropout)
        else:
            conv = GATConv(in_channels=in_channels, out_channels=out_channels, heads=heads, dropout=dropout)
        return conv
    
class GATEncoder(nn.Module):
    def __init__(self, num_node_features, num_blocks, hidden_channels, nout, nhid, dropout, heads, v2):
        super(GATEncoder, self).__init__()
        if type(hidden_channels)==int:
            # First block careful about input channels
            self.gatres_blocks = torch.nn.ModuleList([ResGATBlock(num_node_features, hidden_channels, dropout, heads, v2)])
            # Other blocks
            for _ in range(num_blocks-1):
                self.gatres_blocks.append(ResGATBlock(hidden_channels, hidden_channels, dropout, heads, v2))
            self.mol_hidden1 = nn.Linear(hidden_channels, nhid)
        else:
            # First block careful about input channels
            self.gatres_blocks = torch.nn.ModuleList([ResGATBlock(num_node_features, hidden_channels[0], dropout, heads, v2)])
            # Other blocks
            for i in range(num_blocks-1):
                self.gatres_blocks.append(ResGATBlock(hidden_channels[i], hidden_channels[i+1], dropout, heads, v2))
            self.mol_hidden1 = nn.Linear(hidden_channels[-1], nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        for block in self.gatres_blocks:
            x = block(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class ResTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, heads):
        super(ResTransformerBlock, self).__init__()
        self.transconv1 = TransformerConv(in_channels, out_channels, heads=heads, dropout=dropout)
        self.transconv2 = TransformerConv(heads*out_channels, out_channels, heads=1, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(heads*out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    def forward(self, x, edge_index):
        h = self.transconv1(x, edge_index)
        h = self.bn1(h)
        h = h.relu()
        h = self.transconv2(h, edge_index)
        h = self.bn2(h)
        h = h.relu()
        return (h + x)/np.sqrt(2) # Residual connection with normalization (better for stability?)
    
class RealTransformer(nn.Module):
    def __init__(self, num_node_features, num_blocks, hidden_channels, nout, nhid, dropout, heads):
        super(RealTransformer, self).__init__()
        if type(hidden_channels)==int:
            self.transres_blocks = torch.nn.ModuleList([ResTransformerBlock(num_node_features, hidden_channels, dropout, heads)])
            for _ in range(num_blocks-1):
                self.transres_blocks.append(ResTransformerBlock(hidden_channels, hidden_channels, dropout, heads))
            self.mol_hidden1 = nn.Linear(hidden_channels, nhid)
        else:
            self.transres_blocks = torch.nn.ModuleList([ResTransformerBlock(num_node_features, hidden_channels[0], dropout, heads)])
            for i in range(num_blocks-1):
                self.transres_blocks.append(ResTransformerBlock(hidden_channels[i], hidden_channels[i+1], dropout, heads))
            self.mol_hidden1 = nn.Linear(hidden_channels[-1], nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        for block in self.transres_blocks:
            x = block(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class ResHybridBlock(nn.Module):
    def __init__(self, convname1, convname2, in_channels, out_channels, dropout, heads, v2):
        super(ResHybridBlock, self).__init__()
        self.conv1 = self.init_conv(convname1, in_channels, out_channels, heads, v2, dropout)
        self.conv2 = self.init_conv(convname2, out_channels*heads, out_channels, heads, v2, dropout)
        self.bn1 = nn.BatchNorm1d(heads*out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = h.relu()
        return (h + x)/np.sqrt(2) # Residual connection with normalization (better for stability?)
    def init_conv(self, convname, in_channels, out_channels, heads, v2, dropout):
        if convname == 'GINConv':
            mlp = MLP(
                [in_channels, out_channels, out_channels],
                act='relu',
                act_first=False,
                norm=None,
                norm_kwargs=None,
            )
            return GINConv(mlp)
        elif 'GAT' in convname:
            if v2 or 'v2' in convname or 'V2' in convname:
                return GATv2Conv(in_channels=in_channels, out_channels=out_channels, heads=heads, dropout=dropout)
            else:
                return GATConv(in_channels=in_channels, out_channels=out_channels, heads=heads, dropout=dropout)
        elif convname == 'TransformerConv':
            return TransformerConv(in_channels, out_channels, heads=heads, dropout=dropout)
        elif convname == 'GCNConv':
            return GCNConv(in_channels, out_channels)
    
class HybridEncoder(nn.Module):
    def __init__(self, convname1, convname2, num_node_features, num_blocks, hidden_channels, nout, nhid, dropout, heads, v2):
        super(HybridEncoder, self).__init__()
        if type(hidden_channels)==int:
            self.hybridres_blocks = torch.nn.ModuleList([ResHybridBlock(convname1, convname2, num_node_features, hidden_channels, dropout, heads, v2)])
            for _ in range(num_blocks-1):
                self.hybridres_blocks.append(ResHybridBlock(convname1, convname2, hidden_channels, hidden_channels, dropout, heads, v2))
            self.mol_hidden1 = nn.Linear(hidden_channels, nhid)
        else:
            self.hybridres_blocks = torch.nn.ModuleList([ResHybridBlock(convname1, convname2, num_node_features, hidden_channels[0], dropout, heads, v2)])
            for i in range(num_blocks-1):
                self.hybridres_blocks.append(ResHybridBlock(convname1, convname2, hidden_channels[i], hidden_channels[i+1], dropout, heads, v2))
            self.mol_hidden1 = nn.Linear(hidden_channels[-1], nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        for block in self.hybridres_blocks:
            x = block(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x   
    
class GraphUnet(nn.Module):
    def __init__(self, num_node_features, hidden_dim, nout, nhid, depth_unet=2, pool_ratios=0.5):
        super(GraphUnet, self).__init__()
        self.graph_unet = GUNet(in_channels = num_node_features, hidden_channels=hidden_dim, out_channels=hidden_dim, depth=depth_unet, pool_ratios=pool_ratios)
        self.mol_hidden1 = nn.Linear(hidden_dim, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.graph_unet(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class FineGraphEncoder(nn.Module):
    def __init__(self, model, nout, gamma, depth_mlp=1, learnable_gamma=False):
        super(FineGraphEncoder, self).__init__()
        self.model = model
        self.nout = nout
        if type(gamma) == float:
            self.gamma = torch.ones(1)*gamma
        else:
            self.gamma = gamma
        self.gamma = nn.Parameter(self.gamma, requires_grad=learnable_gamma)
        self.depth_mlp = depth_mlp
        self.mlp = nn.Sequential(*[nn.Linear(self.nout, self.nout) for _ in range(self.depth_mlp)])
        self.ln = nn.LayerNorm((self.nout))
    def forward(self, graph_batch):
        output_graph = self.model(graph_batch)
        return self.ln(output_graph + self.gamma*self.mlp(output_graph))
    
class FineGraphEncoderV2(nn.Module):
    def __init__(self, model, nout, gamma, depth_mlp=1, learnable_gamma=False):
        super(FineGraphEncoderV2, self).__init__()
        self.model = model
        self.nout = nout
        if type(gamma) == float:
            self.gamma = torch.ones(1)*gamma
        else:
            self.gamma = gamma
        self.gamma = nn.Parameter(self.gamma, requires_grad=learnable_gamma)
        self.depth_mlp = depth_mlp
        self.mlp = nn.ModuleList([])
        for _ in range(self.depth_mlp):
            self.mlp.append(nn.Linear(self.nout, self.nout))
            self.mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*self.mlp)
        self.ln = nn.LayerNorm((self.nout))
    def forward(self, graph_batch):
        with torch.no_grad():
            output_graph = self.model(graph_batch)
        return self.ln(output_graph + self.gamma*self.mlp(output_graph))

    

def get_graph_encoder(encoder_type, num_node_features, num_blocks, block_channels, nout, nhid, dropout=0.0, fineclip=False, fineclipv2=False, gamma=1e-4, depth_mlp=1, learnable_gamma=False, initial_state_dict=None, depth_unet=2, deg=None, v2=True, heads=1, convname1='GATConv', convname2='GINConv'):
    if fineclip:
        initial_model = get_graph_encoder(encoder_type, num_node_features, num_blocks, block_channels, nout, nhid, dropout=dropout, depth_unet=depth_unet, deg=deg)
        if initial_state_dict is None:
            return FineGraphEncoder(initial_model, nout, gamma, depth_mlp=depth_mlp, learnable_gamma=learnable_gamma)
        else:
            initial_model.load_state_dict(initial_state_dict)
            for param in initial_model.parameters():
                param.requires_grad = False
            model = FineGraphEncoder(initial_model, nout, gamma, depth_mlp=depth_mlp, learnable_gamma=learnable_gamma)
            return model
    elif fineclipv2:
        initial_model = get_graph_encoder(encoder_type, num_node_features, num_blocks, block_channels, nout, nhid, dropout=dropout, depth_unet=depth_unet, deg=deg)
        if initial_state_dict is None:
            return FineGraphEncoderV2(initial_model, nout, gamma, depth_mlp=depth_mlp, learnable_gamma=learnable_gamma)
        else:
            initial_model.load_state_dict(initial_state_dict)
            for param in initial_model.parameters():
                param.requires_grad = False
            model = FineGraphEncoderV2(initial_model, nout, gamma, depth_mlp=depth_mlp, learnable_gamma=learnable_gamma)
            return model
    elif encoder_type == 'GCN':
        return GCN(num_node_features, nout, nhid, block_channels[0])
    elif encoder_type == 'GraphResNet':
        return GraphResNet(num_node_features, num_blocks, block_channels, nout, nhid)
    elif encoder_type == 'GraphTransformer':
        return GraphTransformer(num_node_features, num_blocks, block_channels[0], nout, nhid, dropout=dropout)
    elif encoder_type == 'GraphTransformerV2':
        return GraphTransformer(num_node_features, num_blocks, block_channels[0], nout, nhid, dropout=dropout, v2=True)
    elif encoder_type == 'ResGraphTransformer':
        return GATEncoder(num_node_features, num_blocks, block_channels, nout, nhid, dropout, heads, v2)
    elif encoder_type == 'RealTransformer':
        return RealTransformer(num_node_features, num_blocks, block_channels, nout, nhid, dropout, heads)
    elif encoder_type == 'GraphUnet':
        return GraphUnet(num_node_features, block_channels[0], nout, nhid, depth_unet, pool_ratios=0.5)
    elif encoder_type == 'GIN':
        return GINEncoder(num_node_features, num_blocks, block_channels[0], nout, nhid, dropout)
    elif encoder_type == 'GINV2':
        return GINEncoderV2(num_node_features, num_blocks, block_channels, nout, nhid, dropout)
    elif encoder_type == 'GraphSAGE':
        return GraphSAGEEncoder(num_node_features, num_blocks, block_channels[0], nout, nhid)
    elif encoder_type == 'GraphPNA':
        return GraphPNAEncoder(num_node_features, num_blocks, block_channels[0], nout, nhid, deg=deg)
    elif encoder_type == 'GraphPNAV2':
        return GraphPNAEncoderV2(num_node_features, num_blocks, block_channels, nout, nhid, deg=deg)
    elif encoder_type == 'HybridEncoder':
        return HybridEncoder(convname1, convname2, num_node_features, num_blocks, block_channels, nout, nhid, dropout, heads, v2)
    else:
        raise NotImplementedError
