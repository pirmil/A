from torch import nn
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CLIP(nn.Module):
    def __init__(self, text_model, graph_model):
        super(CLIP, self).__init__()
        self.text_encoder= text_model
        self.graph_encoder = graph_model
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    def get_text_encoder(self):
        return self.text_encoder
    def get_graph_encoder(self):
        return self.graph_encoder
    
class CLIP_temp(nn.Module):
    def __init__(self, text_model, graph_model, temp, learnable_temp=False):
        super(CLIP_temp, self).__init__()
        self.text_encoder= text_model
        self.graph_encoder = graph_model
        if temp is None:
            self.temp = torch.ones(1)*0.07 # Optimal temperature from original CLIP paper
        elif isinstance(temp, float):
            self.temp = torch.ones(1)*temp
        else:
            self.temp = temp
        self.temp = nn.Parameter(self.temp)
        if not learnable_temp:
            self.temp.requires_grad = False
        
        self.temp_min=0.01
        self.temp_max=10
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded/torch.sqrt(self.temp), text_encoded/torch.sqrt(self.temp)
    def get_text_encoder(self):
        return self.text_encoder
    def get_graph_encoder(self):
        return self.graph_encoder
    
# class FineCLIP(nn.Module):
#     def __init__(self, text_model, graph_model, temp, gamma, nout, depth_mlp=1):
#         super(FineCLIP, self).__init__()
#         self.text_encoder= text_model
#         self.graph_encoder = graph_model
#         if temp is None:
#             self.temp = torch.ones(1)*0.07 # Optimal temperature from original CLIP paper
#         elif isinstance(temp, float):
#             self.temp = torch.ones(1)*temp
#         else:
#             self.temp = temp
#         self.temp_min=0.01
#         self.temp_max=10
#         self.gamma = gamma
#         module_list_mlptext = []
#         module_list_mlpgraph = []
#         for i in range(depth_mlp):
#             module_list_mlptext.append(nn.Linear(nout, nout))
#             module_list_mlpgraph.append(nn.Linear(nout, nout))
#             module_list_mlptext.append(nn.ReLU())
#             module_list_mlpgraph.append(nn.ReLU())
#         module_list_mlpgraph.append(nn.Linear(nout, nout))
#         module_list_mlptext.append(nn.Linear(nout, nout))
#         self.mlptext = nn.Sequential(*module_list_mlptext)
#         self.mlpgraph = nn.Sequential(*module_list_mlpgraph)

# class FineCLIP_temp(nn.Module):
#     def __init__(self, text_model, graph_model, temp, gamma=1e-4, nout=768, learnable_temp=False):
#         super(FineCLIP_temp, self).__init__()
#         self.text_encoder= text_model
#         self.graph_encoder = graph_model
#         if temp is None:
#             self.temp = torch.ones(1)*0.07 # Optimal temperature from original CLIP paper
#         elif isinstance(temp, float):
#             self.temp = torch.ones(1)*temp
#         else:
#             self.temp = temp
#         self.temp = nn.Parameter(self.temp).to(device=device) if learnable_temp else nn.Parameter(self.temp.to(device=device), requires_grad=False)
#         self.temp_min=0.01
#         self.temp_max=10
#         self.gamma = nn.Parameter(torch.ones(nout) * gamma).to(device=device)
#         self.fcs_text = nn.Sequential(
#                 nn.Linear(nout, nout),
#                 nn.GELU(),
#                 nn.Linear(nout, nout)
#         )
#         self.fcs_graph = nn.Sequential(
#             nn.Linear(nout, nout),
#             nn.GELU(),
#             nn.Linear(nout, nout)
#         )
#     def forward(self, graph_batch, input_ids, attention_mask):
#         graph_encoded = torch.div(self.graph_encoder(graph_batch), torch.sqrt(self.temp))
#         text_encoded = torch.div(self.text_encoder(input_ids, attention_mask), torch.sqrt(self.temp))
#         return graph_encoded + self.gamma * self.fcs_graph(graph_encoded), text_encoded + self.gamma * self.fcs_text(text_encoded)
#     def get_text_encoder(self):
#         return self.text_encoder
#     def get_graph_encoder(self):
#         return self.graph_encoder
    
def get_clip(clip_type, text_model, graph_model, temp=None, learnable_temp=False, gamma=1e-4, nout=768, depth_mlp=1, aux_config_file=None):
    if clip_type == 'CLIP':
        return CLIP(text_model, graph_model)
    elif clip_type == 'CLIP_temp':
        return CLIP_temp(text_model, graph_model, temp, learnable_temp)
    # elif clip_type == 'FineCLIP':
    #     return FineCLIP(text_model, graph_model, temp, gamma, nout, depth_mlp, depth_mlp, aux_config_file)
    # elif clip_type == 'FineCLIP_temp':
    #     return FineCLIP_temp(text_model, graph_model, temp, gamma, nout, learnable_temp, depth_mlp, aux_config_file)
    else:
        raise ValueError('clip_type not recognized')
    

