from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from torch import nn
import torch
import os
os.environ['TRANSFORMERS_CACHE'] = '/data/.cache'

class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        try:
            self.model = AutoModel.from_pretrained(model_name)
        except:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.model(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        return encoded_text.last_hidden_state[:,0,:]
    
class FineTextEncoder(nn.Module):
    def __init__(self, model, nout, gamma, depth_mlp=1, learnable_gamma=False):
        super(FineTextEncoder, self).__init__()
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
    def forward(self, input_ids, attention_mask):
        output_text = self.model(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        return self.ln(output_text + self.gamma*self.mlp(output_text))
    
class FineTextEncoderV2(nn.Module):
    def __init__(self, model, nout, gamma, depth_mlp=1, learnable_gamma=False):
        super(FineTextEncoderV2, self).__init__()
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
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            output_text = self.model(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        return self.ln(output_text + self.gamma*self.mlp(output_text))
    

def get_text_encoder_and_tokenizer(model_name, tokenizer_name= None, fineclip=False, fineclipv2 = False, nout=768, gamma=1e-4, depth_mlp=1, learnable_gamma=False, initial_state_dict=None):
    if fineclip:
        if tokenizer_name is None:
            tokenizer_name = model_name
        if initial_state_dict is None:
            return FineTextEncoder(TextEncoder(model_name), nout, gamma, depth_mlp=depth_mlp, learnable_gamma=learnable_gamma), AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            initial_model = TextEncoder(model_name)
            initial_model.load_state_dict(initial_state_dict)
            for param in initial_model.parameters():
                param.requires_grad = False
            model = FineTextEncoder(TextEncoder(model_name), nout, gamma, depth_mlp=depth_mlp, learnable_gamma=learnable_gamma)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            return model, tokenizer
    elif fineclipv2:
        if tokenizer_name is None:
            tokenizer_name = model_name
        if initial_state_dict is None:
            print('initial_state_dict is None, initializing model randomly')
            return FineTextEncoderV2(TextEncoder(model_name), nout, gamma, depth_mlp=depth_mlp, learnable_gamma=learnable_gamma), AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            print('initial_state_dict is not None, initializing model from initial_state_dict')
            initial_model = TextEncoder(model_name)
            initial_model.load_state_dict(initial_state_dict)
            for param in initial_model.parameters():
                param.requires_grad = False
            model = FineTextEncoderV2(TextEncoder(model_name), nout, gamma, depth_mlp=depth_mlp, learnable_gamma=learnable_gamma)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            return model, tokenizer
    else:
        if tokenizer_name is None:
            tokenizer_name = model_name
        return TextEncoder(model_name), AutoTokenizer.from_pretrained(tokenizer_name)