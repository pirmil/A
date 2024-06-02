from dataload.dataloader import get_dataset_and_dataloaders
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from torch_geometric.utils import degree
import torch
import numpy as np

def get_degree_histogram(dataloader_list):
    degree_list = []
    for loader in dataloader_list:
        for data in loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long) #1 for in degree (should be equivalent here?)
            degree_list.append(d)
    np_degree_list = torch.cat(degree_list).numpy()
    max_degree = np.max(np_degree_list)
    hist = np.histogram(np_degree_list, bins=max_degree)
    return hist[0].astype(int), hist[1].astype(int)