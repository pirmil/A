import os
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import Dataset 
from torch_geometric.data import Data
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
import pandas as pd
from torch_geometric.data import DataLoader

class GraphTextDataset(Dataset):
    def __init__(self, root, gt: dict, split, tokenizer=None, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.tokenizer = tokenizer
        self.description = pd.read_csv(os.path.join(self.root, split+'.tsv'), sep='\t', header=None)   
        self.description = self.description.set_index(0).to_dict()
        self.cids = list(self.description[1].keys())
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphTextDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return [f'data_{cid}.pt' for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed/', self.split)

    def download(self):
        pass
        
    def process_graph(self, raw_path):
      edge_index  = []
      x = []
      with open(raw_path, 'r') as f:
        next(f)
        for line in f: 
          if line != "\n":
            edge = *map(int, line.split()), 
            edge_index.append(edge)
          else:
            break
        next(f)
        for line in f: #get mol2vec features:
          substruct_id = line.strip().split()[-1]
          if substruct_id in self.gt.keys():
            x.append(self.gt[substruct_id])
          else:
            x.append(self.gt['UNK'])
        return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0        
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            text_input = self.tokenizer([self.description[1][cid]],
                                   return_tensors="pt", 
                                   truncation=True, 
                                   max_length=256,
                                   padding="max_length",
                                   add_special_tokens=True,)
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index, input_ids=text_input['input_ids'], attention_mask=text_input['attention_mask'])

            torch.save(data, osp.join(self.processed_dir, f'data_{cid}.pt'))
            i += 1

    def plot_graph_with_text(self, limit=None):
        print(f"There are {len(self.raw_paths)} molecules.")
        occ_per_cc = {}
        if limit is None:
            limit = len(self.raw_paths)
        for i, raw_path in enumerate(self.raw_paths):
            cid = int(raw_path.split('/')[-1][:-6])
            edge_index, x = self.process_graph(raw_path)
            plot_graph_with_text_save(edge_index, self.description[1][cid], occ_per_cc)
            if i>=limit:
                break
        print(occ_per_cc)

    def get_stats(self):
        occ_per_edges = {}
        occ_per_nodes = {}
        occ_per_cc = {}
        for i, raw_path in enumerate(self.raw_paths):
            cid = int(raw_path.split('/')[-1][:-6])
            edge_index, x = self.process_graph(raw_path)
            update_stats(edge_index, occ_per_edges, occ_per_cc, occ_per_nodes)
        print(f"cc {occ_per_cc}")
        print(f"nodes {occ_per_nodes}")
        print(f"edges {occ_per_edges}")
        

def update_stats(edge_index, occ_per_edges, occ_per_cc, occ_per_nodes):
    import networkx as nx
    G = nx.Graph()

    for edge in edge_index.t().tolist():
        G.add_edge(edge[0], edge[1])

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_connected_components = nx.number_connected_components(G)

    if num_connected_components in occ_per_cc:
        occ_per_cc[num_connected_components] += 1
    else:
        occ_per_cc[num_connected_components] = 1

    if num_edges in occ_per_edges:
        occ_per_edges[num_edges] += 1
    else:
        occ_per_edges[num_edges] = 1

    if num_nodes in occ_per_nodes:
        occ_per_nodes[num_nodes] += 1
    else:
        occ_per_nodes[num_nodes] = 1

def plot_graph_with_text_save(edge_index, description_text, occ_per_cc, save_path='./images'):
    import os
    import networkx as nx
    import matplotlib.pyplot as plt

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    G = nx.Graph()

    for edge in edge_index.t().tolist():
        G.add_edge(edge[0], edge[1])

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_connected_components = nx.number_connected_components(G)

    if num_connected_components in occ_per_cc:
        occ_per_cc[num_connected_components] += 1
    else:
        occ_per_cc[num_connected_components] = 1

    if occ_per_cc[num_connected_components] <= 2:
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, font_weight='bold')

        image_filename = f"{save_path}/{num_connected_components}_{num_edges}_{num_nodes}_{description_text[:10]}.png"
        plt.savefig(image_filename)
        plt.close()

        log_filename = f"{save_path}/{num_connected_components}_{num_edges}_{num_nodes}_{description_text[:10]}.log"
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Description: {description_text}\n")
            log_file.write(f"Number of Nodes: {num_nodes}\n")
            log_file.write(f"Number of Edges: {num_edges}\n")
            log_file.write(f"Number of Connected Components: {num_connected_components}")

        print(f"Graph image saved at: {image_filename}")
        print(f"Description, nodes, edges, and connected components saved at: {log_filename}")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{self.idx_to_cid[idx]}.pt'))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, f'data_{cid}.pt'))
        return data
    
    
class GraphDataset(Dataset):
    def __init__(self, root, gt: dict, split, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.description = pd.read_csv(os.path.join(self.root, split+'.txt'), sep='\t', header=None)
        self.cids = self.description[0].tolist()
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return [f'data_{cid}.pt' for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed/', self.split)

    def download(self):
        pass
        
    def process_graph(self, raw_path):
      edge_index  = []
      x = []
      with open(raw_path, 'r') as f:
        next(f)
        for line in f: 
          if line != "\n":
            edge = *map(int, line.split()), 
            edge_index.append(edge)
          else:
            break
        next(f)
        for line in f:
          substruct_id = line.strip().split()[-1]
          if substruct_id in self.gt.keys():
            x.append(self.gt[substruct_id])
          else:
            x.append(self.gt['UNK'])
        return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0        
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index)
            torch.save(data, osp.join(self.processed_dir, f'data_{cid}.pt'))
            i += 1
            
    def plot_graph_with_text(self, limit=None):
        print(f"There are {len(self.raw_paths)} molecules.")
        occ_per_cc = {}
        if limit is None:
            limit = len(self.raw_paths)
        for i, raw_path in enumerate(self.raw_paths):
            edge_index, x = self.process_graph(raw_path)
            plot_graph_with_text_save(edge_index, f'test{i}', occ_per_cc)
            if i>=limit:
                break
        print(occ_per_cc)
    def get_stats(self):
        occ_per_edges = {}
        occ_per_nodes = {}
        occ_per_cc = {}
        for i, raw_path in enumerate(self.raw_paths):
            cid = int(raw_path.split('/')[-1][:-6])
            edge_index, x = self.process_graph(raw_path)
            update_stats(edge_index, occ_per_edges, occ_per_cc, occ_per_nodes)
        print(f"cc {occ_per_cc}")
        print(f"nodes {occ_per_nodes}")
        print(f"edges {occ_per_edges}")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{self.idx_to_cid[idx]}.pt'))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, f'data_{cid}.pt'))
        return data
    
    def get_idx_to_cid(self):
        return self.idx_to_cid
    
class TextDataset(TorchDataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = self.load_sentences(file_path)

    def load_sentences(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def get_dataset_and_dataloaders(tokenizer, path='./data', num_workers=8, batch_size=100):
    path_gt = os.path.join(path, 'token_embedding_dict.npy')
    gt = np.load(path_gt, allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root=path, gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root=path, gt=gt, split='train', tokenizer=tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

    idx_to_cid = test_cids_dataset.get_idx_to_cid()

    test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dataset, val_dataset, test_cids_dataset, test_text_dataset, train_loader, val_loader, test_loader, test_text_loader, idx_to_cid