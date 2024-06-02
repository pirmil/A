from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import torch
import numpy as np

def similarities(text_embeddings, graph_embeddings, metric='cosine'):
    if metric == 'cosine':
        similarity = cosine_similarity(text_embeddings, graph_embeddings)
    elif metric == 'euclidean':
        similarity = -euclidean_distances(text_embeddings, graph_embeddings)
    else:
        raise NotImplementedError
    return similarity

def compute_similarities(test_loader, test_text_loader, model, device, metric='cosine'):
    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()
    with torch.no_grad():
        graph_embeddings = []
        for batch in test_loader:
            for output in graph_model(batch.to(device)):
                graph_embeddings.append(output.tolist())

        text_embeddings = []
        for batch in test_text_loader:
            for output in text_model(batch['input_ids'].to(device), 
                                    attention_mask=batch['attention_mask'].to(device)):
                text_embeddings.append(output.tolist())

        #similarity = cosine_similarity(text_embeddings, graph_embeddings)
        similarity = similarities(text_embeddings, graph_embeddings, metric=metric)
    return similarity
    

def compute_autosimilarities(val_loader, model, device, metric='cosine', also_loss=False, criterion=None):
    # graph_model = model.get_graph_encoder()
    # text_model = model.get_text_encoder()
    val_loss = 0
    with torch.no_grad():
        graph_embeddings = []
        text_embeddings = []
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            batch.pop('input_ids')
            attention_mask = batch['attention_mask'].to(device)
            batch.pop('attention_mask')
            graph_batch = batch.to(device)
            x_graph, x_text = model(graph_batch, input_ids, attention_mask)
            current_loss = criterion(x_graph, x_text)   
            for output in x_graph:
                graph_embeddings.append(output.tolist())
            for output in x_text:
                text_embeddings.append(output.tolist())
            val_loss += current_loss.item()
        similarity = similarities(text_embeddings, graph_embeddings, metric=metric)
        val_loss /= len(val_loader)
    if also_loss:
        return similarity, val_loss
    return similarity

def compute_MRR(similarity, also_ij=False):
    """
    Compute MRR with respect to a baseline supposed to be the eye matrix
    """
    similarity = np.array(similarity)
    ranks_i = np.argsort(-similarity, axis=1)+1
    ranks_j = np.argsort(-similarity, axis=0)+1
    match_i = ranks_i == np.arange(1, ranks_i.shape[1]+1).reshape(-1, 1)
    match_j = ranks_j == np.arange(1, ranks_j.shape[0]+1).reshape(1, -1)
    weights_i = match_i * 1.0/np.arange(1, ranks_i.shape[1]+1)
    weights_j = match_j * 1.0/np.arange(1, ranks_j.shape[0]+1).reshape(-1, 1).repeat(ranks_j.shape[1], axis=1)
    weights_i = weights_i.sum(axis=1)
    weights_j = weights_j.sum(axis=0)
    mrr_i = np.mean(weights_i)
    mrr_j = np.mean(weights_j)
    mrr = (mrr_i+mrr_j)/2
    if also_ij:
        return mrr, mrr_i, mrr_j
    return mrr

