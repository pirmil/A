import torch

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

def get_criterion(criterion_name):
    if criterion_name == 'contrastive':
        return contrastive_loss
    else:
        raise ValueError(f'Unknown criterion: {criterion_name}' )