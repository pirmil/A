import torch
import tqdm
from metrics.similarity import compute_autosimilarities, compute_MRR
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ema_constant_print_loss = 0.95


def train(model, train_loader, num_epochs, optimizer, criterion, scheduler=None, val_loader=None, save_path=None, device=device, similarity_metric=None, best_validation_loss=None, best_mrr=None, callback_on_validation=None, callback_args=None, print_progress=False):
    loss = None
    losses = []
    count_iter = 0
    for epoch in range(num_epochs): ## Everything included in scheduler and callbacks
        model.train()
        first_epoch = epoch == 0
        progress_bar = tqdm.tqdm(total=len(train_loader), disable=not (first_epoch and not print_progress))
        for batch in train_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
            current_loss = criterion(x_graph, x_text)   
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            ## If temperature is learnable clip it to allowed values
            if hasattr(model, 'temp'):
                model.temp.data = torch.clamp(model.temp.data, model.temp_min, model.temp_max)
            if loss is None:
                loss = current_loss.item()
            else:
                loss = ema_constant_print_loss*loss + (1-ema_constant_print_loss)*current_loss.item()
            count_iter += 1
            logs = {'Iter' : count_iter, 'Loss': loss}
            progress_bar.set_postfix(logs)
            losses.append(current_loss.item())
            if scheduler is not None:
                scheduler.step()
            progress_bar.update(1)
        progress_bar.close()
        if val_loader is not None:
            model.eval()
            if callback_on_validation is not None:
                best_mrr, best_validation_loss = callback_on_validation(model, val_loader, device, criterion, similarity_metric, optimizer, scheduler, epoch, loss, save_path, best_mrr=best_mrr, best_validation_loss=best_validation_loss,)
            else:
                best_mrr, best_validation_loss = callback_mrr(model, val_loader, device, criterion, similarity_metric, optimizer, scheduler, epoch, loss, save_path=save_path, best_mrr=best_mrr, best_validation_loss=best_validation_loss)
    return losses

def plot_graphs(dataset):
    dataset.plot_graph_with_text()

def get_stats(dataset):
    dataset.get_stats()

def callback_mrr(model, val_loader, device, criterion, similarity_metric, optimizer, scheduler, epoch, loss, save_path, best_mrr = None, best_validation_loss=None, select='mrr'):
    similarity, val_loss = compute_autosimilarities(val_loader, model, device, metric = similarity_metric, also_loss=True, criterion=criterion)
    mrr, mrr_i, mrr_j = compute_MRR(similarity, also_ij=True)
    
    if hasattr(model, 'temp') and model.temp.requires_grad:
        print(f'Epoch {epoch} MRR: {mrr:.4f} MRR_i: {mrr_i:.4f} MRR_j: {mrr_j:.4f} Validation loss: {val_loss:.4f} Temperature: {model.temp.item():.4f}')
    else:
        print(f'Epoch {epoch} MRR: {mrr:.4f} MRR_i: {mrr_i:.4f} MRR_j: {mrr_j:.4f} Validation loss: {val_loss:.4f}')
    if select == 'mrr':
        if best_mrr is None:
            best_mrr = mrr
            save_ckpt(model, optimizer, scheduler, epoch, loss, save_path, select, best_mrr)
        elif mrr > best_mrr:
            best_mrr = mrr
            save_ckpt(model, optimizer, scheduler, epoch, loss, save_path, select, best_mrr)
    elif select == 'loss':
        if best_validation_loss is None:
            best_validation_loss = val_loss
            save_ckpt(model, optimizer, scheduler, epoch, loss, save_path, select, best_validation_loss)
        elif val_loss < best_validation_loss:
            best_validation_loss = val_loss
            save_ckpt(model, optimizer, scheduler, epoch, loss, save_path, select, best_validation_loss)
    return best_mrr, best_validation_loss

def save_ckpt(model, optimizer, scheduler, epoch, loss, save_path, select=None, best_val=None):
    print(f"Epoch {epoch} Best {select}: {best_val:.4f} Saving model to {save_path}")
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'lr_scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    }, save_path)
    print(f'Model saved to {save_path}')
