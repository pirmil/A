from training_procedures.args import opts, ArgumentsReplicationDefault
import os
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import optim

from dataload.dataloader import get_dataset_and_dataloaders
from metrics.similarity import similarities, compute_similarities, compute_autosimilarities
from training_procedures.losses import contrastive_loss, get_criterion
from training_procedures.optimizers_and_schedulers import get_optimizer, get_scheduler
from training_procedures.utils import get_degree_histogram
from training_func import train, callback_mrr, save_ckpt, plot_graphs, get_stats
from models.text_encoders import get_text_encoder_and_tokenizer
from models.graph_encoders import get_graph_encoder
from models.clip import get_clip
import warnings


def main(args: ArgumentsReplicationDefault = None):
    args = opts()
    if args.fineclip:
        if args.initial_ckpt_fine_clip != "":
            ckpt = torch.load(args.initial_ckpt_fine_clip)
            initial_state_dict= ckpt['model_state_dict']
            ## Select all keys starting with text_encoder for the text_encoder_state_dict and remove the text_encoder. part
            initial_state_dict_text = {key.replace('text_encoder.', ''): value for key, value in initial_state_dict.items() if key.startswith('text_encoder.')}
            ## Select all keys starting with graph_encoder for the graph_encoder_state_dict and remove the graph_encoder. part
            initial_state_dict_graph = {key.replace('graph_encoder.', ''): value for key, value in initial_state_dict.items() if key.startswith('graph_encoder.')}
            initial_temperature = initial_state_dict['temp']
        else:
            initial_state_dict_text = None
            initial_state_dict_graph = None
            initial_temperature = args.temperature
    else:
        initial_state_dict_text = None
        initial_state_dict_graph = None
        initial_temperature = args.temperature

    text_encoder, tokenizer = get_text_encoder_and_tokenizer(model_name=args.text_encoder_name, tokenizer_name=args.text_encoder_name, fineclip=args.fineclip, fineclipv2= args.fineclip2, nout=args.nout, gamma=args.gamma, depth_mlp=args.depth_mlp, learnable_gamma=args.learnable_gamma, initial_state_dict=initial_state_dict_text)
    train_dataset, val_dataset, test_cids_dataset, test_text_dataset, train_loader, val_loader, test_loader, test_text_loader, idx_to_cid = get_dataset_and_dataloaders(tokenizer, path='./data', num_workers=args.num_workers, batch_size=args.batch_size)
    if 'PNA' in args.graph_encoder_name:
        h0, _ = get_degree_histogram([train_loader, val_loader, test_loader])
        deg = torch.tensor(h0, dtype=torch.int64)
    else:
        deg = None
    graph_encoder = get_graph_encoder(encoder_type=args.graph_encoder_name, num_node_features=args.num_node_features, num_blocks=args.num_blocks, block_channels=args.block_channels, nout=args.nout, nhid=args.nhid, dropout=args.dropout, fineclip=args.fineclip, gamma=args.gamma, depth_mlp=args.depth_mlp, learnable_gamma=args.learnable_gamma, initial_state_dict=initial_state_dict_graph, depth_unet=args.depth_unet, deg=deg, v2=args.v2, heads=args.heads, convname1=args.convname1, convname2=args.convname2)
    model = get_clip(args.clip_name, text_model=text_encoder, graph_model=graph_encoder, temp=initial_temperature, learnable_temp=args.learnable_temperature, nout=args.nout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #for name, param in model.parameters():
    #    print(name, param.requires_grad)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr,
    #                                 betas=(0.9, 0.999),
    #                                 weight_decay=args.weight_decay)
    optimizer = get_optimizer(args.optimizer, args.lr, model, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = get_scheduler(args.scheduler, args.maxlr, args.warmup, args.minlr, args.restart_steps, args.restart_lrs, optimizer=optimizer)
    print(graph_encoder)
    if args.load_model != "":
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
    criterion = get_criterion(args.loss) ## For now only 'contrastive' is implemented
    
    best_mrr = args.best_mrr
    best_validation_loss = args.best_validation_loss
    save_path = args.save_path
    if save_path == "":
        save_path = f'./ckpt/{args.text_encoder_name}_{args.graph_encoder_name}.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_epochs = args.num_epochs
    callback_on_validation = callback_mrr

    if args.plot_graphs:
        print(f"Training data")
        plot_graphs(train_dataset)
        print(f"Validation data")
        plot_graphs(val_dataset)
        print(f"Test data")
        plot_graphs(test_cids_dataset)
    elif args.get_stats:
        print(f"Training data")
        get_stats(train_dataset)
        print(f"Validation data")
        get_stats(val_dataset)
        print(f"Test data")
        get_stats(test_cids_dataset)
    else:
        losses = train(model, train_loader, num_epochs, optimizer, criterion, scheduler=scheduler, val_loader=val_loader, save_path=save_path, device=device, similarity_metric='cosine', best_validation_loss=best_validation_loss, best_mrr=best_mrr, callback_on_validation=callback_on_validation, callback_args=None, print_progress=args.print_progress_bar)


if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
