from training_procedures.args import opts, ArgumentsReplicationDefault
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import optim

from dataload.dataloader import get_dataset_and_dataloaders
from metrics.similarity import similarities, compute_similarities, compute_autosimilarities
from training_procedures.losses import contrastive_loss, get_criterion
from training_procedures.optimizers_and_schedulers import get_optimizer, get_scheduler
from training_procedures.utils import get_degree_histogram
from training_func import train, callback_mrr, save_ckpt
from models.text_encoders import get_text_encoder_and_tokenizer
from models.graph_encoders import get_graph_encoder
from models.clip import get_clip
from metrics.similarity import compute_similarities

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
            if 'temp' in initial_state_dict:
                initial_temperature = initial_state_dict['temp']
            else:
                initial_temperature = torch.ones(1)*0.07 # Optimal temperature from original CLIP paper
        else:
            initial_state_dict_text = None
            initial_state_dict_graph = None
            initial_temperature = args.temperature
    else:
        initial_state_dict_text = None
        initial_state_dict_graph = None
        initial_temperature = args.temperature
    text_encoder, tokenizer = get_text_encoder_and_tokenizer(model_name=args.text_encoder_name, tokenizer_name=args.text_encoder_name, fineclip=args.fineclip, nout=args.nout, gamma=args.gamma, depth_mlp=args.depth_mlp, learnable_gamma=args.learnable_gamma, initial_state_dict=initial_state_dict_text)
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

    optimizer = get_optimizer(args.optimizer, args.lr, model, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = get_scheduler(args.scheduler, args.maxlr, args.warmup, args.minlr, args.restart_steps, args.restart_lrs, optimizer=optimizer)

    if args.load_model != "":
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
    criterion = get_criterion(args.loss) ## For now only 'contrastive' is implemented
    model.eval()
    similarity = compute_similarities(test_loader, test_text_loader, model, device, metric='cosine')
    solution = pd.DataFrame(similarity)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
    #Get filename of ckpt then use it as filename of submission
    ckpt_filename = args.load_model.split('/')[-1]
    print(ckpt_filename)
    if args.submission_float_format:
        solution.to_csv(f'{args.submission_save_path}submission_{ckpt_filename}.csv', index=False, float_format=args.submission_float_format)
    else:
        solution.to_csv(f'{args.submission_save_path}submission_{ckpt_filename}.csv', index=False)


# def main(args: ArgumentsReplicationDefault = None):
#     args = opts()
#     model_name = 'distilbert-base-uncased'
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
#     model.to(device)

#     print(f'loading best model... ({args.save_path})')
#     checkpoint = torch.load(args.save_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     graph_model = model.get_graph_encoder()
#     text_model = model.get_text_encoder()

#     test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
#     test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

#     idx_to_cid = test_cids_dataset.get_idx_to_cid()

#     test_loader = DataLoader(test_cids_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     graph_embeddings = []
#     for batch in test_loader:
#         for output in graph_model(batch.to(device)):
#             graph_embeddings.append(output.tolist())

#     test_text_loader = TorchDataLoader(test_text_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#     text_embeddings = []
#     with torch.no_grad():
#         for batch in test_text_loader:
#             for output in text_model(batch['input_ids'].to(device), 
#                                     attention_mask=batch['attention_mask'].to(device)):
#                 text_embeddings.append(output.tolist())

#     similarity = cosine_similarity(text_embeddings, graph_embeddings)

#     solution = pd.DataFrame(similarity)
#     solution['ID'] = solution.index
#     solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
#     solution.to_csv('submission.csv', index=False)

if __name__=='__main__':
    main()
