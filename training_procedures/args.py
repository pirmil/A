from argparse import ArgumentParser, Namespace

class ArgumentsReplicationDefault:
    epochs = 5
    batch_size = 32
    lr = 2e-5
    weight_decay = 0.01
    log_interval = 50
    num_workers = 0
    save_path = ""

def opts() -> Namespace:
    """Option Handling Function."""
    parser = ArgumentParser(description="CLIP embedding training script")
    parser.add_argument("--batch-size", type=int, default=100, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--log-interval", type=int, default=50, help="Log interval")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--save-path", type=str, default="./ckpt/dummy.pt", help="Path to the saved checkpoints")
    parser.add_argument("--submission-save-path", type=str, default="/Data/altegradsubmissions/", help="Path to the saved submission")
    parser.add_argument("--submission-float-format", type=str, default='%.5f', help="Float format of the submission")
    parser.add_argument("--text-encoder-name", type=str, default="jonas-luehrs/distilbert-base-uncased-MLM-scirepeval_fos_chemistry", help="Name of the text encoder") # 'distilbert-base-uncased'
    parser.add_argument("--graph-encoder-name", type=str, default="GraphTransformer", help="Name of the graph encoder")
    parser.add_argument("--num-node-features", type=int, default=300, help="Number of node features")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of graph blocks")
    parser.add_argument("--block-channels", type=int, nargs='+', default=[300, 300, 300, 300], help="Number of channels in each graph block")
    parser.add_argument("--nout", type=int, default=768, help="Output dimension of graph encoder")
    parser.add_argument("--nhid", type=int, default=300, help="Hidden dimension of graph encoder")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="InvSqrtwithRestarts", help="Scheduler type")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup steps")
    parser.add_argument("--maxlr", type=float, default=2e-4, help="Maximum learning rate")
    parser.add_argument("--restart-steps", type=int, nargs='+', default=[22500, 45000, 67500, 90000, 1e6], help="Restart steps")
    parser.add_argument("--restart-lrs", type=float, nargs='+', default=[1e-4, 5e-5, 2.5e-5, 1.25e-5, 0.6e-5], help="Restart learning rates")
    parser.add_argument("--minlr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--loss", type=str, default="contrastive", help="Loss type")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature")
    parser.add_argument("--learnable-temperature", action='store_true', help="Learnable temperature")
    parser.add_argument("--clip_name", type=str, default="CLIP_temp", help="Name of the model")
    parser.add_argument("--fineclip", action='store_true', help="Fine-tune CLIP")
    parser.add_argument("--fineclip2", action='store_true', help="Fine-tune CLIP version 2")
    parser.add_argument('--gamma', type=float, default=1e-4, help="Gamma")
    parser.add_argument("--depth-unet", type=int, default=2, help="Depth of the GraphUNet network")
    parser.add_argument("--depth-mlp", type=int, default=1, help="Depth of MLP")
    parser.add_argument("--learnable-gamma", action='store_true', help="Learnable gamma")
    parser.add_argument("--load-model", type=str, default="", help="Path to the model checkpoint")
    parser.add_argument("--best-mrr", type=float, default=0, help="Best MRR")
    parser.add_argument("--best-validation-loss", type=float, default=1000, help="Best validation loss")
    parser.add_argument("--num-epochs", type=int, default=900, help="Number of epochs")
    parser.add_argument("--initial_ckpt_fine_clip", type=str, default="", help="Path to the initial checkpoint of fine-tuned CLIP")
    parser.add_argument("--v2", type=bool, default=True, help="Use GATv2")
    parser.add_argument("--heads", type=int, default=1, help="Number of heads in the multi-head attention")
    parser.add_argument("--print-progress-bar", action='store_true', help="Print progress bar during training")
    parser.add_argument("--convname1", type=str, default='GATv2Conv', help="First graph convolutional layer in a hybrid model")
    parser.add_argument("--convname2", type=str, default='GINConv', help="Second graph convolutional layer in a hybrid model")
    parser.add_argument("--plot-graphs", action='store_true', help='Plot molecules as graphs instead of training models')
    parser.add_argument("--get-stats", action='store_true', help='Get statistics about the data instead of training models')
    args = parser.parse_args()
    return args

