import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    
    # dataset
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--meta_path', type=str)

    # dataloader
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=24)

    # model
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--binary', action='store_true')

    # train
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=40)
    parser.add_argument('--gamma', type=float, default=0.1)

    # etc
    parser.add_argument('--gpus', nargs="+", type=int, default=[0])
    parser.add_argument('--exp_name', type=str, default='exp')

    return parser.parse_args()