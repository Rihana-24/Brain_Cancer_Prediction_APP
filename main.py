import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN model using PyTorch or TensorFlow")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.0001, help="Weight decay (L2 regularization)")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help="Mode of operation: 'train' or 'eval'")
    parser.add_argument('--cuda', action='store_true', help="Use GPU if available")
    parser.add_argument('--framework', type=str, choices=['pytorch', 'tensorflow'], default='pytorch',
                        help="Framework: 'pytorch' or 'tensorflow'")
    return parser.parse_args()


def run_pytorch(args):
    import torch
    from utils import prep_pytorch,prep_tensorflow
    from models.cnn_pytorch import get_pretrained_model
    from models.train_pytorch import Trainer_Pytorch

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using PyTorch on {device}")

    train_dataloader, test_dataloader = prep_pytorch.get_pretrained_model()
    model = get_pretrained_model().to(device)

    if args.mode == 'eval':
        model.load_state_dict(torch.load("model.pth"))

    trainer = Trainer_Pytorch(model, train_dataloader, test_dataloader,
                             args.lr, args.wd, args.epochs, device)

    if args.mode == 'train':
        trainer.train(save=True, plot=True)

    trainer.evaluate()



def run_tensorflow(args):
    import tensorflow as tf
    from utils import prep_tensorflow
    from models.cnn_tensorflow import get_pretrained_model_tf
    from models.train_tensorflow import Trainer_tensorflow

    physical_devices = tf.config.list_physical_devices('GPU')
    device = '/GPU:0' if args.cuda and physical_devices else '/CPU:0'
    print(f"Using TensorFlow on {device}")

    train_dataset, test_dataset = prep_tensorflow.get_pretrained_model_tf()
    with tf.device(device):
        model = get_pretrained_model_tf()

        trainer = Trainer_tensorflow(model, train_dataset, test_dataset,
                                    args.lr, args.wd, args.epochs, device)

        if args.mode == 'train':
            trainer.train(save=True, plot=True)

        trainer.evaluate()


def main():
    args = parse_args()
    if args.framework == 'pytorch':
        run_pytorch(args)
    else:
        run_tensorflow(args)


if __name__ == '__main__':
    main()