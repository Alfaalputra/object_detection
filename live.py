import torch
from src.visualize import detect_live


def main(threshold=0.6):
    model = torch.load('model_25.pt')
    detect_live(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.6)
    args = parser.parse_args()

    main(args.threshold)