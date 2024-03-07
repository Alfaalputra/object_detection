import torch
from src.visualize import detect_video

def main(input_video, output_video, threshold=0.6, fps=30):
    model = torch.load('model_25.pt')
    detect_video(model, input_video, {output_video}, threshold=threshold, fps=fps)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, required=True)
    parser.add_argument('--output_video', type=str, required=True, default='result/output.avi')
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    main(args.input_video, args.output_video, args.threshold, args.fps)