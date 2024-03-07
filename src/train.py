import os
import logging

from datetime import datetime
from torchvision import transforms
from core import Model, Dataset, DataLoader
from utils import xml_to_csv, normalize_transform


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
dt = datetime.now().strftime('%Y%m%d_%H%M%S')


def train(train_dataset, val_dataset, epochs, batch_size, model_name):
    xml_to_csv('data/training/', 'training_labels.csv')
    xml_to_csv('data/validation/', 'validation_labels.csv')

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(saturation=0.3),
        transforms.ToTensor(),
        normalize_transform(),
    ])

    model_name = model_name
    train_dataset =  Dataset('data/training/training_labels.csv', 'data/training/images/', transform=preprocess)
    val_dataset = Dataset('data/validation/validation_labels.csv', 'data/validation/images/')
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = Model([], model_name=model_name)
    losses = model.fit(loader, val_dataset, epochs=epochs, learning_rate=0.001, verbose=True)

    logging.info(f'loss: {losses}')

    model.save('model/{model_name}_{dt}.pth')

    return "model saved to model/{model_name}_{dt}.pth"


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_path', type=str, default='data/training', help='path to training data dir')
    parser.add_argument('--validation_path', type=str, default='data/validation', help='path to validation data dir')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--model_name', type=str, default='fasterrcnn_resnet50_fpn', help='model name')
    args = parser.parse_args()

    train(args.training_path, args.validation_path, args.epochs, args.batch_size, args.model_name)