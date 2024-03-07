import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchvision import transforms
from utils import reverse_normalize, normalize_transform, _is_iterable


def detect_live(model, score_filter=0.6):
    cv2.namedWindow('Faster RCNN')
    try:
        video = cv2.VideoCapture(0)
    except:
        print('No webcam available.')
        return

    while True:
        ret, frame = video.read()
        if not ret:
            break

        labels, boxes, scores = model.predict(frame)
        for i in range(boxes.shape[0]):
            if scores[i] < score_filter:
                continue

            box = boxes[i]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
            if labels:
                cv2.putText(frame, '{}: {}'.format(labels[i], round(scores[i].item(), 2)), (int(box[0]), int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('Faster RCNN', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyWindow('Faster RCNN')
    video.release()


def detect_video(model, input_file, output_file, fps=30, score_filter=0.7):
    video = cv2.VideoCapture(input_file)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scaled_size = 800
    scale_down_factor = min(frame_height, frame_width) / scaled_size
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))
    transform_frame = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(scaled_size),
        transforms.ToTensor(),
        normalize_transform(),
    ])

    while True:
        ret, frame = video.read()
        if not ret:
            break

        transformed_frame = frame
        predictions = model.predict(transformed_frame)

        for label, box, score in zip(*predictions):
            if score < score_filter:
                continue
            
            cv2.rectangle(frame, (int(np.float32(box[0])), int(np.float32(box[1]))), (int(np.float32(box[2])), int(np.float32(box[3]))), (255, 0, 0), 3)
            cv2.putText(frame, '{}: {}'.format(label, round(score.item(), 2)), (int(np.float32(box[0])), int(np.float32(box[1] - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), lineType=cv2.LINE_AA)

        out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    out.release()

    cv2.destroyAllWindows()


def plot_prediction_grid(model, images, dim=None, figsize=None, score_filter=0.6):
    if dim is None:
        dim = (len(images), 1)

    if dim[0] * dim[1] != len(images):
        raise ValueError('Grid dimensions do not match size of list of images')

    fig, axes = plt.subplots(dim[0], dim[1], figsize=figsize)
    index = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            image = images[index]
            preds = model.predict(image)

            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(reverse_normalize(image))
            index += 1
            if dim[0] <= 1 and dim[1] <= 1:
                ax = axes
            elif dim[0] <= 1:
                ax = axes[j]
            elif dim[1] <= 1:
                ax = axes[i]
            else:
                ax = axes[i, j]

            ax.imshow(image)

            for label, box, score in zip(*preds):
                if score >= score_filter:
                    width, height = box[2] - box[0], box[3] - box[1]
                    initial_pos = (box[0], box[1])
                    rect = patches.Rectangle(initial_pos, width, height, linewidth=1,
                                             edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                    ax.text(box[0] + 5, box[1] - 10, '{}: {}'
                            .format(label, round(score.item(), 2)), color='red')
                ax.set_title('Image {}'.format(index))

    plt.show()


def show_labeled_image(image, boxes, labels=None):
    fig, ax = plt.subplots(1)
    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = transforms.ToPILImage()(image)
    ax.imshow(image)
    if boxes.ndim == 1:
        boxes = boxes.view(1, 4)

    if labels is not None and not _is_iterable(labels):
        labels = [labels]

    for i in range(boxes.shape[0]):
        box = boxes[i]
        width, height = (box[2] - box[0]).item(), (box[3] - box[1]).item()
        initial_pos = (box[0].item(), box[1].item())
        rect = patches.Rectangle(initial_pos,  width, height, linewidth=1,
                                 edgecolor='r', facecolor='none')
        if labels:
            ax.text(box[0] + 5, box[1] - 5, '{}'.format(labels[i]), color='red')

        ax.add_patch(rect)

    plt.show()