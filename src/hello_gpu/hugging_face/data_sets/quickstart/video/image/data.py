import torch
from datasets import Image
from torch import Tensor
from torch.utils.data import DataLoader

from src.hello_gpu.hugging_face.data_sets.quickstart.video.data import training_dataset
from src.hello_gpu.hugging_face.data_sets.quickstart.video.image.transformers import create_jitter_transformer


def collate(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["labels"])

    pixels_tensor: Tensor = torch.stack(images)
    labels_tensor: Tensor = torch.tensor(labels)
    return {"pixel_values": pixels_tensor, "labels": labels_tensor}


# Most image models work with RBG images.
# If your dataset contains images in a different mode, you can use the cast_column() function to set the mode to RGB:
image_rgb_training_dataset = training_dataset.cast_column("image", Image(mode="RGB"))
# The Beans dataset contains only RGB images, so this step is unnecessary here.

jitter_colors = create_jitter_transformer()
jittered_image_rgb_training_dataset = image_rgb_training_dataset.with_transform(jitter_colors)

jittered_image_rgb_training_dataloader = DataLoader(jittered_image_rgb_training_dataset, collate_fn=collate, batch_size=4)
