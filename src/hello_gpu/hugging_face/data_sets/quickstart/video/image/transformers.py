from torchvision.transforms import ColorJitter, Compose, ToTensor


def create_jitter_transformer() -> None:
    # 2. Now you can add some data augmentations with any library (Albumentations, imgaug, Kornia) you like.
    # Here, you’ll use torchvision to randomly change the color properties of an image:
    jitterCompose = Compose([ColorJitter(brightness=0.5, hue=0.5), ToTensor()])

    # 3. Create a function to apply your transform to the dataset and generate the model input: pixel_values.
    def transform(examples):
        # FIXME "TypeError: 'JpegImageFile' object is not iterable"
        examples["pixel_values"] = [jitterCompose(image.convert("RGB")) for image in examples["image"]]
        return examples

    return transform
