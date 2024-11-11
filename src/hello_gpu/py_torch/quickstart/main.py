from src.hello_gpu.py_torch.quickstart import data, models, testing, training


def run() -> None:
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training.run(data.training_dataloader, models.model, training.loss_fn, training.optimizer)
        testing.run(data.test_dataloader, models.model, training.loss_fn)
        print("Done!")

    models.save()
