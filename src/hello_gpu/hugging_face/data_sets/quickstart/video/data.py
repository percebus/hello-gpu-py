from datasets import load_dataset

# 1. Load the Beans dataset by providing the load_dataset() function with the dataset name and a dataset split:
training_dataset = load_dataset("beans", split="train")
