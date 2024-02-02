from datasets import load_from_disk
dset = load_from_disk(f"dataset_iter_0")

print(dset[0])
