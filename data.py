import torch
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from config import *

def get_loader(name):
	if name == 'train':
		transform = transforms.Compose([
			transforms.RandomResizedCrop(IMAGE_SIZE),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(MEAN, STD)
		])
		shuffle = True
	elif name == 'val':
		transform = transforms.Compose([
			transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
			transforms.ToTensor(),
			transforms.Normalize(MEAN, STD)
		])
		shuffle = False
	else:
		raise Exception('Unexpected loader type')

	path = Path(DATASET_DIR) / name
	if not path.exists():
		raise Exception('Dataset not exists')

	dataset = datasets.ImageFolder(path, transform = transform)
	num_workers = min(8, os.cpu_count())
	labels = dict(enumerate(dataset.classes))

	return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=num_workers), labels