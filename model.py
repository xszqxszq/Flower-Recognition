import torch
import torch.nn as nn

class VGG16(nn.Module):
	def __init__(self, categories):
		super().__init__()
		self.features = nn.Sequential(
			# Input is 224x224x3
			nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(True),

			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True),
			nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True),

			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),

			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),

			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),

			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.classifier = nn.Sequential(
			nn.Linear(512*7*7, 4096), nn.ReLU(True), nn.Dropout(p=0.5),
			nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(p=0.5),
			nn.Linear(4096, categories)
		)

	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, start_dim = 1)
		x = self.classifier(x)

		return x