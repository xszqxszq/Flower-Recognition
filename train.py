import torch
import tqdm
import json
import datetime
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from model import VGG16
from data import get_loader
from config import *

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	train_loader, train_labels = get_loader('train')
	val_loader, val_labels = get_loader('val')

	assert(train_labels == val_labels)
	with open('classes.json', 'w') as f:
		f.write(json.dumps(train_labels))

	log_dir = Path(LOG_DIR) / datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
	writer = SummaryWriter(log_dir=str(log_dir))

	model = VGG16(CATEGORIES).to(device)

	loss_function = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

	best_acc = 0.0
	step = 0

	for epoch in range(EPOCHS):
		model.train()
		now = tqdm.tqdm(train_loader)
		for index, (images, labels) in enumerate(now):
			optimizer.zero_grad()
			outputs = model(images.to(device))
			loss = loss_function(outputs, labels.to(device))
			loss.backward()
			optimizer.step()

			predicted = torch.argmax(outputs, dim=1)
			train_acc = (predicted == labels.to(device)).sum().item() / BATCH_SIZE
			train_loss = loss.item()

			step += 1

			if step % LOG_INTERVAL == 0:
				writer.add_scalar('train/acc', train_acc, step)
				writer.add_scalar('train/loss', train_loss, step)
				now.desc = f'Epoch {epoch+1}/{EPOCHS}, loss={loss.item():.3f}'

		model.eval()
		accs, losses, total = 0, 0.0, 0
		with torch.no_grad():
			for (images, labels) in tqdm.tqdm(val_loader):
				outputs = model(images.to(device))
				loss = loss_function(outputs, labels.to(device))
				predicted = torch.argmax(outputs, dim=1)

				accs += (predicted == labels.to(device)).sum().item()
				losses += loss.item() * BATCH_SIZE
				total += labels.size(0)

		val_acc = accs / total
		val_loss = losses / total
		writer.add_scalar('val/acc', val_acc, epoch + 1)
		writer.add_scalar('val/loss', val_loss, epoch + 1)
		print(f'Epoch {epoch+1}/{EPOCHS}, val_acc={val_acc:.3f}')
		if val_acc > best_acc:
			best_acc = val_acc
			torch.save(model.state_dict(), MODEL_PATH)

if __name__ == '__main__':
	main()