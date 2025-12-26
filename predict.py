import torch
import sys
import random
import json

from torchvision import transforms
from pathlib import Path
from PIL import Image
from model import VGG16
from config import *

def random_image():
	path = Path(DATASET_DIR) / 'test'
	if not path.exists():
		raise Exception('Dataset not exists')

	images = [file for file in path.glob('**/*') if file.is_file()]
	return random.choice(images)

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = VGG16(CATEGORIES).to(device)
	if not Path(MODEL_PATH).exists():
		raise Exception('model.pth not exists')
	model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

	if not Path(CLASSES_PATH).exists():
		raise Exception('classes.json not exists')
	with open(CLASSES_PATH) as f:
		classes = dict((int(key), value) for key, value in json.loads(f.read()).items())

	transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize(MEAN, STD)
	])

	path = sys.argv[1] if len(sys.argv) > 1 else random_image()
	raw = Image.open(path).convert('RGB')
	image = transform(raw).unsqueeze(0).to(device)

	model.eval()
	with torch.no_grad():
		output = model(image)
		probs = torch.softmax(output, dim=1).squeeze(0).cpu()

	predicted = torch.argmax(probs).item()
	predicted_class = classes[predicted]

	print(f'Image Path: {path}')
	print(f'Predicted Class: {predicted_class}')
	print('Class probabilities:')
	for index, prob in enumerate(probs.tolist()):
		print(f'- {classes[index]}: {prob:.4f}')

if __name__ == '__main__':
	main()