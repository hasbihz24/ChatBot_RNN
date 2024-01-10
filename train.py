import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNetFNN, NeuralNetRNN

# Load intents from JSON file
with open("intents.json", "r", encoding="utf8") as f:
    intents = json.load(f)

# Preprocess data
all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignored_word = ["?", "!", ".", ","]
all_words = [stem(w) for w in all_words if w not in ignored_word]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)  # CrossEnthropyloss

x_train = np.array(x_train)
y_train = np.array(y_train)

# Split data menjadi training dan validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

batch_size = 8
hidden_size = 16
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000

# Initialize datasets and data loaders
train_dataset = ChatDataset(x_train, y_train)
val_dataset = ChatDataset(x_val, y_val)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# Initialize model, loss function, and optimizer
device = torch.device('cpu')
model = NeuralNetRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with validation and F1-score monitoring
for epoch in range(num_epochs):
    model.train()
    for (word, label) in train_loader:
        word = word.to(device)
        label = label.to(device, dtype=torch.int64)
        output = model(word)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        # Evaluate on validation set
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for (word, label) in val_loader:
                word = word.to(device)
                label = label.to(device, dtype=torch.int64)
                output = model(word)
                _, preds = torch.max(output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        # Calculate F1-score
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss={loss.item():.4f}, F1-score={f1:.4f}')

# Save the model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"Training Completed, File saved to {FILE}")