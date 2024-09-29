import json
from nltk_utils import tokenize, stem, bag_of_words # Implementing the utility functions
import numpy as np

import torch
import torch.nn as nn # Importing classes for training neural networks
from torch.utils.data import Dataset, DataLoader # Importing Dataset and Dataloader for creating datasets

from model import NeuralNet

with open('intents.json', 'r') as f: # opening the json file in read mode
    intents = json.load(f) # the "f" variable only refers to the opened file object to read from

all_words = [] # creating an empty list to collect all the patterns in the Json file
tags = [] # creating an empty list to collect all tags in the Json file
xy = [] # creating an empty list to hold both patterns and tags 

# looping over intents file
for i in intents['intents']:
    tag = i['tag']  # getting the tag key in JSON file 
    tags.append(tag)  # appening the tag to our empty tags array

    for pattern in i['patterns']:
        w = tokenize(pattern)   # tokenizing the sentence in the pattern array
        all_words.extend(w) # extending the array into our all_words empty array, not using append because its an array
        xy.append((w, tag))  # adding the tokenized pattern and the corresponding label to the xy list


disregard_words = ['?', '!', '.', ','] # certain punctuation that I want to disregard in my program

# stemming each word in all_words and using list comprehension to disregard certain punctuation/words
all_words = [stem(w) for w in all_words if w not in disregard_words] 
all_words = sorted(set(all_words)) # sorting words and also only taking unique words using "set"
tags = sorted(set(tags)) # sorting tags as well to have unique labels

# Creating the training data
X_train = []  # This is for adding our bag of words
y_train = []  # Associated # of each tag

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)  # This gets the tokenized sentence from above
    X_train.append(bag)  # appending the "bag" content to our X_train data

    label = tags.index(tag) # indexing the label of the tags as they are sorted with 'set' above
    y_train.append(label)

# Creating numpy arrays for both training data
X_train = np.array(X_train)
y_train = np.array(y_train)

# Creating a new dataset using PyTorch's abstract dataset
class ChatDataset(Dataset):
    def __init__(self):      # creating an instance
        self.n_samples = len(X_train)  # storing the # of samples
        self.x_data = X_train
        self.y_data = y_train

    # To later acccess dataset with dataset[idx]
    def __getitem__(self, index):           # retrieving indexed samples from the dataset
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):      # returns total # of samples in the data
        return self.n_samples
    
# Hyperparemeters Initialization
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
batch_size = 8 # defining the samples per batch to load
learning_rate = 0.001
num_epochs = 1000


# Creating an instance of ChatDataset
dataset = ChatDataset()

# Creating the dataloader
# num_workers is for multiproccesing/threading,making the loading faster
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # checking for GPU support if available
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# The training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device).long()

        # forwarding
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward pass and optimizer steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

# Saving the data

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

print(f'training complete, file saved to {FILE}')
