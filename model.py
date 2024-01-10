import torch
import torch.nn as nn

class NeuralNetRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetRNN, self).__init__()
        self.hidden_size = hidden_size

        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # RNN layer
        out, _ = self.rnn(x)

        # ekstrak output layer pada langkah sebelumnya
        out = out[:, :]

        # Output layer
        out = self.fc(out)

        return out
    
class NeuralNetFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetFNN, self).__init__()
        # FNN Layer
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    #output Layer
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
    
