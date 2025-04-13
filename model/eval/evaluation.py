import torch
import torch.nn as nn

class MixedTabularNN(nn.Module):
    def __init__(self, categorical_cardinalities, num_numerical_features, embedding_dim=8, hidden_dims=[128, 64], output_dim=1):
        super(MixedTabularNN, self).__init__()

        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim) for cardinality in categorical_cardinalities
        ])
        self.embedding_output_dim = embedding_dim * len(categorical_cardinalities)

        # Total input = numerical features + all embeddings
        input_dim = num_numerical_features + self.embedding_output_dim

        # Fully connected layers
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = h
        layers.append(nn.Linear(input_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded + [x_num], dim=1)
        return self.model(x)
