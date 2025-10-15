import logging
import os
import sys
import time
import math
import pickle

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import accuracy_score

num_epochs = 10
embed_size = 300  # Must match GloVe dimension (300d)
num_hiddens = 256
num_layers = 4
num_head = 4
dim_feedforward = 1024
batch_size = 32
labels = 2
lr = 1e-4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 512



def length_to_mask(lengths):
    """Converts a tensor of sequence lengths to a boolean mask."""
    max_length = torch.max(lengths)
    # Ensure the new tensor is created on the same device as lengths
    mask = torch.arange(max_length, device=lengths.device).expand(lengths.shape[0], max_length) < lengths.unsqueeze(1)
    return mask


class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    """The main Transformer model for sequence classification."""

    def __init__(self, vocab_size, embedding_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, max_len=512, activation: str = "relu"):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim

        # 1. Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 2. Positional Encoding
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)

        # 3. Transformer Encoder
        # The model dimension (d_model) must match the embedding dimension
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_head,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation=activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 4. Output Layer
        # The input dimension must match the Transformer's output dimension (embedding_dim)
        self.output = nn.Linear(embedding_dim, num_class)

    def forward(self, inputs, lengths):
        # inputs shape: [batch_size, 512]

        # 获取输入张量的固定宽度（也就是 512）
        max_len = inputs.shape[1]

        # 使用真实长度 lengths 和 固定宽度 max_len 来创建尺寸正确的遮罩
        mask = torch.arange(max_len, device=lengths.device).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
        attention_mask = (mask == False)

        inputs = torch.transpose(inputs, 0, 1)
        hidden_states = self.embeddings(inputs)
        hidden_states = self.position_embedding(hidden_states)

        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


# --- Main Execution Block ---

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 1. Load pre-processed data from the pickle file
    logging.info('Loading data from pickle file...')
    pickle_file = os.path.join('..', 'pickle', 'imdb_glove.pickle3')
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
     vocab] = pickle.load(open(pickle_file, 'rb'))
    logging.info('Data loaded successfully!')

    # 2. Initialize the model
    net = Transformer(vocab_size=weight.shape[0],  # Use the shape of the weight matrix for vocab_size
                      embedding_dim=embed_size,
                      num_class=labels,
                      num_head=num_head,
                      num_layers=num_layers,
                      dim_feedforward=dim_feedforward,
                      max_len=MAX_LEN)

    # 3. Load the pre-trained GloVe weights
    net.embeddings.weight.data.copy_(weight)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # 4. Create Datasets and DataLoaders
    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 5. Training and Validation Loop
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0

        # ==================== The core change is here ====================
        # Move validation and final stats update inside the tqdm context manager
        with tqdm(total=len(train_iter), desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            # --- Training Part ---
            net.train()
            for feature, label in train_iter:
                n += 1
                feature = feature.to(device)
                label = label.to(device)
                lengths = (feature != 0).sum(dim=1)

                optimizer.zero_grad()
                score = net(feature, lengths)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()

                train_acc += accuracy_score(torch.argmax(score.cpu().data, dim=1), label.cpu())
                train_loss += loss.item()

                pbar.set_postfix({
                    'train_loss': f'{(train_loss / n):.4f}',
                    'train_acc': f'{(train_acc / n):.2f}'
                })
                pbar.update(1)

            # --- Validation Part (still inside the with block) ---
            net.eval()
            with torch.no_grad():
                for val_feature, val_label in val_iter:
                    m += 1
                    val_feature = val_feature.to(device)
                    val_label = val_label.to(device)
                    val_lengths = (val_feature != 0).sum(dim=1)

                    val_score = net(val_feature, val_lengths)
                    val_loss = loss_function(val_score, val_label)

                    val_acc += accuracy_score(torch.argmax(val_score.cpu().data, dim=1), val_label.cpu())
                    val_losses += val_loss.item()

            end = time.time()
            runtime = end - start

            # --- Final Stats Update (still inside the with block) ---
            # This will update and "freeze" the progress bar with all the final stats
            pbar.set_postfix({
                'train_loss': f'{(train_loss / n):.4f}',
                'train_acc': f'{(train_acc / n):.2f}',
                'val_loss': f'{(val_losses / m):.4f}',
                'val_acc': f'{(val_acc / m):.2f}',
                'time': f'{runtime:.2f}s'
            })

    # 6. Prediction Loop (this part needs no changes)
    test_pred = []
    net.eval()
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for (test_feature,) in test_iter:
                test_feature = test_feature.to(device)
                test_length = (test_feature != 0).sum(dim=1)
                test_score = net(test_feature, test_length)
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())
                pbar.update(1)

    # 7. Save Results (this part needs no changes)
    test_df = pd.read_csv("../tutorialData/testData.tsv", header=0, delimiter="\t", quoting=3)
    result_output = pd.DataFrame(data={"id": test_df["id"], "sentiment": test_pred})
    result_output.to_csv("../results/transformer_glove.csv", index=False, quoting=3)
    logging.info('Results saved to ../results/transformer_glove.csv')