# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:31:03 2024

@author: nhmille
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
import random
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt


def createVocab(hyperparameters, data_directory):
    # Assuming vocab can be found from "data_directory/training_labels"
    training_label_path = os.path.join(os.getcwd(), 'MLDS_hw2_1_data', data_directory, 'training_label.json')
    with open(training_label_path, 'r') as f:
        data = json.load(f)
    
    # Read through and store all captions
    captions = []
    for item in data:
        captions.extend(item['caption']) 
    
    # Count number of occurences for each word
    word_counter = Counter()
    for caption in captions:
        tokens = caption.lower().split()
        word_counter.update(tokens)
    
    # Define vocab list based on occurence count
    vocab = [word for word, count in word_counter.items() if count >= hyperparameters['vocab_count']]

    # Supplemental tokens
    vocab = vocab + ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    
    # Define dict with words as keys and indexs as values
    word2idx = {word: idx for idx, word in enumerate(vocab)} 
    
    # Define dict with indices as keys and words as values
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return vocab, word2idx, idx2word


def createData(data_directory, test=False):
    # Iterate over all files in the training data folder
    # Create a list of tuples containing names and file paths
    paths = []
    
    # Testing mode
    if data_directory != None:
        directory = os.path.join(os.getcwd(), "MLDS_hw2_1_data", data_directory, "feat")       
        
    # Training mode
    else:
        if test:
            directory = os.path.join(os.getcwd(), "MLDS_hw2_1_data", "testing_data", "feat")       
        else:
            directory = os.path.join(os.getcwd(), "MLDS_hw2_1_data", "training_data", "feat") 
        
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            vid_id = os.path.splitext(filename)[0]
            path = os.path.join(directory, filename)
            paths.append((vid_id, path))

    # Testing mode
    if data_directory != None:
        json_filename = data_directory[:-4]
        json_filename = json_filename + "label.json"
        directory = os.path.join(os.getcwd(), "MLDS_hw2_1_data", json_filename)

    # Training mode
    else:
        if test:
            directory = os.path.join(os.getcwd(), "MLDS_hw2_1_data", "testing_label.json")    
        else:
            directory = os.path.join(os.getcwd(), "MLDS_hw2_1_data", "training_label.json")
    with open(directory, 'r') as f:
        caption_data = json.load(f)
    
    # Create a dictionary of captions with ids as keys and captions as values
    caption_dict = {item['id']: item['caption'] for item in caption_data}

    # Extract the caption and create a complate tuple for the data including id, path, and caption    
    complete_data = []
    # Iterate over all paths
    for vid_id, path in paths:
        if vid_id in caption_dict:
            caption = caption_dict[vid_id]
            complete_data.append((vid_id, path, caption))
    return complete_data
    
    
def createDataLists(complete_data, test=False):
    # Iterate through the complete data and extract lists for captions and paths
    path_list = []
    caption_list = []
    for vid_id, path, caption in complete_data:
        for token in caption:
            caption_list.append(token)
            path_list.append(path)
    if test:
        random.shuffle(caption_list)
    
    return path_list, caption_list
    
def readBatch(batch, word2idx):
    # Create tensor lists of batched data
    video_features = []
    tokenized_captions = []
    
    for video_path, caption in batch:
        # Convert captions to tokens
        tokens = [word2idx.get(word, word2idx["<UNK>"]) for word in caption.lower().split()]
        tokens = [word2idx["<BOS>"]] + tokens + [word2idx["<EOS>"]]
        tokenized_captions.append(torch.tensor(tokens, dtype=torch.long))

        # Load video using stored path and add to batch
        video_feature = np.load(video_path)
        video_features.append(torch.tensor(video_feature, dtype = torch.float32))
    # Pad token sequence and stack video features
    padded_captions = pad_sequence(tokenized_captions, batch_first=True, padding_value=word2idx["<PAD>"])
    video_features_tensor = torch.stack(video_features)
    return video_features_tensor, padded_captions
    

class setupDataset(Dataset):
    def __init__(self, path_list, caption_list):
        self.path_list = path_list
        self.caption_list = caption_list
        
    def __len__(self):
        return len(self.caption_list)
    
    def __getitem__(self, idx):
        return self.path_list[idx], self.caption_list[idx]


class Encoder(nn.Module):
    # What defines input_dim?
    def __init__(self, hyperparameters, input_dim=4096):
        super(Encoder, self).__init__()
        hidden_dim = hyperparameters['hidden_dim']
        num_layers = hyperparameters['layers']
        drop = hyperparameters['drop']
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(drop)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, features):
        features = self.dropout(features)
        outputs, (hidden, cell) = self.lstm(features)
        outputs = self.dropout(outputs)
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hyperparameters):
        super(Attention, self).__init__()
        hidden_dim = hyperparameters['hidden_dim']        
        self.scale = hidden_dim**0.5

    def forward(self, hidden, encoder_outputs):
        # Compute dot product between encoder output and hidden state
        weights = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)
        weights = weights/self.scale
        attn_weights = F.softmax(weights, dim=1)

        # Compute context vector by taking dot product between encoder output and weights
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1)


class Decoder(nn.Module):
    def __init__(self, hyperparameters, vocab_size):
        super(Decoder, self).__init__()
        hidden_dim = hyperparameters['hidden_dim']
        num_layers = hyperparameters['layers']
        drop = hyperparameters['drop']
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(2*hidden_dim, hidden_dim, num_layers, batch_first=True)
        # self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(2*hidden_dim, vocab_size)
        # self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.attention = Attention(hyperparameters)
        self.dropout = nn.Dropout(drop)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        
    def forward(self, input_word, hidden, cell, encoder_outputs):
        # Create high-dimensional embedding of input_word
        embedding = self.embedding(input_word).unsqueeze(1)
        embedding = embedding.squeeze(2) if embedding.dim() == 4 else embedding
        output = self.dropout(embedding)
        
        # Calculate context vector using attention network
        context = self.attention(hidden[-1], encoder_outputs)
        
        # print(f"Shape of embedded during BLEU eval: {embedding.shape}")
        # print(f"Shape of context during BLEU eval: {context.shape}") 

        # Input to ltsm is embedding + context (concatenated)
        input_vec = torch.cat((embedding, context.unsqueeze(1)), dim=2)
        # input_vec = embedding
        
        # Propogate through lstm then fully-connected layer to get prediction
        output, (hidden, cell) = self.lstm(input_vec, (hidden, cell))
        output = self.dropout(output)
        
        prediction = self.fc(torch.cat((output.squeeze(1), context), dim=1))
        # prediction = self.fc(output.squeeze(1))   
        return prediction, hidden, cell


class S2S(nn.Module):
    def __init__(self, encoder, decoder, word2idx, hyperparameters):
        super(S2S, self).__init__()
        self.tf = hyperparameters['tf_ratio']
        
        self.encoder = encoder
        self.decoder = decoder
        self.optim = torch.optim.Adam(self.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['decay'])
        self.loss = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        
    def forward(self, features, targets):
        batch_size = targets.size(0)
        max_len = targets.size(1)
        vocab_size = self.decoder.fc.out_features
        
        # Encode features
        encoder_outputs, hidden, cell = self.encoder(features)
        
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)
        input_word = targets[:,0]

        for t in range (1, max_len):
            prediction, hidden, cell = self.decoder(input_word, hidden, cell, encoder_outputs)
            outputs[:,t,:] = prediction
            # Teacher forcing -> Set next input word to be target instead of prediction
            use_teacher_forcing = random.random() < self.tf
            # print(f"Using Teaching Forcing: {use_teacher_forcing}")
            if use_teacher_forcing:
                input_word = targets[:,t] 
            else:
                input_word  = prediction.argmax(1)
        return outputs

    
def train(model, train_loader, test_loader, epochs=200):
    train_loss_history = []
    test_loss_history = []     
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        # Loop to process mini-batches created by train_loader
        for i, (features_batch, captions_batch) in enumerate(train_loader):
            features_batch = features_batch.to(model.device)
            captions_batch = captions_batch.to(model.device)
            model.optim.zero_grad()
            outputs = model(features_batch, captions_batch)
            
            outputs = outputs.view(-1, outputs.size(-1))
            targets = captions_batch.view(-1)
            
            loss = model.loss(outputs, targets)
            loss.backward()
            epoch_loss += loss.item()
            model.optim.step()
            
        avg_train_loss = epoch_loss/len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Evaluate performance on test dataset every 2 epochs
        if (epoch+1)%2 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():  # Disable gradient calculation during evaluation
                for features_batch, captions_batch in test_loader:
                    features_batch = features_batch.to(model.device)
                    captions_batch = captions_batch.to(model.device)
                    
                    outputs = model(features_batch, captions_batch)
                
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = captions_batch.view(-1)
                
                    loss = model.loss(outputs, targets)                
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
            test_loss_history.append(avg_test_loss) 
         
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}')
            
    torch.save(model.state_dict(), 'seq2seq_model.pth')
            
    return train_loss_history, test_loss_history
        

def plotTraining(train_loss_history, test_loss_history, hyperparameters):
    b = hyperparameters["batch_sz"]
    h = hyperparameters["hidden_dim"]
    l = hyperparameters["layers"]
    lr = hyperparameters["lr"]
    tf = hyperparameters["tf_ratio"]
    d = hyperparameters["decay"]
    dr = hyperparameters["drop"]

    filename = f"Batch{b}_HiddenDim_{h}_Layers{l}_LearningRate{lr}_TeachForce{tf}_Decay{d}_Dropout{dr}.png"
    
    epochs = range(1, len(train_loss_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_history, label='Training Loss', color='blue', marker='o')
    
    # Plot test loss (every 2nd epoch)
    test_epochs = range(2, 2*len(test_loss_history) + 1, 2)  # Test loss is collected every 2 epochs
    plt.plot(test_epochs, test_loss_history, label='Testing Loss', color='red', marker='x')
    
    # Add labels and title
    plt.title('Training and Testing Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.grid(True)
    plt.savefig(filename, format='png')
    plt.show()


def createOutputFile(model, test_data, idx2word, word2idx, hyperparameters, name):
    model.eval()
    result = {}
    
    
    b = hyperparameters["batch_sz"]
    h = hyperparameters["hidden_dim"]
    l = hyperparameters["layers"]
    lr = hyperparameters["lr"]
    tf = hyperparameters["tf_ratio"]
    d = hyperparameters["decay"]
    dr = hyperparameters["drop"]

    if name == None:
        output_filename = f"Batch{b}_HiddenDim_{h}_Layers{l}_LearningRate{lr}_TeachForce{tf}_Decay{d}_Dropout{dr}.txt"
    else:
        output_filename = name
    
    with open(output_filename, 'w') as f:
        for vid_id, features, true_captions in test_data:
            # Generate caption with the model
            features = torch.tensor(np.load(features), dtype=torch.float32).unsqueeze(0).to(model.device)
            encoder_outputs, hidden, cell = model.encoder(features)

            # Start caption generation with BOS token
            input_word = torch.tensor([word2idx["<BOS>"]], dtype=torch.long).to(model.device)
            generated_caption = []
            
            for _ in range(20):  # Assume max caption length of 20
                prediction, hidden, cell = model.decoder(input_word, hidden, cell, encoder_outputs)
                top1 = prediction.argmax(1)
                word = idx2word[top1.item()]
                if word == "<EOS>":
                    break
                generated_caption.append(word)
                input_word = top1.unsqueeze(0)

            # Join the generated words into a sentence
            generated_caption_str = ' '.join(generated_caption)
            result[vid_id] = generated_caption_str
            
            # Write video ID and generated caption to output file
            f.write(f"{vid_id},{generated_caption_str}\n")
    
    print(f"Output file '{output_filename}' created.")
    
    
    



    
    
    
    
    
    
    
    
    