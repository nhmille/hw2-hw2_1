import sys
import os
import time
import torch
from torch.utils.data import DataLoader

from HW2_Supp import createVocab, createData, createDataLists, setupDataset, readBatch
from HW2_Supp import Encoder, Decoder, S2S, train, plotTraining, createOutputFile

start_time = time.time()

# test_1 = True

# if test_1:
if len(sys.argv) > 1:
    data_directory = sys.argv[1]
    output_filename = sys.argv[2]
    # data_directory = 'test_testing_data'
    # output_filename = 'out.txt'
    test_only = True
else:
    data_directory = ''
    output_filename = None
    test_only = False

# "best_model" was 128 hidden, 1 layer

hyperparameters ={  "batch_sz": 16,
                    "hidden_dim": 128,
                    "vocab_count": 3,
                    "layers": 1,
                    "lr": 0.00005,
                    "tf_ratio": 0.3,
                    "decay": 1e-4,
                    "drop": 0.5}
    
# @@@@ Change vocab to be defined from the data directory?
# "Directory" should be a folder name
vocab, word2idx, idx2word = createVocab(hyperparameters, data_directory)
vocab_sz=len(word2idx)

encoder = Encoder(hyperparameters)
decoder = Decoder(hyperparameters, vocab_size=vocab_sz)
s2s = S2S(encoder, decoder, word2idx, hyperparameters)

if test_only:
    model_filename = "best_model.pth"
    print(f"Loading saved model from {model_filename}")
    # s2s.load_state_dict(torch.load(model_filename))
    s2s.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
    s2s.eval()

    # Only need to evaluate on testing data    
    testing_data = createData(data_directory=data_directory)
    createOutputFile(s2s, testing_data, idx2word, word2idx, hyperparameters, name=output_filename)


else:
    training_data = createData(data_directory=data_directory)
    testing_data = createData(data_directory=data_directory, test=True)

    path_list, caption_list = createDataLists(training_data)
    path_list_t, caption_list_t = createDataLists(testing_data)

    training_dataset = setupDataset(path_list, caption_list)
    testing_dataset = setupDataset(path_list_t, caption_list_t)

    train_dataloader = DataLoader(training_dataset, batch_size=hyperparameters['batch_sz'], shuffle=True, collate_fn=lambda batch: readBatch(batch, word2idx))
    test_dataloader = DataLoader(testing_dataset, batch_size=hyperparameters['batch_sz'], shuffle=True, collate_fn=lambda batch: readBatch(batch, word2idx))

    train_loss_history, test_loss_history = train(s2s, train_dataloader, test_dataloader)

    plotTraining(train_loss_history, test_loss_history, hyperparameters)

    print(time.time() - start_time)

    createOutputFile(s2s, testing_data, idx2word, word2idx, hyperparameters, name=None)








