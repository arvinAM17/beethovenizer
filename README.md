# Project description

This project aims as practice for an LSTM model, that embeds the input data, and predicts an output. 

The first attempt of this model was trained on the tiny Shakespeare text, with the goal of predicting the next word given a set of words. 
The eventual goal of this project is to train it on Beethoven's musical notes (using data available in Music21), so that given a sequence of notes, it could predict the next few notes in the style of Beethoven.

## Data processing
With the text data, we need to create a representation that's suitable for a neural network. 
We first get the data, and sent_tokenize each sentence. We remove sentences with too few or too many characters, since they would be outliers and skew the model. 
We then word_tokenize each sentence, removing sentences with too few or too many words. 
We then remove rare words, and create a dictionary of words to an index, (stoi) and the reverse dictionary (itos), which will allow us to think of words/tokens as one hot encodings, and also make the lookup of words simpler. We also add special tokens to the dictionaries, such as EOS (end of sentence), SOS (start of sentence), UNK (unknown token), PAD (padding token).
With this dictionary, we turn our sentences into arrays of encodings, and finally have our raw data in numerical format.
In order to use this data as a data loader and train/test our model, I've also made another consideration. Since each sentece has a different length, if we want to batch our data, the batches require same length sentences. 
A simple approach to this would be to make every sentence have the same length as the longest one. This approach has 2 caveats. It both makes the data unnecessarily big, and it also creates a lot of fluff within the training data, since we would have to look at or ignore a lot of padding for a lot of data points. 
My idea was to bucket batches based on sentence length. In order to do this, we would sort sentences by length, and have batches consist of similar-length sentences. That way, we minimize the data size, and minimize the total number of paddings, while still using the benefits of batching our data. 

## Model architecture
The architecture of this model is a simple one. 
- It uses an embedding layer to embed the input. This layer also considers the padding index, since our model does require a fixed length input within each batch, so we might need some padding per input.
- The output of the embedding layer is given to an LSTM block, with 2 layers.
- The short-term output of the last LSTM layer is given to a fully connected layer, that transforms that output into scores of our vocabulary of tokens. The best score is selected as the predicted token.

The loss function for this model is a simple CrossEntropy loss, since we're artifically considering logits for each token as the output of the model. An argmax output, works really well with a crossentropy on the softmax of the outputs. 
We optimize the model using an AdamW optimizer, which has an L2 regularization term inside. 
