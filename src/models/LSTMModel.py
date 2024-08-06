import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Input, Dropout, LSTM, Activation, Bidirectional

from src.models.BaseModel import BaseModel


class LSTMModel(BaseModel):
    def __init__(self):
        super().__init__("LSTM")
        self.wordsToIndex = None
        self.wordVecMap = None
        self._read_glove()

    def _read_glove(self):
        '''
            read GloVE embeddings and build two dictionaries word to vector map, and word to index map
        '''
        with open('data/glove.6B.50d.txt', 'r') as f:
            wordVecMap = {}
            gloveWords = set()
            for line in f:
                line = line.strip().split()
                gloveWords.add(line[0])
                wordVecMap[line[0]] = np.array(line[1:], dtype=np.float64)

            index = 1
            wordsToIndex = {}
            for word in sorted(gloveWords):
                wordsToIndex[word] = index
                index = index + 1

        self.wordVecMap = wordVecMap
        self.wordsToIndex = wordsToIndex

    def set_max_len(self, maxLen):
        '''
            set maximum length of text.
        '''
        self.maxLen = maxLen

    def glove_embedding_layer(self):
        '''
            Freeze training of embedding layer, and load GloVE vectors

            Returns:
            embeddingLayer: Keras embedding layer
        '''
        vocabSize = len(self.wordsToIndex) + 1

        # Find the shape of embedding matrix GLoVE vectors
        embDim = self.wordVecMap[list(self.wordVecMap.keys())[0]].shape[0]

        # Set the embedding layer, this layer is not trainable because
        # glove vectors and embeddings are pre-trained
        embeddingLayer = Embedding(vocabSize, embDim, trainable=False)

        # zero initialisation of embedding matrix
        embMatrix = np.zeros((vocabSize, embDim))


        # Build the embedding layer
        embeddingLayer.build((None,))

        # Filling the respective values in the embedding matrix
        # by finding words in the dictionary
        for word, idx in self.wordsToIndex.items():
            embMatrix[idx, :] = self.wordVecMap[word]

        # Set the pretrained weights
        embeddingLayer.set_weights([embMatrix])

        return embeddingLayer

    def tokenize(self, X):
        '''
            Zero pad smaller sentences so that all the texts are of equal length. Array of strings to array of Glove word indices

            @params: X : numpy array, the input array of strings
            @return: xIndices: numpy array, Array of indices (string to vector conversion) 
        '''
        noOfRecords = X.shape[0]

        # Fill in a 2D matrix of size noOfRecords X maxLen with zeroes so that
        # shorter sentences of zeroes automatically padded at the end before input
        xIndices = np.zeros((noOfRecords, self.maxLen))

        for record in range(noOfRecords):
            rowWords = X[record].split()
            
            col = 0

            for word in rowWords:
                if word in self.wordsToIndex:
                    xIndices[record, col] = self.wordsToIndex[word]
                    col = col + 1
        print(xIndices.shape)
        return xIndices

    def build_model(self):
        '''
            Build model
        '''
        print("the length is", self.maxLen)
        inputLayer = Input((self.maxLen, ), dtype='int32')
        embeddingLayer = self.glove_embedding_layer()
        embeddings = embeddingLayer(inputLayer)

        # Add Bidirectional LSTM layer with 128 dimensional hidden state
        X = Bidirectional(LSTM(128, return_sequences=True))(embeddings)
        # Prevent overfitting
        X = Dropout(0.5)(X)
        # Add Bidirectional LSTM layer with 128 dimensional hidden state
        X = Bidirectional(LSTM(128, return_sequences=False))(X)
        # Prevent overfitting
        X = Dropout(0.5)(X)
        # 6 classes
        X = Dense(6)(X)
        X = Activation('softmax')(X)
        model = Model(inputLayer, X)
        self.model = model

    def model_compile(self):
        '''
            compile model with categorical_crossentropy and adam optimization
            Also prints the summary of model compiled
        '''
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def model_fit(self, X, Y, xTest=None, yTest=None, epochs=20):
        '''
            Fit the model build against training data
            @param: X: dataframe column Training set input text column
            @param: Y: dataframe column Training set output label column
            @param: xTest: dataframe column testing set input text column
            @param: yTest: dataframe column testing set output label column
            @epochs: int number of epochs
            @return history the epoch training history
        '''
        xInd = self.tokenize(np.asarray(X))
        yTrainOneHot = self.convert_to_one_hot(np.asarray(Y), 6)
        xTestInd = self.tokenize(np.asarray(xTest))
        yTestOneHot = self.convert_to_one_hot(np.asarray(yTest), 6)

        # Validation data is sent to calculate validation accuracy in each epoch
        history = self.model.fit(xInd, yTrainOneHot, epochs = epochs, batch_size = 1000, shuffle=True, validation_data=(xTestInd, yTestOneHot))
        return history

    def model_evaluate(self, xTest, yTest):
        '''
            evaluate model performance against test data
        '''
        xTestInd = self.tokenize(np.asarray(xTest))
        yTrainOh = self.convert_to_one_hot(np.asarray(yTest), 6)
        loss, acc = self.model.evaluate(xTestInd, yTrainOh)
        return loss, acc

    def model_predict(self, sentenceArr):
        '''
            Predict on own data
        '''
        sentenceIndices = self.tokenize(sentenceArr)
        pred = np.argmax(self.model.predict(sentenceIndices))
        return pred