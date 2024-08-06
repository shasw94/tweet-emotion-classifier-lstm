import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from src.models.LSTMModel import LSTMModel

try:
    nltk.data.find('tokenizers/punkt')
    nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')


def read_raw_data(path='data/text.csv'):
    return pd.read_csv(path)


def preprocess_data(rawData):
    cleanData = rawData.copy()

    # remove non-alphabetic characters
    cleanData['text'] = cleanData['text'].str.replace(r'[^\w\s]', '', regex=True)

    # remove multiple whitespaces
    cleanData['text'] = cleanData['text'].str.replace(r'\s+', ' ', regex=True)

    # remove digits
    cleanData['text'] = cleanData['text'].str.replace(r'\d+', '', regex=True)

    # remove extra spaces
    cleanData['text'] = cleanData['text'].str.lower()

    # remove stop words
    stop = stopwords.words('english')

    # remove stop words
    cleanData["text"] = cleanData['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    # drop duplicates in text column after cleaning
    cleanData = cleanData.drop_duplicates()

    return cleanData


def main():
    rawData = read_raw_data()
    cleanData = preprocess_data(rawData)
    xTrain, xTest, yTrain, yTest = train_test_split(cleanData['text'], cleanData['label'], test_size=0.2,
                                                    random_state=20)
    methods = {'LSTM': LSTMModel() }
    for key, val in methods.items():
        model = val.load_model(key)
        if model is None:
            val.calculate_max_len(xTrain)
            val.build_model()
            val.model_compile()
            yTrainOneHot = val.convert_to_one_hot(np.asarray(yTrain), 6)
            history = val.model_fit(xTrain, yTrainOneHot)
            val.save_model(key+'.keras')

        val.draw_training_graphs(history)
        model = val.model
        print(val.model_evaluate(xTest, yTest))
        test = np.array(['i am delighted today', 'i feel like crying'])
        print(val.model_predict(test))


# main()
