from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.BaseModel import BaseModel
import numpy as np

class SVMModel(BaseModel):
    def __init__(self):
        super().__init__("LSTM")

    def calculate_max_len(self, X):
        self.maxLen = 79
        # self.maxLen = len(max(X, key=len).split())

    def tokenize(self,X):
        vectorizer = TfidfVectorizer()
        vectorizer.fit(X)
        return vectorizer.transform(X)

    def build_model(self):
        self.model = SVC()

    def model_compile(self):
        self.model.summary()

    def model_fit(self, X, Y, xTest=None, yTest=None):
        param_grid = {'C': [0.1, 1], 'gamma': [0.1, 1], 'kernel': ['linear']}
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5)
        X = self.tokenize(X)
        grid_search.fit(X, Y)
        print("Best Parameters:", grid_search.best_params_)

        # Make predictions on the test data using the best model
        y_pred = grid_search.predict(self.tokenize(xTest))

        # Print classification report
        print("Classification Report:")
        print(classification_report(yTest, y_pred))

    def model_evaluate(self, xTest, yTest):
        pass

    def model_predict(self, sentenceArr):
        pass