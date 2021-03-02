import sentence_transformers
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
import os


class FAQ_BERT(object):
    """ Class for predicting a continuous value from 0..1 for a given question-answer pair

    :param model_path: model path
    :param loss_type: loss type used during model training
    """ 
    def __init__(self, model_path, loss_type="triplet"):
        
        self.model_path = model_path
        self.loss_type = loss_type

        if self.loss_type not in {'triplet', 'softmax'}:
            raise ValueError('loss_type not exist')

        self.model = None
        if self.loss_type == "triplet":
            self.model = SentenceTransformer(self.model_path)
        elif self.loss_type == "softmax":
            self.model = CrossEncoder(self.model_path, num_labels=1)

    def predict(self, question, answer):
        """ Predict score for question-answer pair 
        The higher the score, question-answer pair is relevant.
        The lower the score, question-answer pair is relevant.
        A score of 1 represents positive label whereas 0 a negative label
        
        :param question: input question
        :param answer: input answer
        :return: score
        """        
        score = 0
        pair = [question, answer]
        if self.loss_type == "triplet":
            paraphrases = util.paraphrase_mining(self.model, pair)
            for paraphrase in paraphrases:
                score, _, _ = paraphrase
        elif self.loss_type == "softmax":
            score = self.model.predict(pair, convert_to_numpy=True, show_progress_bar=False)
            score = float(score)
        return score
