import sentence_transformers
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
from shared.utils import isDir
import numpy as np
import os


class FAQ_BERT(object):
    """ Class for predicting a continuous value from 0..1 for a given question-answer pair

    :param bert_model_path: trained FAQ BERT model path
    """ 
    def __init__(self, bert_model_path):
        
        self.bert_model_path = bert_model_path
        self.abs_path = ""
        self.model_path = ""
        self.model_dirname = ""
        
        dirs = self.bert_model_path.split('/')

        # Get model directory name
        if len(dirs) > 1:
            self.abs_path = self.bert_model_path.rsplit("/", 1)[0]
            self.model_dirname = self.bert_model_path.rsplit("/", 1)[-1]
            self.model_path = self.abs_path + "/" + self.model_dirname
        else:
            self.model_dirname = self.bert_model_path
            self.model_path = self.bert_model_path

        # Extract loss_type from model dirname
        self.params = self.model_dirname.split("_")
        self.loss_type = self.params[0]
       
        if not isDir(self.model_path):
            raise ValueError("model not found")
        
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
