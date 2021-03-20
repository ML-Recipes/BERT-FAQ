from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from sentence_transformers.cross_encoder.evaluation import  CEBinaryClassificationEvaluator
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.readers import TripletReader
from sentence_transformers.readers import InputExample
from sklearn.model_selection import train_test_split
from shared.utils import make_dirs, dump_to_json
from torch.utils.data import DataLoader
import pandas as pd 
import numpy as np
import logging
import torch
import math


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

class FAQ_BERT_Finetuning(object):
    """ Class for finetuning BERT on FAQ triplet dataset

    :param loss_type: train loss type triplet/softmax
    :param query_type: query type faq/user_query
    :param neg_type: negative type simple/hard
    :param version: model version number
    :param epochs: number of iterations
    :param batch_size: number of samples per training batch
    :param pre_trained_name: BERT pre-trained model name
    :param evaluation_steps: evaluation steps
    :param test_size: total number of samples in test set
    :param num_labels: param used for loss_type="softmax"
    """
    def __init__(self, loss_type="triplet", query_type='faq', neg_type='simple', version="1.1", epochs=4, batch_size=32, 
                 pre_trained_name='distilbert-base-uncased', evaluation_steps=1000, test_size=0.20, num_labels=1):
        
        self.loss_type = loss_type
        self.query_type = query_type
        self.neg_type = neg_type
        self.version = version
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.num_labels = num_labels
        self.pre_trained_name = pre_trained_name
        self.evaluation_steps = evaluation_steps
        self.model = None
        self.train_df = None
        self.test_df = None
        self.val_df = None

        if self.loss_type not in {'triplet', 'softmax'}:
            raise ValueError('loss_type not exist')
        
        self.bert_model = None
        if self.loss_type == "triplet":
          word_embedding_model = models.Transformer(self.pre_trained_name)

          # # Apply mean pooling to get one fixed sized sentence vector
          pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                          pooling_mode_mean_tokens=True,
                                          pooling_mode_cls_token=False,
                                          pooling_mode_max_tokens=False)

          self.bert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        elif self.loss_type == "softmax":
            self.bert_model = CrossEncoder(self.pre_trained_name, num_labels=self.num_labels)

    def split_train_val_test_sets(self, df):
        """ Generate train, test and validation sets from dataframe 
        
        :param df: dataframe
        :return: train, test, val sets
        """
        train, temp = train_test_split(df, test_size=self.test_size)
        val, test = train_test_split(temp, test_size=0.5)
        return train, val, test
    
    def generate_triplets(self, df):
        """ Generate triplets from a given DataFrame 
        
        :param df: pandas DataFrame
        :return: list of triplets 
        """
        triplets = []
        for _, row in df.iterrows():
            if self.loss_type == "triplet":
                question = row['question']
                positive = row['positive']
                negative = row['negative']
                triplets.append(InputExample(texts=[question, positive, negative], label=0))
            elif self.loss_type == "softmax":
                question = row['question']
                answer = row['answer']
                label = row['label']
                triplets.append(InputExample(texts=[question, answer], label=label))
        return triplets

    def create_model(self, df, output_path):
        """ Finetune BERT model on FAQ dataset and generate model at given path

        :param df: pandas DataFrame
        :param output_path: path to save model
        """

        try:
            
            logging.info("Generating directory structure")
            # Define output path
            subdir = "{}_{}_{}_{}".format(self.loss_type, self.neg_type, self.query_type, self.version)

            models_path = output_path + "/models/" + subdir
            eval_path = output_path + "/evaluation/" + subdir
            label_index_path = output_path + "/label_index/"

            # Create directories
            make_dirs(models_path)
            make_dirs(eval_path)

            logging.info("Generating triplets for training")
            # split dataframe into train, test, validation
            self.train_df, self.val_df, self.test_df = self.split_train_val_test_sets(df)
            
            # generate triplets for train, test, validation
            train_samples = self.generate_triplets(self.train_df)
            test_samples = self.generate_triplets(self.val_df)
            val_samples = self.generate_triplets(self.test_df)

            logging.info("Training model")
            self.model = self.train(self.bert_model, train_samples, val_samples, models_path)
            
            logging.info("Evaluating model")
            self.model = self.evaluate(test_samples, models_path)

            # Dump train, test, val to csv files
            self.train_df.to_csv(eval_path + "/train.csv", index=False)
            self.test_df.to_csv(eval_path + "/test.csv", index=False)
            self.val_df.to_csv(eval_path + "/val.csv", index=False)

        except Exception:
            logging.error('error occured', exc_info=True)   


    def train(self, bert_model, train_samples, val_samples, output_path):
        """ Train model using BERT pre-trained model using train, val triplets 
        
        :param bert_model: BERT pre-trained model
        :param train_samples: train triplets
        :param val_samples: validation triplets
        :param output_path: path to save model
        :return: trained model
        """
        model = None

        if self.loss_type == "triplet":
            # generate train dataloader
            train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=self.batch_size)
            train_loss = losses.TripletLoss(model=bert_model)
            evaluator = TripletEvaluator.from_input_examples(val_samples, name='val')
            
            # train model
            warmup_steps = int(len(train_dataloader) * self.epochs * 0.1) # 10% of train data
            model = bert_model.fit(
                train_objectives=[(train_dataloader, train_loss)], evaluator=evaluator,
                epochs=self.epochs, evaluation_steps=self.evaluation_steps, 
                warmup_steps=warmup_steps, output_path=output_path
            )
        elif self.loss_type == "softmax":
            # generate train dataloader
            train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=self.batch_size)
            evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_samples)
      
            # train the model
            warmup_steps = int(len(train_dataloader)*self.epochs/self.batch_size*0.1) #10% of train data
            model = bert_model.fit(train_dataloader=train_dataloader, evaluator=evaluator,
                epochs=self.epochs, evaluation_steps=self.evaluation_steps, loss_fct=None,
                warmup_steps=warmup_steps, output_path=output_path,
            )
        
        return model

    def evaluate(self, test_samples, output_path):
        """ Evaluate generated model using test triplets

        :param test_samples: test triplets
        :param output_path: path to load saved model
        :return: trained model
        """
        
        model = None
        
        if self.loss_type == "triplet":
            # load the stored model and evaluate its performance on test set
            model = SentenceTransformer(output_path)
            test_evaluator = TripletEvaluator.from_input_examples(test_samples, name='test')
            test_evaluator(model, output_path=output_path)
        
        elif self.loss_type == "softmax":
            # load the stored model and evaluate its performance on test set
            model = CrossEncoder(output_path, num_labels=self.num_labels)
            test_evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples, name='test')
            test_evaluator(model, output_path=output_path)

        return model

    