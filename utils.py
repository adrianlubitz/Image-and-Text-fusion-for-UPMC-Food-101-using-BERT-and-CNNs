'''
Some utils to import in the notebook to make processing easier
'''

# System imports
import os
import re
import glob
# 3rd party imports
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import bert
import tensorflow_hub as hub
from sklearn.utils import shuffle



# local imports

# end file header
__author__      = 'Adrian Lubitz'

class Data():
    def __init__(self, test_path='texts/test_titles.csv', train_path='texts/train_titles.csv', load='test') -> None:

        # Parameters setting: images width and height, depth, number if classes, input shape
        self.batch_size =  80
        self.img_width = 299
        self.img_height = 299
        self.depth = 3
        self.max_length = 20 #Setup according to the text
        self.vec_get_missing = np.vectorize(self.get_missing, signature='(),(m,n)->(),(),()')  
        pos_load = ['test', 'train', 'both']
        self.colnames=['image_path', 'text', 'food']
        if load == 'train' or load == 'both':
            self.train = pd.read_csv(train_path, names=self.colnames, header=None, sep = ',', index_col=['image_path'])
            self.train = self.train.sort_values('image_path')
            self.train = self.add_not_found('images/train/*/*.jpg', self.train)
        if load == 'test' or load == 'both':
            self.test = pd.read_csv(test_path, names=self.colnames, header=None, sep = ',', index_col=['image_path'])
            self.test = self.test.sort_values('image_path')
            self.test = self.add_not_found('images/test/*/*.jpg', self.test)
        if load not in pos_load:
            raise ValueError(f'load must be one of {pos_load}')


        self.nClasses = self.test.food.nunique()
        self.Classes = self.test.food.unique()
        self.input_shape = (self.img_width, self.img_height, self.depth)
        self.vec_load_image = np.vectorize(self.load_image, signature = '()->(r,c,d),(s)')
        self.vec_get_text = np.vectorize(self.get_texts)
        self.TAG_RE = re.compile(r'<[^>]+>')
        self.vec_preprocess_text = np.vectorize(self.preprocess_text)
        self.vec_get_ids = np.vectorize(self.get_ids, signature = '(),(),()->(n)')
        BertTokenizer = bert.bert_tokenization.FullTokenizer
        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                    trainable=False)
        vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
        self.vec_get_masks = np.vectorize(self.get_masks, signature = '(),(),()->(n)')
        self.vec_get_segments = np.vectorize(self.get_segments, signature = '(),(),()->(n)')

    def get_missing(self, file, df):
        parts = file.split(os.sep)
        idx = parts[-1]
        cls = parts[-2]
        indexes = df[:,0]
        classes = df[:,2]

        if idx in indexes:
            text = df[idx == indexes][0,1]
            return pd.NA, pd.NA, pd.NA
        else:
            text = df[cls == classes][0,1]
            
        return idx, text, cls   


    def add_not_found(self, path, df):
        files = glob.glob(path)
        df = df.reset_index()
        idxs, texts, cls = self.vec_get_missing(files, df.values)
        
        found = pd.DataFrame({"text": texts,
                                "food": cls,
                            "image_path": idxs})
        na = found.isna().sum().values[0]
        if na<found.shape[0]:
            df = df.append(found)
        df = df.drop_duplicates(subset='image_path', keep='first').dropna()
        df = df.set_index('image_path')
        df = shuffle(df, random_state = 0)
        return df    

    def clean(self, i, tokens):
        try:
            this_token = tokens[i]
            next_token = tokens[i+1]
        except:
            return tokens
        if '##' in next_token:
            tokens.remove(next_token)
            tokens[i] = this_token + next_token[2:]
            tokens = self.clean(i, tokens)
            return tokens
        else:
            i = i+1
            tokens = self.clean(i, tokens)
            return tokens

    def clean_text(self, array):
        array = array[(array!=0) & (array != 101) & (array != 102)]
        tokens = self.tokenizer.convert_ids_to_tokens(array)
        tokens = self.clean(0, tokens)
        text = ' '.join(tokens)
        return text


# Images preprocessing
    def load_image(self, path):
        path = path.decode('utf-8')
        image = cv2.imread(path)
        image = cv2.resize(image, (self.img_width, self.img_height))
        image = image/255
        image = image.astype(np.float32)
        parts = path.split(os.sep)
        labels = parts[-2] == self.Classes 
        labels = labels.astype(np.int32)
        
        return image, labels
        
    

    # Dataset creation

    def prepare_data(self, paths):
        #Images and labels
        images, labels = tf.numpy_function(self.vec_load_image, 
                                        [paths], 
                                        [tf.float32, 
                                            tf.int32])
        
        
        [ids, segments, masks, ] = tf.numpy_function(self.prepare_text, 
                                                [paths], 
                                                [tf.int32, 
                                                tf.int32,
                                                tf.int32])
        images.set_shape([None, self.img_width, self.img_height, self.depth])
        labels.set_shape([None, self.nClasses])
        ids.set_shape([None, self.max_length])
        masks.set_shape([None, self.max_length])
        segments.set_shape([None, self.max_length])
        return ({"input_word_ids": ids, 
                "input_mask": masks,  
                "segment_ids": segments, 
                "image": images},
                {"class": labels})


    def get_texts(self, path):
        path = path.decode('utf-8')
        parts = path.split(os.sep)
        image_name = parts[-1]
        is_train = parts[-3] == 'train'
        if is_train:
            df = self.train
        else:
            df = self.test

        text = df['text'][image_name]
        return text
    

    def preprocess_text(self, sen):
        # Removing html tags
        sentence = self.remove_tags(sen)
        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = sentence.lower()
        return sentence

    def remove_tags(self, text):
        return self.TAG_RE.sub('', text)

    def get_ids(self, text, tokenizer, max_length):
        """Token ids from Tokenizer vocab"""
        tokens, length = self.get_tokens(text, tokenizer)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = np.asarray(token_ids + [0] * (max_length-length))
        return input_ids

    def get_tokens(self, text, tokenizer):
        tokens = tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        length = len(tokens)
        if length > self.max_length:
            tokens = tokens[:self.max_length]
        return tokens, length  

    def get_segments(self, text, tokenizer, max_length):
        """Segments: 0 for the first sequence, 1 for the second"""
        tokens, length = self.get_tokens(text, tokenizer)
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return np.asarray(segments + [0] * (max_length - len(tokens)))


    def prepare_text(self, paths):
        #Preparing texts
        
        texts = self.vec_get_text(paths)
        
        text_array = self.vec_preprocess_text(texts)
        ids = self.vec_get_ids(text_array, 
                        self.tokenizer, 
                        self.max_length).squeeze().astype(np.int32)
        masks = self.vec_get_masks(text_array,
                            self.tokenizer,
                            self.max_length).squeeze().astype(np.int32)
        segments = self.vec_get_segments(text_array,
                                    self.tokenizer,
                                    self.max_length).squeeze().astype(np.int32)
        
        return ids, segments, masks


    def get_masks(self, text, tokenizer, max_length):
        """Mask for padding"""
        tokens, length = self.get_tokens(text, tokenizer)
        return np.asarray([1]*len(tokens) + [0] * (max_length - len(tokens)))


    # Images loading using tf.data
    def tf_data(self, path, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        paths = tf.data.Dataset.list_files(path)
        paths = paths.batch(64)
        dataset = paths.map(self.prepare_data, tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.unbatch()
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        return dataset   