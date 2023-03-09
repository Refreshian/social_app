import io
import re
import json
import nltk
import pandas as pd
# import tensorflow_text
import os
from sklearn import manifold
from nltk.corpus import stopwords
import tensorflow_text

nltk.download('stopwords')
import tensorflow_hub as hub
from tqdm import tqdm
path_to_files = '/home/dev/social_app/data'

class json_ba_not_empty_text():

    def __init__(self):

        os.chdir(path_to_files)
        self.df = pd.DataFrame()
        self.embed = hub.load("universal-sentence-encoder-multilingual_3")

    def open_file(self, filename):

        try: 
            with io.open(filename, encoding='utf-8', mode='r') as train_file:
                dict_train = json.load(train_file, strict=False)
       
        except:
            a = []
            with open(filename, encoding='utf-8', mode='r') as file:
                for line in file:
                    a.append(line)
            dict_train = []
            for i in range(len(a)):
                try:
                    dict_train.append(ast.literal_eval(a[i]))
                except:
                    continue
            dict_train = [x[0] for x in dict_train]

        self.df = pd.DataFrame(dict_train)

        # тексты
        self.df_text = self.df[['text']]

        # метаданные
        columns = list(self.df.columns)
        columns.remove('text')
        self.df_meta = pd.concat([pd.DataFrame.from_records(self.df['authorObject'].values), self.df[columns]], axis=1)

    def preprocess_texts(self):

        regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

        def words_only(text, regex=regex):
            try:
                return " ".join(regex.findall(text))
            except:
                return ""

        mystopwords = stopwords.words('russian') + ['это', 'наш', 'тыс', 'млн', 'млрд', 'также', 'т', 'д', 'URL',
                                                    'i', 's', 'v', 'info', 'a', 'подробнее', 'который', 'год',
                                                    ' - ', '-', 'В', '—', '–', '-', 'в', 'который']

        def preprocess_text(text):
            text = text.lower().replace("ё", "е")
            text = re.sub('((www\[^\s]+)|(https?://[^\s]+))', 'URL', text)
            text = re.sub('@[^\s]+', 'USER', text)
            text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
            text = re.sub(' +', ' ', text)
            return text.strip()

        def remove_stopwords(text, mystopwords=mystopwords):
            try:
                return " ".join([token for token in text.split() if not token in mystopwords])
            except:
                return ""

        self.df_text['text'] = self.df_text['text'].apply(words_only)
        self.df_text['text'] = self.df_text['text'].apply(preprocess_text)
        self.df_text['text'] = self.df_text['text'].apply(remove_stopwords)

        self.sent_ru = self.df_text['text'].values

    def create_embed(self):
        self.embed_list = []
        for sent in tqdm(self.sent_ru):
            self.embed_list.append(self.embed(sent)[0].numpy())

        names = self.df_meta['fullname'].values.tolist()
        names = [x if x != '' else 'None' for x in names]
        self.names_list = names