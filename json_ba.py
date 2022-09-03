import io
import re
import json
import nltk
import pandas as pd
import tensorflow_text
import os, psutil
from sklearn import manifold
from nltk.corpus import stopwords
nltk.download('stopwords')
import tensorflow_hub as hub


class json_ba():

    def __init__(self):

        self.df = pd.DataFrame()
        self.embed = hub.load("universal-sentence-encoder-multilingual_3")
        a = []
        

    def open_file(self, filename):
    
        with io.open(filename, encoding='utf-8', mode='r') as train_file:
            dict_train = json.load(train_file)

        self.df = pd.DataFrame(dict_train)

        # тексты 
        self.df_text = self.df[['text']]

        # метаданные
        columns = list(self.df.columns)
        columns.remove('text')
        self.df_meta = pd.concat([pd.DataFrame.from_records(self.df['authorObject'].values), self.df[columns]], axis=1)
    
    
    def preprocess_text(self):
    
        regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

        def words_only(text, regex=regex):
            try:
                return " ".join(regex.findall(text))
            except:
                return ""


        mystopwords = stopwords.words('russian') + ['это', 'наш' , 'тыс', 'млн', 'млрд', 'также',  'т', 'д', 'URL', 
                                                    'i', 's', 'v', 'info', 'a', 'подробнее', 'который', 'год', 
                                                   ' - ', '-','В','—', '–', '-', 'в', 'который']

        def preprocess_text(text):
            text = text.lower().replace("ё", "е")
            text = re.sub('((www\[^\s]+)|(https?://[^\s]+))','URL', text)
            text = re.sub('@[^\s]+','USER', text)
            text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
            text = re.sub(' +',' ', text)
            return text.strip()

        def  remove_stopwords(text, mystopwords = mystopwords):
            try:
                return " ".join([token for token in text.split() if not token in mystopwords])
            except:
                return ""


        self.df_text['text'] = self.df_text['text'].apply(words_only)
        self.df_text['text'] = self.df_text['text'].apply(preprocess_text)
        self.df_text['text'] = self.df_text['text'].apply(remove_stopwords)

        self.sent_ru = self.df_text['text'].values
    
    
    def create_embed(self):
        a = []
        for sent in self.sent_ru:
            a.append(self.embed(sent)[0].numpy())

        self.dff = pd.DataFrame(a)

        
    def tsne_create(self):
        
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        x_tsne = tsne.fit_transform(self.dff.values)

        self.coord_list = []

        for i in range(len(x_tsne.tolist())):
            self.coord_list.append(', '.join([str(x) for x in x_tsne.tolist()[i]]))

        names = self.df_meta['fullname'].values.tolist()
        names = [x if x != '' else 'None' for x in names]
        self.names_list = []

        for i in range(len(names)):
            self.names_list.append(names[i])

        print("!!!!!+++++=====")
        print(self.names_list[:2])