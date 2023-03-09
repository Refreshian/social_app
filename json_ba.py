import io
import re
import json
import nltk
import pandas as pd
# import tensorflow_text
import os
from sklearn import manifold
from nltk.corpus import stopwords
import numpy

nltk.download('stopwords')
import tensorflow_hub as hub


class json_ba():

    def __init__(self):

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

        df = pd.DataFrame(dict_train)


        # метаданные
        # разбивка и сборка соцмедиа и СМИ в один датафрэйм с данными
        df_meta = pd.DataFrame()

        # случай выгрузки темы только по СМИ
        if 'hubtype' not in df.columns:

            dff = df
            dff['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
            df_meta_smi_only = dff[['timeCreate', 'hub', 'toneMark', 'audience', 'url', 'text', 'citeIndex']]
            # df_meta_smi_only.columns = ['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'text', 'citeIndex']
            df_meta_smi_only['fullname'] = dff['hub']
            df_meta_smi_only['author_type'] = 'Новости'
            df_meta_smi_only['hubtype'] = 'Новости'
            df_meta_smi_only['type'] = 'Новости'
            df_meta_smi_only['er'] = 0
            df_meta_smi_only['date'] = [x[:10] for x in df_meta_smi_only.index]
        #     df_meta_smi_only = df_meta_smi_only[columns]

            df_meta = df_meta_smi_only


        if 'hubtype' in df.columns:

            for i in range(2): # новости или соцмедиа

                    if i == 0:
                        dff = df[df['hubtype'] != 'Новости']
                        if dff.shape[0] != 0:

                            dff['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                            df_meta_socm = dff[['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'er', 'hubtype', 'text', 'type']]
                            df_meta_socm['fullname'] = pd.DataFrame.from_records(dff['authorObject'].values)['fullname'].values
                            df_meta_socm['author_type'] = pd.DataFrame.from_records(dff['authorObject'].values)['author_type'].values
                            df_meta_socm['date'] = [x[:10] for x in df_meta_socm.index]

                    if i == 1:
                        dff = df[df['hubtype'] == 'Новости']
                        if dff.shape[0] != 0:
                            dff['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                            df_meta_smi = dff[['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'er', 'hubtype', 'text', 'citeIndex']]
                            df_meta_smi['fullname'] = dff['hub']
                            df_meta_smi['author_type'] = 'Новости'
                            df_meta_smi['hubtype'] = 'Новости'
                            df_meta_smi['type'] = 'Новости'
                            df_meta_smi['date'] = [x[:10] for x in df_meta_smi.index]

            if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
                df_meta = pd.concat([df_meta_socm, df_meta_smi])
            elif 'df_meta_smi' and 'df_meta_socm' not in locals():
                df_meta = df_meta_smi
            else:
                df_meta = df_meta_socm



        self.df = df_meta

        # тексты
        self.df_text = self.df[['text']]
        self.df_meta = df_meta.drop('text', axis=1)

        # метаданные
        # columns = list(self.df.columns)
        # columns.remove('text')
        # self.df_meta = pd.concat([pd.DataFrame.from_records(self.df['authorObject'].values), self.df[columns]], axis=1)

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
        a = []
        for sent in self.sent_ru:
            # a.append(self.embed(sent)[0].numpy())
            a.append([numpy.round(x, 5) for x in self.embed(sent)[0].numpy()])

        self.dff = pd.DataFrame(a)

    def tsne_create(self):

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        x_tsne = tsne.fit_transform(self.dff.values)

        self.coord_list = []

        for i in range(len(x_tsne.tolist())):
            self.coord_list.append(', '.join([str(x) for x in x_tsne.tolist()[i]]))

        names = self.df_meta['fullname'].values.tolist()
        names = [x if x != '' else 'None' for x in names]

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

        self.names_list = [words_only(x) if type(x) != float else 'None' for x in names]
        self.names_list = [preprocess_text(x) if type(x) != float else 'None' for x in names]
        self.names_list = [remove_stopwords(x) if type(x) != float else 'None' for x in names]
        self.names_list = ['None' if x == '' else x for x in self.names_list]
