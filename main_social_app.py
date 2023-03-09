import collections
import datetime
from datetime import timedelta
import glob
import html as htl
import codecs
import json
import os
import io
import re
from collections import Counter, OrderedDict, defaultdict
from operator import itemgetter
import functools as ft
import time
from sklearn.cluster import KMeans
from threading import Thread
from markupsafe import Markup
from os import listdir
from os.path import isfile, join
from gensim.models.ldamodel import LdaModel
from tqdm import tqdm
import ast
from sklearn import manifold

import requests
import sys
import traceback
import urllib
import shutil
import plotly

import flask
import numpy
import pandas as pd
from json_ba import json_ba
from io import BytesIO

from pandas_datareader import data as web
from scipy import stats

import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI 

from werkzeug.security import check_password_hash, generate_password_hash
from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory, flash
from models import Users

from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from multiprocessing import Process, freeze_support
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import gensim

from bertopic import BERTopic
topic_model_rus = BERTopic(low_memory=True,verbose=True,calculate_probabilities=True,language="russian", min_topic_size=3)
topic_model_multi = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True)

app = Flask(__name__)
app.debug = True
app.secret_key = 'a(d)fs#$T12eF#4-key'
app.permanent_session_lifetime = timedelta(days=15)

# path_to_files = "/home/centos/home/centos/social_app/data"
path_to_files = '/home/dev/social_app/data'
path_to_embed_save = '/home/dev/social_app/data/embed_storage'
path_bert_topic_data = '/home/dev/social_app/data/BertTopic'
path_Lda_topic_data = '/home/dev/social_app/data/LdaTopic'
templates_path = '/home/dev/social_app/templates'
path_tsne_data = '/home/dev/social_app/data/TSNE'
path_projector_files = '/home/dev/social_app/data/projector_files'


# path_to_Bert_models = 'D:/social_app_vsc/data/Tonality/BertModels/rubert-tiny'
path_themes_tonality_models = '/home/dev/social_app/data/tonality/tonality_themes_models'


session = flask.session
app.config['UPLOAD_FOLDER'] = path_to_files


from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
# from wtforms import SelectMultipleField, Form

engine = create_engine("postgresql://postgres:ffsfds&fdv12w@localhost:5432/datadb")
session_factory = sessionmaker(bind=engine)
session_scoped = scoped_session(session_factory)


# postgresql://user:password@localhost:5432/v2tdb
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:ffsfds&fdv12w@localhost:5432/datadb'
SQLALCHEMY_TRACK_MODIFICATIONS = False
db = SQLAlchemy(app)



regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")
def words_only(text, regex=regex):
    try:
        return " ".join(regex.findall(text))
    except:
        return ""

def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('((www\[^\s]+)|(https?://[^\s]+))','URL', text)
    text = re.sub('@[^\s]+','USER', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +',' ', text)
    return text.strip()

mystopwords = stopwords.words('russian') + ['это', 'наш' , 'тыс', 'млн', 'млрд', 'также',  'т', 'д', 'URL',
                                            'i', 's', 'v', 'info', 'a', 'подробнее', 'который', 'год',
                                        ' - ', '-','В','—', '–', '-', 'в', 'который']

def  remove_stopwords(text, mystopwords = mystopwords):
    try:
        return " ".join([token for token in text.split() if not token in mystopwords])
    except:
        return ""

# def parse():
#     # parsing json & create embedding
#     X = json_ba()
#     X.open_file(session['filename'])
#     X.preprocess_texts()
#     X.create_embed()
#     X.tsne_create()
#     return X


@app.route("/")
def hello():
    return redirect(url_for('index'))


@app.route("/index", endpoint='index')
def menu():
    return render_template('index.html')


@app.route('/authors', methods=['GET', 'POST'], endpoint='authors')
def graph():

    json_files = [x + '.json' for x in os.listdir(path_tsne_data)]
    len_files = len(json_files)

    if 'send' in request.values.to_dict(flat=True):
        filename = request.values.to_dict(flat=True)['file_choose']

        os.chdir(path_tsne_data + '/' + filename.replace('.json', ''))
        tsne_data = codecs.open(filename.replace('.json', '_data_tsne.txt'), 'r', encoding='utf-8').read()
        tsne_data = json.loads(tsne_data)

        names = tsne_data['author_name_str']
        coord_list_str = tsne_data['coord_list_str']

        return render_template('visualizer.html', names=names, coord=coord_list_str, files=json_files, len_files=len_files)
 
    return render_template('authors_cluster.html', files=json_files, len_files=len_files)



@app.route('/tsne', methods=('GET', 'POST'))
def tsne():
    return render_template('visualizer.html')


@app.route('/tonality_landscape', methods=['GET', 'POST'], endpoint='tonality_landscape')
def tonality():

    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files)

    if 'send' in request.values.to_dict(flat=True):

        session['filename'] = request.values.to_dict(flat=True)['file_choose']

        # parsing json
        try: 
            with io.open(session['filename'], encoding='utf-8', mode='r') as train_file:
                dict_train = json.load(train_file, strict=False)
       
        except:
            a = []
            with open(session['filename'], encoding='utf-8', mode='r') as file:
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
        columns = ['hub', 'toneMark', 'fullname', 'date']

        # случай выгрузки темы только по СМИ
        if 'hubtype' not in df.columns:

            dff = df
            dff['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
            df_meta_smi_only = dff[['timeCreate', 'hub', 'toneMark']]
            df_meta_smi_only['fullname'] = dff['hub']
            df_meta_smi_only.dropna(subset=['timeCreate'], inplace=True)
            df_meta_smi_only = df_meta_smi_only.set_index(['timeCreate'])
            df_meta_smi_only['date'] = [x[:10] for x in df_meta_smi_only.index]
            df_meta_smi_only = df_meta_smi_only[columns]

            df_meta = df_meta_smi_only
            
            
        if 'hubtype' in df.columns:

            for i in range(2): # новости или соцмедиа

                    if i == 0:
                        dff = df[df['hubtype'] != 'Новости']
                        if dff.shape[0] != 0:
                            dff['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                            df_meta_socm = pd.concat([pd.DataFrame.from_records(dff['authorObject'].values), dff], axis=1)
                            df_meta_socm.dropna(subset=['timeCreate'], inplace=True)
                            df_meta_socm = df_meta_socm.set_index(['timeCreate'])
                            df_meta_socm['date'] = [x[:10] for x in df_meta_socm.index]
                            df_meta_socm = df_meta_socm[columns]
                            df_meta_socm['toneMark'] = df_meta_socm['toneMark'].astype(int)
                    #         df_meta_socm.reset_index(inplace=True)

                    if i == 1:
                        dff = df[df['hubtype'] == 'Новости']
                        if dff.shape[0] != 0:
                            dff['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                            df_meta_smi = dff[['timeCreate', 'hub', 'toneMark']]
                            df_meta_smi['fullname'] = dff['hub']
                            df_meta_smi.dropna(subset=['timeCreate'], inplace=True)
                            df_meta_smi = df_meta_smi.set_index(['timeCreate'])
                            df_meta_smi['date'] = [x[:10] for x in df_meta_smi.index]
                            df_meta_smi = df_meta_smi[columns]
                    #         df_meta_smi.reset_index(inplace=True)
            

            if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
                df_meta = pd.concat([df_meta_socm, df_meta_smi])
            elif 'df_meta_smi' and 'df_meta_socm' not in locals():
                df_meta = df_meta_smi
            else:
                df_meta = df_meta_socm

        def date_reverse(date):
            lst = date.split('-')
            temp = lst[1]
            lst[1] = lst[2]
            lst[2] = temp
            return lst

        data_start = '-'.join(date_reverse('-'.join(
            [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][::-1])))
        data_stop = '-'.join(date_reverse('-'.join(
            [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][::-1])))

        mask = (df_meta['date'] >= data_start) & (df_meta['date'] <= data_stop)
        df_meta = df_meta.loc[mask]
        df_meta.reset_index(inplace=True)

        if df_meta.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
            flash('По запросу найдено 0 сообщений (проверьте даты и/или другие условия)')
            return redirect(url_for('tonality_landscape'))

        # негатив и позитив по площадкам (соцмедиа)
        # hub_neg = Counter(df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] == 1)]['hub'].values)
        # hub_pos = Counter(df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] == -1)]['hub'].values)

        hub_neg = Counter(df_meta[df_meta['toneMark'] == 1]['hub'].values)
        hub_pos = Counter(df_meta[df_meta['toneMark'] == -1]['hub'].values)

        # df_meta['date'] = [x[:10] for x in df_meta.index]  # столюец с датами без часов/минут/сек
        # neg_tabl = df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] == 1)]
        # pos_table = df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] == -1)]
        neg_tabl = df_meta[df_meta['toneMark'] == 1]
        pos_table = df_meta[df_meta['toneMark'] == -1]

        neg_list_data = list(OrderedDict(Counter(neg_tabl['date'].values)).values())
        pos_list_data = list(OrderedDict(Counter(pos_table['date'].values)).values())

        neg_list_name = list(Counter(neg_tabl['date'].values[::-1]).keys())
        pos_list_name = list(Counter(pos_table['date'].values[::-1]).keys())

        data_tonality_hub_neg_data = [x[1] for x in sorted((hub_neg).items(), key=itemgetter(1), reverse=True)]
        data_tonality_hub_pos_data = [x[1] for x in sorted((hub_pos).items(), key=itemgetter(1), reverse=True)]
        data_tonality_hub_neg_name = [x[0] for x in sorted((hub_neg).items(), key=itemgetter(1), reverse=True)]
        data_tonality_hub_pos_name = [x[0] for x in sorted((hub_pos).items(), key=itemgetter(1), reverse=True)]

        superDonatName = request.values.to_dict(flat=True)['file_choose'].split('_')[0].replace('.json', '')

        neg_authors = []
        for i in range(len(list(hub_neg.keys()))):
            neg_authors.append([list(hub_neg.keys())[i],
                                list(Counter(
                                    neg_tabl[neg_tabl['hub'] == list(hub_neg.keys())[i]]['fullname'].values).keys()),
                                list(Counter(
                                    neg_tabl[neg_tabl['hub'] == list(hub_neg.keys())[i]]['fullname'].values).values())])

        neg1 = [x[0] for x in neg_authors]
        neg2 = [x[1] for x in neg_authors]
        neg3 = [x[2] for x in neg_authors]

        neg2 = [['None' if type(x) == float else x for x in group] for group in neg2]
        neg2 = [[x.replace('"', '') for x in group] for group in neg2]  # для корректной передачи в javaS
        neg2 = [[words_only(x) for x in group] for group in neg2]  # для корректной передачи в javaS


        pos_authors = []
        for i in range(len(list(hub_pos.keys()))):
            pos_authors.append([list(hub_pos.keys())[i],
                                list(Counter(
                                    pos_table[pos_table['hub'] == list(hub_pos.keys())[i]]['fullname'].values).keys()),
                                list(Counter(
                                    pos_table[pos_table['hub'] == list(hub_pos.keys())[i]][
                                        'fullname'].values).values())])

        pos1 = [x[0] for x in pos_authors]
        pos2 = [x[1] for x in pos_authors]
        pos3 = [x[2] for x in pos_authors]

        ar = [[x if type(x) == float else 0 for x in group] for group in pos2]
        ar = [x for x in ar if x != 0]
        print(ar)
        pos2 = [['None' if type(x) == float else x for x in group] for group in pos2]
        pos2 = [[x.replace('"', '') for x in group] for group in pos2]  # для корректной передачи в javaS
        # pos2 = [[words_only(x) for x in group] for group in pos2]  # для корректной передачи в javaS
        # pos2 = [[x.replace('"', '') if type(x) != float else x for x in group] for group in pos2]

        len_neg = numpy.sum(data_tonality_hub_neg_data)
        len_pos = numpy.sum(data_tonality_hub_pos_data)

        percent_pos = int(numpy.sum(data_tonality_hub_neg_data))
        percent_neg = int(numpy.sum(data_tonality_hub_pos_data))

        if percent_pos == 0 and percent_neg == 0: # если не было негативных или позитивных сообщений
            flash('Не найдено негативных и позитивных сообщений!')
            return redirect(url_for('tonality_landscape'))

        if percent_neg == 0:
            percent_neg = 0
            percent_pos = 100
        elif percent_pos == 0:
            percent_neg = 100
        else:
            count_sum = percent_pos + percent_neg
            percent_pos = np.round((percent_pos / count_sum), 2)*100
            percent_neg = np.round((percent_neg / count_sum), 2)*100

        data_tonality_hub_neg_name = [x.replace('"', '') for x in data_tonality_hub_neg_name]
        data_tonality_hub_pos_name = [x.replace('"', '') for x in data_tonality_hub_pos_name]

        # data_tonality_hub_neg_name = [words_only(x) for x in data_tonality_hub_neg_name]
        # data_tonality_hub_pos_name = [words_only(x) for x in data_tonality_hub_pos_name]

        neg1 = [x.replace('"', '') for x in neg1]
        pos1 = [x.replace('"', '') for x in pos1]

        neg1 = [words_only(x) for x in neg1]
        pos1 = [words_only(x) for x in pos1]

        data = {
            "neg_list_data": neg_list_data,
            "neg_list_name": neg_list_name,
            "pos_list_data": pos_list_data,
            "pos_list_name": pos_list_name,

            "data_tonality_hub_neg_data": data_tonality_hub_neg_data,
            "data_tonality_hub_pos_data": data_tonality_hub_pos_data,
            "data_tonality_hub_neg_name": data_tonality_hub_neg_name,
            "data_tonality_hub_pos_name": data_tonality_hub_pos_name,

            "superDonatName": superDonatName,

            "hub_neg_val": list(hub_neg.values()),
            "hub_pos_val": list(hub_pos.values()),

            "neg1": neg1,
            "neg2": neg2,
            "neg3": neg3,

            "pos1": pos1,
            "pos2": pos2,
            "pos3": pos3,

            "percent_pos": percent_pos,
            "percent_neg": percent_neg,

            "len_neg": int(len_neg),
            "len_pos": int(len_pos),
        }


        date = data_start + ' : ' + data_stop

        return render_template('tonality_landscape.html', files=json_files, len_files=len_files, datagraph=data,
                            object_name=superDonatName, date=date)

    # data = {
    #     'neg_list_data': [11, 12, 16, 9],
    #     'neg_list_name': ['S', 'M', 'F', 'K'],
    #     'pos_list_data': [8, 6, 7, 9],
    #     'pos_list_name': ['H', 'G', 'T', 'P'],
    #     "superDonatName": 'test'
    # }

    return render_template('tonality_landscape.html', files=json_files, len_files=len_files)


@app.route('/datalake', methods=['GET', 'POST'], endpoint='datalake')
def show_data():
    if request.method == 'POST' and 'filename' in request.files:
        uploaded_file = request.files['filename']
        if uploaded_file.filename != '':
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))
            session['filename'] = uploaded_file.filename
        return redirect(url_for('start_create_embed'))

    os.chdir(path_to_files)

    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    name_files = [x.split('_')[0] for x in json_files]
    name_files = [x.split('.')[0] for x in name_files]
    name_files = list(set(name_files))
    len_files = len(json_files)

    files_dict = {}
    for i in range(len(name_files)):
        files_dict[name_files[i]] = [x for x in json_files if name_files[i] in x]

    return render_template('datalake.html', files=json_files, name_files=name_files, files_dict=files_dict,
                           len_files=len_files)


# https://stackoverflow.com/questions/48994440/execute-a-function-after-flask-returns-response/63080968#63080968
@app.route('/start_create_embed', methods=['GET', 'POST'], endpoint='start_create_embed')
def start_task():
    def do_work(filename):

        print('Start Preprocess file data!')

        os.chdir(path_to_files)
        embed = hub.load("universal-sentence-encoder-multilingual_3")

        # parsing json
        try: 
            with io.open('Як141 - аудитория.json', encoding='utf-8', mode='r') as train_file:
                dict_train = json.load(train_file, strict=False)

        except:
            a = []
            with open('Як141 - аудитория.json', encoding='utf-8', mode='r') as file:
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
        # метаданных
        df = df[['text', 'url', 'hub', 'timeCreate']]

        print('start')
        print(datetime.datetime.now())

        regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")
        def words_only(text, regex=regex):
            try:
                return " ".join(regex.findall(text))
            except:
                return ""

        def preprocess_text(text):
            text = text.lower().replace("ё", "е")
            text = re.sub('((www\[^\s]+)|(https?://[^\s]+))','URL', text)
            text = re.sub('@[^\s]+','USER', text)
            text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
            text = re.sub(' +',' ', text)
            return text.strip()

        mystopwords = stopwords.words('russian') + ['это', 'наш' , 'тыс', 'млн', 'млрд', 'также',  'т', 'д', 'URL',
                                                    'i', 's', 'v', 'info', 'a', 'подробнее', 'который', 'год',
                                                ' - ', '-','В','—', '–', '-', 'в', 'который']

        def  remove_stopwords(text, mystopwords = mystopwords):
            try:
                return " ".join([token for token in text.split() if not token in mystopwords])
            except:
                return ""

        print('preprocess')
        
        sent_ru = df['text'].values
        sent_ru = [preprocess_text(x) for x in sent_ru]
        sent_ru = [remove_stopwords(x) for x in sent_ru]
        sent_ru = [words_only(x) for x in sent_ru]

        df['text'] = df['text'].apply(preprocess_text)
        # df['text'] = df['text'].apply(remove_stopwords)
        df['text'] = df['text'].apply(words_only)
        
        print("EMDED")
        ### BERT EMBED
        emb_list = []
        for sent in sent_ru:
            emb_list.append(embed(sent))

        print(datetime.datetime.now())

        a = []
        for i in range(len(emb_list)):
            a.append(emb_list[i][0].numpy())

        ### FASTTEXT EMBED
        # path_embed = "D://Загрузки//"
        # os.chdir(path_embed)
        # ft_model = _FastText(model_path='cc.ru.300.bin')

        # def generateVector(sentence):
        #     return ft_model.get_sentence_vector(sentence)

        # a = []
        # for sent in sent_ru:
        #     a.append(generateVector(sent))

        import textwrap as tw

        def short_text(text):
            dedented_text = tw.dedent(text)
            short = tw.shorten(dedented_text, 500)
            return short
            
        df['text'] = df['text'].apply(short_text)
        

        dff = pd.DataFrame(a)

        # https://gist.github.com/komasaru/ed07018ae246d128370a1693f5dd1849
        def shorten(url_long): # делаем ссылки короткими для отображения в web

            URL = "http://tinyurl.com/api-create.php"
            try:
                url = URL + "?" \
                    + urllib.parse.urlencode({"url": url_long})
                res = requests.get(url)
                if res.text == 'Error':
                    return url_long
                else:
                    return res.text

            except Exception as e:
                raise

        # create short url links
        # df.loc[:, 'url'] = [shorten(x) for x in df['url'].values]
        # create active url links
        # df['url'] = '<a href="' + df['url'] + '">' + df['url'] + '</a>'


        # date = datetime.datetime.now().strftime("%m.%d.%Y %H:%M:%S")

        os.chdir(path_to_embed_save)
        df_fin = pd.concat([df, dff], axis=1)
        df_fin.to_csv(os.path.join(path_to_embed_save, filename.replace('.json', '.csv')))
        print('done BertEmbed')
        
        # create and save BertTopic data
        texts = [x['text'] for x in dict_train] 
        texts = [words_only(x) for x in texts]
        texts = [preprocess_text(x) for x in texts]
        texts = [remove_stopwords(x) for x in texts]    

        topic_model = BERTopic(language="russian", calculate_probabilities=True, verbose=True)
        topics, probs = topic_model.fit_transform(texts)


        # словарь с данными для графиков BertTopic
        Bert_dicts = {}
        Bert_dicts['topics'] = topics
        Bert_dicts['probs'] = probs.tolist()

        os.chdir(path_bert_topic_data)
        name_folder = filename.replace('.json', '')
        os.mkdir(name_folder)
        os.chdir(name_folder)
        with open(filename.replace('.json', '') + '.txt', 'w') as my_file:
            json.dump(Bert_dicts, my_file)

        topic_model.save(filename.replace('json', 'pt'), save_embedding_model=False)
        
        print('done BerTopic')
        del topic_model
        del texts
        del df_fin
        del df
        del embed

        # Подготовка данных для tsne
        # parsing json & create embedding
        print('Start tsne emb')
        os.chdir(path_to_files)
        embed = hub.load("universal-sentence-encoder-multilingual_3")

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

                    if i == 1:
                        dff = df[df['hubtype'] == 'Новости']
                        if dff.shape[0] != 0:
                            dff['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                            df_meta_smi = dff[['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'er', 'hubtype', 'text', 'citeIndex']]
                            df_meta_smi['fullname'] = dff['hub']
                            df_meta_smi['author_type'] = 'Новости'
                            df_meta_smi['hubtype'] = 'Новости'
                            df_meta_smi['type'] = 'Новости'

            if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
                df_meta = pd.concat([df_meta_socm, df_meta_smi])
            elif 'df_meta_smi' and 'df_meta_socm' not in locals():
                df_meta = df_meta_smi
            else:
                df_meta = df_meta_socm


        # тексты
        df_text = df_meta[['text']]

        # метаданные
        # columns = list(df.columns)
        # columns.remove('text')
        # df_meta = pd.concat([pd.DataFrame.from_records(df['authorObject'].values), df[columns]], axis=1)


        regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

        def words_only(text, regex=regex):
            try:
                return " ".join(regex.findall(text))
            except:
                return ""

        mystopwords = ['это', 'наш', 'тыс', 'млн', 'млрд', 'также', 'т', 'д', 'URL',
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

        df_text['text'] = df_text['text'].apply(words_only)
        df_text['text'] = df_text['text'].apply(preprocess_text)
        df_text['text'] = df_text['text'].apply(remove_stopwords)

        sent_ru = df_text['text'].values

        a = []
        for sent in sent_ru:
            # a.append(embed(sent)[0].numpy())
            a.append([numpy.round(x, 8) for x in embed(sent)[0].numpy()])
        
        embed_list = a

        dff = pd.DataFrame(a)


        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        x_tsne = tsne.fit_transform(dff.values)

        coord_list = []

        for i in range(len(x_tsne.tolist())):
            coord_list.append(', '.join([str(x) for x in x_tsne.tolist()[i]]))

        names = df_meta['fullname'].values.tolist()
        names = [x if x != '' else 'None' for x in names]

        regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")


        names_list = [words_only(x) if type(x) != float else 'None' for x in names]
        names_list = [preprocess_text(x) if type(x) != float else 'None' for x in names]
        names_list = [remove_stopwords(x) if type(x) != float else 'None' for x in names]
        names_list = ['None' if x == '' else x for x in names_list]

        name_str = '\n'.join(names_list)
        coord_list_str = '\n'.join(coord_list)

        os.chdir(path_tsne_data)
        os.mkdir(filename.replace('.json', ''))
        os.chdir(path_tsne_data + '/' + filename.replace('.json', ''))

        # сохранение данных для tsne
        dict_tsne = {}
        dict_tsne['author_name_str'] = name_str
        dict_tsne['coord_list_str'] = coord_list_str

        with open(filename.replace('.json', '') + '_data_tsne.txt', 'w') as my_file:
            json.dump(dict_tsne, my_file)

        print('done tsne embed')
        del dict_tsne

        ### Подготовка данных для LdaTopic
        os.chdir(path_to_files)
        print('stassrt') 
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

        df['text'] = df['text'].apply(preprocess_text)
        df['text'] = df['text'].apply(remove_stopwords)
        df['text'] = df['text'].apply(words_only)
        sent_ru = df['text'].values.tolist()

        text_clean= []
        for index, row in df.iterrows():
                text_clean.append(row['text'].split())

        from gensim.models import Phrases
        bigram = Phrases(text_clean) # Создаем биграммы на основе корпуса
        trigram = Phrases(bigram[text_clean])# Создаем триграммы на основе корпуса

        for idx in range(len(text_clean)):
            for token in bigram[text_clean[idx]]:
                if '_' in token:
                    # Токен это биграмма, добавим в документ.
                    text_clean[idx].append(token)
            for token in trigram[text_clean[idx]]:
                if '_' in token:
                    # Токен это три грамма, добавим в документ.
                    text_clean[idx].append(token)


        from gensim.corpora.dictionary import Dictionary
        from numpy import array

        dictionary = Dictionary(text_clean)
        dictionary.filter_extremes(no_below=10, no_above=0.1)
        #Создадим словарь и корпус для lda модели
        corpus = [dictionary.doc2bow(doc) for doc in text_clean] 

        print('start again')
        # def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
        #     """
        #     Подсчет c_v когерентности для различного количества тем
        #     dictionary : Gensim словарь
        #     corpus : Gensim корпус
        #     texts : Список текстов
        #     limit : Максимальное количество тем
            
        #     model_list : Список LDA моделей
        #     coherence_values :Когерентности, соответствующие модели LDA с количеством тем
        #     """
        #     coherence_values = []
        #     model_list = []
        #     for num_topics in range(start, limit, step):
        #         model=LdaMulticore(corpus=corpus,id2word=dictionary, num_topics=num_topics)
        #         model_list.append(model)
        #         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        #         coherence_values.append(coherencemodel.get_coherence())

        #     return model_list, coherence_values
            
        # # Вызовем функцию и посчитаем
        # model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=text_clean, start=2, limit=40, step=1)
        # optimal_topic_num =coherence_values.index(np.max(coherence_values))
        # optimal_topic_num = optimal_topic_num + 2 # добавляем 2 так как расчет когерентности начиается с минимум 2х кластеров
        print('start LdaMdel')
        model=LdaModel(corpus=corpus,id2word=dictionary, num_topics=32)
        print('stop LdaMdel')
        # save model_LDA
        
        os.chdir(path_Lda_topic_data)
        os.mkdir(filename.replace('.json', ''))
        os.chdir(path_Lda_topic_data + '/' + filename.replace('.json', ''))
        model.save('Model_LDA' + filename.replace('.json', ''))

        #save lda corpus
        with open('corpus.txt', 'w') as my_file:
            json.dump(corpus, my_file)

        # save lda dictionary
        dictionary.save('dictionary')

        # save Lda model
        p = gensimvis.prepare(model, corpus, dictionary)
        os.chdir(templates_path)
        # pyLDAvis.save_html(p, filename.replace('.json', '_LdaTemplate.html'))

        if os.path.exists(filename.replace('.json', '_LdaTemplate.html')):
            # delete the file
            os.remove(filename.replace('.json', '_LdaTemplate.html'))
            pyLDAvis.save_html(p, filename.replace('.json', '_LdaTemplate.html'))
        else:
            # if the file does not exist.
            pyLDAvis.save_html(p, filename.replace('.json', '_LdaTemplate.html'))

        print('done LdaMdel')

        ### Подготовка данных для Projector
        print('Start Projector')
        # from json_ba_not_empty_text import json_ba_not_empty_text
        # X = json_ba_not_empty_text()
        # X.open_file(filename)
        # X.preprocess_texts()
        # print('Start Embed')
        # X.create_embed()
        # print('Stop Embed')

        # round_embed_list = []
        # for i in tqdm(range(len(sent_ru))):
        #     round_embed_list.append([np.round(x, 7) for x in embed_list[i]])

        tsv_embed_list = []
        for i in range(len(embed_list)):
            tsv_embed_list.append('\t'.join([str(x) for x in embed_list[i]]))

        # save projector data
        import csv

        os.chdir(path_projector_files) 
        if not os.path.exists(filename.replace('.json', '')):
            os.mkdir(filename.replace('.json', ''))
            os.chdir(path_projector_files + '/' + filename.replace('.json', ''))
        else:
            os.chdir(path_projector_files + '/' + filename.replace('.json', ''))

        with open(filename.replace('.json', '.tsv'), 'w') as f:
            for line in tsv_embed_list:
                f.write(f"{line}\n")

        test_names = names_list
        with open(filename.replace('.json', '.txt'), 'w', encoding="utf-8") as f:
            for line in test_names:
                f.write(f"{line}\n")

        os.chdir(path_projector_files)
        with open(filename.replace('.json', '.tsv'), 'w') as f:
            for line in tsv_embed_list:
                f.write(f"{line}\n")
        test_names = names_list
        with open(filename.replace('.json', '.txt'), 'w', encoding="utf-8") as f:
            for line in test_names:
                f.write(f"{line}\n")

        print('Done Projector')


    thread = Thread(target=do_work, kwargs={'filename': request.args.get('filename', session['filename'])})
    del session['filename']
    thread.start()
    return redirect(url_for('datalake'))


@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = '/home/dev/social_app/data/'
    file_path = file_path + filename
    return send_file(file_path, as_attachment=True)


@app.route('/delete-file/<filename>')
def delete_files_tut(filename):

    os.chdir(path_to_files)
    if filename in [f for f in listdir(path_to_files) if isfile(join(path_to_files, f))]:
        os.remove(path_to_files + '/' + filename) # удаляем файл из datalake

    os.chdir(path_bert_topic_data)
    if filename.replace('.json', '') in next(os.walk(path_bert_topic_data))[1]:
        shutil.rmtree(path_bert_topic_data + '/' + filename.replace('.json', '')) # удаляем папку из BertTopic

    os.chdir(path_to_embed_save)
    if filename.replace('.json', '.csv') in os.listdir((path_to_embed_save)):
        os.remove(path_to_embed_save + '/' + filename.replace('.json', '.csv')) # удаляем файл из Embeddings

    os.chdir(path_Lda_topic_data)
    if filename.replace('.json', '') in next(os.walk(path_Lda_topic_data))[1]:
        shutil.rmtree(path_Lda_topic_data + '/' + filename.replace('.json', '')) # удаляем папку из LdaTopic

    os.chdir(path_tsne_data)
    if filename.replace('.json', '') in next(os.walk(path_tsne_data))[1]:
        shutil.rmtree(path_tsne_data + '/' + filename.replace('.json', '')) # удаляем папку из TSNE

    os.chdir(templates_path)
    if filename.replace('.json', '_LdaTemplate.html') in os.listdir((templates_path)):
        print(f'yes')
        os.remove(templates_path + '/' + filename.replace('.json', '_LdaTemplate.html')) # удаляем Lda-html файл из Templates

    os.chdir(path_projector_files)
    if filename.replace('.json', '.tsv') in [f for f in listdir(path_projector_files) if isfile(join(path_projector_files, f))]:
        os.remove(path_projector_files + '/' + filename.replace('.json', '.tsv')) # удаляем файл .tsv из projector folder
    os.chdir(path_projector_files)
    if filename.replace('.json', '.txt') in [f for f in listdir(path_projector_files) if isfile(join(path_projector_files, f))]:
        os.remove(path_projector_files + '/' + filename.replace('.json', '.txt')) # удаляем файл .txt из projector folder

    os.chdir(path_projector_files)
    if filename.replace('.json', '') in next(os.walk(path_projector_files))[1]:
        shutil.rmtree(path_projector_files + '/' + filename.replace('.json', '')) # удаляем папку из Projector


    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    name_files = [x.split('_')[0] for x in json_files]
    name_files = [x.split('.')[0] for x in name_files]
    name_files = list(set(name_files))
    len_files = len(json_files)

    files_dict = {}
    for i in range(len(name_files)):
        files_dict[name_files[i]] = [x for x in json_files if name_files[i] in x]

    return render_template('datalake.html', files=json_files, name_files=name_files, files_dict=files_dict,
                           len_files=len_files)


@app.route('/information_graph', methods=['GET', 'POST'], endpoint='information_graph')
def information_graph():
    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files)

    if 'send' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)['text_search'] == '':

        session['filename'] = request.values.to_dict(flat=True)['file_choose']

        # parsing json
        try: 
            with io.open(session['filename'], encoding='utf-8', mode='r') as train_file:
                dict_train = json.load(train_file, strict=False)

        except:
            a = []
            with open(session['filename'], encoding='utf-8', mode='r') as file:
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
            df_meta_smi_only = dff[['timeCreate', 'hub', 'toneMark', 'audience', 'url', 'text']]
            df_meta_smi_only.columns = ['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'text']
            df_meta_smi_only['fullname'] = dff['hub']
            df_meta_smi_only['author_type'] = 'Новости'
            df_meta_smi_only['hubtype'] = 'Новости'
            df_meta_smi_only['type'] = 'Новости'
            df_meta_smi_only['er'] = 0
            df_meta_smi_only.dropna(subset=['timeCreate'], inplace=True)
            df_meta_smi_only = df_meta_smi_only.set_index(['timeCreate'])
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
                            df_meta_socm.dropna(subset=['timeCreate'], inplace=True)
                            df_meta_socm = df_meta_socm.set_index(['timeCreate'])
                            df_meta_socm['date'] = [x[:10] for x in df_meta_socm.index]

                    if i == 1:
                        dff = df[df['hubtype'] == 'Новости']
                        if dff.shape[0] != 0:
                            dff['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                            df_meta_smi = dff[['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'er', 'hubtype', 'text']]
                            df_meta_smi['fullname'] = dff['hub']
                            df_meta_smi['author_type'] = 'Новости'
                            df_meta_smi['hubtype'] = 'Новости'
                            df_meta_smi['type'] = 'Новости'
                            df_meta_smi.dropna(subset=['timeCreate'], inplace=True)
                            df_meta_smi = df_meta_smi.set_index(['timeCreate'])
                            df_meta_smi['date'] = [x[:10] for x in df_meta_smi.index]

            if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
                df_meta = pd.concat([df_meta_socm, df_meta_smi])
            elif 'df_meta_smi' and 'df_meta_socm' not in locals():
                df_meta = df_meta_smi
            else:
                df_meta = df_meta_socm

        def date_reverse(date):  # фильтрация по дате/календарик
            lst = date.split('-')
            temp = lst[1]
            lst[1] = lst[2]
            lst[2] = temp
            return lst

        if 'daterange' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)[
            'daterange'] != '01/01/2022 - 01/12/2022':
            data_start = '-'.join(date_reverse('-'.join(
                [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][::-1])))
            data_stop = '-'.join(date_reverse('-'.join(
                [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][::-1])))

        mask = (df_meta['date'] >= data_start) & (df_meta['date'] <= data_stop)
        df_meta = df_meta.loc[mask]
        df_meta.reset_index(inplace=True)

        if df_meta.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
            flash('По запросу найдено 0 сообщений')
            return redirect(url_for('information_graph'))

        if 'hub_select' in request.values.to_dict(flat=True):
            hub_select = request.form.getlist('hub_select')
            df_meta = df_meta.loc[df_meta['hub'].isin(hub_select)]

        # if request.values.to_dict(flat=True)['text_min'] != '':
        #     df_meta = df_meta.iloc[int(request.values.to_dict(flat=True)['text_min']):]

        # if request.values.to_dict(flat=True)['text_max'] != '':
        #     df_meta = df_meta.iloc[:int(request.values.to_dict(flat=True)['text_max'])]

        # if request.values.to_dict(flat=True)['text_min'] != '' and request.values.to_dict(flat=True)['text_max'] != '':
        #     df_meta = df_meta.iloc[int(request.values.to_dict(flat=True)['text_min']):int(
        #         request.values.to_dict(flat=True)['text_max'])]

        if df_meta.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
            flash('По запросу найдено 0 сообщений')
            return redirect(url_for('information_graph'))

        df_meta_filter = df_meta
        df_meta = pd.DataFrame()

        # фильтрация по типу сообщения
        if 'posts' in request.values.to_dict(flat=True):
            if request.values.to_dict(flat=True)['posts'] == 'on':
                df_meta = pd.concat([df_meta, df_meta_filter[
                    (df_meta_filter['type'] == 'Пост') | (df_meta_filter['type'] == 'Комментарий')]], ignore_index=True)

        if 'reposts' in request.values.to_dict(flat=True):
            if request.values.to_dict(flat=True)['reposts'] == 'on':
                df_meta = pd.concat([df_meta, df_meta_filter[
                    (df_meta_filter['type'] == 'Репост') | (df_meta_filter['type'] == 'Репост с дополнением')]],
                                    ignore_index=True)

        if 'smi' in request.values.to_dict(flat=True):
            if request.values.to_dict(flat=True)['smi'] == 'on':
                df_meta = pd.concat([df_meta, df_meta_filter[df_meta_filter['hubtype'] == 'Новости']],
                                    ignore_index=True)

        if 'posts' not in request.values.to_dict(flat=True) and 'reposts' not in request.values.to_dict(
                flat=True) and 'smi' not in request.values.to_dict(flat=True):
            df_meta = df_meta_filter

        if df_meta.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
            flash('По запросу найдено 0 сообщений')
            return redirect(url_for('information_graph'))

        df_data_rep = df_meta[['fullname', 'url', 'author_type', 'text', 'er', 'hub', 'audienceCount']]
        if 'smi' in request.values.to_dict(flat=True):
            df_rep_auth = list(df_data_rep['hub'].values)
        else:
            df_rep_auth = list(df_data_rep['fullname'].values)
        data_rep_er = list(df_data_rep['er'].values)

        all_hubs = list(df_data_rep['hub'].values)
        all_hubs = [words_only(x) for x in all_hubs]
        df_rep_auth = [words_only(x) for x in df_rep_auth]
        data_audience = list(df_data_rep['audienceCount'].values)

        for i in range(len(df_rep_auth) - 1):
            if df_rep_auth[i + 1] == df_rep_auth[i]:
                df_rep_auth[i + 1] = df_rep_auth[i] + ' '

        f = lambda A, n=1: [[df_rep_auth[i], df_rep_auth[i + n]] for i in range(0, len(df_rep_auth) - 1,
                                                                                n)]  # ф-ия разбивки авторов на последовательности [[1, 2], [2,3]...]
        df_rep_auth_inverse = f(df_rep_auth.append(df_rep_auth[-1]))

        theme = request.values.to_dict(flat=True)['file_choose'].split('_')[0]

        er = [int(z) for z in [int(y) for y in [5 if x == 0 else x + 5 for x in data_rep_er]]]
        er = [numpy.mean(er) if x > 5 * numpy.mean(er) else x for x in er]
        er[0] = int(numpy.max(er) + 2)

        hubs = Counter(df_meta['hub'].values)
        hubs = hubs.most_common()
        hubs = [x[0] for x in hubs]
        hubs = [words_only(x) for x in hubs]
        data_audience = [int(z) for z in [int(y) for y in [5 if x == 0 else x for x in data_audience]]]

        data = {
            "df_rep_auth": df_rep_auth_inverse[:100],
            "data_rep_er": er[:100],
            "data_rep_audience": data_audience[:100],
            "data_authors": df_rep_auth[:100],
            "authors_count": len(set(df_rep_auth)),
            "len_messages": df_meta.shape[0],
            "data_hub": hubs,
            "all_hubs": all_hubs[:100]
        }


        ### second graph
        df_meta.fullname.fillna("СМИ", inplace=True)
        df_meta = df_meta[['fullname', 'url', 'er', 'hub', 'audienceCount', 'hubtype', 'timeCreate']]
        df_meta = df_meta
        # время к unix формату для js 
        df_meta['timeCreate'] = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - 
                                    datetime.datetime(1970, 1, 1)).total_seconds() * 1000) for x in df_meta['timeCreate'].values]
        
        unique_hubs = list(df_meta['hub'].unique())
        # https://stackoverflow.com/questions/17426292/how-to-create-a-dictionary-of-two-pandas-dataframe-columns
        multivalue_dict = defaultdict(list)
        authors_name = [] # список с именами авторов
        authors_urls = [] # список urls текстов

        for i in range(len(unique_hubs)):
            
            df = df_meta[df_meta['hub'] == unique_hubs[i]]
            
            for idx,row in df.iterrows():
                multivalue_dict[row['hub']].append([row['timeCreate'], row['audienceCount'], 
                words_only(row['fullname']), row['url']])
            

        multivalue_dict = dict(multivalue_dict)
        multivalue_dict = json.dumps(multivalue_dict)

        # with open('file.txt', 'w') as file:
        #     file.write(json.dumps(multivalue_dict))

        return render_template('information_graph.html', theme=theme, len_files=len_files, files=json_files,
                               data=data, multivalue_dict=multivalue_dict)


    if 'send' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)['text_search'] != '':

        session['filename'] = request.values.to_dict(flat=True)['file_choose']

        # parsing json
        try: 
            with io.open(session['filename'], encoding='utf-8', mode='r') as train_file:
                dict_train = json.load(train_file, strict=False)

        except:
            a = []
            with open(session['filename'], encoding='utf-8', mode='r') as file:
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
            df_meta_smi_only = dff[['timeCreate', 'hub', 'toneMark', 'audience', 'url', 'text']]
            df_meta_smi_only.columns = ['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'text']
            df_meta_smi_only['fullname'] = dff['hub']
            df_meta_smi_only['author_type'] = 'Новости'
            df_meta_smi_only['hubtype'] = 'Новости'
            df_meta_smi_only['type'] = 'Новости'
            df_meta_smi_only['er'] = 0
            df_meta_smi_only.dropna(subset=['timeCreate'], inplace=True)
            df_meta_smi_only = df_meta_smi_only.set_index(['timeCreate'])
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
                            df_meta_socm.dropna(subset=['timeCreate'], inplace=True)
                            df_meta_socm = df_meta_socm.set_index(['timeCreate'])
                            df_meta_socm['date'] = [x[:10] for x in df_meta_socm.index]

                    if i == 1:
                        dff = df[df['hubtype'] == 'Новости']
                        if dff.shape[0] != 0:
                            dff['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                            df_meta_smi = dff[['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'er', 'hubtype', 'text']]
                            df_meta_smi['fullname'] = dff['hub']
                            df_meta_smi['author_type'] = 'Новости'
                            df_meta_smi['hubtype'] = 'Новости'
                            df_meta_smi['type'] = 'Новости'
                            df_meta_smi.dropna(subset=['timeCreate'], inplace=True)
                            df_meta_smi = df_meta_smi.set_index(['timeCreate'])
                            df_meta_smi['date'] = [x[:10] for x in df_meta_smi.index]

            if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
                df_meta = pd.concat([df_meta_socm, df_meta_smi])
            elif 'df_meta_smi' and 'df_meta_socm' not in locals():
                df_meta = df_meta_smi
            else:
                df_meta = df_meta_socm

        def date_reverse(date):  # фильтрация по дате/календарик
            lst = date.split('-')
            temp = lst[1]
            lst[1] = lst[2]
            lst[2] = temp
            return lst

        if 'daterange' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)[
            'daterange'] != '01/01/2022 - 01/12/2022':
            data_start = '-'.join(date_reverse('-'.join(
                [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][::-1])))
            data_stop = '-'.join(date_reverse('-'.join(
                [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][::-1])))

        mask = (df_meta['date'] >= data_start) & (df_meta['date'] <= data_stop)
        df_meta = df_meta.loc[mask]
        df_meta.reset_index(inplace=True)

        if df_meta.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
            flash('По запросу найдено 0 сообщений')
            return redirect(url_for('information_graph'))

        if set(df_meta['hub'].values) == {"telegram.org"}:  # если все сообщения только ТГ

            search_lst = request.values.to_dict(flat=True)['text_search'].split(',')
            search_lst = [x.split('или') for x in search_lst]
            search_lst = [[x.strip().lower() for x in group] for group in search_lst]

            index_table = []
            text_val = df_meta['text'].values
            text_val = [x.lower() for x in text_val]

            for j in range(len(text_val)):
                a = []
                for i in range(len(search_lst)):
                    if [item for item in search_lst[i] if item in text_val[j]] != []:
                        a.append([item for item in search_lst[i] if item in text_val[j]])
                if len(a) == len(search_lst):
                    index_table.append(df_meta.index[j])

            df_meta = df_meta.loc[index_table]

            # if request.values.to_dict(flat=True)['text_min'] != '':
            #     df_meta = df_meta.iloc[int(request.values.to_dict(flat=True)['text_min']):]

            # if request.values.to_dict(flat=True)['text_max'] != '':
            #     df_meta = df_meta.iloc[:int(request.values.to_dict(flat=True)['text_max'])]

            # if request.values.to_dict(flat=True)['text_min'] != '' and request.values.to_dict(flat=True)[
            #     'text_max'] != '':
            #     df_meta = df_meta.iloc[int(request.values.to_dict(flat=True)['text_min']):int(
            #         request.values.to_dict(flat=True)['text_max'])]

            if df_meta.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
                flash('По запросу найдено 0 сообщений')
                return redirect(url_for('information_graph'))

            df_data_rep = df_meta[['fullname', 'url', 'author_type', 'text', 'audienceCount', 'hub', 'er', 'type']].sort_index(axis=0)
            df_rep_auth = list(df_data_rep['fullname'].values)
            data_rep_er = list(df_data_rep['er'].values)
            data_audience = list(df_data_rep['audienceCount'].values)
            all_hubs = list(df_data_rep['hub'].values)

            all_hubs = [words_only(x) for x in all_hubs]
            df_rep_auth = [words_only(x) for x in df_rep_auth]

            for i in range(len(df_rep_auth) - 1):
                if df_rep_auth[i + 1] == df_rep_auth[i]:
                    df_rep_auth[i + 1] = df_rep_auth[i] + ' '

            f = lambda A, n=1: [[df_rep_auth[i], df_rep_auth[i + n]] for i in range(0, len(df_rep_auth) - 1,
                                                                                    n)]  # ф-ия разбивки авторов на последовательности [[1, 2], [2,3]...]

            df_rep_auth_inverse = f(df_rep_auth.append(df_rep_auth[-1]))

            theme = request.values.to_dict(flat=True)['file_choose'].split('_')[0]

            data_rep_er = [int(z) for z in [int(y) for y in [5 if x == 0 else x + 5 for x in data_rep_er]]]
            data_rep_er = [numpy.mean(data_rep_er) if x > 5 * numpy.mean(data_rep_er) else x for x in data_rep_er]
            data_rep_er[0] = int(numpy.max(data_rep_er) + 2)

            hubs = Counter(df_meta['hub'].values)
            hubs = hubs.most_common()
            hubs = [x[0] for x in hubs]
            hubs = [words_only(x) for x in hubs]

            data_audience = [int(z) for z in [int(y) for y in [5 if x == 0 else x for x in data_audience]]]

            data = {
                "df_rep_auth": df_rep_auth_inverse[:100],
                "data_rep_er": data_rep_er[:100],
                "data_rep_audience": data_audience[:100],
                "data_authors": df_rep_auth[:100],
                "authors_count": len(set(df_rep_auth)),
                "len_messages": df_meta.shape[0],
                "data_hub": hubs,
                "all_hubs": all_hubs[:100]
            }

            ### second graph
            df_meta.fullname.fillna("СМИ", inplace=True)
            df_meta = df_meta[['fullname', 'url', 'er', 'hub', 'audienceCount', 'hubtype', 'timeCreate']]
            df_meta = df_meta
            # время к unix формату для js 
            df_meta['timeCreate'] = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - 
                                        datetime.datetime(1970, 1, 1)).total_seconds() * 1000) for x in df_meta['timeCreate'].values]
            
            unique_hubs = list(df_meta['hub'].unique())
            # https://stackoverflow.com/questions/17426292/how-to-create-a-dictionary-of-two-pandas-dataframe-columns
            multivalue_dict = defaultdict(list)
            authors_name = [] # список с именами авторов
            authors_urls = [] # список urls текстов

            for i in range(len(unique_hubs)):
                
                df = df_meta[df_meta['hub'] == unique_hubs[i]]
                
                for idx,row in df.iterrows():
                    multivalue_dict[row['hub']].append([row['timeCreate'], row['audienceCount'], 
                    words_only(row['fullname']), row['url']])
                

            multivalue_dict = dict(multivalue_dict)

            print(multivalue_dict)

            return render_template('information_graph.html', theme=theme, len_files=len_files, files=json_files,
                                   data=data, multivalue_dict=multivalue_dict)

        df_meta_filter = df_meta
        df_meta = pd.DataFrame()

        # фильтрация по типу сообщения
        if 'posts' in request.values.to_dict(flat=True):
            if request.values.to_dict(flat=True)['posts'] == 'on':
                df_meta = pd.concat([df_meta, df_meta_filter[
                    (df_meta_filter['type'] == 'Пост') | (df_meta_filter['type'] == 'Комментарий')]], ignore_index=True)

        if 'reposts' in request.values.to_dict(flat=True):
            if request.values.to_dict(flat=True)['reposts'] == 'on':
                df_meta = pd.concat([df_meta, df_meta_filter[
                    (df_meta_filter['type'] == 'Репост') | (df_meta_filter['type'] == 'Репост с дополнением')]],
                                    ignore_index=True)

        if 'smi' in request.values.to_dict(flat=True):
            if request.values.to_dict(flat=True)['smi'] == 'on':
                df_meta = pd.concat([df_meta, df_meta_filter[df_meta_filter['hubtype'] == 'Новости']],
                                    ignore_index=True)

        if 'posts' not in request.values.to_dict(flat=True) and 'reposts' not in request.values.to_dict(
                flat=True) and 'smi' not in request.values.to_dict(flat=True):
            df_meta = df_meta_filter

        if df_meta.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
            flash('По запросу найдено 0 сообщений')
            return redirect(url_for('information_graph'))

        # if request.values.to_dict(flat=True)['text_min'] != '':
        #     df_meta = df_meta.iloc[int(request.values.to_dict(flat=True)['text_min']):]

        # if request.values.to_dict(flat=True)['text_max'] != '':
        #     df_meta = df_meta.iloc[:int(request.values.to_dict(flat=True)['text_max'])]

        # if request.values.to_dict(flat=True)['text_min'] != '' and request.values.to_dict(flat=True)['text_max'] != '':
        #     df_meta = df_meta.iloc[int(request.values.to_dict(flat=True)['text_min']):int(
        #         request.values.to_dict(flat=True)['text_max'])]

        search_lst = request.values.to_dict(flat=True)['text_search'].split(',')
        search_lst = [x.split('или') for x in search_lst]
        search_lst = [[x.strip().lower() for x in group] for group in search_lst]

        index_table = []
        text_val = df_meta['text'].values
        text_val = [x.lower() for x in text_val]

        for j in range(len(text_val)):
            a = []
            for i in range(len(search_lst)):
                if [item for item in search_lst[i] if item in text_val[j]] != []:
                    a.append([item for item in search_lst[i] if item in text_val[j]])
            if len(a) == len(search_lst):
                index_table.append(df_meta.index[j])

        df_meta = df_meta.loc[index_table]
        if df_meta.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
            flash('По запросу найдено 0 сообщений')
            return redirect(url_for('information_graph'))

        df_data_rep = df_meta[['fullname', 'url', 'author_type', 'text', 'er', 'hub', 'audienceCount']].sort_index(axis=0)
        df_rep_auth = list(df_data_rep['fullname'].values)
        data_rep_er = list(df_data_rep['er'].values)
        all_hubs = list(df_data_rep['hub'].values)

        all_hubs = [words_only(x) for x in all_hubs]
        df_rep_auth = [words_only(x) for x in df_rep_auth]

        for i in range(len(df_rep_auth) - 1):
            if df_rep_auth[i + 1] == df_rep_auth[i]:
                df_rep_auth[i + 1] = df_rep_auth[i] + ' '

        f = lambda A, n=1: [[df_rep_auth[i], df_rep_auth[i + n]] for i in range(0, len(df_rep_auth) - 1,
                                                                                n)]  # ф-ия разбивки авторов на последовательности [[1, 2], [2,3]...]
        df_rep_auth_inverse = f(df_rep_auth.append(df_rep_auth[-1]))

        theme = request.values.to_dict(flat=True)['file_choose'].split('_')[0]

        er = [int(z) for z in [int(y) for y in [5 if x == 0 else x + 5 for x in data_rep_er]]]
        er = [numpy.mean(er) if x > 5 * numpy.mean(er) else x for x in er]
        er[0] = int(numpy.max(er) + 2)

        hubs = Counter(df_meta['hub'].values)
        hubs = hubs.most_common()
        hubs = [x[0] for x in hubs]
        hubs = [words_only(x) for x in hubs]

        data_audience = list(df_data_rep['audienceCount'].values)
        data_audience = [int(z) for z in [int(y) for y in [5 if x == 0 else x for x in data_audience]]]

        data = {
            "df_rep_auth": df_rep_auth_inverse[:100],
            "data_rep_er": er[:100],
            "data_rep_audience": data_audience[:100],
            "data_authors": df_rep_auth[:100],
            "authors_count": len(set(df_rep_auth)),
            "len_messages": df_meta.shape[0],
            "data_hub": hubs,
            "all_hubs": all_hubs[:100]
        }

        ### second graph
        df_meta.fullname.fillna("СМИ", inplace=True)
        df_meta = df_meta[['fullname', 'url', 'er', 'hub', 'audienceCount', 'hubtype', 'timeCreate']]
        df_meta = df_meta
        # время к unix формату для js 
        df_meta['timeCreate'] = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - 
                                    datetime.datetime(1970, 1, 1)).total_seconds() * 1000) for x in df_meta['timeCreate'].values]
        
        unique_hubs = list(df_meta['hub'].unique())
        # https://stackoverflow.com/questions/17426292/how-to-create-a-dictionary-of-two-pandas-dataframe-columns
        multivalue_dict = defaultdict(list)
        for i in range(len(unique_hubs)):
            
            df = df_meta[df_meta['hub'] == unique_hubs[i]]
            for idx,row in df.iterrows():
                multivalue_dict[row['hub']].append([row['timeCreate'], row['audienceCount'], 
                words_only(row['fullname']), row['url']])

        multivalue_dict = dict(multivalue_dict)

        return render_template('information_graph.html', theme=theme, len_files=len_files, files=json_files,
                               data=data, multivalue_dict=multivalue_dict)

    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files) 

    # data
    data = {
        "df_rep_auth": ['A', 'G', 'K', 'M'],
        "data_rep_audience": [11, 12, 15, 8],
        "data_rep_er": [25, 29, 34, 18],
        "authors_count": 4,
        }

    multivalue_dict = {"facebook.com": [[1660516886000, 1875, 'Константин Лазарев', 'http://www.facebook.com/100000048821988/posts/5774082512603319'], 
        [1660133911000, 187779, 'Стартапы и бизнес', 'http://www.facebook.com/169345889757001/posts/5710404055651129']], 
        "ok.ru": [[1660511171000, 604, 'Валерий Жадан', 'http://www.ok.ru/group/52118611624148/topic/155101385145812'], 
        [1660510999000, 695, 'Валерий Жадан', 'http://www.ok.ru/group/52074498621689/topic/154563267526905']]}

    return render_template('information_graph.html', len_files=len_files, files=json_files,
                            multivalue_dict=multivalue_dict)


@app.route('/media_rating', methods=['GET', 'POST'], endpoint='media_rating')
def media_rating():
    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files)

    if 'send' in request.values.to_dict(flat=True):

        session['filename'] = request.values.to_dict(flat=True)['file_choose']

        # parsing json
        try: 
            with io.open(session['filename'], encoding='utf-8', mode='r') as train_file:
                dict_train = json.load(train_file, strict=False)

        except:
            a = []
            with open(session['filename'], encoding='utf-8', mode='r') as file:
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
            df_meta_smi_only.dropna(subset=['timeCreate'], inplace=True)
            df_meta_smi_only = df_meta_smi_only.set_index(['timeCreate'])
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
                            df_meta_socm.dropna(subset=['timeCreate'], inplace=True)
                            df_meta_socm = df_meta_socm.set_index(['timeCreate'])
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
                            df_meta_smi.dropna(subset=['timeCreate'], inplace=True)
                            df_meta_smi = df_meta_smi.set_index(['timeCreate'])
                            df_meta_smi['date'] = [x[:10] for x in df_meta_smi.index]

            if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
                df_meta = pd.concat([df_meta_socm, df_meta_smi])
            elif 'df_meta_smi' and 'df_meta_socm' not in locals():
                df_meta = df_meta_smi
            else:
                df_meta = df_meta_socm

        def date_reverse(date):
            lst = date.split('-')
            temp = lst[1]
            lst[1] = lst[2]
            lst[2] = temp
            return lst


        data_start = '-'.join(date_reverse('-'.join(
            [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][::-1])))
        data_stop = '-'.join(date_reverse('-'.join(
            [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][::-1])))

        mask = (df_meta['date'] >= data_start) & (df_meta['date'] <= data_stop)
        df_meta = df_meta.loc[mask]
        df_meta.reset_index(inplace=True)

        print(df_meta.columns)

        if df_meta.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
            flash('По запросу найдено 0 сообщений (проверьте даты и/или другие условия)')
            return redirect(url_for('media_rating')) 

        if set(df_meta['hub'].values) == {"telegram.org"}:

            df_meta = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['hub'] == "telegram.org")]

            # negative smi
            df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] == -1)][
                ['fullname', 'audienceCount']].values

            dict_neg = {}
            for i in range(len(df_hub_siteIndex)):

                if df_hub_siteIndex[i][0] not in dict_neg.keys():

                    dict_neg[df_hub_siteIndex[i][0]] = []
                    dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                else:
                    dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

            list_neg = [list(set(x)) for x in dict_neg.values()]
            list_neg = [[0] if x[0] == 'n/a' else x for x in list_neg if x != 'n/a']
            list_neg = [int(x[0]) if x[0] != '' else 0 for x in list_neg]

            for i in range(len(list_neg)):
                dict_neg[list(dict_neg.keys())[i]] = list_neg[i]

            dict_neg = dict(sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))

            dict_neg_hubs_count = dict(
                Counter(list(
                    df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] == -1)]['fullname'])))

            fin_neg_dict = defaultdict(tuple)
            for d in (dict_neg, dict_neg_hubs_count):  # you can list as many input dicts as you want here
                for key, value in d.items():
                    fin_neg_dict[key] += (value,)

            list_neg_smi = list(fin_neg_dict.keys())
            list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
            list_neg_smi_massage_count = [x[1] for x in fin_neg_dict.values()]

            # positive smi
            df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] == 1)][
                ['fullname', 'audienceCount']].values

            dict_pos = {}
            for i in range(len(df_hub_siteIndex)):

                if df_hub_siteIndex[i][0] not in dict_pos.keys():

                    dict_pos[df_hub_siteIndex[i][0]] = []
                    dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                else:
                    dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

            list_pos = [list(set(x)) for x in dict_pos.values()]
            list_pos = [[0] if x[0] == 'n/a' else x for x in list_pos if x != 'n/a']
            list_pos = [int(x[0]) if x[0] != '' else 0 for x in list_pos]

            for i in range(len(list_pos)):
                dict_pos[list(dict_pos.keys())[i]] = list_pos[i]

            dict_pos = dict(sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))

            dict_pos_hubs_count = dict(
                Counter(list(
                    df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] == 1)]['fullname'])))

            fin_pos_dict = defaultdict(tuple)
            for d in (dict_pos, dict_pos_hubs_count):  # you can list as many input dicts as you want here
                for key, value in d.items():
                    fin_pos_dict[key] += (value,)

            list_pos_smi = list(fin_pos_dict.keys())
            list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
            list_pos_smi_massage_count = [x[1] for x in fin_pos_dict.values()]

            # data to bobble graph
            bobble = []
            df_tonality = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] != 0)][
                ['fullname', 'audienceCount', 'toneMark', 'url']].values
            index_ton = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] != 0)][
                ['timeCreate']].values.tolist()
            date_ton = [x[0] for x in index_ton]
            date_ton = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime.datetime(1970, 1,
                                                                                                    1)).total_seconds() * 1000)
                        for x in date_ton]

            for i in range(len(df_tonality)):
                if df_tonality[i][2] == -1:
                    bobble.append([date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1, df_tonality[i][3]])
                elif df_tonality[i][2] == 1:
                    bobble.append([date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1, df_tonality[i][3]])

            for i in range(len(bobble)):
                if bobble[i][3] == 1:
                    bobble[i][3] = "#32ff32"
                else:
                    bobble[i][3] = "#FF3232"


            list_neg_smi = [words_only(x) for x in list_neg_smi]
            list_pos_smi = [words_only(x) for x in list_pos_smi]
            name_bobble = [x[1] for x in bobble]
            name_bobble = [words_only(x) for x in name_bobble]

            data = {
                "neg_smi_name": list_neg_smi[:100],
                "neg_smi_count": list_pos_smi_massage_count[:100],
                "neg_smi_rating": list_neg_smi_index[:100],
                "pos_smi_name": list_pos_smi[:100],
                "pos_smi_count": list_pos_smi_massage_count[:100],
                "pos_smi_rating": list_pos_smi_index[:100],

                "date_bobble": [x[0] for x in bobble],
                "name_bobble": name_bobble,
                "index_bobble": [x[2] for x in bobble],
                "z_index_bobble": [1] * len(bobble),
                "tonality_index_bobble": [x[3] for x in bobble],
                "tonality_url": [x[4] for x in bobble],
            }

            theme = request.values.to_dict(flat=True)['file_choose'].split('_')[0]


            return render_template('media_rating.html', len_files=len_files, files=json_files, data=data, theme=theme)


        df_meta = df_meta[df_meta['hubtype'] == 'Новости']

        # negative smi
        if df_meta.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
            flash('В данной теме/файле не собраны упоминания из Новостей для их отображения и анализа')
            return redirect(url_for('media_rating'))

        df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] == -1)][
            ['hub', 'citeIndex']].values

        dict_neg = {}
        for i in range(len(df_hub_siteIndex)):

            if df_hub_siteIndex[i][0] not in dict_neg.keys():

                dict_neg[df_hub_siteIndex[i][0]] = []
                dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

            else:
                dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

        list_neg = [list(set(x)) for x in dict_neg.values()]
        list_neg = [[0] if x[0] == 'n/a' else x for x in list_neg if x != 'n/a']
        list_neg = [int(x[0]) if x[0] != '' else 0 for x in list_neg]

        for i in range(len(list_neg)):
            dict_neg[list(dict_neg.keys())[i]] = list_neg[i]

        dict_neg = dict(sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))

        dict_neg_hubs_count = dict(
            Counter(list(df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] == -1)]['hub'])))

        fin_neg_dict = defaultdict(tuple)
        for d in (dict_neg, dict_neg_hubs_count):  # you can list as many input dicts as you want here
            for key, value in d.items():
                fin_neg_dict[key] += (value,)

        list_neg_smi = list(fin_neg_dict.keys())
        list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
        list_neg_smi_massage_count = [x[1] for x in fin_neg_dict.values()]

        # positive smi
        df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] == 1)][
            ['hub', 'citeIndex']].values

        dict_pos = {}
        for i in range(len(df_hub_siteIndex)):

            if df_hub_siteIndex[i][0] not in dict_pos.keys():

                dict_pos[df_hub_siteIndex[i][0]] = []
                dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

            else:
                dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

        list_pos = [list(set(x)) for x in dict_pos.values()]
        list_pos = [[0] if x[0] == 'n/a' else x for x in list_pos if x != 'n/a']
        list_pos = [int(x[0]) if x[0] != '' else 0 for x in list_pos]

        for i in range(len(list_pos)):
            dict_pos[list(dict_pos.keys())[i]] = list_pos[i]

        dict_pos = dict(sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))

        dict_pos_hubs_count = dict(
            Counter(list(df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] == 1)]['hub'])))

        fin_pos_dict = defaultdict(tuple)
        for d in (dict_pos, dict_pos_hubs_count):  # you can list as many input dicts as you want here
            for key, value in d.items():
                fin_pos_dict[key] += (value,)

        list_pos_smi = list(fin_pos_dict.keys())
        list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
        list_pos_smi_massage_count = [x[1] for x in fin_pos_dict.values()]

        # data to bobble graph
        bobble = []
        df_tonality = df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] != 0)][
            ['hub', 'citeIndex', 'toneMark', 'url']].values
        index_ton = df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] != 0)][
            ['timeCreate']].values.tolist()
        date_ton = [x[0] for x in index_ton]
        date_ton = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime.datetime(1970, 1,
                                                                                                1)).total_seconds() * 1000)
                    for x in date_ton]


        for i in range(len(df_tonality)):
            if df_tonality[i][2] == -1:
                bobble.append([date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1, df_tonality[i][3]])
            elif df_tonality[i][2] == 1:
                bobble.append([date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1, df_tonality[i][3]])

        for i in range(len(bobble)):
            if bobble[i][3] == 1:
                bobble[i][3] = "#32ff32"
            else:
                bobble[i][3] = "#FF3232"

        data = {
            "neg_smi_name": list_neg_smi[:20],
            "neg_smi_count": list_pos_smi_massage_count[:20],
            "neg_smi_rating": list_neg_smi_index[:20],
            "pos_smi_name": list_pos_smi[:20],
            "pos_smi_count": list_pos_smi_massage_count[:20],
            "pos_smi_rating": list_pos_smi_index[:20],

            "date_bobble": [x[0] for x in bobble],
            "name_bobble": [x[1] for x in bobble],
            "index_bobble": [x[2] for x in bobble],
            "z_index_bobble": [1] * len(bobble),
            "tonality_index_bobble": [x[3] for x in bobble], 
            "tonality_url": [x[4] for x in bobble],
        }


        theme = request.values.to_dict(flat=True)['file_choose'].split('_')[0]

        data = json.dumps(data)

        return render_template('media_rating.html', len_files=len_files, files=json_files, data=data, theme=theme)

    if 'send' not in request.values.to_dict(flat=True):
        data = {
            "neg_smi_name": ['mk', 'aif'],
            "neg_smi_count": [12, 11],
            "pos_smi_name": ['lenta', 'koms'],
            "pos_smi_rating": [15, 8],
            "neg_smi_rating": [1512, 1245],
            "pos_smi_rating": [214, 2512]
        }

        return render_template('media_rating.html', len_files=len_files, files=json_files, data=data)


@app.route('/voice', methods=['GET', 'POST'], endpoint='voice')
def voice():
    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files)


    if 'send' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)['text_search'] != '':

        session['filename'] = request.values.to_dict(flat=True)['file_choose']

        # parsing json
        try: 
            with io.open(session['filename'], encoding='utf-8', mode='r') as train_file:
                dict_train = json.load(train_file, strict=False)

        except:
            a = []
            with open(session['filename'], encoding='utf-8', mode='r') as file:
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
            df_meta_smi_only.dropna(subset=['timeCreate'], inplace=True)
            df_meta_smi_only = df_meta_smi_only.set_index(['timeCreate'])
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
                            df_meta_socm.dropna(subset=['timeCreate'], inplace=True)
                            df_meta_socm = df_meta_socm.set_index(['timeCreate'])
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
                            df_meta_smi.dropna(subset=['timeCreate'], inplace=True)
                            df_meta_smi = df_meta_smi.set_index(['timeCreate'])
                            df_meta_smi['date'] = [x[:10] for x in df_meta_smi.index]

            if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
                df_meta = pd.concat([df_meta_socm, df_meta_smi])
            elif 'df_meta_smi' and 'df_meta_socm' not in locals():
                df_meta = df_meta_smi
            else:
                df_meta = df_meta_socm

        def date_reverse(date):
            lst = date.split('-')
            temp = lst[1]
            lst[1] = lst[2]
            lst[2] = temp 
            return lst

        data_start = '-'.join(date_reverse('-'.join(
            [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][::-1])))
        data_stop = '-'.join(date_reverse('-'.join(
            [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][::-1])))

        mask = (df_meta['date'] >= data_start) & (df_meta['date'] <= data_stop)
        df_meta = df_meta.loc[mask]
        df_meta.reset_index(inplace=True)

        if df_meta.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
            flash('По запросу найдено 0 сообщений')
            return redirect(url_for('voice'))

        # df_meta = df_meta[:10]

        search_lst = request.values.to_dict(flat=True)['text_search'].split(',')
        search_lst = [x.split('или') for x in search_lst]
        search_lst = [[x.strip().lower() for x in group] for group in search_lst]

        text_val = df_meta['text'].values
        text_val = [x.lower() for x in text_val]

        dict_count = {key: [] for key in [x[0].capitalize() for x in
                                          search_lst]}  # словарь с названием продукта и индексом его встречаемости в таблице текстов
        list_keys = list(dict_count.keys())

        for j in range(len(text_val)):
            a = []
            for i in range(len(search_lst)):
                if [item for item in search_lst[i] if item in text_val[j]] != []:
                    dict_count[list_keys[i]].append(j)

        df_meta['toneMark'] = df_meta['toneMark'].map({0: 'Neutral', -1: 'Negative', 1: 'Positive'})

        ### блок для расчета диаграммы sunkey

        # учет кол-ва источников по объекту
        list_sunkey_hubs = []

        for i in range(len(list_keys)):
            dict_hubs = Counter(df_meta.loc[dict_count[list_keys[i]]]['hub'])
            dict_hubs = dict_hubs.most_common()
            list_hubs = [[x[0], list_keys[i], x[1]] for x in dict_hubs]
            list_sunkey_hubs.append(list_hubs)

        # учет кол-ва постов, репостов, комментариев
        list_sunkey_post_type = []
        list_type_post = []  # cписок для понимания, какие типы постов были для каждого объекта поиска (необходимо для тональности по типам далее)

        for i in range(len(list_keys)):
            list_sunkey_post = Counter(df_meta.loc[dict_count[list_keys[i]]]['type'])
            list_type_post.append(list(list_sunkey_post.keys()))  # добавляем какие типы постов встречались для объекта
            list_sunkey_post = list_sunkey_post.most_common()
            list_sunkey_post = [[list_keys[i], x[0], x[1]] for x in list_sunkey_post]
            list_sunkey_post_type.append(list_sunkey_post)

        # учет тональности по каждому типу сообщений
        list_sunkey_tonality = []  # кол-во тональности постов, репостов, комментариев
        tonality_type_posts = {}  # словарь для сбора отдельно кол-ва позитива, нейтрала и негавтива по типу источников в объекте поиска
        # {'Платон': [{'Комментарий': [('Neutral', 16), ('Negative', 6), ('Positive', 5)], 'Пост':...

        for i in range(len(list_keys)):
            df = df_meta.loc[dict_count[list_keys[i]]]
            a = []
            d = {}
            for j in range(len(list_type_post[i])):
                list_sunkey_ton = Counter(df[df['type'] == list_type_post[i][j]]['toneMark'])
                list_sunkey_ton = list_sunkey_ton.most_common()
                d[list_type_post[i][j]] = list_sunkey_ton

            tonality_type_posts[list_keys[i]] = d

        val_tonality_type_posts = list(tonality_type_posts.values())
        tonality_by_post_type = []  # собираем все типы сообщений, их тональность и кол-во ['Комментарий', 'Neutral', 16],
        # ['Комментарий', 'Negative', 6], ['Комментарий', 'Positive', 5], ['Пост', 'Neutral', 13]...
        for i in range(len(val_tonality_type_posts)):
            for k, v in val_tonality_type_posts[i].items():
                for l in range(len(v)):
                    tonality_by_post_type.append([k, v[l][0], v[l][1]])

        list_sunkey_hubs = [item for sublist in list_sunkey_hubs for item in sublist]
        list_sunkey_post_type = [item for sublist in list_sunkey_post_type for item in sublist]
        # tonality_by_post_type = [item for sublist in tonality_by_post_type for item in sublist]

        fin_list = []  # итоговый список с данными для sunkey ['facebook.com', 'Платон', 10], ['vk.com', 'Платон', 7], ['ok.ru', 'Платон', 6]...
        if list_sunkey_hubs is not None:
            fin_list.append(list_sunkey_hubs)
        if list_sunkey_post_type is not None:
            fin_list.append(list_sunkey_post_type)
        if tonality_by_post_type is not None:
            fin_list.append(tonality_by_post_type)

        fin_list = [item for sublist in fin_list for item in sublist]

        ### data to sunkey diagram done ###

        # продолжение расчета для первой/круговой диаграммы

        for key, val in dict_count.items():  # получаем словарь с количеством позитива, негатива и нейтрала ({'Которых': {'Neutral': 153}, 'Стоимость': {'Neutral': 18}})
            dict_count[key] = dict(Counter(list(df_meta[['toneMark']].loc[dict_count[key]]['toneMark'].values)))

        for key, val in dict_count.items():
            if 'Neutral' not in dict_count[key]:
                dict_count[key]['Neutral'] = 0
            if 'Negative' not in dict_count[key]:
                dict_count[key]['Negative'] = 0
            if 'Positive' not in dict_count[key]:
                dict_count[key]['Positive'] = 0


        dict_count = collections.OrderedDict(sorted(dict_count.items()))

        a = []  # список с кол-ом поз, нег и нейтр по продуктам (поиску) пример списка - [[0, 0, 153], [0, 0, 18]]
        for key, val in dict_count.items():
            a.append([dict_count[key]["Negative"], dict_count[key]["Positive"], dict_count[key]["Neutral"]])

        

        theme = request.values.to_dict(flat=True)['file_choose'].split('_')[0]
        dict_names = {key: key[0] for key in [x for x in list(dict_count.keys())]}
        dict_names = collections.OrderedDict(sorted(dict_names.items()))


        data = {
            "negative": [x[0] for x in a],
            "positive": [x[1] for x in a],
            "neutral": [x[2] for x in a],
            "list_sunkey_hubs": list_sunkey_hubs,
            "list_sunkey_post_type": list_sunkey_post_type,
            "tonality_by_post_type": tonality_by_post_type,
            "fin_list": fin_list,
            "names": list(dict_names.keys()),

            "hubs": [x[0] for x in list_sunkey_hubs],
            "type_message": list(set([x[1] for x in list_sunkey_post_type])),
            "tonality": list(set([x[1] for x in tonality_by_post_type])),
        }

        data = json.dumps(data)

        return render_template('voice.html', files=json_files, len_files=len_files, data=data, theme=theme,
                               dict_names=dict_names)

    return render_template('voice.html', files=json_files, len_files=len_files)



@app.route('/Kmeans', methods=["GET", "POST"], endpoint='kmeans')
def Kmeans():


    from sqlalchemy.dialects import postgresql

    os.chdir(path_to_files)
    json_files = [x for x in os.listdir(path_to_embed_save) if '.csv' in x]
    len_files = len(json_files)

    if 'send' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)['file_choose'] != 'Выбрать файл':

        print(datetime.datetime.now())

        cluster_num = request.values.to_dict(flat=True)['clusters_choose']
        # query = db.session.query(EmbedTable.data_embed).filter_by(filename=request.values.to_dict(flat=True)['file_choose']).order_by(EmbedTable.id.asc()).first()
        # numpy_arr = ast.literal_eval(query[0])

        # print(df.head())
        # метаданных
        # df = df[['text', 'url', 'hub', 'timeCreate']]
        os.chdir(path_to_embed_save)
        df = pd.read_csv(request.values.to_dict(flat=True)['file_choose'].replace('.json', '.csv'))
        
        # timestamp to date
        df['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                 df['timeCreate'].values]

        df = df.set_index(['timeCreate'])  # индекс - время создания поста

        def date_reverse(date):  # фильтрация по дате/календарик
            lst = date.split('-')
            temp = lst[1]
            lst[1] = lst[2]
            lst[2] = temp
            return lst


        if 'daterange' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)['daterange'] != '01/01/2022 - 01/12/2022':
            data_start = '-'.join(date_reverse('-'.join(
                [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][::-1])))
            data_stop = '-'.join(date_reverse('-'.join(
                [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][::-1])))

            df = df.loc[data_stop:data_start]  # фильтрация данных по дате в календарике

        df = df.reset_index() # возвращаем индексы к 0, 1, 2, 3 для дальнейшей итерации по ним

        if df.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
            flash('По запросу найдено 0 сообщений')
            return redirect(url_for('kmeans'))


        print('start kmeans clusterisation')
        print(datetime.datetime.now())

        clusters = KMeans(n_clusters = int(cluster_num))
        df_clust = df.loc[:, ~df.columns.isin(['text', 'url', 'hub', 'timeCreate'])]
        clusters.fit(df_clust.values)

        df = df[['text', 'url', 'hub']]
        df['label'] = clusters.labels_

        clusters_len = np.arange(1, 50)


        df.drop_duplicates(subset=['text'], inplace=True)
        columns = df.columns
        df = df.reset_index()

        list_count_labels = list(Counter(df['label'].values).keys())
        count_clusters = Counter(df['label'].values) # кол-во сообщений в каждом кластере

        a = []
        df_values = df.values

        for i in range(len(list_count_labels)):
            for j in range(len(df_values)):
                if list_count_labels[i] == df_values[j][4]:
                    a.append(df_values[j])


        df = pd.DataFrame(a)
        df.drop(0, axis=1, inplace=True)
        df.columns = columns
        df.drop_duplicates(subset=['text'], inplace=True)
        df['value'] = df['label'].map(count_clusters)
        df = df.sort_values(by='value', ascending=False)
        df.drop(['value'], axis=1, inplace=True)

        # https://gist.github.com/komasaru/ed07018ae246d128370a1693f5dd1849
        def shorten(url_long): # делаем ссылки короткими для отображения в web

            URL = "http://tinyurl.com/api-create.php"
            try:
                url = URL + "?" \
                    + urllib.parse.urlencode({"url": url_long})
                res = requests.get(url)
                if res.text == 'Error':
                    return url_long
                else:
                    return res.text

            except Exception as e:
                raise

        # create short url links
        # df.loc[:, 'url'] = [shorten(x) for x in df['url'].values]
        # create active url links
        # df['url'] = '<a href="' + df['url'] + '">' + df['url'] + '</a>'

        # данные для первого графика (heat)
        name_cluster_column = ['Cluster_' + str(x) for x in df['label'].values]

        df_count_hub = df[['hub']]
        df_count_hub['cluster_name'] = name_cluster_column

        dict_count_hub = {}
        df_count_hub_values = df_count_hub.values

        for i in range(len(df_count_hub_values)): # получаем словарь вида {'Cluster_4': ['telegra.ph', 'vk.com', 'vk.com', ...
            if df_count_hub_values[i][1] not in dict_count_hub.keys():
                dict_count_hub[df_count_hub_values[i][1]] = []
                dict_count_hub[df_count_hub_values[i][1]].append(df_count_hub_values[i][0])

            else:
                dict_count_hub[df_count_hub_values[i][1]].append(df_count_hub_values[i][0])


        for key, val in dict_count_hub.items(): # получаем словарь с подсчетом каждого источника вида 'Cluster_1': Counter({'telegram.org': 21, 'vk.com': 34, 'instagram.com': 8, ...
            dict_count_hub[key] = Counter(val)

        # преобразуем Counter словарь с кол-ом источников по каждому кластеру к массиву для передачи во front
        # {'instagram.com': 6, 'telegram.org': 24, 'banki.ru': 11, 'vk.com': 58,..
        for key, val in dict_count_hub.items():
            dict_count_hub[key] = dict(val)

        a = [[dict_count_hub[group] for val in group] for group in dict_count_hub]
        hubs_cluster_id = [list(x[0].keys()) for x in a] # ['telegra.ph', 'vk.com', 'zen.yandex.ru', 'ok.ru', 'youtube.com', 'telegram.org', 'apple.com'
        hubs_cluster_values = [list(x[0].values()) for x in a] # [6, 137, 11, 12, 13, 56, 1, 6, 20, 11, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1], [6,

        hub_parents = []
        for i in range(len(hubs_cluster_values)):
            hub_parents.append([list(dict_count_hub.keys())[i] for x in hubs_cluster_values[i]])

        data = {
                "cluster_names": list(dict_count_hub.keys()), # кластеры ['Cluster_4', 'Cluster_7', 'Cluster_2', 'Cluster_5', 'Cluster_0', 'Cluster_3', 'Cluster_1', 'Cluster_6']
                "cluster_values": list(Counter(df['label'].values).values()), # сколько сообщений в кластере [287, 216, 211, 141, 128, 111, 68, 4]
                "hubs_cluster_id": hubs_cluster_id,
                "hubs_cluster_values": hubs_cluster_values,
                "cluster_parent": hub_parents,
        }

        data_to_table = list(df.values)
        array_num = list(numpy.arange(0, df.shape[0]))
        array_num = [float(x) for x in array_num]


        texts = [x[0] for x in data_to_table]
        url = [x[1] for x in data_to_table]
        hubs = [x[2] for x in data_to_table]
        cluster = [x[3] for x in data_to_table]

        regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")
        def words_only(text, regex=regex):
            try:
                return " ".join(regex.findall(text))
            except:
                return ""

        def preprocess_text(text):
            text = text.lower().replace("ё", "е")
            text = re.sub('((www\[^\s]+)|(https?://[^\s]+))','URL', text)
            text = re.sub('@[^\s]+','USER', text)
            text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
            text = re.sub(' +',' ', text)
            return text.strip()

        mystopwords = stopwords.words('russian') + ['это', 'наш' , 'тыс', 'млн', 'млрд', 'также',  'т', 'д', 'URL',
                                                    'i', 's', 'v', 'info', 'a', 'подробнее', 'который', 'год',
                                                ' - ', '-','В','—', '–', '-', 'в', 'который']

        def  remove_stopwords(text, mystopwords = mystopwords):
            try:
                return " ".join([token for token in text.split() if not token in mystopwords])
            except:
                return ""

        data_table = {
            "number": array_num, 
            "texts": [remove_stopwords(x) for x in texts],
            "url": url,
            "hub": hubs,
            "cluster": cluster
        }

        # df['text'] = df['text'].apply(preprocess_text) 
        # df['text'] = df['text'].apply(remove_stopwords)
        # df['text'] = df['text'].apply(words_only)
        

        return render_template('Kmeans.html', len_files=len_files, files=json_files, clusters_len=clusters_len, data=data,
                                tables=[df.to_html(classes='data', render_links=True, escape=False)], titles=df.columns.values, 
                                data_table=data_table)


    data = {
    "cluster_names": ['Cluster_4', 'Cluster_7', 'Cluster_2'],
    "cluster_values": [287, 216, 211],
    "hubs_cluster_id": ['VK', 'OK', 'Fb'], 
    "hubs_cluster_values": [11, 12, 8],
    "cluster_parent": ['Cluster_4', 'Cluster_7', 'Cluster_2'] 
    }

    clusters_len = np.arange(1, 50)
    return render_template('Kmeans.html', len_files=len_files, files=json_files, clusters_len=clusters_len, data=data)



@app.route('/ml_tonality', methods=["GET", "POST"])
def test():

    return render_template('tonality_option.html')


@app.route('/competitors', methods=["GET", "POST"], endpoint='competitors')
def test():
    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files)

    another_graph = []  # данные для отображения графиков bubbles и далее
    if 'selected_files' in request.values.to_dict(flat=False):
        files = request.values.to_dict(flat=False)['selected_files']
        filenames = files.copy()

        min_date = []
        max_date = []

        for i in range(len(files)):
            print(i)
            
            try: 
                with io.open(files[i], encoding='utf-8', mode='r') as train_file:
                    files_df = json.load(train_file, strict=False)
                    # заменяем audience на audienceCount для случаев только СМИ
                    files_df = [{"audienceCount" if k == "audience" else k:v for k,v in x.items()} for x in files_df]
                    
                    # добавляем в СМИ объект 'authorObject'
                    for i in range(len(files_df)):
                        if 'authorObject' not in files_df[i]:
                            authorObject = {}
                            authorObject['url'] = files_df[i]['url']
                            authorObject['author_type'] = files_df[i]['categoryName']
                            authorObject['sex'] = 'n/a'
                            authorObject['age'] = 'n/a'
                            files_df[i]['authorObject'] = authorObject
                            
                        if 'hubtype' not in files_df[i]:
                            files_df[i]['hubtype'] = 'Новости'
                    
                    # поиск минимальной и максимальной даты (самой ранней и самой поздней)
                    min_date.append(np.min([x['timeCreate'] for x in [item for item in files_df]]))
                    max_date.append(np.max([x['timeCreate'] for x in [item for item in files_df]]))
                    another_graph.append(files_df)
                    

            except:
                a = []
                with open(files[i], encoding='utf-8', mode='r') as file:
                    for line in file:
                        a.append(line)
                dict_train = []
                for i in range(len(a)):
                    try:
                        dict_train.append(ast.literal_eval(a[i]))
                    except:
                        continue
                dict_train = [x[0] for x in dict_train]
                # заменяем audience на audienceCount для случаев только СМИ
                dict_train = [{"audienceCount" if k == "audience" else k:v for k,v in x.items()} for x in dict_train]
                # добавляем в СМИ объект 'authorObject'
                for i in range(len(dict_train)):
                    authorObject = {}
                    authorObject['url'] = dict_train[i]['url']
                    authorObject['author_type'] = dict_train[i]['categoryName']
                    authorObject['sex'] = 'n/a'
                    authorObject['age'] = 'n/a'
                    dict_train[i]['authorObject'] = authorObject
                    dict_train[i]['hubtype'] = 'Новости'
                    
                files_df = dict_train

                # поиск минимальной и максимальной даты (самой ранней и самой поздней)
                min_date.append(np.min([x['timeCreate'] for x in [item for item in files_df]]))
                max_date.append(np.max([x['timeCreate'] for x in [item for item in files_df]]))

                another_graph.append(files_df)

            dates = [min_date, max_date]

        min_date = np.min(dates[0])
        max_date = np.max(dates[1])

        # для фиксации далее файла, в котором будет минимальная/начальная дата для старта графика
        min_date_number_file = dates[0].index(min_date)

        for i in range(len(another_graph)):

            files[i] = pd.DataFrame(another_graph[i])
            # метаданные
            files[i] = files[i][['text', 'url', 'hub', 'timeCreate', 'audienceCount']]

            #     files[i] = files[i].set_index(['timeCreate'])  # индекс - время создания поста

            def date_reverse(date):  # фильтрация по дате/календарик
                lst = date.split('-')
                temp = lst[1]
                lst[1] = lst[2]
                lst[2] = temp
                return lst

            files[i]['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                      files[i]['timeCreate'].values]

            files[i] = files[i].set_index(['timeCreate'])


            if 'daterange' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)[
                'daterange'] != datetime.datetime.now().strftime("%m/%d/%Y") + ' - ' + datetime.datetime.now().strftime("%m/%d/%Y"):
                data_start = '-'.join(date_reverse('-'.join(
                    [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][
                    ::-1])))
                data_stop = '-'.join(date_reverse('-'.join(
                    [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][
                    ::-1])))
                
                files[i] = files[i].reset_index()
                mask = (files[i]['timeCreate'] >= data_start) & (files[i]['timeCreate'] <= data_stop) # фильтрация данных по дате в календарике
                files[i] = files[i].loc[mask]


            files[i]['timeCreate'] = pd.to_datetime(files[i]['timeCreate'])

            # разбиваем даные на 10-минутки (отрезки по 10 мин для группировки далее)
            bins = pd.date_range(datetime.datetime.fromtimestamp(min_date).strftime('%Y-%m-%d %H:%M:%S'),
                                 datetime.datetime.fromtimestamp(max_date).strftime('%Y-%m-%d %H:%M:%S'), freq='10T')
            files[i]['bins'] = pd.cut(files[i]['timeCreate'], bins)

            files[i]['bins_left'] = pd.IntervalIndex(pd.cut(files[i]['timeCreate'], bins)).left
            files[i]['bins_right'] = pd.IntervalIndex(pd.cut(files[i]['timeCreate'], bins)).right

            files[i]['bins_time_unix'] = files[i]['bins_left'].to_numpy().astype(numpy.int64) // 10 ** 9

            files[i].index = files[i]

            # оставляем только время
            files[i] = files[i][['bins_time_unix', 'bins_left', 'bins_right']]

            # подсчет кол-ва собщений за каждые n минут
            files[i] = pd.DataFrame(files[i][['bins_time_unix']].value_counts())
            files[i]['bins_time_unix'] = [x[0] for x in list(files[i].index)]
            files[i].columns = ['count', 'time']
            files[i] = files[i][['time', 'count']]
            # сортируем по дате от минимальной до конечной, убираем 1ую строчку (с отрицат.значением вермени)
            files[i] = files[i].sort_values(by='time').loc[1:]
            files[i].columns = ['time_' + str(i), filenames[i]]

        # объединяем данные в финальную таблицу
        # добавляем колонку с индексом для объединения талиц в 1 табл.
        for i in range(len(files)):
            files[i]['index_col'] = [x[0] for x in files[i].index]

        # ф-ия объединения нескольких таблиц в одну
        df_fin = ft.reduce(lambda left, right: pd.merge(left, right, on=['index_col']), files)

        # убираем одинаковые столбцы с датами и index_col (такие же значения есть в индексе), оставляем кол-во сообщений за каждые 10мин
        df_fin = df_fin[[x for x in df_fin.columns if 'time' not in x]]
        df_fin.index = df_fin['index_col']
        df_fin = df_fin[[x for x in df_fin.columns if 'index_col' not in x]]

        df_fin['date'] = [x for x in df_fin.index]
        df_fin = df_fin[df_fin.columns[::-1]]

        # приводим дату к unix-формату
        a = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in df_fin['date'].values]
        a = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') -
                  datetime.datetime(1970, 1, 1)).total_seconds() * 1000) for x in a]
        df_fin['date'] = a

        dg = df_fin.to_csv(index=False)

        # bobbles
        a = None
        data = {}
        data["news_smi_name"] = []
        data["news_smi_count"] = []
        data["news_smi_rating"] = []
        count_data = 50  # сколько данных/кругов выводить внутри

        for i in range(len(another_graph)):

            files[i] = pd.DataFrame(another_graph[i])
            columns = ['citeIndex', 'timeCreate', 'toneMark', 'hubtype', 'hub', 'audienceCount', 'url']

            # метаданные
            files[i] = pd.concat([pd.DataFrame.from_records(files[i]['authorObject'].values), files[i][columns]],
                                 axis=1)

            files[i]['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                      files[i]['timeCreate'].values]

            files[i] = files[i].set_index(['timeCreate'])  # индекс - время создания поста

            def date_reverse(date):
                lst = date.split('-')
                temp = lst[1]
                lst[1] = lst[2]
                lst[2] = temp
                return lst

            if 'daterange' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)[
                'daterange'] != datetime.datetime.now().strftime("%m/%d/%Y") + ' - ' + datetime.datetime.now().strftime("%m/%d/%Y"):
                data_start = '-'.join(date_reverse('-'.join(
                    [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][
                    ::-1])))
                data_stop = '-'.join(date_reverse('-'.join(
                    [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][
                    ::-1])))

                files[i] = files[i].loc[data_stop: data_start]

            files[i]['timeCreate'] = list(files[i].index)

            if files[i].shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
                flash('По запросу найдено 0 сообщений (проверьте даты и/или другие условия)')

            if set(files[i]['hub'].values) == {"telegram.org"}:

                a = files[i]
                a = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['hub'] == "telegram.org")]

                # negative smi
                df_hub_siteIndex = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == -1)][
                    ['fullname', 'audienceCount']].values

                dict_neg = {}
                for i in range(len(df_hub_siteIndex)):

                    if df_hub_siteIndex[i][0] not in dict_neg.keys():

                        dict_neg[df_hub_siteIndex[i][0]] = []
                        dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                    else:
                        dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                list_neg = [list(set(x)) for x in dict_neg.values()]
                list_neg = [[0] if x[0] == 'n/a' else x for x in list_neg if x != 'n/a']
                list_neg = [int(x[0]) for x in list_neg]

                for i in range(len(list_neg)):
                    dict_neg[list(dict_neg.keys())[i]] = list_neg[i]

                dict_neg = dict(sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))

                dict_neg_hubs_count = dict(
                    Counter(list(
                        a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == -1)]['fullname'])))

                fin_neg_dict = defaultdict(tuple)
                for d in (dict_neg, dict_neg_hubs_count):  # you can list as many input dicts as you want here
                    for key, value in d.items():
                        fin_neg_dict[key] += (value,)

                list_neg_smi = list(fin_neg_dict.keys())
                list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
                list_neg_smi_massage_count = [x[1] for x in fin_neg_dict.values()]

                # positive smi
                df_hub_siteIndex = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == 1)][
                    ['fullname', 'audienceCount']].values

                dict_pos = {}
                for i in range(len(df_hub_siteIndex)):

                    if df_hub_siteIndex[i][0] not in dict_pos.keys():

                        dict_pos[df_hub_siteIndex[i][0]] = []
                        dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                    else:
                        dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                list_pos = [list(set(x)) for x in dict_pos.values()]
                list_pos = [[0] if x[0] == 'n/a' else x for x in list_pos if x != 'n/a']
                list_pos = [int(x[0]) for x in list_pos]

                for i in range(len(list_pos)):
                    dict_pos[list(dict_pos.keys())[i]] = list_pos[i]

                dict_pos = dict(sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))

                dict_pos_hubs_count = dict(
                    Counter(list(
                        a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == 1)]['fullname'])))

                fin_pos_dict = defaultdict(tuple)
                for d in (dict_pos, dict_pos_hubs_count):  # you can list as many input dicts as you want here
                    for key, value in d.items():
                        fin_pos_dict[key] += (value,)

                list_pos_smi = list(fin_pos_dict.keys())
                list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
                list_pos_smi_massage_count = [x[1] for x in fin_pos_dict.values()]

                # data to bobble graph
                bobble = []
                df_tonality = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] != 0)][
                    ['fullname', 'audienceCount', 'toneMark', 'url']].values
                index_ton = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] != 0)][
                    ['timeCreate']].values.tolist()
                date_ton = [x[0] for x in index_ton]
                date_ton = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime.datetime(1970, 1,
                                                                                                        1)).total_seconds() * 1000)
                            for x in date_ton]

                for i in range(len(df_tonality)):
                    if df_tonality[i][2] == -1:
                        bobble.append(
                            [date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1, df_tonality[i][4]])
                    elif df_tonality[i][2] == 1:
                        bobble.append(
                            [date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1, df_tonality[i][4]])

                for i in range(len(bobble)):
                    if bobble[i][3] == 1:
                        bobble[i][3] = "#32ff32"
                    else:
                        bobble[i][3] = "#FF3232"

                list_neg_smi = [words_only(x) for x in list_neg_smi]
                list_pos_smi = [words_only(x) for x in list_pos_smi]
                name_bobble = [x[1] for x in bobble]
                name_bobble = [words_only(x) for x in name_bobble]

                data = {
                    "neg_smi_name": list_neg_smi[:100],
                    "neg_smi_count": list_pos_smi_massage_count[:100],
                    "neg_smi_rating": list_neg_smi_index[:100],
                    "pos_smi_name": list_pos_smi[:100],
                    "pos_smi_count": list_pos_smi_massage_count[:100],
                    "pos_smi_rating": list_pos_smi_index[:100],

                    "date_bobble": [x[0] for x in bobble],
                    "name_bobble": name_bobble,
                    "index_bobble": [x[2] for x in bobble],
                    "z_index_bobble": [1] * len(bobble),
                    "tonality_index_bobble": [x[3] for x in bobble],
                    "tonality_url": [x[4] for x in bobble],
                }

                data = json.dumps(data)

                return render_template('competitors.html', len_files=len_files, files=json_files, data=data)

            a = files[i]
            files[i] = files[i][files[i]['hubtype'] == 'Новости']

            # smi
            df_hub_siteIndex = a[a['hubtype'] == 'Новости'][['hub', 'citeIndex']].values

            dict_news = {}
            for i in range(len(df_hub_siteIndex)):

                if df_hub_siteIndex[i][0] not in dict_news.keys():

                    dict_news[df_hub_siteIndex[i][0]] = []
                    dict_news[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                else:
                    dict_news[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

            dict_news_hubs_count = dict(
                Counter(list(a[a['hubtype'] == 'Новости']['hub'])))

            fin_news_dict = defaultdict(tuple)
            for d in (dict_news, dict_news_hubs_count):  # you can list as many input dicts as you want here
                for key, value in d.items():
                    fin_news_dict[key] += (value,)

            list_news_smi = list(fin_news_dict.keys())
            list_news_smi_index = [x[0] for x in fin_news_dict.values()]
            list_news_smi_massage_count = [x[1] for x in fin_news_dict.values()]

            # убираем n/a, берем максимальное значение индекса СМИ за период
            for i in range(len(list_news_smi_index)):
                list_news_smi_index[i] = [int(x) for x in list_news_smi_index[i] if type(x) != str]

                if list_news_smi_index[i] == []:
                    list_news_smi_index[i] = [0]
                list_news_smi_index[i] = np.max(list_news_smi_index[i])

            data["news_smi_name"].append(list_news_smi[:count_data])
            data["news_smi_count"].append(list_news_smi_massage_count[:count_data])
            data["news_smi_rating"].append([int(x) for x in list_news_smi_index][:count_data])

        datas = {
            "news_smi_name": data["news_smi_name"],
            "news_smi_count": data["news_smi_count"],
            "news_smi_rating": data["news_smi_rating"]
        }

        # bubble2 socmedia
        a = None
        data = {}
        data["list_socmedia"] = []
        data["list_socmedia_massage_count"] = []
        data["list_socmedia_index"] = []

        for i in range(len(another_graph)):

            files[i] = pd.DataFrame(another_graph[i])
            columns = ['citeIndex', 'timeCreate', 'toneMark', 'hubtype', 'hub', 'audienceCount', 'url']

            # метаданные
            files[i] = pd.concat([pd.DataFrame.from_records(files[i]['authorObject'].values), files[i][columns]],
                                 axis=1)

            files[i]['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                      files[i]['timeCreate'].values]

            files[i] = files[i].set_index(['timeCreate'])  # индекс - время создания поста

            def date_reverse(date):
                lst = date.split('-')
                temp = lst[1]
                lst[1] = lst[2]
                lst[2] = temp
                return lst

            if 'daterange' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)[
                'daterange'] != datetime.datetime.now().strftime("%m/%d/%Y") + ' - ' + datetime.datetime.now().strftime("%m/%d/%Y"):
                data_start = '-'.join(date_reverse('-'.join(
                    [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][
                    ::-1])))
                data_stop = '-'.join(date_reverse('-'.join(
                    [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][
                    ::-1])))

                files[i] = files[i].loc[data_stop: data_start]

            files[i]['timeCreate'] = list(files[i].index)

            if files[i].shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
                flash('По запросу найдено 0 сообщений (проверьте даты и/или другие условия)')

            if set(files[i]['hub'].values) == {"telegram.org"}:

                a = files[i]
                a = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['hub'] == "telegram.org")]

                # negative smi
                df_hub_siteIndex = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == -1)][
                    ['fullname', 'audienceCount']].values

                dict_neg = {}
                for i in range(len(df_hub_siteIndex)):

                    if df_hub_siteIndex[i][0] not in dict_neg.keys():

                        dict_neg[df_hub_siteIndex[i][0]] = []
                        dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                    else:
                        dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                list_neg = [list(set(x)) for x in dict_neg.values()]
                list_neg = [[0] if x[0] == 'n/a' else x for x in list_neg if x != 'n/a']
                list_neg = [int(x[0]) for x in list_neg]

                for i in range(len(list_neg)):
                    dict_neg[list(dict_neg.keys())[i]] = list_neg[i]

                dict_neg = dict(sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))

                dict_neg_hubs_count = dict(
                    Counter(list(
                        a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == -1)]['fullname'])))

                fin_neg_dict = defaultdict(tuple)
                for d in (dict_neg, dict_neg_hubs_count):  # you can list as many input dicts as you want here
                    for key, value in d.items():
                        fin_neg_dict[key] += (value,)

                list_neg_smi = list(fin_neg_dict.keys())
                list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
                list_neg_smi_massage_count = [x[1] for x in fin_neg_dict.values()]

                # positive smi
                df_hub_siteIndex = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == 1)][
                    ['fullname', 'audienceCount']].values

                dict_pos = {}
                for i in range(len(df_hub_siteIndex)):

                    if df_hub_siteIndex[i][0] not in dict_pos.keys():

                        dict_pos[df_hub_siteIndex[i][0]] = []
                        dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                    else:
                        dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                list_pos = [list(set(x)) for x in dict_pos.values()]
                list_pos = [[0] if x[0] == 'n/a' else x for x in list_pos if x != 'n/a']
                list_pos = [int(x[0]) for x in list_pos]

                for i in range(len(list_pos)):
                    dict_pos[list(dict_pos.keys())[i]] = list_pos[i]

                dict_pos = dict(sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))

                dict_pos_hubs_count = dict(
                    Counter(list(
                        a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == 1)]['fullname'])))

                fin_pos_dict = defaultdict(tuple)
                for d in (dict_pos, dict_pos_hubs_count):  # you can list as many input dicts as you want here
                    for key, value in d.items():
                        fin_pos_dict[key] += (value,)

                list_pos_smi = list(fin_pos_dict.keys())
                list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
                list_pos_smi_massage_count = [x[1] for x in fin_pos_dict.values()]

                # data to bobble graph
                bobble = []
                df_tonality = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] != 0)][
                    ['fullname', 'audienceCount', 'toneMark', 'url']].values
                index_ton = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] != 0)][
                    ['timeCreate']].values.tolist()
                date_ton = [x[0] for x in index_ton]
                date_ton = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime.datetime(1970, 1,
                                                                                                        1)).total_seconds() * 1000)
                            for x in date_ton]

                for i in range(len(df_tonality)):
                    if df_tonality[i][2] == -1:
                        bobble.append(
                            [date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1, df_tonality[i][4]])
                    elif df_tonality[i][2] == 1:
                        bobble.append(
                            [date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1, df_tonality[i][4]])

                for i in range(len(bobble)):
                    if bobble[i][3] == 1:
                        bobble[i][3] = "#32ff32"
                    else:
                        bobble[i][3] = "#FF3232"

                list_neg_smi = [words_only(x) for x in list_neg_smi]
                list_pos_smi = [words_only(x) for x in list_pos_smi]
                name_bobble = [x[1] for x in bobble]
                name_bobble = [words_only(x) for x in name_bobble]

                data = {
                    "neg_smi_name": list_neg_smi[:100],
                    "neg_smi_count": list_pos_smi_massage_count[:100],
                    "neg_smi_rating": list_neg_smi_index[:100],
                    "pos_smi_name": list_pos_smi[:100],
                    "pos_smi_count": list_pos_smi_massage_count[:100],
                    "pos_smi_rating": list_pos_smi_index[:100],

                    "date_bobble": [x[0] for x in bobble],
                    "name_bobble": name_bobble,
                    "index_bobble": [x[2] for x in bobble],
                    "z_index_bobble": [1] * len(bobble),
                    "tonality_index_bobble": [x[3] for x in bobble],
                    "tonality_url": [x[4] for x in bobble],
                }

                data = json.dumps(data)

                return render_template('competitors.html', len_files=len_files, files=json_files, data=data)

            a = files[i]
            files[i] = files[i][files[i]['hubtype'] != 'Новости']

            df_hub_siteIndex = a[a['hubtype'] != 'Новости'][['hub', 'audienceCount']].values

            dict_socmedia = {}
            for i in range(len(df_hub_siteIndex)):

                if df_hub_siteIndex[i][0] not in dict_socmedia.keys():

                    dict_socmedia[df_hub_siteIndex[i][0]] = []
                    dict_socmedia[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                else:
                    dict_socmedia[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

            dict_socmedia_hubs_count = dict(
                Counter(list(a[a['hubtype'] != 'Новости']['hub'])))

            fin_news_dict = defaultdict(tuple)
            for d in (dict_socmedia, dict_socmedia_hubs_count):  # you can list as many input dicts as you want here
                for key, value in d.items():
                    fin_news_dict[key] += (value,)

            list_socmedia = list(fin_news_dict.keys())
            list_socmedia_index = [x[0] for x in fin_news_dict.values()]
            list_socmedia_massage_count = [x[1] for x in fin_news_dict.values()]

            # убираем n/a, берем максимальное значение индекса СМИ за период
            for i in range(len(list_socmedia_index)):
                list_socmedia_index[i] = [int(x) for x in list_socmedia_index[i] if type(x) != str]

                if list_socmedia_index[i] == []:
                    list_socmedia_index[i] = [0]
                list_socmedia_index[i] = np.max(list_socmedia_index[i])

            data["list_socmedia"].append(list_socmedia[:count_data])
            data["list_socmedia_massage_count"].append(list_socmedia_massage_count[:count_data])
            data["list_socmedia_index"].append(list_socmedia_index[:count_data])

        # итоговые данные для 2х графикоков СМИ и Соцмедиа (bubbles)
        datas["list_socmedia"] = data["list_socmedia"]
        datas["list_socmedia_massage_count"] = data["list_socmedia_massage_count"]
        datas["list_socmedia_rating"] = [[int(x) for x in item] for item in data["list_socmedia_index"]]
        datas["filename"] = filenames

        # bubbles3 - график с динамикой отношения в СМИ и соцмедиа (динамика bubbles)
        date_bobble = []
        name_bobble = []
        index_bobble = []
        z_index_bobble = []
        tonality_index_bobble = []
        tonality_url = []

        for j in range(len(another_graph)):

            df = pd.DataFrame(another_graph[j])

            # метаданные
            columns = ['citeIndex', 'timeCreate', 'toneMark', 'hubtype', 'hub', 'audienceCount', 'url']
            # columns.remove('text')
            df_meta = pd.concat(
                [pd.DataFrame([x['authorObject'] if 'authorObject' in x else '' for x in another_graph[j]]),
                 df[columns]], axis=1)
            # timestamp to date
            df_meta['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                     df_meta['timeCreate'].values]

            df_meta = df_meta.set_index(['timeCreate'])  # индекс - время создания поста

            def date_reverse(date):
                lst = date.split('-')
                temp = lst[1]
                lst[1] = lst[2]
                lst[2] = temp
                return lst

            if 'daterange' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)[
                'daterange'] != datetime.datetime.now().strftime("%m/%d/%Y") + ' - ' + datetime.datetime.now().strftime("%m/%d/%Y"):
                data_start = '-'.join(date_reverse('-'.join(
                    [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][
                    ::-1])))
                data_stop = '-'.join(date_reverse('-'.join(
                    [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][
                    ::-1])))

            df_meta['timeCreate'] = list(df_meta.index)
            df_meta = df_meta[df_meta['hubtype'] == 'Новости']

            # negative smi
            df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] == -1)][
                ['hub', 'citeIndex']].values

            dict_neg = {}
            for i in range(len(df_hub_siteIndex)):

                if df_hub_siteIndex[i][0] not in dict_neg.keys():

                    dict_neg[df_hub_siteIndex[i][0]] = []
                    dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                else:
                    dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

            list_neg = [list(set(x)) for x in dict_neg.values()]
            list_neg = [[0] if x[0] == 'n/a' else x for x in list_neg if x != 'n/a']
            list_neg = [int(x[0]) if x[0] != '' else 0 for x in list_neg]

            for i in range(len(list_neg)):
                dict_neg[list(dict_neg.keys())[i]] = list_neg[i]

            dict_neg = dict(sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))

            dict_neg_hubs_count = dict(
                Counter(list(df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] == -1)]['hub'])))

            fin_neg_dict = defaultdict(tuple)
            for d in (dict_neg, dict_neg_hubs_count):  # you can list as many input dicts as you want here
                for key, value in d.items():
                    fin_neg_dict[key] += (value,)

            list_neg_smi = list(fin_neg_dict.keys())
            list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
            list_neg_smi_massage_count = [x[1] for x in fin_neg_dict.values()]

            # positive smi
            df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] == 1)][
                ['hub', 'citeIndex']].values

            dict_pos = {}
            for i in range(len(df_hub_siteIndex)):

                if df_hub_siteIndex[i][0] not in dict_pos.keys():

                    dict_pos[df_hub_siteIndex[i][0]] = []
                    dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                else:
                    dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

            list_pos = [list(set(x)) for x in dict_pos.values()]
            list_pos = [[0] if x[0] == 'n/a' else x for x in list_pos if x != 'n/a']
            list_pos = [int(x[0]) if x[0] != '' else 0 for x in list_pos]

            for i in range(len(list_pos)):
                dict_pos[list(dict_pos.keys())[i]] = list_pos[i]

            dict_pos = dict(sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))

            dict_pos_hubs_count = dict(
                Counter(list(df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] == 1)]['hub'])))

            fin_pos_dict = defaultdict(tuple)
            for d in (dict_pos, dict_pos_hubs_count):  # you can list as many input dicts as you want here
                for key, value in d.items():
                    fin_pos_dict[key] += (value,)

            list_pos_smi = list(fin_pos_dict.keys())
            list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
            list_pos_smi_massage_count = [x[1] for x in fin_pos_dict.values()]

            # data to bobble graph
            bobble = []
            df_tonality = df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] != 0)][
                ['hub', 'citeIndex', 'toneMark', 'url']].values
            index_ton = df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] != 0)][
                ['timeCreate']].values.tolist()
            date_ton = [x[0] for x in index_ton]
            date_ton = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime.datetime(1970, 1,
                                                                                                    1)).total_seconds() * 1000)
                        for x in date_ton]

            for i in range(len(df_tonality)):
                if df_tonality[i][2] == -1:
                    bobble.append([date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1, df_tonality[i][4]])
                elif df_tonality[i][2] == 1:
                    bobble.append([date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1, df_tonality[i][4]])

            colors_red = ['#8B0000', '#FF4500', '#FFA07A']
            colors_green = ['#006400', '#00FF00', '#8FBC8F']

            for i in range(len(bobble)):
                if bobble[i][3] == 1:
                    bobble[i][3] = colors_green[j]
                else:
                    bobble[i][3] = colors_red[j]

            list_neg_smi = [words_only(x) for x in list_neg_smi]
            list_pos_smi = [words_only(x) for x in list_pos_smi]
            names_bobble = [x[1] for x in bobble]  # названия источников
            names_bobble = [words_only(x) for x in names_bobble]  # названия источников

            count_date = 50  # сколько данных взять из списков
            date_bobble.append([x[0] for x in bobble][:count_date])
            name_bobble.append(names_bobble[:count_date])
            index_bobble.append([x[2] for x in bobble][:count_date])
            z_index_bobble.append([1] * len(bobble[:count_date]))
            tonality_index_bobble.append([x[3] for x in bobble][:count_date])
            tonality_url.append([x[4] for x in bobble][:count_date])

        # финальные данные для графика динамики с конкурентами по СМИ
        data_chart_3 = {"date_bobble": date_bobble,  # дата поста
                        "name_bobble": name_bobble,  # имя источника
                        "index_bobble": index_bobble,  # индекс СМИ
                        "z_index_bobble": z_index_bobble,
                        "tonality_index_bobble": tonality_index_bobble,  # цвет шаров
                        "tonality_url": tonality_url,  # ссылка на пост
                        "filenames": filenames}

        # bubbles4 - график с динамикой отношения в соцмедиа (динамика bubbles)
        date_bobble = []
        name_bobble = []
        index_bobble = []
        z_index_bobble = []
        tonality_index_bobble = []
        tonality_url = []

        for j in range(len(another_graph)):

            df = pd.DataFrame(another_graph[j])

            # метаданные
            columns = ['citeIndex', 'timeCreate', 'toneMark', 'hubtype', 'hub', 'audienceCount', 'url']
            # columns.remove('text')
            df_meta = pd.concat(
                [pd.DataFrame([x['authorObject'] if 'authorObject' in x else '' for x in another_graph[j]]),
                 df[columns]], axis=1)
            # timestamp to date
            df_meta['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                     df_meta['timeCreate'].values]

            df_meta = df_meta.set_index(['timeCreate'])  # индекс - время создания поста

            def date_reverse(date):
                lst = date.split('-')
                temp = lst[1]
                lst[1] = lst[2]
                lst[2] = temp
                return lst

            if 'daterange' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)[
                'daterange'] != datetime.datetime.now().strftime("%m/%d/%Y") + ' - ' + datetime.datetime.now().strftime("%m/%d/%Y"):
                data_start = '-'.join(date_reverse('-'.join(
                    [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][
                    ::-1])))
                data_stop = '-'.join(date_reverse('-'.join(
                    [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][
                    ::-1])))

            df_meta['timeCreate'] = list(df_meta.index)
            df_meta = df_meta[df_meta['hubtype'] != 'Новости']

            # negative Соцмедиа
            df_hub_siteIndex = df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] == -1)][
                ['hub', 'audienceCount']].values

            dict_neg = {}
            for i in range(len(df_hub_siteIndex)):

                if df_hub_siteIndex[i][0] not in dict_neg.keys():

                    dict_neg[df_hub_siteIndex[i][0]] = []
                    dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                else:
                    dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

            list_neg = [list(set(x)) for x in dict_neg.values()]
            list_neg = [[0] if x[0] == 'n/a' else x for x in list_neg if x != 'n/a']
            list_neg = [int(x[0]) for x in list_neg]

            for i in range(len(list_neg)):
                dict_neg[list(dict_neg.keys())[i]] = list_neg[i]

            dict_neg = dict(sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))

            dict_neg_hubs_count = dict(
                Counter(list(df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] == -1)]['hub'])))

            fin_neg_dict = defaultdict(tuple)
            for d in (dict_neg, dict_neg_hubs_count):  # you can list as many input dicts as you want here
                for key, value in d.items():
                    fin_neg_dict[key] += (value,)

            list_neg_smi = list(fin_neg_dict.keys())
            list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
            list_neg_smi_massage_count = [x[1] for x in fin_neg_dict.values()]

            # positive Соцмедиа
            df_hub_siteIndex = df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] == 1)][
                ['hub', 'audienceCount']].values

            dict_pos = {}
            for i in range(len(df_hub_siteIndex)):

                if df_hub_siteIndex[i][0] not in dict_pos.keys():

                    dict_pos[df_hub_siteIndex[i][0]] = []
                    dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

                else:
                    dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

            list_pos = [list(set(x)) for x in dict_pos.values()]
            list_pos = [[0] if x[0] == 'n/a' else x for x in list_pos if x != 'n/a']
            list_pos = [int(x[0]) for x in list_pos]

            for i in range(len(list_pos)):
                dict_pos[list(dict_pos.keys())[i]] = list_pos[i]

            dict_pos = dict(sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))

            dict_pos_hubs_count = dict(
                Counter(list(df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] == 1)]['hub'])))

            fin_pos_dict = defaultdict(tuple)
            for d in (dict_pos, dict_pos_hubs_count):  # you can list as many input dicts as you want here
                for key, value in d.items():
                    fin_pos_dict[key] += (value,)

            list_pos_smi = list(fin_pos_dict.keys())
            list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
            list_pos_smi_massage_count = [x[1] for x in fin_pos_dict.values()]

            # data to bobble graph
            bobble = []
            df_tonality = df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] != 0)][
                ['hub', 'audienceCount', 'toneMark', 'url']].values
            index_ton = df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] != 0)][
                ['timeCreate']].values.tolist()
            date_ton = [x[0] for x in index_ton]
            date_ton = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime.datetime(1970, 1,
                                                                                                    1)).total_seconds() * 1000)
                        for x in date_ton]

            for i in range(len(df_tonality)):
                if df_tonality[i][2] == -1:
                    bobble.append([date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1, df_tonality[i][4]])
                elif df_tonality[i][2] == 1:
                    bobble.append([date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1, df_tonality[i][4]])

            colors_red = ['#8B0000', '#FF4500', '#FFA07A']
            colors_green = ['#006400', '#00FF00', '#8FBC8F']

            for i in range(len(bobble)):
                if bobble[i][3] == 1:
                    bobble[i][3] = colors_green[j]
                else:
                    bobble[i][3] = colors_red[j]

            list_neg_smi = [words_only(x) for x in list_neg_smi]
            list_pos_smi = [words_only(x) for x in list_pos_smi]
            names_bobble = [x[1] for x in bobble]  # названия источников
            names_bobble = [words_only(x) for x in names_bobble]  # названия источников 

            count_date = 50  # сколько данных взять из списков
            date_bobble.append([x[0] for x in bobble][:count_date])
            name_bobble.append(names_bobble[:count_date])
            index_bobble.append([x[2] for x in bobble][:count_date])
            z_index_bobble.append([1] * len(bobble[:count_date]))
            tonality_index_bobble.append([x[3] for x in bobble][:count_date])
            tonality_url.append([x[4] for x in bobble][:count_date])

        # финальные данные для графика динамики с конкурентами по Соцмедиа
        data_chart_4 = {"date_bobble": date_bobble,  # дата поста
                        "name_bobble": name_bobble,  # имя источника
                        "index_bobble": index_bobble,  # аудитория поста Соцмедиа
                        "z_index_bobble": z_index_bobble,
                        "tonality_index_bobble": tonality_index_bobble,  # цвет шаров
                        "tonality_url": tonality_url,  # ссылка на пост
                        "filenames": filenames}

        datas = json.dumps(datas)
        data_chart_3 = json.dumps(data_chart_3)
        data_chart_4 = json.dumps(data_chart_4)

        return render_template('competitors.html', files=json_files, len_files=len_files, dg=dg, data=datas,
                               data_chart_3=data_chart_3, data_chart_4=data_chart_4)

    return render_template('competitors.html', files=json_files, len_files=len_files)



@app.route('/login', methods=["GET", "POST"])
def login():

    if request.method == "POST":
        session.permanent = True
        username = request.form["email"]
        password = request.form["psw"]
        session['user'] = username


        enter_passw = db.session.query(Users.password).filter_by(email=username).order_by(Users.id.asc()).first()[0]

        error = None

        if username is None:
            error = "Введен некорректный e-mail!"

        elif db.session.query(Users.email).filter_by(email=username).order_by(Users.id.asc()).first() is None:
            error = "Пользователь не найден! Войдите под своим пользователем или зарегистрируйте нового."

        elif not check_password_hash(enter_passw, password):
            error = "Введен неверный пароль!"

        elif error is None:
            return redirect(url_for("index"))

        flash(error)

    return render_template("login.html")


@app.route('/register', methods=("GET", "POST"))
def register():

    if request.method == "POST":
        username = request.form["name"]
        email = request.form["email"]
        psw = request.form["psw"]
        psw2 = request.form["psw2"]

        if request.form["psw"] == request.form["psw2"]:
            hash_pass = generate_password_hash(request.form["psw"])

        error = None

        if not username:
            error = "Введите Имя"
        elif not email:
            error = "Укажите ваш E-mail"
        elif not psw or not psw2:
            error = "Введите пароль!"

        if error is None:

            if db.session.query(Users.email).filter_by(email=email).order_by(Users.id.asc()).first():

                return f"Пользователь {email} уже зарегистрирован, пожалуйста, смените {email} или зарегистрируйте новый E-mail"

            else:

                date = datetime.datetime.now().strftime("%m.%d.%Y %H:%M:%S")

                session_scoped.add(Users(user_name=username,
                                  email=email,
                                  password=hash_pass,
                                  time_created=str(date)))
                session_scoped.commit()

                # Success, go to the login page.
                return redirect(url_for("login"))

        flash(error)

    return render_template("register.html")


@app.route('/external_factors', endpoint='external_factors', methods=['GET', 'POST'])
def external_factors():

    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files)

    # подготовка данных с биржи
    # https://www.moex.com/s205
    df = pd.read_excel('MOEX_codes.xlsx')

    # словарь для отображения названия фондового актива во фронт
    df_name_active = df[['Код базисного актива на срочном рынке', 'Название базисного актива']]
    df_name_active = df_name_active.set_index('Код базисного актива на срочном рынке').to_dict()

    if request.method == "POST":

        # https://www.moex.com/s205
        # https://www.moex.com/ru/marketdata/?g=4#/mode=groups&group=4&collection=3&boardgroup=57&data_type=current&category=main

        if 'send' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)[
            'file_choose'] != 'Выбрать файл':

            # parsing json
            try: 
                with io.open(request.values.to_dict(flat=True)['file_choose'], encoding='utf-8', mode='r') as train_file:
                    dict_train = json.load(train_file, strict=False)
        
            except:
                a = []
                with open(request.values.to_dict(flat=True)['file_choose'], encoding='utf-8', mode='r') as file:
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
            df = df[['text', 'url', 'hub', 'timeCreate']]
            # df_meta = pd.concat([pd.DataFrame.from_records(df['authorObject'].values), df[columns]], axis=1)
            # timestamp to date
            df['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                df['timeCreate'].values]

            df = df.set_index(['timeCreate'])  # индекс - время создания поста

            def date_reverse(date):  # фильтрация по дате/календарик
                lst = date.split('-')
                temp = lst[1]
                lst[1] = lst[2]
                lst[2] = temp
                return lst

            if 'daterange' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)[
                'daterange'] != '01/01/2022 - 01/12/2022':
                data_start = '-'.join(date_reverse('-'.join(
                    [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][
                    ::-1])))
                data_stop = '-'.join(date_reverse('-'.join(
                    [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][
                    ::-1])))

                df = df.loc[data_stop:data_start]  # фильтрация данных по дате в календарике

            df = df.reset_index()  # возвращаем индексы к 0, 1, 2, 3 для дальнейшей итерации по ним

            if df.shape[0] == 0:  # если по запросу найдено 0 сообщений - вывести flash
                flash('По запросу найдено 0 сообщений')
                return redirect(url_for('Gauge'))

            df = df[['timeCreate']]
            df.columns = ['Дата']

            def date_to_day(date_str):
                return datetime.datetime.strptime(date_str[:10], "%Y-%m-%d")

            df['Дата'] = df['Дата'].apply(date_to_day)

            # Counter with initial values
            counter = Counter(df['Дата'].values)
            # Counter({'a': 2, 'b': 1})

            date = [pd.Timestamp(x) for x in list(counter.keys())]
            values = [list(counter.values())]

            df = pd.DataFrame(zip(date, values[0]))
            df.index = df[0]
            df.drop(0, axis=1, inplace=True)
            df.columns = ['Count_messages']

            # prepare moex data
            external_factor = request.values.to_dict(flat=True)['external_factor']
            f = web.DataReader(external_factor, 'moex', start=data_start, end=data_stop)
            f = f[['CLOSE']]

            # correlation
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
            df_fin = df.join(f, lsuffix='_caller', rsuffix='_other')
            df_fin.dropna(inplace=True)
            df_fin = df_fin.iloc[::-1]
            corr = stats.pearsonr(df_fin['CLOSE'].values, df_fin['Count_messages'].values)

            # данные для графика сравнительной динамики сообщений и индекса на бирже
            unixtime = [int(time.mktime(x.timetuple())) for x in list(df_fin.index)]
            unixtime_ms = [x * 1000 for x in unixtime]
            unixtime_ms = [(x + 86400000) for x in
                           unixtime_ms]  # чтобы корректно отображалось на фронт (нужно добавить 1 день)

            # имена данных -какой файл и какие биржевые значения
            names = [request.values.to_dict(flat=True)['file_choose'].split('_')[0],
                     df_name_active['Название базисного актива'][external_factor]]
            names = [x.replace('"', '') for x in names]  # удаление "" для корректного парсинга js

            values_dinamic = []
            for i in range(len(names)):  # собираем итоговый list с данными для графика динамики

                if i == 0:  # перевести из float в int значения кол-ва сообщений
                    values_dinamic.append([names[i], [int(x) for x in df_fin[df_fin.columns[i]].values.tolist()]])
                else:
                    values_dinamic.append([names[i], df_fin[df_fin.columns[i]].values.tolist()])

            # подготовка данных от датах для оси x
            unixtime = [int(time.mktime(x.timetuple())) for x in list(df_fin.index)]
            unixtime_ms = [x * 1000 for x in unixtime]  # для unix-js умножаем на 1000
            unixtime_ms = [(x + 86400000) for x in
                           unixtime_ms]  # чтобы корректно отображалось на фронт (нужно добавить 1 день)

            # приведеение к 2м знакам и взятие первого значения из corr (-0.05892622513253115, 0.9410737748674689)
            corr = np.round(corr[0], 2)

            data = {
                "corr": str(corr),
                "data": values_dinamic,
                "unixtime_ms": unixtime_ms,
            }

            return render_template('external_factors.html', files=json_files, len_files=len_files, data=data,
                                   df_name_active=df_name_active['Название базисного актива'])

    return render_template('external_factors.html', files=json_files, len_files=len_files,
                           df_name_active=df_name_active['Название базисного актива'])


@app.route('/BertTopic', methods=['GET', 'POST'], endpoint='BertTopic')
def BertTopic():

    os.chdir(path_to_files)
    json_files = [x + '.json' for x in os.listdir(path_bert_topic_data)]
    len_files = len(json_files)

    if request.method == 'POST':
        
        filename = request.values.to_dict(flat=False)['file_choose'][0]
        os.chdir(path_bert_topic_data + '/' + filename.replace('.json', ''))
        obj_Bert = codecs.open(filename.replace('.json', '') + '.txt', 'r', encoding='utf-8').read()
        data_BertTopic = json.loads(obj_Bert)

        topics = data_BertTopic['topics']
        probs = data_BertTopic['probs']
        probs = np.array(probs)

        os.chdir(path_bert_topic_data + '/' + filename.replace('.json', ''))
        topic_model = BERTopic.load(filename.replace('.json', '.pt'))

        print('start modelling')

        # topic_model.visualize_topics()
        topic_model_visual = topic_model.visualize_topics()

        # topic_model.visualize_distribution(probs[500], min_probability=0.015)
        # topic_model_propab = topic_model.visualize_distribution(probs[500], min_probability=0.015)

        # topic_model.visualize_hierarchy(top_n_topics=50)
        topic_model_hierarchy = topic_model.visualize_hierarchy(top_n_topics=70)

        # topic_model.visualize_barchart(top_n_topics=5)
        topic_model_barchart = topic_model.visualize_barchart(top_n_topics=8)

        # topic_model.visualize_heatmap(n_clusters=20, width=1000, height=1000)
        topic_model_heatmap = topic_model.visualize_heatmap(n_clusters=20, width=1000, height=1000)

        # obj = plotly.offline.plot(topic_model_visual, include_plotlyjs=True, output_type='div')
        print('stop modelling')
        
        # https://stackoverflow.com/questions/61682730/plotly-offline-plot-output-type-div-not-working-inside-html-how-to-embed-plo
        def my_bar_chart(fig):
            my_bar_chart = plotly.offline.plot(fig, output_type='div', include_plotlyjs=False).replace('"align":"left"', '"align":"center"').replace('plotly-graph-div"', 'center"').replace('"align":"left"', '"align":"center"').replace('Intertopic Distance Map', 'Распределение тем').replace('style="height:650px; width:650px;"', 'style="height:860px; width:1000px;"').replace('"width":650', '"width":850').replace('"height":860', '"height":1000').replace('height":650', 'height":800').replace('width":650', 'width":850')
            return Markup(my_bar_chart)

        chart_topic_model_vis = my_bar_chart(topic_model_visual)
        # chart_topic_model_propab = my_bar_chart(topic_model_propab)
        chart_topic_model_hierarchy = my_bar_chart(topic_model_hierarchy).replace('Hierarchical Clustering', 'Иерархия тем')
        chart_topic_model_barchart = my_bar_chart(topic_model_barchart).replace('Topic Word Scores', 'ТОП-слова в темах')
        chart_topic_model_heatmap = my_bar_chart(topic_model_heatmap).replace('Similarity Matrix', 'Матрица схожести тем').replace('Similarity Score', 'Степень')

        print('done')

        return render_template('BertTopic.html', chart_topic_model_vis=chart_topic_model_vis, chart_topic_model_hierarchy=chart_topic_model_hierarchy, chart_topic_model_barchart=chart_topic_model_barchart, chart_topic_model_heatmap=chart_topic_model_heatmap, len_files=len_files, files=json_files)

    return render_template('BertTopic.html', len_files=len_files, files=json_files)


@app.route('/classification', methods=['GET', 'POST'], endpoint='classification')
def clusterisation():
    return render_template('classification.html')


@app.route('/LdaTopic', methods=['GET', 'POST'], endpoint='ldatopic')
def ldatopic():

    os.chdir(path_to_files)
    json_files = [x + '.json' for x in os.listdir(path_Lda_topic_data)]
    len_files = len(json_files)

    if request.method == 'POST':
        filename = request.values.to_dict(flat=False)['file_choose'][0]
        LdaTopic = filename.replace('.json', '') + '_LdaTemplate.html'
        session['LDA_file'] = LdaTopic # сохраняем в сессии имя файла для отображения
        return render_template('LdaTopic.html', files=json_files, len_files=len_files, LdaTopic=LdaTopic)

    return render_template('LdaTopic.html', files=json_files, len_files=len_files)


# https://stackoverflow.com/questions/60874060/flask-set-local-html-as-the-src-of-the-iframe
@app.route('/show_frame')
def show():

    HTMLfile = session['LDA_file']
    del session['LDA_file']
    return render_template(HTMLfile)


@app.route('/tonality_fit', methods=['GET', 'POST'], endpoint='tonality_fit')
def tonality_fit():

    import os
    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files) 

    if request.method == 'POST':

        if 'filename' in request.files and 'file_choose' in request.values.to_dict(flat=True):
            uploaded_file = request.files['filename']
            if uploaded_file.filename != '' and request.values.to_dict(flat=True)['file_choose'] != '':
                
                os.chdir(path_themes_tonality_models)
                if not os.path.exists(request.values.to_dict(flat=True)['file_choose'].replace('.json', '')):
                    os.mkdir(request.values.to_dict(flat=True)['file_choose'].replace('.json', ''))
                os.chdir(request.values.to_dict(flat=True)['file_choose'].replace('.json', ''))
                uploaded_file.save(uploaded_file.filename)
                # session['train_tonality_file'] = uploaded_file.filename
    
            import datasets
            import evaluate
            import pandas as pd
            import numpy as np
            from datasets import Dataset
            from sklearn.model_selection import train_test_split
            from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                                    TrainingArguments, Trainer)

            import nltk
            nltk.download('stopwords')
            from nltk.corpus import stopwords
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


            # Загружаем данные
            df = pd.read_csv(uploaded_file.filename)
            df.dropna(inplace=True)
            df.columns = ['text','label']
            df = df.drop_duplicates(subset=['text'])
            df['text'] = df['text'].apply(preprocess_text)
            df['text'] = df['text'].apply(remove_stopwords)
            df['text'] = df['text'].apply(words_only)
            df['label'] = df['label'].astype(int)

            # Конвертируем датафрейм в Dataset
            train, test = train_test_split(df, test_size=0.3)
            train = Dataset.from_pandas(train)
            test = Dataset.from_pandas(test)

            # Выполняем предобработку текста
            tokenizer = AutoTokenizer.from_pretrained(
                'cointegrated/rubert-tiny')

            def tokenize_function(examples):
                return tokenizer(examples['text'], padding='max_length', truncation=True)

            tokenized_train = train.map(tokenize_function)
            tokenized_test = test.map(tokenize_function)

            # Загружаем предобученную модель
            model = AutoModelForSequenceClassification.from_pretrained(
                'cointegrated/rubert-tiny',
                num_labels=3)

            # Задаем параметры обучения
            training_args = TrainingArguments(
                output_dir = 'test_trainer_log',
                evaluation_strategy = 'epoch',
                per_device_train_batch_size = 6,
                per_device_eval_batch_size = 6,
                num_train_epochs = 3,
                report_to='none')

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                return metric.compute(predictions=predictions, references=labels)

            # Определяем как считать метрику
            from datasets import load_metric
            metric = load_metric("recall")
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                return metric.compute(predictions=predictions, references=labels, average="weighted")


            # Выполняем обучение
            trainer = Trainer(
                model = model,
                args = training_args,
                train_dataset = tokenized_train,
                eval_dataset = tokenized_test,
                compute_metrics = compute_metrics)

            print('Start train')
            trainer.train()
            print('Stop train')

            # Сохраняем модель
            save_directory = path_themes_tonality_models + '/' + request.values.to_dict(flat=True)['file_choose'].replace('.json', '')
            #tokenizer.save_pretrained(save_directory)
            model.config.id2label = {0: 'Neutral', 1: 'Negative', 2: 'Positive'}
            model.config.label2id = {'Neutral': 0, 'Negative': 1, 'Positive': 2}
            model.save_pretrained(save_directory)

            return render_template('tonality_fit.html', files=json_files, len_files=len_files)


    return render_template('tonality_fit.html', files=json_files, len_files=len_files)



@app.route('/tonality_check', methods=['GET', 'POST'], endpoint='tonality_check')
def tonality_check():

    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files) 


    if request.method == 'POST':

        print(request.values.to_dict(flat=True))

        ## код отправки текста на проверку
        from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification

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

        MODEL_NAME = path_themes_tonality_models + '/' + request.values.to_dict(flat=True)['thing'].replace('.json', '')
        tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny')
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        text = request.values.to_dict(flat=True)['text_search']
        text_to_show = text
        text = words_only(remove_stopwords(preprocess_text(text)))
        prediction = pipe(text, return_all_scores=True)
        print('=====')
        print(type(prediction))
        prediction = [(x['label'], str(np.round(x['score'], 3))) for x in prediction[0]]
        prediction = '; '.join([': '.join(x) for x in prediction])
        print(prediction)


        print(request.values.to_dict(flat=True))
        thing = request.values.to_dict(flat=True)['thing']
        json_files = [x for x in json_files if x != thing]
        return render_template('check_tonality.html', files=json_files, len_files=len_files, thing=thing, text=text_to_show, prediction=prediction)
    
    if 'text_search' in request.values.to_dict(flat=True):
        if request.method == 'GET':
            text = request.values.to_dict(flat=True)['text_search']
            thing = request.form.get('thing')
            print(request.values.to_dict(flat=True))
            json_files = [x for x in json_files if x != thing]
            return render_template('check_tonality.html', files=json_files, len_files=len_files, thing=thing, text=text)

    if 'text_search' not in request.values.to_dict(flat=True):
        if request.method == 'GET':
            thing = request.form.get('thing')
            print(request.values.to_dict(flat=True))
            json_files = [x for x in json_files if x != thing]
            return render_template('check_tonality.html', files=json_files, len_files=len_files, thing=thing)



@app.route('/data', methods=['GET', 'POST'], endpoint='data')
def data():
    return render_template('data.html')


@app.route('/author_cluster_files', methods=['GET', 'POST'], endpoint='author_cluster_files')
def data():

    data_folders = [x for x in next(os.walk(path_projector_files))[1]]
    len_folders = len(data_folders)
    print(data_folders)

    return render_template('author_cluster_files.html', data_folders=data_folders, len_folders=len_folders)


@app.route('/open-folder-projector/<foldername>')
def open_folder_projector(foldername):

    file_path = path_projector_files + '/' + foldername
    files = [x for x in os.listdir(file_path)]
    print(files)
    return render_template('projector_files.html', files=files)


@app.route('/projector-file-load/<filename>')
def projector_file_load(filename):

    file_path = file_path = path_projector_files + '/' + filename
    return send_file(file_path, as_attachment=True)


@app.route('/projector', methods=['GET', 'POST'], endpoint='projector')
def projector():
    return render_template('projector.html')

@app.route('/clusterisation', methods=['GET', 'POST'], endpoint='clusterisation')
def author_clusterisation():
    return render_template('clusterisation.html')


if __name__ == "__main__":
    app.run(host='146.185.208.165', port=8000, debug=True)