import datetime
import glob
import html as htl
import json
import os
import io
import re
from collections import Counter, OrderedDict, defaultdict
from operator import itemgetter
import time
import flask
import numpy 
import pandas as pd
import jaal
from jaal import Jaal
from json_ba import json_ba
from io import BytesIO
from dash import Dash, dcc, html

from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory, flash
from jaal import Jaal

app = Flask(__name__)
app.debug = True
app.secret_key = 'a(d)fs#$T12eF#4-key'

path_to_files = "C:\\Users\\NUC\\PycharmProjects\\clusters\\data"
session = flask.session
app.config['UPLOAD_FOLDER'] = path_to_files

regex = re.compile("[А-Яа-яЁё:=!\).\()A-z\_\%/|0-9]+")
def words_only(text, regex=regex):
    try:
        return " ".join(regex.findall(text))
    except:
        return ""


@app.route("/")
def hello():
    return redirect(url_for('index'))


@app.route("/index", endpoint='index')
def menu():
    return render_template('index.html')


@app.route('/authors', methods=['GET', 'POST'], endpoint='authors')
def graph():
    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files)

    if 'go' in request.values.to_dict(flat=True):
        session['filename'] = request.values.to_dict(flat=True)['file_choose']

        # parsing json & create embedding
        X = json_ba()
        X.open_file(session['filename'])
        X.preprocess_text()
        X.create_embed()
        X.tsne_create()

        name_str = ','.join(X.names_list)
        # b = [', '.join([str(x[0]), str(x[1])]) for x in X.coord_list]
        coord_list_str = '\n'.join(X.coord_list)

        # return str(coord_list_str)
        return render_template('visualizer.html', names=name_str, coord=coord_list_str)

    return render_template('authors_cluster.html', files=json_files, len_files=len_files)


@app.route('/tsne', methods=('GET', 'POST'))
def tsne():
    return render_template('visualizer.html')


@app.route('/graph', methods=("GET", "POST"))
def graph():
    # columns = ['from', 'to', 'weight', 'strenght']
    # edge_df = pd.DataFrame([['Alex', 'Matt', 3, 12], ['Dot', 'Carl', 18, 4]])
    # edge_df.columns = columns
    #
    # columns = ['id', 'gender']
    # node_df = pd.DataFrame([['Alex', 'male'], ['Matt', 'female'],
    #                         ['Dot', 'male'], ['Carl', 'male']])
    # node_df.columns = columns

    X = Jaal(edge_df, node_df)
    X.create(app)

    return flask.redirect('http://127.0.0.1:5000/graph/')
    # render_template()


@app.route('/tonality_landscape', methods=['GET', 'POST'], endpoint='tonality_landscape')
def tonality():
    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files)

    if 'send' in request.values.to_dict(flat=True):

        session['filename'] = request.values.to_dict(flat=True)['file_choose']

        # parsing json
        with io.open(session['filename'], encoding='utf-8', mode='r') as train_file:
            dict_train = json.load(train_file)

        df = pd.DataFrame(dict_train)

        # метаданные
        columns = list(df.columns)
        # columns.remove('text')
        df_meta = pd.concat([pd.DataFrame.from_records(df['authorObject'].values), df[columns]], axis=1)
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

        data_start = '-'.join(date_reverse('-'.join(
            [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][::-1])))
        data_stop = '-'.join(date_reverse('-'.join(
            [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][::-1])))
        df_meta = df_meta.loc[data_stop: data_start]

        # негатив и позитив по площадкам (соцмедиа)
        hub_neg = Counter(df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] == 1)]['hub'].values)
        hub_pos = Counter(df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] == -1)]['hub'].values)

        df_meta['date'] = [x[:10] for x in df_meta.index]  # столюец с датами без часов/минут/сек
        neg_tabl = df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] == 1)]
        pos_table = df_meta[(df_meta['hubtype'] != 'Новости') & (df_meta['toneMark'] == -1)]

        neg_list_data = list(OrderedDict(Counter(neg_tabl['date'].values)).values())
        pos_list_data = list(OrderedDict(Counter(pos_table['date'].values)).values())

        neg_list_name = list(Counter(neg_tabl['date'].values[::-1]).keys())
        pos_list_name = list(Counter(pos_table['date'].values[::-1]).keys())

        data_tonality_hub_neg_data = [x[1] for x in sorted((hub_neg).items(), key=itemgetter(1), reverse=True)]
        data_tonality_hub_pos_data = [x[1] for x in sorted((hub_pos).items(), key=itemgetter(1), reverse=True)]
        data_tonality_hub_neg_name = [x[0] for x in sorted((hub_neg).items(), key=itemgetter(1), reverse=True)]
        data_tonality_hub_pos_name = [x[0] for x in sorted((hub_pos).items(), key=itemgetter(1), reverse=True)]

        superDonatName = request.values.to_dict(flat=True)['file_choose'].split('_')[0]

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

        neg2 = [[x.replace('"', '') for x in group] for group in neg2]  # для корректной передачи в html java

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

        pos2 = [[x.replace('"', '') for x in group] for group in pos2]  # для корректной передачи в html java

        # neg_authors = {}
        # for i in range(len(list(hub_neg.keys()))):
        #     neg_authors[list(hub_neg.keys())[i]] = dict(
        #         Counter(neg_tabl[neg_tabl['hub'] == list(hub_neg.keys())[i]]['fullname'].values))

        regex = re.compile("[А-Яа-я:=!\).\()A-z\_\%/|ёЁ]+")
        def words_only(text, regex=regex):
            try:
                return " ".join(regex.findall(text))
            except:
                return ""

        neg_list_name = [words_only(x) for x in neg_list_name]
        pos_list_name = [words_only(x) for x in pos_list_name]
        data_tonality_hub_neg_name = [words_only(x) for x in data_tonality_hub_neg_name]
        data_tonality_hub_pos_name = [words_only(x) for x in data_tonality_hub_pos_name]

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
            "pos3": pos3
        }

        date = data_start + ' : ' + data_stop

        # print(neg_authors['apple.com'])
        return render_template('tonality_landscape.html', files=json_files, len_files=len_files, datagraph=data,
                               object_name=superDonatName, date=date)

    data = {
        'neg_list_data': [11, 12, 16, 9],
        'neg_list_name': ['S', 'M', 'F', 'K'],
        'pos_list_data': [8, 6, 7, 9],
        'pos_list_name': ['H', 'G', 'T', 'P'],
        "superDonatName": 'test'
    }

    return render_template('tonality_landscape.html', files=json_files, len_files=len_files, datagraph=data)


@app.route('/datalake', methods=['GET', 'POST'], endpoint='datalake')
def show_data():
    if request.method == 'POST' and 'filename' in request.files:
        uploaded_file = request.files['filename']
        if uploaded_file.filename != '':
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))
        return redirect(url_for('datalake'))

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


@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = path_to_files + filename
    return send_file(file_path, as_attachment=True, attachment_filename='')


@app.route('/information_graph', methods=['GET', 'POST'], endpoint='information_graph')
def information_graph():
    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files)

    if 'send' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)['text_search'] == '':

        session['filename'] = request.values.to_dict(flat=True)['file_choose']

        # parsing json
        with io.open(session['filename'], encoding='utf-8', mode='r') as train_file:
            dict_train = json.load(train_file)

        df = pd.DataFrame(dict_train)

        # метаданные
        columns = ['text', 'er', 'timeCreate', 'type', 'hubtype', 'hub', 'audienceCount']
        # columns.remove('text')
        df_meta = pd.concat([pd.DataFrame.from_records(df['authorObject'].values), df[columns]], axis=1)
        # timestamp to date
        df_meta['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                 df_meta['timeCreate'].values]
        df_meta = df_meta.set_index(['timeCreate'])  # индекс - время создания поста

        def date_reverse(date): # фильтрация по дате/календарик
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

            df_meta = df_meta.loc[data_stop: data_start] # фильтрация данных по дате в календарике
        
        df_meta['timeCreate'] = list(df_meta.index)

        if df_meta.shape[0] == 0: # если по запросу найдено 0 сообщений - вывести flash 
            flash('По запросу найдено 0 сообщений')
            return redirect(url_for('information_graph'))

        if 'hub_select' in request.values.to_dict(flat=True):
            hub_select = request.form.getlist('hub_select')
            df_meta = df_meta.loc[df_meta['hub'].isin(hub_select)]

        if request.values.to_dict(flat=True)['text_min'] != '':
            df_meta = df_meta.iloc[int(request.values.to_dict(flat=True)['text_min']):]

        if request.values.to_dict(flat=True)['text_max'] != '':
            df_meta = df_meta.iloc[:int(request.values.to_dict(flat=True)['text_max'])]

        if request.values.to_dict(flat=True)['text_min'] != '' and request.values.to_dict(flat=True)['text_max'] != '':
            df_meta = df_meta.iloc[int(request.values.to_dict(flat=True)['text_min']):int(
                request.values.to_dict(flat=True)['text_max'])]
        
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

        if df_meta.shape[0] == 0: # если по запросу найдено 0 сообщений - вывести flash 
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
            "df_rep_auth": df_rep_auth_inverse,
            "data_rep_er": er,
            "data_rep_audience": data_audience,
            "data_authors": df_rep_auth,
            "authors_count": len(set(df_rep_auth)),
            "len_messages": df_meta.shape[0],
            "data_hub": hubs,
            "all_hubs": all_hubs
        }

        return render_template('information_graph.html', theme=theme, len_files=len_files, files=json_files,
                               data=data)


    if 'send' in request.values.to_dict(flat=True) and request.values.to_dict(flat=True)['text_search'] != '':

        session['filename'] = request.values.to_dict(flat=True)['file_choose']

        # parsing json
        with io.open(session['filename'], encoding='utf-8', mode='r') as train_file:
            dict_train = json.load(train_file)

        df = pd.DataFrame(dict_train)

        # метаданные
        columns = ['text', 'er', 'timeCreate', 'hub', 'audienceCount', 'type']
        # columns.remove('text')
        df_meta = pd.concat([pd.DataFrame.from_records(df['authorObject'].values), df[columns]], axis=1)
        # timestamp to date
        df_meta['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                 df_meta['timeCreate'].values]
        df_meta = df_meta.set_index(['timeCreate'])  # индекс - время создания поста

        def date_reverse(date): # фильтрация по дате/календарик
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

            df_meta = df_meta.loc[data_stop: data_start] # фильтрация данынх по дате в календарике
        
        df_meta['timeCreate'] = list(df_meta.index)


        if df_meta.shape[0] == 0: # если по запросу найдено 0 сообщений - вывести flash 
            flash('По запросу найдено 0 сообщений')
            return redirect(url_for('information_graph'))


        if set(df_meta['hub'].values) == {"telegram.org"}: # если все сообщения только ТГ

            if request.values.to_dict(flat=True)['text_min'] != '':
                df_meta = df_meta.iloc[int(request.values.to_dict(flat=True)['text_min']):]

            if request.values.to_dict(flat=True)['text_max'] != '':
                df_meta = df_meta.iloc[:int(request.values.to_dict(flat=True)['text_max'])]

            if request.values.to_dict(flat=True)['text_min'] != '' and request.values.to_dict(flat=True)[
                'text_max'] != '':
                df_meta = df_meta.iloc[int(request.values.to_dict(flat=True)['text_min']):int(
                    request.values.to_dict(flat=True)['text_max'])]

            search_lst = request.values.to_dict(flat=True)['text_search'].split(',')
            search_lst = [x.split('или') for x in search_lst]
            search_lst = [[x.strip() for x in group] for group in search_lst]

            index_table = []
            text_val = df_meta['text'].values

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

            df_data_rep = df_meta[['fullname', 'url', 'author_type', 'text', 'audienceCount', 'hub', 'er', 'type']]
            df_rep_auth = list(df_data_rep['fullname'].values)
            data_rep_er = list(df_data_rep['er'].values)
            data_audience = list(df_data_rep['audienceCount'].values)
            all_hubs = list(df_data_rep['hub'].values)
            print('!!!')


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
                "df_rep_auth": df_rep_auth_inverse,
                "data_rep_er": data_rep_er,
                "data_rep_audience": data_audience,
                "authors_count": len(set(df_rep_auth)),
                "len_messages": df_meta.shape[0],
                "data_authors": df_rep_auth,
                "data_hub": hubs,
                "all_hubs": all_hubs
            }

            return render_template('information_graph.html', theme=theme, len_files=len_files, files=json_files,
                                   data=data)


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

        if df_meta.shape[0] == 0: # если по запросу найдено 0 сообщений - вывести flash 
            flash('По запросу найдено 0 сообщений')
            return redirect(url_for('information_graph'))

        if request.values.to_dict(flat=True)['text_min'] != '':
            df_meta = df_meta.iloc[int(request.values.to_dict(flat=True)['text_min']):]

        if request.values.to_dict(flat=True)['text_max'] != '':
            df_meta = df_meta.iloc[:int(request.values.to_dict(flat=True)['text_max'])]

        if request.values.to_dict(flat=True)['text_min'] != '' and request.values.to_dict(flat=True)['text_max'] != '':
            df_meta = df_meta.iloc[int(request.values.to_dict(flat=True)['text_min']):int(
                request.values.to_dict(flat=True)['text_max'])]

        search_lst = request.values.to_dict(flat=True)['text_search'].split(',')
        search_lst = [x.split('или') for x in search_lst]
        search_lst = [[x.strip() for x in group] for group in search_lst]

        index_table = []
        text_val = df_meta['text'].values

        for j in range(len(text_val)):
            a = []
            for i in range(len(search_lst)):
                if [item for item in search_lst[i] if item in text_val[j]] != []:
                    a.append([item for item in search_lst[i] if item in text_val[j]])
            if len(a) == len(search_lst):
                index_table.append(df_meta.index[j])

        df_meta = df_meta.loc[index_table]
        if df_meta.shape[0] == 0: # если по запросу найдено 0 сообщений - вывести flash 
            flash('По запросу найдено 0 сообщений')
            return redirect(url_for('information_graph'))

        df_data_rep = df_meta[['fullname', 'url', 'author_type', 'text', 'er', 'hub', 'audienceCount']]
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
            "df_rep_auth": df_rep_auth_inverse,
            "data_rep_er": er,
            "data_rep_audience": data_audience,
            "data_authors": df_rep_auth,
            "authors_count": len(set(df_rep_auth)),
            "len_messages": df_meta.shape[0],
            "data_hub": hubs,
            "all_hubs": all_hubs
        }


        return render_template('information_graph.html', theme=theme, len_files=len_files, files=json_files,
                               data=data)

    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files)

    data = {
        "df_rep_auth": ['A', 'G', 'K', 'M'],
        "data_rep_audience": ['11', '12', '15', '8']

    }

    return render_template('information_graph.html', len_files=len_files, files=json_files, data=data)


@app.route('/media_rating', methods=['GET', 'POST'], endpoint='media_rating')
def media_rating():
    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files)

    if 'send' in request.values.to_dict(flat=True):

        session['filename'] = request.values.to_dict(flat=True)['file_choose']

        # parsing json
        with io.open(session['filename'], encoding='utf-8', mode='r') as train_file:
            dict_train = json.load(train_file)

        df = pd.DataFrame(dict_train)

        # метаданные
        columns = ['citeIndex', 'timeCreate', 'toneMark', 'hubtype', 'hub', 'audienceCount']
        # columns.remove('text')
        df_meta = pd.concat([pd.DataFrame.from_records(df['authorObject'].values), df[columns]], axis=1)
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


        data_start = '-'.join(date_reverse('-'.join(
            [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[0].split('/')][::-1])))
        data_stop = '-'.join(date_reverse('-'.join(
            [x.strip() for x in request.values.to_dict(flat=True)['daterange'].split('-')[1].split('/')][::-1])))
        df_meta = df_meta.loc[data_stop: data_start]
        df_meta['timeCreate'] = list(df_meta.index)

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
            list_neg = [int(x[0]) for x in list_neg]

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
            list_pos = [int(x[0]) for x in list_pos]

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
                ['fullname', 'audienceCount', 'toneMark']].values
            index_ton = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] != 0)][
                ['timeCreate']].values.tolist()
            date_ton = [x[0] for x in index_ton]
            date_ton = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime.datetime(1970, 1,
                                                                                                    1)).total_seconds() * 1000)
                        for x in date_ton]

            for i in range(len(df_tonality)):
                if df_tonality[i][2] == -1:
                    bobble.append([date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1])
                elif df_tonality[i][2] == 1:
                    bobble.append([date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1])

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
                "tonality_index_bobble": [x[3] for x in bobble]
            }

            theme = request.values.to_dict(flat=True)['file_choose'].split('_')[0]

            print("=====!!!!!+++++")
            # print(data)

            return render_template('media_rating.html', len_files=len_files, files=json_files, data=data, theme=theme)

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
        list_neg = [int(x[0]) for x in list_neg]

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
        list_pos = [int(x[0]) for x in list_pos]

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
            ['hub', 'citeIndex', 'toneMark']].values
        index_ton = df_meta[(df_meta['hubtype'] == 'Новости') & (df_meta['toneMark'] != 0)][
            ['timeCreate']].values.tolist()
        date_ton = [x[0] for x in index_ton]
        date_ton = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime.datetime(1970, 1,
                                                                                                1)).total_seconds() * 1000)
                    for x in date_ton]

        for i in range(len(df_tonality)):
            if df_tonality[i][2] == -1:
                bobble.append([date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1])
            elif df_tonality[i][2] == 1:
                bobble.append([date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1])

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
            "tonality_index_bobble": [x[3] for x in bobble]
        }

        print("=====!!!!!&&&&&")
        print(data)
        theme = request.values.to_dict(flat=True)['file_choose'].split('_')[0]

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
        with io.open(session['filename'], encoding='utf-8', mode='r') as train_file:
            dict_train = json.load(train_file)

        df = pd.DataFrame(dict_train)

        # метаданные
        columns = ['text', 'er', 'timeCreate', 'type', 'hubtype', 'hub', 'toneMark']
        # columns.remove('text')
        df_meta = pd.concat([pd.DataFrame.from_records(df['authorObject'].values), df[columns]], axis=1)
        # timestamp to date
        df_meta['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                 df_meta['timeCreate'].values]

        search_lst = request.values.to_dict(flat=True)['text_search'].split(',')
        search_lst = [x.split('или') for x in search_lst]
        search_lst = [[x.strip() for x in group] for group in search_lst]

        text_val = df_meta['text'].values

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

        a = []  # список с кол-ом поз, нег и нейтр по продуктам (поиску) пример списка - [[0, 0, 153], [0, 0, 18]]
        for key, val in dict_count.items():
            a.append([dict_count[key]["Negative"], dict_count[key]["Positive"], dict_count[key]["Neutral"]])

        theme = request.values.to_dict(flat=True)['file_choose'].split('_')[0]

        dict_names = {key: key[0] for key in [x for x in list(dict_count.keys())]}

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

        print("=====!!!!!!&&&&&")
        print(data["type_message"])
        return render_template('voice.html', files=json_files, len_files=len_files, data=data, theme=theme,
                               dict_names=dict_names)

    return render_template('voice.html', files=json_files, len_files=len_files)


@app.route('/test', methods=["GET", "POST"])
def test():

    os.chdir(path_to_files)
    json_files = [pos_json for pos_json in os.listdir(os.getcwd()) if pos_json.endswith('.json')]
    len_files = len(json_files)


    filename = 'Rosbank_15.08.2022-21.08.2022_telegram.json'
    # 'Rosbank_08.08.2022-14.08.2022_telegram.json'

    # parsing json
    with io.open(filename, encoding='utf-8', mode='r') as train_file:
        dict_train = json.load(train_file)

    df = pd.DataFrame(dict_train)

    # метаданные
    columns = ['citeIndex', 'timeCreate', 'toneMark', 'hubtype', 'hub', 'audienceCount']
    # columns.remove('text')
    df_meta = pd.concat([pd.DataFrame.from_records(df['authorObject'].values), df[columns]], axis=1)
    # timestamp to date
    df_meta['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                             df_meta['timeCreate'].values]


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
    list_neg = [int(x[0]) for x in list_neg]

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
    list_pos = [int(x[0]) for x in list_pos]

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
        ['fullname', 'audienceCount', 'toneMark']].values
    index_ton = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] != 0)][
        ['timeCreate']].values.tolist()
    date_ton = [x[0] for x in index_ton]
    date_ton = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime.datetime(1970, 1,
                                                                                            1)).total_seconds() * 1000)
                for x in date_ton]

    for i in range(len(df_tonality)):
        if df_tonality[i][2] == -1:
            bobble.append([date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1])
        elif df_tonality[i][2] == 1:
            bobble.append([date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1])

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
        "neg_smi_name": list_neg_smi[:20],
        "neg_smi_count": list_pos_smi_massage_count[:20],
        "neg_smi_rating": list_neg_smi_index[:20],
        "pos_smi_name": list_pos_smi[:20],
        "pos_smi_count": list_pos_smi_massage_count[:20],
        "pos_smi_rating": list_pos_smi_index[:20],

        "date_bobble": [x[0] for x in bobble],
        "name_bobble": name_bobble,
        "index_bobble": [x[2] for x in bobble],
        "z_index_bobble": [1] * len(bobble),
        "tonality_index_bobble": [x[3] for x in bobble]
    }

    with open('file.txt', 'w') as file:
        file.write(json.dumps(data))  # use `json.loads` to do the reverse

    theme = filename.split('_')[0]

    print("=====!!!!!+++++")
    print(data)

    return render_template('media_rating.html', len_files=len_files, files=json_files, data=data, theme=theme)