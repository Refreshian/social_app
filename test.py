# import os
# import datetime
# from collections import OrderedDict, Counter

# import pandas as pd
# import json
# import io

# os.chdir("C:\\Users\\User\\Desktop\\data")
# with io.open('Platon_24.08.2022-29.08.2022.json', encoding='utf-8', mode='r') as train_file:
#     dict_train = json.load(train_file)

# df = pd.DataFrame(dict_train)

# # метаданные
# columns = ['text', 'er', 'timeCreate', 'type', 'hubtype', 'hub', 'toneMark']
# # columns.remove('text')
# df_meta = pd.concat([pd.DataFrame.from_records(df['authorObject'].values), df[columns]], axis=1)
# # timestamp to date
# df_meta['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
#                          df_meta['timeCreate'].values]

# search_lst = 'платон'.split(',')
# search_lst = [x.split('или') for x in search_lst]
# search_lst = [[x.strip() for x in group] for group in search_lst]

# df_meta['toneMark'] = df_meta['toneMark'].map({0: 'Neutral', -1: 'Negative', 1: 'Positive'})
# text_val = df_meta['text'].values
# dict_count = {key: [] for key in [x[0].capitalize() for x in
#                                   search_lst]}  # словарь с названием продукта и индексом его встречаемости в таблице текстов
# list_keys = list(dict_count.keys())

# for j in range(len(text_val)):
#     for i in range(len(search_lst)):
#         if [item for item in search_lst[i] if item in text_val[j]] != []:
#             dict_count[list_keys[i]].append(j)

# list_sunkey_hubs = []

# ### учет кол-ва источников по объекту
# for i in range(len(list_keys)):
#     dict_hubs = Counter(df_meta.loc[dict_count[list_keys[i]]]['hub'])
#     dict_hubs = dict_hubs.most_common()
#     list_hubs = [[x[0], list_keys[i], x[1]] for x in dict_hubs]
#     list_sunkey_hubs.append(list_hubs)


# ### учет кол-ва постов, репостов, комментариев
# list_sunkey_post_type = []
# list_type_post = [] # cписок для понимания, какие типы постов были для каждого объекта поиска (необходимо для тональности по типам далее)

# for i in range(len(list_keys)):
#     list_sunkey_post = Counter(df_meta.loc[dict_count[list_keys[i]]]['type'])
#     list_type_post.append(list(list_sunkey_post.keys())) # добавляем какие типы постов встречались для объекта
#     list_sunkey_post = list_sunkey_post.most_common()
#     list_sunkey_post = [[list_keys[i], x[0], x[1]] for x in list_sunkey_post]
#     list_sunkey_post_type.append(list_sunkey_post)


# ### учет тональности по каждому типу сообщений
# list_sunkey_tonality = [] # кол-во тональности постов, репостов, комментариев
# tonality_type_posts = {} # словарь для сбора отдельно кол-ва позитива, нейтрала и негавтива по типу источников в объекте поиска
#                             # {'Платон': [{'Комментарий': [('Neutral', 16), ('Negative', 6), ('Positive', 5)], 'Пост':...

# for i in range(len(list_keys)):
#     df = df_meta.loc[dict_count[list_keys[i]]]
#     a = []
#     d = {}
#     for j in range(len(list_type_post[i])):
#         list_sunkey_ton = Counter(df[df['type'] == list_type_post[i][j]]['toneMark'])
#         list_sunkey_ton = list_sunkey_ton.most_common()
#         d[list_type_post[i][j]] = list_sunkey_ton

#     tonality_type_posts[list_keys[i]] = d


# val_tonality_type_posts = list(tonality_type_posts.values())
# tonality_by_post_type = [] # собираем все типы сообщений, их тональность и кол-во ['Комментарий', 'Neutral', 16],
#                             # ['Комментарий', 'Negative', 6], ['Комментарий', 'Positive', 5], ['Пост', 'Neutral', 13]...
# for i in range(len(val_tonality_type_posts)):
#         for k, v in val_tonality_type_posts[i].items():
#             for l in range(len(v)):
#                 tonality_by_post_type.append([k, v[l][0], v[l][1]])

# list_sunkey_hubs = [item for sublist in list_sunkey_hubs for item in sublist]
# list_sunkey_post_type = [item for sublist in list_sunkey_post_type for item in sublist]
# # tonality_by_post_type = [item for sublist in tonality_by_post_type for item in sublist]



# fin_list = []
# fin_list.append(list_sunkey_hubs)
# fin_list.append(list_sunkey_post_type)
# fin_list.append(tonality_by_post_type)

import requests
import sys
import traceback
import urllib



# https://gist.github.com/komasaru/ed07018ae246d128370a1693f5dd1849
def shorten(url_long): # делаем ссылки корткими для отображения в web 

    URL = "http://tinyurl.com/api-create.php"
    try:
        url = URL + "?" \
            + urllib.parse.urlencode({"url": url_long})
        res = requests.get(url)
        print("   LONG URL:", url_long)
        print("  SHORT URL:", res.text)
    except Exception as e:
        raise

# X = UrlShortenTinyurl()
print(shorten('https://adpass.ru/reklama-prinesla-vk-polovinu-vyruchki/'))
