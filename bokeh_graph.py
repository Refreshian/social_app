import pandas as pd
import networkx
import matplotlib.pyplot as plt
import numpy as np
import io
import json
import datetime
import re

from bokeh.models import HoverTool
from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, OpenURL, TapTool
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
from bokeh.transform import linear_cmap
from networkx.algorithms import community
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges, LabelSet


regex = re.compile("[А-Яа-яЁё:=!\).\()A-z\_\%/|0-9]+")


def words_only(text, regex=regex):
    try:
        return " ".join(regex.findall(text))
    except:
        return ""


class bokeh_show:

    def __init__(filename):
        pass

    def open_preprocess_file(self, filename=None, got_df=None):

        if filename != None:
            with io.open(filename, encoding='utf-8', mode='r') as train_file:
                dict_train = json.load(train_file)
            df_meta = pd.DataFrame(dict_train)

        if filename == None:
            df_meta = got_df
            df_meta['timeCreate'] = [datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S') for x in
                                     df_meta['timeCreate'].values]
            df_meta.columns = ['fullname', 'url', 'er', 'hub', 'audienceCount', 'hubtype',
                               'Time', 'viewsCount', 'toneMark']

        # метаданные
        columns = ['text', 'er', 'timeCreate', 'type', 'hubtype',
                   'hub', 'audienceCount', 'viewsCount', 'url', 'toneMark']
        # columns.remove('text')
        if 'authorObject' in df_meta.columns:
            df_meta = pd.concat([pd.DataFrame.from_records(
                df_meta['authorObject'].values).drop('url', axis=1), df_meta[columns]], axis=1)
            # timestamp to date
            df_meta['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                     df_meta['timeCreate'].values]
            # индекс - время создания поста
            df_meta = df_meta.set_index(['timeCreate'])

            def date_reverse(date):  # фильтрация по дате/календарик
                lst = date.split('-')
                temp = lst[1]
                lst[1] = lst[2]
                lst[2] = temp
                return lst

            df_meta['Time'] = list(df_meta.index)
            df_meta = df_meta.iloc[::-1]

        df_meta[['fullname']] = df_meta[['fullname']].fillna('')

        fullname_val = df_meta['fullname'].values
        hub_val = df_meta['hub'].values

        for i in range(df_meta.shape[0]):
            if fullname_val[i] == '':
                fullname_val[i] = hub_val[i]

        df_meta['fullname'] = fullname_val

        df_meta_values = df_meta[['fullname',
                                  'audienceCount', 'viewsCount']].values
        weight = []

        for i in range(df_meta.shape[0]):
            if df_meta_values[i][2] == '' and df_meta_values[i][1] != 0:
                weight.append('н/д')
            elif df_meta_values[i][2] == '' and df_meta_values[i][1] == 0:
                weight.append('н/д')
            elif df_meta_values[i][2] != '' and df_meta_values[i][1] != 0:
                weight.append(
                    int(np.round(int(df_meta_values[i][2]) / int(df_meta_values[i][1]), 2)*100))
            else:
                try:
                    weight.append(int(np.round(int(df_meta_values[i][2]) / int(df_meta_values[i][1]), 2)*100))
                except:
                    weight.append(0)

        df_meta['weight'] = weight

        df_data_rep = df_meta[['fullname', 'url', 'er', 'hub', 'audienceCount', 'viewsCount', 'weight',
                               'toneMark', 'hub', 'Time', 'toneMark']]
        data_rep_er = list(df_data_rep['er'].values)

        all_hubs = list(df_data_rep['hub'].values)
        # all_hubs = [words_only(x) for x in all_hubs]

        df_rep_auth = list(df_data_rep['fullname'].values)
#         df_rep_auth = [words_only(x) for x in df_rep_auth]
        data_audience = list(df_data_rep['audienceCount'].values)

        df_rep_auth = [x if x != '' else 'None' for x in df_rep_auth]

        for i in range(len(df_rep_auth) - 1):
            if df_rep_auth[i + 1] == df_rep_auth[i]:
                df_rep_auth[i + 1] = df_rep_auth[i] + ' '

        def f(A, n=1): return [[df_rep_auth[i].strip(), df_rep_auth[i + n].strip()] for i in range(0, len(df_rep_auth) - 1,
                                                                                                   n)]  # ф-ия разбивки авторов на последовательности [[1, 2], [2,3]...]
        df_rep_auth_inverse = f(df_rep_auth.append(df_rep_auth[-1]))

        def return_1(str_int):
            if str_int == '':
                return 'н/д'
            else:
                return str_int

        df_meta['viewsCount'] = df_meta['viewsCount'].apply(return_1)
        df_meta.loc[:, ['viewsCount']] = df_meta.loc[:,
                                                     ['viewsCount']].fillna(1)

        self.got_df = pd.DataFrame(df_rep_auth_inverse)
        self.got_df['Weight'] = df_meta['weight'].values
        self.got_df['ToneMark'] = df_meta['toneMark'].values
        self.got_df['AudienceCount'] = df_meta['audienceCount'].values
        self.got_df['viewsCount'] = df_meta['viewsCount'].values
        self.got_df['hub'] = df_meta['hub'].values
        self.got_df['er'] = df_meta['er'].values
        self.got_df['time'] = df_meta['Time'].values
        self.got_df['toneMark'] = df_meta['toneMark'].values
        self.got_df['url'] = df_meta['url'].values
        self.got_df['toneMark'] = self.got_df['toneMark'].map(
            {-1: 'Негатив', 1:  'Позитив', 0: 'Нейтральная'})

        self.got_df.columns = ['Source', 'Target', 'Weight', 'ToneMark',
                               'AudienceCount', 'viewsCount', 'hub', 'er', 'time', 'toneMark', 'url']

    def bokeh(self):

        # self.got_df = self.got_df[self.got_df['Source']
        #                           != self.got_df['Target']]

        namesval = self.got_df.values

        G = networkx.from_pandas_edgelist(
            self.got_df, 'Source', 'Target', 'Weight')

        degrees = dict(networkx.degree(G))
        networkx.set_node_attributes(G, name='degree', values=degrees)

        # создание значения аудитория для tooltip
        audience_dict = pd.Series(
            self.got_df['AudienceCount'].values, index=self.got_df['Source']).to_dict()
        networkx.set_node_attributes(G, name='audience', values=audience_dict)

        # создание значения "просмотры" для tooltip
        viewsCount_dict = pd.Series(
            self.got_df['viewsCount'].values, index=self.got_df['Source']).to_dict()
        networkx.set_node_attributes(
            G, name='viewsCount', values=viewsCount_dict)

        # создание значения "источник" для tooltip
        hub_dict = pd.Series(
            self.got_df['hub'].values, index=self.got_df['Source']).to_dict()
        networkx.set_node_attributes(G, name='hub', values=hub_dict)

        # создание значения "время" для tooltip
        time_dict = pd.Series(
            self.got_df['time'].values, index=self.got_df['Source']).to_dict()
        networkx.set_node_attributes(G, name='time', values=time_dict)

        # создание значения "реакций" для tooltip
        er_dict = pd.Series(
            self.got_df['er'].values, index=self.got_df['Source']).to_dict()
        networkx.set_node_attributes(
            G, name='er', values=er_dict)

        # создание значения "тональность" для tooltip
        tone_dict = pd.Series(
            self.got_df['toneMark'].values, index=self.got_df['Source']).to_dict()
        networkx.set_node_attributes(G, name='tone', values=tone_dict)

        # создание значения "url" для tooltip
        url_dict = pd.Series(
            self.got_df['url'].values, index=self.got_df['Source']).to_dict()
        networkx.set_node_attributes(G, name='url', values=url_dict)

        names = list(self.got_df['Source'].values)

        print(len(hub_dict))
        print(len(tone_dict))
        print(self.got_df.shape)

        # размер узла
        number_to_adjust_by = 5
        adjusted_node_size = dict(
            [(node, degree+number_to_adjust_by) for node, degree in networkx.degree(G)])
        networkx.set_node_attributes(
            G, name='adjusted_node_size', values=adjusted_node_size)

        communities = community.greedy_modularity_communities(G)

        list_of_colors = ['#007f5f',  # позитив + много просмотров (в % от аудитории, > 25%)
                          # позитив + не много просмотров (в % от аудитории, 5-25%)
                          '#55a630',
                          # позитив и мало просмотров (в % от аудитории, < 5%)
                          '#aacc00',

                          # негатив + много просмотров (в % от аудитории, > 25%)
                          '#d00000',
                          # негатив + не много просмотров (в % от аудитории, 5-25%)
                          '#dc2f02',
                          # негатив и мало просмотров (в % от аудитории, < 5%)
                          '#e85d04',

                          # нейтрал + много просмотров (в % от аудитории, > 25%)
                          '#ea9010',
                          # нейтрал + не много просмотров (в % от аудитории, 5-25%)
                          '#adb5bd',
                          # нейтрал и мало просмотров (в % от аудитории, < 5%)
                          '#ced4da',

                          '#bc4749',  # нет данных по просмотрам, но большая аудитория
                          '#f8f9fa',  # нет данных по аудитории
                          ]

        # https://coolors.co/palettes/popular/green
        # https://coolors.co/palettes/popular/red
        # https://coolors.co/palettes/popular/gray

        count = 0
        countnd = 0

        modularity_class = {}
        modularity_color = {}
        self.got_df_values = self.got_df.values

        for i in range(len(self.got_df_values)):

            if self.got_df_values[i][2] != 'н/д':
                countnd += 1

                if self.got_df_values[i][2] > 25 and self.got_df_values[i][3] == 1:
                    # позитив + много просмотров (в % от аудитории, > 25%)
                    modularity_color[names[i]] = list_of_colors[0]

                elif 5 < self.got_df_values[i][2] < 25 and self.got_df_values[i][3] == 1:
                    # позитив + не много просмотров (в % от аудитории, 5-25%)
                    modularity_color[names[i]] = list_of_colors[1]

                elif self.got_df_values[i][2] < 25 and self.got_df_values[i][3] == 1:
                    # позитив и мало просмотров (в % от аудитории, < 5%)
                    modularity_color[names[i]] = list_of_colors[2]

                elif self.got_df_values[i][2] > 25 and self.got_df_values[i][3] == -1:
                    # негатив + много просмотров (в % от аудитории, > 25%)
                    modularity_color[names[i]] = list_of_colors[9]

                elif 5 < self.got_df_values[i][2] < 25 and self.got_df_values[i][3] == -1:
                    # негатив + не много просмотров (в % от аудитории, 5-25%)
                    modularity_color[names[i]] = list_of_colors[9]

                elif self.got_df_values[i][2] < 5 and self.got_df_values[i][3] == -1:
                    # негатив и мало просмотров (в % от аудитории, < 5%)
                    modularity_color[names[i]] = list_of_colors[5]

                elif self.got_df_values[i][2] > 25 and self.got_df_values[i][3] == 0:
                    # нейтрал + много просмотров (в % от аудитории, > 25%)
                    modularity_color[names[i]] = list_of_colors[6]

                elif 5 < self.got_df_values[i][2] < 25 and self.got_df_values[i][3] == 0:
                    # нейтрал + не много просмотров (в % от аудитории, 5-25%)
                    modularity_color[names[i]] = list_of_colors[7]

                elif self.got_df_values[i][2] < 5 and self.got_df_values[i][3] == 0:
                    # нейтрал и мало просмотров (в % от аудитории, < 5%)
                    modularity_color[names[i]] = list_of_colors[8]

                elif self.got_df_values[i][4] == 0:
                    # нет данных по аудитории
                    modularity_color[names[i]] = list_of_colors[10]

            if self.got_df_values[i][2] == 'н/д':
                if self.got_df_values[i][4] > 5000:
                    count += 1
                    # нет данных по просмотрам, но большая аудитория
                    modularity_color[names[i]] = list_of_colors[9]

                elif self.got_df_values[i][4] < 5000:
                    count += 1
                    # аудитория менее 5000
                    modularity_color[names[i]] = list_of_colors[8]

            if self.got_df_values[i][4] > 20000:
                count += 1
                # аудитория более 20000
                modularity_color[names[i]] = list_of_colors[4]

        # выделяем цветом начало развития событий - первого автора
        modularity_color[namesval[0][0]] = '#f62828'

        # выделяем цветом первых 5 авторов в цепочке
        first_10 = self.got_df['Source'].values[:10]

        for i in range(len(first_10)):
            modularity_color[first_10[i]] = '#BE28F6'

        # Add modularity class and color as attributes from the network above
        networkx.set_node_attributes(G, modularity_class, 'modularity_class')
        networkx.set_node_attributes(G, modularity_color, 'modularity_color')

        from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges

        # Choose colors for node and edge highlighting
        node_highlight_color = 'white'
        edge_highlight_color = 'blue'

        # Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
        size_by_this_attribute = 'adjusted_node_size'
        color_by_this_attribute = 'modularity_color'

        # Pick a color palette — Blues8, Reds8, Purples8, Oranges8, Viridis8
        color_palette = Blues8

        # Choose a title!
        title = 'Граф распространения информации'

        # Establish which categories will appear when hovering over each node
        HOVER_TOOLTIPS = [
            ("Автор", "@index"),
            ("Degree", "@degree"),
            ("Modularity Class", "@audience"),
            ("viewsCount Class", "@viewsCount"),

        ]

        # Create a plot — set dimensions, toolbar, and title
        self.plot = figure(tooltips=HOVER_TOOLTIPS,
                           tools="pan,wheel_zoom,save,reset,tap", active_scroll='wheel_zoom',
                           x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title, width=600, height=600)

        # show the tooltip
        hover = self.plot.select(dict(type=HoverTool))
        hover.tooltips = [("Автор", "@index"),
                          ("Аудитория", "@audience"),
                          ("Тональность", "@tone"),
                          ("Просмотров", "@viewsCount"),
                          ("Источник", "@hub"),
                          ("Реакций", "@er"),
                          ("Время", "@time")]
        hover.mode = 'mouse'

        url = "@url"
        taptool = self.plot.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

        # Create a network graph object
        # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
        network_graph = from_networkx(
            G, networkx.spring_layout, scale=10, center=(0, 0))

        # Set node sizes and colors according to node degree (color as category from attribute)
        network_graph.node_renderer.glyph = Circle(
            size=size_by_this_attribute, fill_color=color_by_this_attribute)
        # Set node highlight colors
        network_graph.node_renderer.hover_glyph = Circle(
            size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)
        network_graph.node_renderer.selection_glyph = Circle(
            size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)

        # Set edge opacity and width
        network_graph.edge_renderer.glyph = MultiLine(
            line_alpha=0.5, line_width=2)
        # Set edge highlight colors
        network_graph.edge_renderer.selection_glyph = MultiLine(
            line_color=edge_highlight_color, line_width=5)
        network_graph.edge_renderer.hover_glyph = MultiLine(
            line_color=edge_highlight_color, line_width=5)

        # Highlight nodes and edges
        network_graph.selection_policy = NodesAndLinkedEdges()
        network_graph.inspection_policy = NodesAndLinkedEdges()

        self.plot.renderers.append(network_graph)
        self.plot.sizing_mode = 'scale_width'

        # Add Labels
        x, y = zip(*network_graph.layout_provider.graph_layout.values())
        node_labels = list(G.nodes())
        source = ColumnDataSource(
            {'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
        labels = LabelSet(x='x', y='y', text='name', source=source,
                          background_fill_color='white', text_font_size='12px', background_fill_alpha=.7)
        self.plot.renderers.append(labels)

        # show(self.plot)
        # save(plot, filename=f"{title}.html")
