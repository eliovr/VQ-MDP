import time

import numpy as np
import pandas as pd

import plotly.graph_objs as go
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from sklearn import decomposition


class Scatterplot:
    def __init__(self):
        self.mdp = decomposition.PCA(n_components=2)
        self.point_size = 7
        self.data = np.array([])
        self.selected_data = []
        self.mdp_time = 0
        self.mdp_counts = 0
        self.state = ''

        self.layout = dict(
            xaxis={'showticklabels': False, 'zeroline': False},
            yaxis={'showticklabels': False, 'zeroline': False},
            margin={'l': 20, 'b': 20, 't': 20, 'r': 20},
            hovermode='closest',
            dragmode='lasso',
            clickmode='event+select'
        )
        self.controls = html.Div(id='scatterplot-controls', className='ml-2', children=[
            html.Button(type='button', className='btn btn-info dropdown-toggle btn-sm', **{'data-toggle': 'dropdown'}, children=[
                html.Span(className='glyphicon glyphicon-cog', children='Scatterplot')
            ]),
            html.Ul(className='dropdown-menu w-50', children=[
                html.Li(className='dropdown-header', children='Point size (px)'),
                html.Li(children=dcc.Slider(
                    id='point-size', value=self.point_size,
                    min=1, max=17, step=1,
                    marks={x: '{}'.format(x) for x in range(1, 18, 2)}))
            ]),
            html.Div(id='scatterplot-message', children='', className='alert alert-warning small', style={'display': 'none'})
        ])

    def abstract(self, prototypes):
        projection = self.mdp.fit_transform(prototypes)
        self.data = np.transpose(projection)
        return self.data

    def figure(self, projection):
        indices = np.arange(0, projection.size)

        figure_data = {
            'data': [go.Scatter(
                x = projection[0], y = projection[1],
                mode = 'markers',
                # opacity = .5,
                customdata = indices,
                marker = dict(
                    size = self.point_size
                ))],
            'layout': self.layout
        }

        if len(self.selected_data) > 0:
            figure_data['data'][0]['selectedpoints'] = self.selected_data

        return figure_data


    def visualize(self, prototypes):
        start = time.time()
        projection = self.abstract(prototypes)
        self.mdp_time += time.time() - start
        self.mdp_counts += 1
        self.state = 'avg time: {:10.2f}s'.format(self.mdp_time/self.mdp_counts)
        return self.figure(projection)


    def update(self):
        return self.figure(self.data)

    def selected_ids(self, relayout_data):
        """
        Translate relayoutData (from zoom interaction) into a vector of
        selected prototypes ids.
        """
        x1 = relayout_data.get('xaxis.range[0]', 0)
        y1 = relayout_data.get('yaxis.range[0]', 0)
        x2 = relayout_data.get('xaxis.range[1]', 0)
        y2 = relayout_data.get('yaxis.range[1]', 0)

        xs = self.data[0]
        ys = self.data[1]
        ks = np.arange(0, len(xs))
        return ks[(xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)]

    def register_listener(self, app):
        @app.callback(Output('scatterplot-message', 'children'),
            [Input('point-size', 'value')])
        def controls_listener(size):
            self.point_size = size
            return 'PCA: Nothing to report'



class ParallelCoordinates:
    def __init__(self):
        self.data = []
        self.selected_data = []
        self.layout = dict(
            width = 800,
            margin = {'l': 0}
        )
        self.controls = html.Div()


    def abstract(self, prototypes):
        self.data = np.transpose(prototypes[:, :20])
        return self.data


    def figure(self, dimensions):
        indices = []
        if len(dimensions) > 0:
            indices = np.arange(0, len(dimensions[0]))

        dim_elems = [dict(
            range = [np.min(d), np.max(d)],
            values = d
        ) for d in dimensions]

        figure_data = {
            'data': [go.Parcoords(
                dimensions = list(dim_elems),
                customdata = indices
            )],
            'layout': self.layout
        }

        if len(self.selected_data) > 0:
            lines = pd.DataFrame({'key': indices})
            lines['selected'] = 0
            lines.loc[lines['key'].isin(self.selected_data), 'selected'] = 1
            figure_data['data'][0]['line'] = {
                'color': lines['selected'],
                'colorscale': [[0, 'lightgray'], [1, 'blue']]
            }

        return figure_data


    def visualize(self, prototypes):
        dimensions = self.abstract(prototypes)
        return self.figure(dimensions)


    def update(self):
        return self.figure(self.data)


    def register_listener(self, app):
        pass
