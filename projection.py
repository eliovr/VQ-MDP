import time

import numpy as np
import pandas as pd

import plotly.graph_objs as go
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from sklearn import decomposition
from sklearn.cluster import OPTICS


class Scatterplot:
    def __init__(self):
        self.mdp = decomposition.PCA(n_components=2)
        self.point_size = 7
        self.data = np.array([])
        self.mdp_time = 0
        self.mdp_counts = 0
        self.state_str = ''

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
        return np.transpose(projection)

    def view(self, projection, selected_data):
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

        if len(selected_data) > 0:
            figure_data['data'][0]['selectedpoints'] = selected_data

        return figure_data


    def visualize(self, prototypes=[], selected_data=[]):
        if len(prototypes) > 0:
            start = time.time()
            self.data = self.abstract(prototypes)
            self.mdp_time += time.time() - start
            self.mdp_counts += 1

        return self.view(self.data, selected_data), self.get_state()

    def get_state(self):
        avg_time = 0
        if self.mdp_counts > 0:
            avg_time = self.mdp_time/self.mdp_counts

        return 'avg time: {:10.2f}s'.format(avg_time)

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


class ReachabilityPlot:
    def __init__(self):
        self.mdp = OPTICS()
        self.data = []
        self.model = None
        self.mdp_time = 0
        self.mdp_counts = 0
        self.layout = dict(
            width = 750, height = 280,
            margin = {'l': 0, 't': 0},
            xaxis={'showticklabels': False, 'zeroline': False},
            dragmode='lasso'
        )
        self.controls = html.Div(id='reachabilityplot-controls', className='ml-2', children=[
            html.Button(type='button', className='btn btn-info dropdown-toggle btn-sm', **{'data-toggle': 'dropdown'}, children=[
                html.Span(className='glyphicon glyphicon-cog', children='Reachability plot')
            ]),
            html.Ul(className='dropdown-menu w-50', children=[
                html.Li(className='dropdown-header', children='Distance metric'),
                html.Li(className='stop-propagation', children=[dcc.Dropdown(
                    id='reachplot-metric',
                    style={'padding': '0 15px 0 15px'},
                    options=[
                        {'label': 'Euclidean', 'value': 'euclidean'},
                        {'label': 'Manhattan', 'value': 'manhattan'},
                        {'label': 'Cosine', 'value': 'cosine'}
                    ],
                    value='euclidean')]),
                html.Li(className='dropdown-header', children='Xi'),
                html.Li(children=dcc.Slider(
                    id='reachplot-xi', value=.05,
                    min=0, max=1, step=.05,
                    marks={x/10: '{}'.format(x/10) for x in range(0, 10, 2)}))
            ]),
            html.Div(id='reachplot-message', children='', className='alert alert-warning small', style={'display': 'none'})
        ])

    def abstract(self, prototypes):
        self.model = self.mdp.fit(prototypes)
        return self.model.reachability_[self.model.ordering_]

    def view(self, projection, selected_data):
        indices = np.arange(0, len(projection))[self.model.ordering_]

        figure_data = {
            'data': [go.Bar(
                y = projection,
                customdata = indices)],
            'layout': self.layout
        }

        if len(selected_data) > 0:
            selection = np.where(np.isin(indices, selected_data))
            figure_data['data'][0]['selectedpoints'] = selection[0]

        return figure_data


    def visualize(self, prototypes=[], selected_data=[]):
        if len(prototypes) > 0:
            start = time.time()
            self.data = self.abstract(prototypes)
            self.mdp_time += time.time() - start
            self.mdp_counts += 1

        return self.view(self.data, selected_data), self.get_state()

    def get_state(self):
        if self.mdp_counts > 0:
            return 't: {:10.2f}s'.format(self.mdp_time/self.mdp_counts)
        else:
            return 't: 0.0s'

    def register_listener(self, app):
        @app.callback(Output('reachplot-message', 'children'),
            [Input('reachplot-metric', 'value'),
            Input('reachplot-xi', 'value')])
        def controls_listener(metric, xi):
            self.mdp.set_params(metric=metric, xi=xi)
            return 'params: {}'.format(self.mdp.get_params())


class ParallelCoordinates:
    def __init__(self):
        self.data = []
        self.layout = dict(
            width = 750, height = 300,
            margin = {'l': 0, 't': 0},
            xaxis_tickformat = '{:10.1f}'
        )
        self.controls = html.Div()


    def abstract(self, prototypes):
        _, n_dim = prototypes.shape
        n_dim = min(n_dim, 20)      # for the sake of testing.
        return np.transpose(prototypes[:, :n_dim])

    def view(self, dimensions, selected_data):
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

        if len(selected_data) > 0:
            lines = pd.DataFrame({'key': indices})
            lines['selected'] = 0
            lines.loc[lines['key'].isin(selected_data), 'selected'] = 1
            figure_data['data'][0]['line'] = {
                'color': lines['selected'],
                'colorscale': [[0, 'lightgray'], [1, 'blue']]
            }

        return figure_data

    def visualize(self, prototypes=[], selected_data=[]):
        if len(prototypes) > 0:
            self.data = self.abstract(prototypes)

        return self.view(self.data, selected_data), self.get_state()

    def get_state(self):
        return ''

    def register_listener(self, app):
        pass
