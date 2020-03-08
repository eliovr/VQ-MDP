import time

import numpy as np
import pandas as pd

import plotly.graph_objs as go
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from sklearn import decomposition
from sklearn import manifold
from sklearn.cluster import OPTICS


class Scatterplot:
    def __init__(self):
        # self.mdp = decomposition.PCA(n_components=2)
        self.mdp = manifold.MDS(n_components=2, n_init=1, max_iter=100)
        # self.mdp = manifold.TSNE(n_components=2, n_iter=500)
        self.point_size = 7
        self.data = np.array([])
        self.color_values = []
        self.exe_time = 0
        self.mdp_counts = 0
        self.state_str = ''

        self.layout = dict(
            xaxis={'showticklabels': False, 'zeroline': False},
            yaxis={'showticklabels': False, 'zeroline': False},
            margin={'l': 20, 'b': 15, 't': 20, 'r': 20},
            hovermode='closest',
            dragmode='lasso',
            clickmode='event+select'
        )
        self.controls = html.Div(id='scatterplot-controls', className='input-group input-group-sm ml-2', children=[
            html.Button(type='button', className='btn btn-info dropdown-toggle form-control', **{'data-toggle': 'dropdown'}, children=[
                html.Span(className='glyphicon glyphicon-cog', children='MDS')
            ]),
            html.Ul(className='dropdown-menu w-100', children=[
                html.Li(className='dropdown-header', children='n init'),
                html.Li(children=dcc.Slider(
                    id='mds-ninit', value=self.mdp.get_params()['n_init'],
                    min=0, max=10, step=1,
                    marks={x: '{}'.format(x) for x in range(0, 12, 2)})),

                html.Li(className='dropdown-header', children='Iterations'),
                html.Li(children=dcc.Slider(
                    id='mds-maxiter', value=self.mdp.get_params()['max_iter'],
                    min=0, max=500, step=50,
                    marks={x: '{}'.format(x) for x in range(0, 600, 100)})),

                # html.Li(className='dropdown-header', children='Perplexity'),
                # html.Li(children=dcc.Slider(
                #     id='tsne-perplexity', value=self.mdp.get_params()['perplexity'],
                #     min=5, max=50, step=5,
                #     marks={x: '{}'.format(x) for x in range(5, 55, 5)})),
                #
                # html.Li(className='dropdown-header', children='Learning rate'),
                # html.Li(children=dcc.Slider(
                #     id='tsne-learning-rate', value=self.mdp.get_params()['learning_rate'],
                #     min=0, max=1000, step=100,
                #     marks={x: '{}'.format(x) for x in range(0, 1200, 200)})),

                html.Li(className='dropdown-divider'),

                html.Li(className='dropdown-header', children='Point size (px)'),
                html.Li(children=dcc.Slider(
                    id='point-size', value=self.point_size,
                    min=1, max=17, step=1,
                    marks={x: '{}'.format(x) for x in range(1, 18, 2)}))
            ]),
            html.Div(className='input-group-append', children=[
                html.Span(id='scatterplot-state', className='input-group-text', children=' - ')
            ]),
            html.Div(id='scatterplot-message', children='', className='alert alert-warning small', style={'display': 'none'})
        ])

    def abstract(self, prototypes):
        # --- PCA or MDS.
        n_prototypes = len(prototypes)
        if n_prototypes > 0:
            if len(self.data) == n_prototypes:
                self.data = self.mdp.fit_transform(prototypes, init=self.data)
            else:
                self.data = self.mdp.fit_transform(prototypes)
            self.color_values = np.var(prototypes, axis=1)

        return np.transpose(self.data), self.color_values

        # --- PCA.
        # projection = self.mdp.fit_transform(prototypes)
        # return np.transpose(projection)

        # --- TSNE.
        # if len(self.data) == n_prototypes:
        #     self.mdp.set_params(init=self.data)
        # if n_prototypes > 0:
        #     self.data = self.mdp.fit_transform(prototypes)
        # return np.transpose(self.data)

    def view(self, projection, selected_data=[], color_values=[]):
        indices = np.arange(0, projection.size)

        figure_data = {
            'data': [go.Scatter(
                x = projection[0], y = projection[1],
                mode = 'markers',
                customdata = indices,
                marker = dict(
                    size = self.point_size,
                    color = color_values,
                    colorscale = 'sunsetdark'
                ))],
            'layout': self.layout
        }

        if len(selected_data) > 0:
            figure_data['data'][0]['selectedpoints'] = selected_data

        return figure_data


    def visualize(self, prototypes=[], selected_data=[]):
        start = time.time()
        abstraction, color_by = self.abstract(prototypes)
        view = self.view(abstraction, selected_data=selected_data, color_values=color_by)
        self.exe_time += time.time() - start
        self.mdp_counts += 1

        return view, self.get_state()


    def get_state(self):
        state = 't: 0.0s | stress: 0'
        if self.mdp_counts > 0:
            t = self.exe_time/self.mdp_counts
            stress = self.mdp.stress_
            state = 't: {:10.2f}s | stress: {:10.0f}'.format(t, stress)
            # divergence = self.mdp.kl_divergence_
            # state = 't: {:10.2f}s | divergence: {:10.2f}'.format(t, divergence)
        return state


    def selected_ids(self, relayout_data):
        """
        Translate relayoutData (from zoom interaction) into a vector of
        selected prototypes ids.
        """
        x1 = relayout_data.get('xaxis.range[0]', 0)
        y1 = relayout_data.get('yaxis.range[0]', 0)
        x2 = relayout_data.get('xaxis.range[1]', 0)
        y2 = relayout_data.get('yaxis.range[1]', 0)
        proj = np.transpose(self.data)
        xs = proj[0]
        ys = proj[1]
        ks = np.arange(0, len(xs))
        return ks[(xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)]


    def register_listener(self, app):
        @app.callback(
            Output('scatterplot-message', 'children'),
            [Input('mds-ninit', 'value'),
            Input('mds-maxiter', 'value'),
            Input('point-size', 'value')])
        def tsne_controls_listener(mds_ninit, mds_maxiter, point_size):
            self.point_size = point_size
            self.mdp.set_params(n_init=mds_ninit, max_iter=mds_maxiter)
            return 'MDS: Nothing to report'

        # @app.callback(
        #     Output('scatterplot-message', 'children'),
        #     [Input('tsne-perplexity', 'value'),
        #     Input('tsne-learning-rate', 'value'),
        #     Input('point-size', 'value')])
        # def tsne_controls_listener(tsne_perplexity, tsne_learning_rate, point_size):
        #     self.point_size = point_size
        #     self.mdp.set_params(perplexity=tsne_perplexity, learning_rate=tsne_learning_rate)
        #     return 't-SNE: Nothing to report'


class ReachabilityPlot:
    def __init__(self):
        self.mdp = OPTICS()
        self.data = []
        self.color_values = []
        self.model = None
        self.exe_time = 0
        self.mdp_counts = 0
        self.layout = dict(
            width = 750, height = 260,
            margin = {'l': 15, 't': 15},
            xaxis={'showticklabels': False, 'zeroline': False},
            dragmode='lasso'
        )
        self.controls = html.Div(id='reachabilityplot-controls', className='input-group input-group-sm ml-2', children=[
            html.Button(type='button', className='btn btn-info dropdown-toggle form-control', **{'data-toggle': 'dropdown'}, children=[
                html.Span(className='glyphicon glyphicon-cog', children='OPTICS')
            ]),
            html.Ul(className='dropdown-menu w-100', children=[
                html.Li(className='dropdown-header', children='Distance metric'),
                html.Li(className='stop-propagation', children=[dcc.Dropdown(
                    id='optics-metric',
                    style={'padding': '0 15px 0 15px'},
                    options=[
                        {'label': 'Euclidean', 'value': 'euclidean'},
                        {'label': 'Manhattan', 'value': 'manhattan'},
                        {'label': 'Cosine', 'value': 'cosine'}
                    ],
                    value='euclidean')]),

                html.Li(className='dropdown-header', children='Min samples'),
                html.Li(children=dcc.Slider(
                    id='optics-min-samples', value=self.mdp.get_params()['min_samples'],
                    min=0, max=20, step=1,
                    marks={x: '{}'.format(x) for x in range(0, 25, 5)})),

                html.Li(className='dropdown-header', children='Xi'),
                html.Li(children=dcc.Slider(
                    id='optics-xi', value=.05,
                    min=0, max=1, step=.05,
                    marks={x/10: '{}'.format(x/10) for x in range(0, 12, 2)}))
            ]),
            html.Div(className='input-group-append', children=[
                html.Span(id='reachplot-state', className='input-group-text', children=' - ')
            ]),
            html.Div(id='optics-message', children='', className='alert alert-warning small', style={'display': 'none'})
        ])


    def abstract(self, prototypes):
        self.model = self.mdp.fit(prototypes)
        color_values = np.var(prototypes, axis=1)
        return self.model.reachability_[self.model.ordering_], color_values


    def view(self, projection, selected_data=[], color_values=[]):
        indices = self.model.ordering_

        figure_data = {
            'data': [go.Bar(
                y = projection,
                customdata = indices,
                marker = dict(
                    color = color_values[indices],
                    colorscale = 'sunsetdark'
                ))],
            'layout': self.layout
        }

        if len(selected_data) > 0:
            selection = np.where(np.isin(indices, selected_data))
            figure_data['data'][0]['selectedpoints'] = selection[0]

        return figure_data


    def visualize(self, prototypes=[], selected_data=[]):
        if len(prototypes) > 0:
            start = time.time()
            self.data, self.color_values = self.abstract(prototypes)
            self.exe_time += time.time() - start
            self.mdp_counts += 1

        view = self.view(self.data, selected_data=selected_data, color_values=self.color_values)

        return view, self.get_state()


    def get_state(self):
        state = 't: 0.0s'
        if self.mdp_counts > 0:
            state = 't: {:10.2f}s'.format(self.exe_time/self.mdp_counts)
        return state


    def register_listener(self, app):
        @app.callback(
            Output('optics-message', 'children'),
            [Input('optics-metric', 'value'),
            Input('optics-min-samples', 'value'),
            Input('optics-xi', 'value')])
        def controls_listener(metric, min_samples, xi):
            self.mdp.set_params(metric=metric, min_samples=min_samples, xi=xi)
            return 'params: {}'.format(self.mdp.get_params())


class ParallelCoordinates:
    def __init__(self):
        self.data = [],
        self.color_values = []
        self.min_spacing = 70
        self.layout = dict(
            width = 700, height = 290,
            margin = {'l': 20, 't': 40},
            xaxis_tickformat = '{:10.1f}'
        )
        self.controls = html.Div()


    def abstract(self, prototypes):
        ks = prototypes
        if type(ks) is list:
            ks = np.array(ks)
        _, n_dim = ks.shape
        # n_dim = min(n_dim, 20)      # for the sake of testing.
        return np.transpose(ks[:, :n_dim]), np.var(prototypes, axis=1)


    def view(self, dimensions, selected_data=[], column_names=[], color_values=[]):
        indices = []
        n_dimensions = len(dimensions)

        if n_dimensions > 0:
            indices = np.arange(0, len(dimensions[0]))
            w1 = self.layout['width']
            w2 = self.min_spacing * n_dimensions
            self.layout['width'] = max(w1, w2)

        if len(column_names) <= 0:
            column_names = ['col_{}'.format(i) for i in indices]

        dim_elems = [dict(
            # range = [np.min(d), np.max(d)],
            values = d,
            label = n
        ) for (d, n) in zip(dimensions, column_names)]

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

        elif len(color_values) > 0:
            figure_data['data'][0]['line'] = {
                'color': color_values,
                'colorscale': 'sunsetdark'
            }

        return figure_data


    def visualize(self, prototypes=[], selected_data=[], column_names=[]):
        if len(prototypes) > 0:
            self.data, self.color_values = self.abstract(prototypes)

        view = self.view(self.data,
            selected_data=selected_data,
            column_names=column_names,
            color_values=self.color_values)

        return view, self.get_state()


    def get_state(self):
        return ''


    def register_listener(self, app):
        pass
