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

        self.layout = dict(
            xaxis={'showticklabels': False, 'zeroline': False},
            yaxis={'showticklabels': False, 'zeroline': False},
            margin={'l': 20, 'b': 20, 't': 20, 'r': 20},
            hovermode='closest',
            dragmode='lasso',
            clickmode='event+select'
        )
        self.controls = html.Div(id='pca-controls', style={'margin': '10px'}, children=[
            html.Div(className='card', children=[
                html.Div(className='card-header', children='PCA + Scatterplot'),
                html.Div(className='card-body', children=[
                    html.Div(className='form-group', children=[
                        html.Label('Point size', className='small'),
                        dcc.Slider(id='point-size', value=self.point_size,
                            min=1, max=17, step=1,
                            marks={x: '{}'.format(x) for x in range(1, 18, 2)})
                    ]),
                    html.Div(id='pca-message', children='', className='alert alert-warning small', style={'display': 'none'})
                ]),
            ])
        ])


    def abstract(self, prototypes):
        start = time.time()
        projection = self.mdp.fit_transform(prototypes)
        self.data = np.transpose(projection)
        self.mdp_time += time.time() - start
        self.mdp_counts += 1
        return self.data


    def figure(self, projection, selected_data):
        self.selected_data = selected_data
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


    def visualize(self, prototypes, selected_data):
        projection = self.abstract(prototypes)
        return self.figure(projection, selected_data)


    def update(self, selected_data):
        return self.figure(self.data, selected_data)

    def state(self):
        return 'Avg MDP time: {:10.2f}'.format(self.mdp_time/self.mdp_counts)


    def register_listener(self, app):
        @app.callback(Output('pca-message', 'children'),
            [Input('point-size', 'value')])
        def controls_listener(size):
            self.point_size = size

            # return '{}'.format(self.selected_data)
            return 'PCA: Nothing to report'



class ParallelCoordinates:
    def __init__(self):
        self.data = []
        self.selected_data = []
        self.layout = dict(
            width = 1000,
            margin = {'l': 20}
        )
        self.controls = html.Div()


    def abstract(self, prototypes):
        self.data = np.transpose(prototypes[:, :20])
        return self.data


    def figure(self, dimensions, selected_data):
        self.selected_data = selected_data
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


    def visualize(self, prototypes, selected_data):
        dimensions = self.abstract(prototypes)
        return self.figure(dimensions, selected_data)


    def update(self, selected_data):
        return self.figure(self.data, selected_data)


    def register_listener(self, app):
        pass
