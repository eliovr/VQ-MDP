import time
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from sklearn import cluster
import pandas as pd
import numpy as np

class MiniBatchKMeans:
    def __init__(self):
        self.spawns = []
        self.fitting_time = 0
        self.sample_count = 0
        self.state = ''

        self.optimizer = cluster.MiniBatchKMeans(
            n_clusters=300,
            random_state=0,
            batch_size=500,
            reassignment_ratio=.01)

        self.controls = html.Div(id='kmeans-controls', className='ml-2', children=[
            html.Button(type='button', className='btn btn-info dropdown-toggle btn-sm', **{'data-toggle': 'dropdown'}, children=[
                html.Span(className='glyphicon glyphicon-cog', children='Quantizer')
            ]),
            html.Ul(className='dropdown-menu w-50', children=[
                html.Li(className='dropdown-header', children='# of prototypes'),
                html.Li(children=dcc.Slider(
                    id='prototypes', value=self.optimizer.n_clusters,
                    min=100, max=500, step=10,
                    marks={x: '{}'.format(x) for x in range(100, 600, 100)})),

                html.Li(className='dropdown-header', children='Batch size'),
                html.Li(children=dcc.Slider(
                    id='batch-size', value=self.optimizer.batch_size,
                    min=0, max=1000, step=50,
                    marks={x: '{}'.format(x) for x in range(0, 1200, 200)})),

                html.Li(className='dropdown-header', children='Reassignment ratio'),
                html.Li(children=dcc.Slider(
                    id='reassignment-ratio', value=self.optimizer.reassignment_ratio,
                    min=0, max=1, step=.01,
                    marks={x/10: '{}'.format(x/10) for x in range(0, 12, 2)}))
            ]),
            html.Div(id='kmeans-message', children='', className='alert alert-warning small', style={'display': 'none'})
        ])


    def learn(self, sample):
        """
        Runs the vector quantization on the sample, and returns the fitted prototypes.
        If the sample size is larged than the number of prototypes, then just
        return the sample.
        """
        if len(sample) > self.optimizer.n_clusters:
            start = time.time()
            self.optimizer = self.optimizer.partial_fit(sample)
            self.fitting_time += time.time() - start
            self.sample_count += 1
            self.state = 'avg fitting time: {:10.2f}s; spawns: {}'.format(self.fitting_time/self.sample_count, len(self.spawns))
            return self.get_prototypes()
        else:
            self.state = 'Sample size ({}) < prototypes ({}). No fitting took place.'.format(len(sample), self.optimizer.n_clusters)
            return sample

    def reset(self):
        """
        Clears the values of the partially trained model.
        """
        if len(self.spawns) > 0:
            self.unspawn()
        else:
            self.optimizer = cluster.MiniBatchKMeans(**self.optimizer.get_params(deep=False))

    def get_prototypes(self):
        return self.optimizer.cluster_centers_

    def spawn(self):
        self.spawns.append(dict(
            optimizer = self.optimizer
        ))
        self.optimizer = cluster.MiniBatchKMeans(**self.optimizer.get_params(deep=False))
        return self

    def unspawn(self):
        if len(self.spawns) > 0:
            parent = self.spawns.pop()
            self.optimizer = parent['optimizer']
        return self

    def predict(self, data):
        return self.optimizer.predict(data)

    def predict_select(self, data, select_ks):
        predictions = self.predict(data).reshape(len(data), 1)
        k_data = np.append(predictions, data.values, axis=1)
        k_data = pd.DataFrame(k_data)
        return k_data.loc[k_data[0].isin(select_ks)].drop(0, axis=1)

    def register_listener(self, app):
        @app.callback(Output('kmeans-message', 'children'),
            [Input('prototypes', 'value'),
            Input('batch-size', 'value'),
            Input('reassignment-ratio', 'value')])
        def kmeans_listener(n_prototypes, batch_size, reassignment_ratio):
            if hasattr(self.optimizer, 'counts_') and n_prototypes != self.optimizer.n_clusters:
                delattr(self.optimizer, 'counts_')

            self.optimizer.set_params(
                n_clusters=n_prototypes,
                batch_size=batch_size,
                reassignment_ratio=reassignment_ratio)

            return 'params: {}'.format(str(self.optimizer.get_params(deep=False)))
