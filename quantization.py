import time
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from sklearn import cluster
from sklearn.metrics.pairwise import euclidean_distances

from pyspark.sql.functions import udf
from pyspark.sql.functions import *
from pyspark.sql.types import BooleanType

import pandas as pd
import numpy as np
from numpy.linalg import norm

class MiniBatchKMeans:
    def __init__(self):
        self.spawns = []
        self.exe_time = 0
        self.sample_count = 0
        self.optimizer = cluster.MiniBatchKMeans(
            n_clusters=200,
            random_state=None,
            batch_size=500,
            reassignment_ratio=.01)

        self.controls = html.Div(id='kmeans-controls', className='input-group input-group-sm ml-2', children=[
            html.Button(type='button', className='btn btn-info dropdown-toggle form-control', **{'data-toggle': 'dropdown'}, children=[
                html.Span(className='glyphicon glyphicon-cog', children='Quantizer')
            ]),
            html.Ul(className='dropdown-menu w-100', children=[
                html.Li(className='dropdown-header', children='# of prototypes'),
                html.Li(children=dcc.Slider(
                    id='prototypes', value=self.optimizer.n_clusters,
                    min=100, max=500, step=10,
                    marks={x: '{}'.format(x) for x in range(100, 600, 100)})),

                html.Li(className='dropdown-header', children='Reassignment ratio'),
                html.Li(children=dcc.Slider(
                    id='reassignment-ratio', value=self.optimizer.reassignment_ratio,
                    min=0, max=1, step=.01,
                    marks={x/10: '{}'.format(x/10) for x in range(0, 12, 2)}))
            ]),
            html.Div(className="input-group-append", children=[
                html.Span(id='quantizer-state', className='input-group-text', children=' - ')
            ]),
            html.Div(id='kmeans-message', children='', className='alert alert-warning small', style={'display': 'none'})
        ])


    def learn(self, sample):
        """
        Runs the vector quantization on the sample, and returns the fitted prototypes.
        If the sample size is larged than the number of prototypes, then just
        return the sample.
        """
        m = []

        if len(sample) > self.optimizer.n_clusters:
            start = time.time()
            self.optimizer = self.optimizer.partial_fit(sample)
            self.exe_time += time.time() - start
            self.sample_count += 1
            m = self.get_prototypes()
        else:
            m = sample

        return m, self.get_state()


    def reset(self):
        """
        Clears the values of the partially trained model.
        """
        self.optimizer = cluster.MiniBatchKMeans(**self.optimizer.get_params(deep=False))


    def get_prototypes(self):
        """
        Returns the array of (latest trained) prototypes. I remove the two prototypes
        with the lowest and highest L2-norm. It is an uggly thing but, for some reason
        there's always a 'stray' prototypes which messes with the plots.
        """
        ks = self.optimizer.cluster_centers_
        ns = norm(ks, axis=1)
        ns = (ns != np.min(ns)) & (ns != np.max(ns))
        return ks[ns]


    def get_state(self):
        """
        Returns a string with a state message.
        """
        state = 't: 0.0s | inertia: 0'
        if self.sample_count > 0:
            t = self.exe_time/self.sample_count
            inertia = self.optimizer.inertia_
            state = 't: {:10.2f}s | inertia: {:10.0f}'.format(t, inertia)
        return state


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
        # Pandas DataFrame.
        if type(data) is pd.core.frame.DataFrame:
            predictions = self.predict(data).reshape(len(data), 1)
            k_data = np.append(predictions, data.values, axis=1)
            k_data = pd.DataFrame(k_data)
            return k_data.loc[k_data[0].isin(select_ks)].drop(0, axis=1)

        # Spark DataFrame.
        else:
            ks = self.get_prototypes()
            def is_selected(arr):
                distances = np.sqrt(norm(np.power(arr - ks, 2), axis=1))
                kid = np.argmin(distances)
                return (kid in select_ks)

            is_selected_udf = udf(is_selected, BooleanType())
            cols = data.columns
            return data.filter(is_selected_udf(array(*cols))).cache()


    def register_listener(self, app):
        @app.callback(Output('kmeans-message', 'children'),
            [Input('prototypes', 'value'),
            Input('reassignment-ratio', 'value')])
        def kmeans_listener(n_prototypes, reassignment_ratio):
            if hasattr(self.optimizer, 'counts_') and n_prototypes != self.optimizer.n_clusters:
                delattr(self.optimizer, 'counts_')

            self.optimizer.set_params(
                n_clusters=n_prototypes,
                reassignment_ratio=reassignment_ratio)

            return 'params: {}'.format(str(self.optimizer.get_params(deep=False)))
