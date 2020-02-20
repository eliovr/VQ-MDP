import time
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from sklearn import cluster


class MiniBatchKMeans:
    def __init__(self):
        self.spawns = []
        self.fitting_time = 0
        self.sample_count = 0
        self.optimizer = cluster.MiniBatchKMeans(
            n_clusters=300,
            random_state=0,
            batch_size=100,
            max_iter=5)

        self.controls = html.Div(id='kmeans-controls', style={'margin': '10px'}, children=[
            html.Div(className='card', children=[
                html.Div(className='card-header', children='Mini batch K-Means'),
                html.Div(className='card-body', children=[
                    html.Div(className='form-group', children=[
                        html.Label('Number of prototypes', className='small'),
                        dcc.Input(id='prototypes', className='form-control form-control-sm',
                                    value=self.optimizer.n_clusters, type='number', min='100', max='500', step='10')
                    ]),
                    html.Div(className='form-group', children=[
                        html.Label('Batch size (# of samples)', className='small'),
                        dcc.Input(id='batch-size', className='form-control form-control-sm',
                                    value=self.optimizer.batch_size, type='number', min='50', max='1000', step='50')
                    ]),
                    html.Div(className='form-group', children=[
                        html.Label('Max iterations per sample', className='small'),
                        dcc.Input(id='max-iterations', className='form-control form-control-sm',
                                    value=self.optimizer.max_iter, type='number', min='1', max='200')
                    ]),
                    html.Div(id='kmeans-message', children='', className='alert alert-warning small', style={'display': 'none'})
                ]),
            ])
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
            return self.get_prototypes()
        else:
            return sample

    def state(self):
        return 'Avg quantization time: {:10.2f}'.format(self.fitting_time/self.sample_count)

    def reset(self):
        """
        Clears the values of the partially trained model.
        """
        if len(self.spawns) > 0:
            self.unspawn()
        else:
            self.optimizer = cluster.MiniBatchKMeans(**self.optimizer.get_params(deep=False))

    def predict(self, data):
        return self.optimizer.predict(data)

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

    def register_listener(self, app):
        @app.callback(Output('kmeans-message', 'children'),
            [Input('prototypes', 'value'),
            Input('batch-size', 'value'),
            Input('max-iterations', 'value')])
        def kmeans_listener(n_prototypes, batch_size, max_iter):
            if hasattr(self.optimizer, 'counts_') and n_prototypes != self.optimizer.n_clusters:
                delattr(self.optimizer, 'counts_')

            self.optimizer.set_params(
                n_clusters=n_prototypes,
                batch_size=batch_size,
                max_iter=max_iter)

            return 'params: {}'.format(str(self.optimizer.get_params(deep=False)))
