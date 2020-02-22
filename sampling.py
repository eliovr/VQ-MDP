import time

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output


class Sampler:
    def __init__(self, data):
        self.spawns = []
        self.data = data
        self.sample_count = 0
        self.sampling_time = 0
        self.state = ''

        self.sample_size = 500
        self.sample_frequency = .1

        self.controls = html.Div(id='sampler-controls', className='ml-2', children=[
            html.Button(type='button', className='btn btn-info dropdown-toggle btn-sm', **{'data-toggle': 'dropdown'}, children=[
                html.Span(className='glyphicon glyphicon-cog', children='Sampler')
            ]),
            html.Ul(className='dropdown-menu w-50', children=[
                html.Li(className='dropdown-header', children='Sample size (# of points)'),
                html.Li(children=dcc.Slider(
                    id='sample-size', value=self.sample_size,
                    min=0, max=4000, step=50,
                    marks={x: '{}K'.format(x/1000) for x in range(0, 5000, 1000)})),

                html.Li(className='dropdown-header', children='Sample frequency'),
                html.Li(children=dcc.Slider(
                    id='sample-frequency', value=self.sample_frequency,
                    min=0, max=1, step=.05,
                    marks={x/10: '{}'.format(x/10) for x in range(0, 12, 2)}))
            ]),
            html.Div(id='sampler-message', children='', className='alert alert-warning small', style={'display': 'none'})
        ])


    def sample(self):
        if self.sample_frequency > 0:
            time.sleep(self.sample_frequency)

        if len(self.data.index) > self.sample_size:
            start = time.time()
            x = self.data.sample(n=self.sample_size).values
            self.sampling_time += time.time()
            self.sample_count += 1
            self.state = 'samples: {}; spawns: {}'.format(self.sample_count, len(self.spawns))
            return x
        else:
            self.state = 'Data points ({}) < sample size ({}); spawns: {}'.format(len(self.data.index), self.sample_size, len(self.spawns))
            return self.data.values

    def reset(self):
        self.sample_count = 0
        self.sampling_time = 0
        if len(self.spawns) > 0:
            self.unspawn()

    def spawn(self, data):
        self.spawns.append(dict(
            data = self.data,
            sample_count = self.sample_count
        ))

        self.data = data
        self.sample_count = 0
        return self

    def unspawn(self):
        if len(self.spawns) > 0:
            parent = self.spawns.pop()
            self.data = parent['data']
            self.sample_count = parent['sample_count']
        return self

    def register_listener(self, app):
        @app.callback(Output('sampler-message', 'children'),
            [Input('sample-size', 'value'),
            Input('sample-frequency', 'value')])
        def controls_listener(size, frequency):
            self.sample_size = size
            self.sample_frequency = frequency

            return 'Size of data: {}'.format(len(self.data.index))
