import time

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output

# import pandas as pd

class Sampler:
    def __init__(self, data):
        # ---- State
        self.spawns = []
        self.data = data
        self.relayout = None
        self.sample_count = 0
        self.sampling_time = 0

        # ---- Parameters
        self.sample_size = 1000
        self.sample_frequency = .1

        self.controls = html.Div(id='sampler-controls', style={'margin': '10px'}, children=[
            html.Div(className='card', children=[
                html.Div(className='card-header', children='Sampler'),
                html.Div(className='card-body', children=[
                    html.Div(className='form-group', children=[
                        html.Label('Sample size (thousand points)', className='small'),
                        dcc.Slider(id='sample-size', value=self.sample_size,
                            min=50, max=10000, step=50,
                            marks={x: '{}'.format(x/1000) for x in range(0, 12000, 2000)})
                    ]),
                    html.Div(className='form-group', children=[
                        html.Label('Sample frequency (seconds)', className='small'),
                        dcc.Slider(id='sample-frequency', value=self.sample_frequency,
                            min=0, max=1, step=.05,
                            marks={x: '{}'.format(x) for x in range(0, 2)})
                    ]),
                    html.Div(id='sampler-message', children='', className='alert alert-warning small', style={'display': 'none'})
                ]),
            ])
        ])


    def sample(self):
        if self.sample_frequency > 0:
            time.sleep(self.sample_frequency)

        if len(self.data.index) > self.sample_size:
            start = time.time()
            x = self.data.sample(n=self.sample_size).values
            self.sampling_time += time.time()
            self.sample_count += 1
            return x
        else:
            return self.data.values

    def state(self):
        return 'Samples taken: {}'.format(self.sample_count)

    def reset(self):
        if len(self.spawns) > 0:
            self.unspawn()

    def spawn(self, data, relayoutData):
        self.spawns.append(dict(
            data = self.data,
            relayout = self.relayout
        ))

        self.data = data
        self.relayout = relayoutData
        return self

    def unspawn(self):
        if len(self.spawns) > 0:
            parent = self.spawns.pop()
            self.data = parent['data']
            self.relayout = parent['relayout']
        return self

    def register_listener(self, app):
        @app.callback(Output('sampler-message', 'children'),
            [Input('sample-size', 'value'),
            Input('sample-frequency', 'value')])
        def controls_listener(size, frequency):
            self.sample_size = size
            self.sample_frequency = frequency

            return 'Size of data: {}'.format(len(self.data.index))
