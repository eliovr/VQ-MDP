import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from sampling import Sampler
from quantization import MiniBatchKMeans
from projection import Scatterplot, ParallelCoordinates

import pandas as pd
import numpy as np


class States:
    STOP = 0
    TRAIN_UPDATE_LOOP = 1
    TRAIN_UPDATE_STOP = 2
    UPDATE_STOP = 3

data = pd.read_csv('/home/elio/apps/notebooks/data/isolet.csv', header=None)
sampler = Sampler(data=data)
quantizer = MiniBatchKMeans()
scatterplot = Scatterplot()
parcoor = ParallelCoordinates()


external_scripts = [
    {
        'src': 'https://code.jquery.com/jquery-3.4.1.slim.min.js',
        'integrity': 'sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n',
        'crossorigin': 'anonymous'
    },{
        'src': 'https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js',
        'integrity': 'sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo',
        'crossorigin': 'anonymous'
    },{
        'src': 'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js',
        'integrity': 'sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6',
        'crossorigin': 'anonymous'
    }
]
external_stylesheets = [{
    'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css',
    'rel': 'stylesheet',
    'integrity': 'sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh',
    'crossorigin': 'anonymous'
}]

app = dash.Dash(__name__,
                external_scripts=external_scripts,
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

app.layout = html.Div(className='container', style={'margin-top': '10px'}, children=[
        html.Div(id='hidden-state-container', children=html.Div(id='hidden-state', children=States.STOP), style={'display': 'none'}),
        html.Div(id='hidden-interaction-state', children=States.STOP, style={'display': 'none'}),
        html.Div(className='row', children=[
            html.Div(id='controls', className='form-inline col-5', children=[
                html.Div(className='btn-group', children=[
                    html.Button('Reset', id='button-reset', type='button', className='btn btn-light', n_clicks=0),
                    html.Button('Run', id='button-run', type='button', className='btn btn-secondary', n_clicks=0)
                ]),
                sampler.controls, quantizer.controls, scatterplot.controls
            ]),
            html.Div(id='state', className='col-7', children=[
                html.Div(id='message-board', children='', className='alert alert-warning small')
            ])
        ]),
        html.Div(id='graphs', className='row', children=[
            html.Div(className='col-5', children=[
                dcc.Graph(id='scatterplot-graph', figure={'layout': scatterplot.layout}),
            ]),
            html.Div(className='col-7', children=[
                dcc.Graph(id='parcoor-graph', figure={'layout': parcoor.layout})
            ])
        ])
    ])

sampler.register_listener(app)
quantizer.register_listener(app)
scatterplot.register_listener(app)
parcoor.register_listener(app)

def state_message():
    return 'Sampler: {} | Quantizer: {} | Projection: {}'.format(sampler.state, quantizer.state, scatterplot.state)

@app.callback(
    [Output('button-run', 'children'),
    Output('hidden-state', 'children')],
    [Input('button-run', 'n_clicks')],
    [State('hidden-state', 'children')])
def btn_run_clicked(clicks, hidden_state):
    if clicks == -1:
        return 'Run', States.TRAIN_UPDATE_STOP
    elif hidden_state in (States.STOP, States.TRAIN_UPDATE_STOP):
        return 'Pause', States.TRAIN_UPDATE_LOOP
    else:
        return 'Run', States.STOP

@app.callback(
    Output('button-run', 'n_clicks'),
    [Input('button-reset', 'n_clicks')])
def btn_reset_clicked(clicks):
    if clicks > 0:
        quantizer.reset()
        sampler.reset()
        return -1

    return dash.no_update

@app.callback(
    [Output('scatterplot-graph', 'figure'),
    Output('parcoor-graph', 'figure'),
    Output('hidden-state-container', 'children'),
    Output('message-board', 'children')],
    [Input('hidden-state', 'children'),
    Input('hidden-interaction-state', 'children')])
def traing_and_update(hidden_state, blabla):
    next_state = dash.no_update
    sp_graph = dash.no_update
    pc_graph = dash.no_update

    if (hidden_state in (States.TRAIN_UPDATE_LOOP, States.TRAIN_UPDATE_STOP)
        or (len(sampler.spawns) > 0 and sampler.sample_count == 0)):
        x = sampler.sample()
        m = quantizer.learn(x)
        sp_graph = scatterplot.visualize(m)
        pc_graph = parcoor.visualize(m)

        if hidden_state == States.TRAIN_UPDATE_LOOP:
            next_state = html.Div(id='hidden-state', children=States.TRAIN_UPDATE_LOOP)

    elif sampler.sample_count > 0:
        sp_graph = scatterplot.update()
        pc_graph = parcoor.update()

    return sp_graph, pc_graph, next_state, 'Samples: {}, Old state: {}, new state: {}, level: {}'.format(sampler.sample_count, hidden_state, next_state, len(sampler.spawns))


@app.callback(
    Output('hidden-interaction-state', 'children'),
    [Input('scatterplot-graph', 'selectedData'),
    Input('scatterplot-graph', 'relayoutData'),
    Input('parcoor-graph', 'restyleData')])
def user_interaction(lasso_selection, zoom_selection, parcoor_filter):
    if lasso_selection:
        selected_data = [p['customdata'] for p in lasso_selection['points']]
        scatterplot.selected_data = selected_data
        parcoor.selected_data = selected_data
        return 0

    elif zoom_selection:
        if 'dragmode' in zoom_selection and len(scatterplot.selected_data) == 0:
            raise PreventUpdate

        elif not 'dragmode' in zoom_selection:
            scatterplot.selected_data = []
            parcoor.selected_data = []

            if 'xaxis.range[0]' in zoom_selection:
                selected_ks = scatterplot.selected_ids(zoom_selection)
                zoomed_data = quantizer.predict_select(sampler.data, selected_ks)
                sampler.spawn(zoomed_data)
                quantizer.spawn()

            elif 'xaxis.autorange' in zoom_selection:
                sampler.unspawn()
                quantizer.unspawn()
                m = quantizer.get_prototypes()
                scatterplot.visualize(m)
                parcoor.visualize(m)

            return 0

    else:
        return 0


if __name__ == '__main__':
    app.run_server(debug=True)
