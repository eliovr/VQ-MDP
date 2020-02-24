import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from sampling import Sampler
from quantization import MiniBatchKMeans
from projection import Scatterplot, ParallelCoordinates, ReachabilityPlot

import pandas as pd
import numpy as np
import json


class States:
    STOP = 0
    TRAIN_UPDATE_LOOP = 1
    TRAIN_UPDATE_ONCE = 2
    UPDATE_STOP = 3

data = pd.read_csv('/home/elio/apps/notebooks/data/isolet.csv', header=None)
sampler = Sampler(data=data)
quantizer = MiniBatchKMeans()
scatterplot = Scatterplot()
parcoor = ParallelCoordinates()
reachplot = ReachabilityPlot()


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
    html.Div(id='hidden-states', style={'display': 'none'}, children=[
        html.Div(id='training-state-container', children=html.Div(id='hidden-state', children=States.STOP)),
        html.Div(id='interaction-state', children='[]'),
        html.Div(id='dragmode-state', children='lasso'),
    ]),

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
            dcc.Graph(id='scatterplot-graph', figure={'layout': scatterplot.layout})]),
        html.Div(className='col-7', children=[
            dcc.Graph(id='parcoor-graph', figure={'layout': reachplot.layout})])
        # html.Div(className='col-7', children=[
        #     dcc.Graph(id='parcoor-graph', figure={'layout': parcoor.layout})])
    ])
])

sampler.register_listener(app)
quantizer.register_listener(app)
scatterplot.register_listener(app)
# parcoor.register_listener(app)
reachplot.register_listener(app)

def state_message():
    return 'Sampler: {} | Quantizer: {} | Projection: {}'.format(sampler.state, quantizer.state, scatterplot.state)

@app.callback(
    [Output('button-run', 'children'),
    Output('hidden-state', 'children')],
    [Input('button-run', 'n_clicks')],
    [State('hidden-state', 'children')])
def btn_run_clicked(clicks, hidden_state):
    if clicks == -1:
        return 'Run', States.TRAIN_UPDATE_ONCE
    elif hidden_state in (States.STOP, States.TRAIN_UPDATE_ONCE):
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
    Output('training-state-container', 'children'),
    Output('message-board', 'children')],
    [Input('hidden-state', 'children'),
    Input('interaction-state', 'children')])
def traing_and_update(hidden_state, selection):
    next_state = dash.no_update
    sp_graph = dash.no_update
    pc_graph = dash.no_update
    sp_state = scatterplot.get_state()
    pc_state = parcoor.get_state()
    selected_data = []

    if selection: selected_data = json.loads(selection)

    if (hidden_state in (States.TRAIN_UPDATE_LOOP, States.TRAIN_UPDATE_ONCE)
        or (len(sampler.spawns) > 0 and sampler.sample_count == 0)):
        x = sampler.sample()
        m = quantizer.learn(x)
        sp_graph, sp_state = scatterplot.visualize(m, selected_data=selected_data)
        # pc_graph, pc_state = parcoor.visualize(m, selected_data=selected_data)
        pc_graph, pc_state = reachplot.visualize(m, selected_data=selected_data)

        if hidden_state == States.TRAIN_UPDATE_LOOP:
            next_state = html.Div(id='hidden-state', children=States.TRAIN_UPDATE_LOOP)

    # update graph in case of, e.g., user lasso interaction.
    elif sampler.sample_count > 0:
        sp_graph, sp_state = scatterplot.visualize(selected_data=selected_data)
        pc_graph, pc_state = reachplot.visualize(selected_data=selected_data)
        # pc_graph, pc_state = parcoor.visualize(selected_data=selected_data)

    state = 'Samples: {}, Old state: {}, new state: {}'.format(sampler.sample_count, hidden_state, next_state)

    return sp_graph, pc_graph, next_state, state


@app.callback(
    [Output('interaction-state', 'children'),
    Output('dragmode-state', 'children')],
    [Input('scatterplot-graph', 'selectedData'),
    Input('scatterplot-graph', 'relayoutData'),
    Input('parcoor-graph', 'restyleData')],
    [State('dragmode-state', 'children'),
    State('interaction-state', 'children')])
def user_interaction(lasso_selection, zoom_selection, parcoor_filter, dragmode, selection_state):
    selected_data = dash.no_update
    dragmode_state = dash.no_update

    # user made a lasso selection.
    if lasso_selection:
        selection = [p['customdata'] for p in lasso_selection['points']]
        selected_data = json.dumps(selection)

    elif zoom_selection:
        # user is changing interaction tool in scatterplot.
        if 'dragmode' in zoom_selection:
            if zoom_selection['dragmode'] != dragmode:
                dragmode_state = zoom_selection['dragmode']

        # user is zooming in.
        elif dragmode != 'zoomed-in':
            selected_data = '[]'
            dragmode_state = 'zoomed-in'

            # user is zooming in.
            if 'xaxis.range[0]' in zoom_selection:
                selected_ks = scatterplot.selected_ids(zoom_selection)
                zoomed_data = quantizer.predict_select(sampler.data, selected_ks)
                sampler.spawn(zoomed_data)
                quantizer.spawn()

            # user is zooming out.
            elif 'xaxis.autorange' in zoom_selection:
                sampler.unspawn()
                quantizer.unspawn()
                m = quantizer.get_prototypes()
                scatterplot.visualize(m)
                reachplot.visualize(m)
                # parcoor.visualize(m)

    return selected_data, dragmode_state


if __name__ == '__main__':
    app.run_server(debug=True)
