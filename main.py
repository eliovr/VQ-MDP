import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from sampling import PandasSampler, SparkSampler
from quantization import MiniBatchKMeans
from projection import Scatterplot, ParallelCoordinates, ReachabilityPlot

from pyspark.sql import SparkSession
from pyspark.sql.types import *

import pandas as pd
import numpy as np
import json, time


# data_path = '/home/elio/datasets/expressions_parquet'
# data_path = '/home/elio/datasets/gas_sensor_parquet'
# data_path = '/home/elio/datasets/biometrics_parquet'
# data_path = '/home/elio/datasets/gsod_parquet'
# data_path = '/home/elio/datasets/chembl_parquet_scaled'
# data = spark.read.parquet(data_path).drop('hba', 'hbd', 'hba_lipinski', 'hdb_lipinski', 'num_lipinski_ro5_violations', 'rtb', 'num_ro5_violations', 'num_alerts')

data_path = '/home/elio/datasets/emtab6961_parquet'
use_spark = True
data = None
sampler = None
exe_time = .0

if use_spark:
    spark = SparkSession.builder.appName("VQ-MDP").getOrCreate()
    # spark = SparkSession.builder.config('spark.sql.codegen.wholeStage', 'false').appName("VQ-MDP").getOrCreate()
    data = spark.read.parquet(data_path)
    sampler = SparkSampler(data=data)
else:
    data = pd.read_csv(data_path)
    data = data[data.columns[1:-1]]
    sampler = PandasSampler(data=data)

quantizer = MiniBatchKMeans()
scatterplot = Scatterplot()
parcoor = ParallelCoordinates()
reachplot = ReachabilityPlot()

class States:
    STOP = 0
    TRAIN_UPDATE_LOOP = 1
    TRAIN_UPDATE_ONCE = 2
    UPDATE_STOP = 3


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

    html.Div(className='row form-inline col-12 mb-1', children=[
            html.Div(className='btn-group mr-3', children=[
                html.Button('Reset', id='button-reset', type='button', className='btn btn-secondary', n_clicks=0),
                html.Button('Run', id='button-run', type='button', className='btn btn-light', n_clicks=0)
            ]),
            sampler.controls, html.Small(id='sampler-state', className='mx-1 px-1 border', children=' - '),
            quantizer.controls, html.Small(id='quantizer-state', className='mx-1 px-1 border', children=' - '),
            html.Small(id='message-board', className='px-1 ml-2 float-right', children='')
    ]),

    html.Div(id='graphs', className='row', children=[
        html.Div(className='col-6', children=[
            dcc.Graph(id='scatterplot-graph', figure={'layout': scatterplot.layout}),
            scatterplot.controls, html.Small(id='scatterplot-state', className='mx-1 px-1 border', children=' - ')
        ]),

        html.Div(className='col-6', children=[
            reachplot.controls, html.Small(id='reachplot-state', className='mx-1 px-1 border', children=' - '),
            dcc.Graph(id='reachplot-graph', style={'margin': '5px 0 -40px 0'}, figure={'layout': reachplot.layout}),
            dcc.Graph(id='parcoor-graph', className="width-750", figure={'layout': parcoor.layout})
        ])
    ]),

    # html.Button('Explain', id='button-explain', type='button', className='btn btn-warning', n_clicks=0),
    # html.Div(id='explanation-board', children='', className='alert alert-warning small'),

    # html.Div(id='state', className='col-12', style={'display': 'none'}, children=[
    #     html.Div(id='message-board', children='', className='alert alert-warning small')
    # ])
])

sampler.register_listener(app)
quantizer.register_listener(app)
scatterplot.register_listener(app)
parcoor.register_listener(app)
reachplot.register_listener(app)


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
    global exe_time
    if clicks > 0:
        quantizer.reset()
        sampler.reset()
        exe_time = .0
        return -1

    return dash.no_update


# @app.callback(
#     Output('explanation-board', 'children'),
#     [Input('button-explain', 'n_clicks')],
#     [State('interaction-state', 'children')])
# def btn_reset_clicked(clicks, selection):
#     selected_data = json.loads(selection)
#     explanation = ''
#
#     if len(selected_data) > 0:
#         X = quantizer.get_prototypes()
#         y = np.arange(0, len(X))
#         y = np.isin(y, selected_data)
#         # explanation = 'y: {}, s: {}'.format(y, selected_data)
#         model = RuleListClassifier(max_iter=1000, class1label="selected", verbose=False)
#         model.fit(X, y)
#         explanation = model
#
#     return explanation

@app.callback(
    [Output('scatterplot-graph', 'figure'),
    Output('reachplot-graph', 'figure'),
    Output('parcoor-graph', 'figure'),
    Output('training-state-container', 'children'),
    Output('message-board', 'children'),

    Output('sampler-state', 'children'),
    Output('quantizer-state', 'children'),
    Output('scatterplot-state', 'children'),
    Output('reachplot-state', 'children')],

    [Input('hidden-state', 'children'),
    Input('interaction-state', 'children')])
def traing_and_update(hidden_state, selection):
    global exe_time
    next_state = dash.no_update
    sp_graph = dash.no_update
    rp_graph = dash.no_update
    pc_graph = dash.no_update
    sampler_state = dash.no_update
    quantizer_state = dash.no_update
    sp_state = dash.no_update
    rp_state = dash.no_update
    state_msg = ''
    selected_data = []
    cols = data.columns

    if selection:
        selected_data = json.loads(selection)

    if (hidden_state in (States.TRAIN_UPDATE_LOOP, States.TRAIN_UPDATE_ONCE)
        or (len(sampler.spawns) > 0 and sampler.sample_count == 0)):

        start = time.time()
        x, sampler_state = sampler.sample()
        m, quantizer_state = quantizer.learn(x)
        sp_graph, sp_state = scatterplot.visualize(m, selected_data=selected_data)
        rp_graph, rp_state = reachplot.visualize(m, selected_data=selected_data)
        pc_graph, _ = parcoor.visualize(m, selected_data=selected_data, column_names=cols)
        exe_time += time.time() - start

        if hidden_state == States.TRAIN_UPDATE_LOOP:
            next_state = html.Div(id='hidden-state', children=States.TRAIN_UPDATE_LOOP)

    # update graph in case of, e.g., user lasso interaction.
    elif sampler.sample_count > 0:
        sp_graph, sp_state = scatterplot.visualize(selected_data=selected_data)
        rp_graph, rp_state = reachplot.visualize(selected_data=selected_data)
        pc_graph, _ = parcoor.visualize(selected_data=selected_data, column_names=cols)

    if exe_time > 0:
        state_msg =  'Avg time (all): {:10.2f}s'.format(exe_time / sampler.sample_count)

    return (sp_graph, rp_graph, pc_graph, next_state,
        state_msg,
        sampler_state,
        quantizer_state,
        sp_state,
        rp_state)


@app.callback(
    [Output('interaction-state', 'children'),
    Output('dragmode-state', 'children')],
    [Input('scatterplot-graph', 'selectedData'),
    Input('scatterplot-graph', 'relayoutData'),
    Input('reachplot-graph', 'selectedData'),
    Input('parcoor-graph', 'restyleData')],
    [State('dragmode-state', 'children'),
    State('interaction-state', 'children')])
def user_interaction(lasso_selection, zoom_selection, rp_selection, parcoor_filter, dragmode, selection_state):
    selected_data = '[]'
    dragmode_state = dash.no_update

    # user made a lasso selection in the scatterplot.
    if lasso_selection:
        selection = [p['customdata'] for p in lasso_selection['points']]
        selected_data = json.dumps(selection)

    # user made a lasso selection in the reachabilityplot.
    elif rp_selection:
        selection = [p['customdata'] for p in rp_selection['points']]
        selected_data = json.dumps(selection)

    elif zoom_selection:
        # user is changing interaction tool in scatterplot.
        if 'dragmode' in zoom_selection:
            if zoom_selection['dragmode'] != dragmode:
                dragmode_state = zoom_selection['dragmode']
            selected_data = dash.no_update

        # user is zooming in or out.
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
                parcoor.visualize(m)

    return selected_data, dragmode_state


if __name__ == '__main__':
    app.run_server(debug=True)
