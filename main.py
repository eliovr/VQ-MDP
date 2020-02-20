import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output

from sampling import Sampler
from quantization import MiniBatchKMeans
from projection import Scatterplot, ParallelCoordinates

import pandas as pd
import numpy as np


data = pd.read_csv('/home/elio/apps/notebooks/data/isolet.csv', header=None)
# sampler = Sampler('/home/elio/apps/notebooks/data/isolet.csv')
sampler = Sampler(data=data)
quantizer = MiniBatchKMeans()
scatterplot = Scatterplot()
parcoor = ParallelCoordinates()


external_scripts = [{
    'src': 'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js',
    'integrity': 'sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6',
    'crossorigin': 'anonymous'
}]
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

app.layout = html.Div(
    html.Div(className='container', style={'margin-top': '10px'}, children=[
        html.Div(id='hidden-state-container', children=html.Div(id='hidden-state', children=0), style={'display': 'none'}),
        html.Div(className='row', children=[
            html.Div(id='controls', className='col-4', children=[
                html.Div(className='btn-group btn-block', children=[
                    html.Button('Reset', id='button-reset', type='button', className='btn btn-light', n_clicks=0),
                    html.Button('Run', id='button-run', type='button', className='btn btn-secondary', n_clicks=0)
                ]),
                sampler.controls,
                quantizer.controls,
                scatterplot.controls
            ]),
            html.Div(id='graphs', className='col-8', children=[
                dcc.Graph(id='scatterplot-graph', figure={'layout': scatterplot.layout}),
                dcc.Graph(id='parcoor-graph', figure={'layout': parcoor.layout}),]),
            html.Div(id='message-board', children='', className='alert alert-warning small')
        ])
    ]))


sampler.register_listener(app)
quantizer.register_listener(app)
scatterplot.register_listener(app)
parcoor.register_listener(app)

def state_message():
    return '{}; {}; {}'.format(sampler.state(), quantizer.state(), scatterplot.state())

@app.callback(
    [Output('button-run', 'children'),
    Output('hidden-state', 'children')],
    [Input('button-run', 'n_clicks')])
def btn_run_clicked(clicks):
    if clicks == 0 or clicks % 2 == 0: # paused / idle.
        return 'Run', 0
    else:
        return 'Pause', 1


@app.callback(
    Output('button-run', 'n_clicks'),
    [Input('button-reset', 'n_clicks')])
def btn_reset_clicked(clicks):
    quantizer.reset()
    sampler.reset()
    return 0


@app.callback(
    [Output('scatterplot-graph', 'figure'),
    Output('parcoor-graph', 'figure'),
    Output('hidden-state-container', 'children'),
    Output('message-board', 'children')],
    [Input('hidden-state', 'children'),
    Input('parcoor-graph', 'restyleData'), # [{'dimensions[17].constraintrange': [[-0.9162413986093053, -0.5785513652552432]]}, [0]]
    Input('scatterplot-graph', 'selectedData'),
    Input('scatterplot-graph', 'relayoutData')])
def training_loop(is_running, pc_selected_data, sp_selected_data, sp_relayout_data):
    global sampler, quantizer

    sp_selected = []
    pc_selected = []
    selected_data = []
    is_lasso = False
    is_zooming = False
    selected_ks = None

    if sp_selected_data:
        sp_selected = [int(p['customdata']) for p in sp_selected_data['points']]
        selected_data = sp_selected
        if sp_selected != scatterplot.selected_data:
            is_lasso = True

    if (sp_relayout_data
        and 'xaxis.range[0]' in sp_relayout_data
        and sp_relayout_data != sampler.relayout):

        x1 = sp_relayout_data.get('xaxis.range[0]', 0)
        y1 = sp_relayout_data.get('yaxis.range[0]', 0)
        x2 = sp_relayout_data.get('xaxis.range[1]', 0)
        y2 = sp_relayout_data.get('yaxis.range[1]', 0)

        xs = scatterplot.data[0]
        ys = scatterplot.data[1]
        ks = np.arange(0, len(xs))
        selected_ks = ks[(xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)]

        predictions = quantizer.predict(data).reshape(len(data), 1)
        k_data = np.append(predictions, data.values, axis=1)
        k_data = pd.DataFrame(k_data)

        zoomed_data = k_data.loc[k_data[0].isin(selected_ks)].drop(0, axis=1)
        sampler.spawn(zoomed_data, sp_relayout_data)
        quantizer.spawn()
        is_zooming = True

    # training is running.
    if is_running == 1 or is_zooming:
        x = sampler.sample()
        m = quantizer.learn(x)
        v1 = scatterplot.visualize(m, selected_data)
        v2 = parcoor.visualize(m, selected_data)

        if is_zooming:
            return v1, v2, html.Div(id='hidden-state', children=0), state_message()
        else:
            return v1, v2, html.Div(id='hidden-state', children=1), state_message()

    # a selection happened when not training.
    elif is_lasso:
        v1 = scatterplot.update(selected_data)
        v2 = parcoor.update(selected_data)
        return v1, v2, html.Div(id='hidden-state', children=0), state_message()

    else:
        raise PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True)
