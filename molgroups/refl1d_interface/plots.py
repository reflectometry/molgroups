"""Plots for the Refl1D webview interface
    Used with MolgroupsExperiment.register_webview_plot
"""

import concurrent.futures
import csv
import dill
import io
import multiprocessing
import time
import numpy as np
import plotly.graph_objs as go

from bumps.dream.state import MCMCDraw
from bumps.dream.stats import credible_interval
from bumps.webview.server.custom_plot import CustomWebviewPlot
from bumps.plotutil import form_quantiles
from refl1d.names import FitProblem, Experiment
from refl1d.webview.server.colors import COLORS

from .layers import MolgroupsLayer

def hex_to_rgb(hex_string):
    r_hex = hex_string[1:3]
    g_hex = hex_string[3:5]
    b_hex = hex_string[5:7]
    return int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)

def exportdata_to_csv(export_data: dict) -> str:
    # Write CSV data for export
    with io.StringIO() as f:
        writer = csv.DictWriter(f, fieldnames=export_data.keys())
        writer.writeheader()
        for i in range(len(export_data.get('z', []))):
            writer.writerow({k: v[i] for k, v in export_data.items()})

        return f.getvalue()

# =============== CVO plot ================
def cvo_plot(layer: MolgroupsLayer, model: Experiment | None = None, problem: FitProblem | None = None):
    # component volume occupancy plot

    # compile moldat
    moldat = {}
    group_names = {}
    for group in [layer.base_group] + layer.add_groups + layer.overlay_groups:
        for k, v in group._group_names.items():
            # propagate frac_replacement to subgroups
            for gp in v:
                group._stored_profile[gp]['frac_replacement'] = group._stored_profile['frac_replacement']

            # merge group names
            if k not in group_names.keys():
                group_names[k] = []
            group_names[k] += v

        
        moldat.update(group._stored_profile)

    # define normarea
    normarea = layer.base_group._stored_profile['normarea']

    fig = go.Figure()
    traces = []
    export_data = {}
    MOD_COLORS = COLORS[1:]
    color_idx = 1
    sumarea = 0
    for lbl, item in group_names.items():
        area = 0
        for gp in item:
            if gp in moldat.keys():
                zaxis = moldat[gp]['zaxis']
                newarea = moldat[gp]['area'] / moldat[gp]['frac_replacement']
                area += np.maximum(0, newarea)
            else:
                print(f'Warning: {gp} not found')

        export_data.setdefault('z', zaxis)
        color = MOD_COLORS[color_idx % len(MOD_COLORS)]
        plotly_color = ','.join(map(str, hex_to_rgb(color)))
        traces.append(go.Scatter(x=zaxis,
                                y=area / normarea,
                                legendgroup=lbl,
                                mode='lines',
                                name=lbl,
                                line=dict(color=color)))
        export_data[lbl] = area / normarea
        traces.append(go.Scatter(x=zaxis,
                                y=area / normarea,
                                legendgroup=lbl,
                                mode='lines',
                                line=dict(width=0),
                                fill='tozeroy',
                                fillcolor=f'rgba({plotly_color},0.3)',
                                showlegend=False
                                ))
        color_idx += 1
        sumarea += area

    color = COLORS[0]
    plotly_color = ','.join(map(str, hex_to_rgb(color)))
    
    traces.append(go.Scatter(x=zaxis,
                                y=sumarea / normarea,
                                mode='lines',
                                name='buffer',
                                legendgroup='buffer',
                                line=dict(color=color)))
    export_data['buffer'] = sumarea/ normarea
    traces.append(go.Scatter(x=zaxis,
                                y=sumarea / normarea,
                                mode='lines',
                                line=dict(width=0),
                                legendgroup='buffer',
                                fill='tonexty',
                                fillcolor=f'rgba({plotly_color},0.3)',
                                showlegend=False
                                ))    
    traces.append(go.Scatter(x=zaxis,
                                y=[1.0] * len(zaxis),
                                legendgroup='buffer',
                                mode='lines',
                                line=dict(color=color, width=0),
                                showlegend=False))

    
    fig.add_traces(traces[::-1])

    fig.update_layout(
        title='Component Volume Occupancy',
        template = 'plotly_white',
        xaxis_title=dict(text='z (\u212b)'),
        yaxis_title=dict(text='volume occupancy'),
        legend=dict(yanchor='top',
                    xanchor='right',
                    x=0.99,
                    y=0.99),
        yaxis_range=[0, 1]
    )

    return CustomWebviewPlot(fig_type='plotly',
                                plotdata=fig,
                                exportdata=exportdata_to_csv(export_data))

# =============== Uncertainty plot ================
# parallelization code adapted from refl1d.errors
def _initialize_worker(shared_serialized_problem, model_index, layer_name):

    global _shared_problem
    _shared_problem = dill.loads(shared_serialized_problem[:])

    global _layer_name
    _layer_name = layer_name

    global _model_index
    _model_index = model_index

_shared_problem = None  # used by multiprocessing pool to hold problem

def _worker_eval_plot_point(point):
    return _calc_profile(_shared_problem, _model_index, _layer_name, point)

def _calc_profile(problem: FitProblem | None, model_index: int, layer_name: str, pt: np.ndarray | list) -> tuple[dict, float]:

    problem.setp(pt)
    model: Experiment = list(problem.models)[model_index]
    model.update()
    model.nllf()
    if hasattr(model, 'parts'):
        for p in model.parts:
            if hasattr(p.sample, 'molgroups_layer'):
                if p.sample.molgroups_layer.name == layer_name:
                    layer = p.sample.molgroups_layer
                    break
    else:
        layer = model.sample.molgroups_layer
    imoldat = {}
    for group in [layer.base_group] + layer.add_groups + layer.overlay_groups:
        for k, v in group._group_names.items():
            for gp in v:
                group._stored_profile[gp]['frac_replacement'] = group._stored_profile['frac_replacement']

        imoldat.update(group._stored_profile)

    normarea = layer.base_group._stored_profile['normarea']

    return imoldat, normarea

def cvo_uncertainty_plot(layer: MolgroupsLayer, model: Experiment | None = None, problem: FitProblem | None = None, state: MCMCDraw | None = None, n_samples: int = 50):

    # TODO: allow groups to label some items as uncertainty groups and use the median or best for others

    if state is None:
        return cvo_plot(layer, model, problem)
    
    fig = go.Figure()
    traces = []
    uncertainty_traces = []
    MOD_COLORS = COLORS[1:]
    color_idx = 1
    sumarea = 0

    group_names = {}
    for group in [layer.base_group] + layer.add_groups + layer.overlay_groups:
        for k, v in group._group_names.items():
            # merge group names
            if k not in group_names.keys():
                group_names[k] = []
            group_names[k] += v

    statdata: dict[str, list[np.ndarray]] = {lbl: [] for lbl in group_names.keys()}
    statnormarea = []
    print('Starting CVO uncertainty calculation...')
    init_time = time.time()

    # condition the points draw (adapted from bumps.errplot.calc_errors_from_state)
    points = state.draw().points
    if points.shape[0] < n_samples:
        n_samples = points.shape[0]
    points = points[np.random.permutation(len(points) - 1)]
    points = points[-n_samples:-1]
    #print('\n'.join(['%i\t%s' % a for a in enumerate(state.labels)]))

    # set up a parallel calculation
    model_index = list(problem.models).index(model)

    with multiprocessing.Manager() as manager:
        shared_serialized_problem = manager.Array("B", dill.dumps(problem))
        #args = [(shared_serialized_problem, point) for point in points]

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=None, initializer=_initialize_worker, initargs=(shared_serialized_problem, model_index, layer.name)
        ) as executor:
            results = executor.map(_worker_eval_plot_point, points)

    for (imoldat, normarea) in results:
        for lbl, item in group_names.items():
            area = 0
            for gp in item:
                if gp in imoldat.keys():
                    zaxis = imoldat[gp]['zaxis']    
                    newarea = imoldat[gp]['area'] / imoldat[gp]['frac_replacement']
                    area += np.maximum(0, newarea)
            statdata[lbl].append(area / normarea)
        statnormarea.append(normarea)
    print(f'CVO uncertainty calculation done after {time.time() - init_time} seconds')

    export_data = {'z': zaxis}

    for lbl, statlist in statdata.items():
        color = MOD_COLORS[color_idx % len(MOD_COLORS)]
        plotly_color = ','.join(map(str, hex_to_rgb(color)))
        med_area = np.median(statlist, axis=0)
        med_norm_area = np.median(statnormarea)
        #print(lbl, med_area.shape, zaxis.shape)
        _, q = form_quantiles(statlist, (68,))
        for lo, hi in q:
            uncertainty_traces.append(go.Scatter(x=zaxis,
                                    y=lo, # * med_norm_area / normarea,
                                    mode='lines',
                                    legendgroup=lbl,
                                    line=dict(color=color, width=1),
                                    hoverinfo="skip",
                                    fill='tonexty',
                                    fillcolor=f'rgba({plotly_color},0.3)',
                                    showlegend=False
                                    ))
            export_data[lbl + ' lower 68p CI'] = lo
            export_data[lbl + ' upper 68p CI'] = hi
            uncertainty_traces.append(go.Scatter(x=zaxis,
                                    y=hi, # * med_norm_area / normarea,
                                    showlegend=False,
                                    legendgroup=lbl, 
                                    opacity=0.3,
                                    hoverinfo="skip",
                                    mode='lines',
                                    name=lbl,
                                    line=dict(color=color, width=1)))
        uncertainty_traces.append(go.Scatter(x=zaxis,
                            y=med_area, # * med_norm_area / normarea,
                            mode='lines',
                            name=lbl,
                            legendgroup=lbl,
                            line=dict(color=color)))
        export_data[lbl] = med_area

        color_idx += 1
        sumarea += med_area * med_norm_area

    """
    for lbl, item in group_names.items():
        area = 0
        for gp in item:
            if gp in moldat.keys():
                zaxis = moldat[gp]['zaxis']
                area += np.maximum(0, moldat[gp]['area'])
            else:
                print(f'Warning: {gp} not found')

        color = MOD_COLORS[color_idx % len(MOD_COLORS)]
        plotly_color = ','.join(map(str, hex_to_rgb(color)))
        traces.append(go.Scatter(x=zaxis,
                                 y=area / normarea,
                                 mode='lines',
                                 name=lbl,
                                 line=dict(color=color)))
        traces.append(go.Scatter(x=zaxis,
                                 y=area / normarea,
                                 mode='lines',
                                 line=dict(width=0),
                                 fill='tozeroy',
                                 fillcolor=f'rgba({plotly_color},0.3)',
                                 showlegend=False
                                 ))
        color_idx += 1
        sumarea += area
    """
    color = COLORS[0]
    plotly_color = ','.join(map(str, hex_to_rgb(color)))
    
    buffer_traces = []

    buffer_traces.append(go.Scatter(x=zaxis,
                                y=sumarea / med_norm_area,
                                mode='lines',
                                name='buffer',
                                legendgroup='buffer',
                                line=dict(color=color)))
    export_data['buffer'] = sumarea / med_norm_area
    buffer_traces.append(go.Scatter(x=zaxis,
                                y=sumarea / med_norm_area,
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor=f'rgba({plotly_color},0.3)',
                                legendgroup='buffer',
                                showlegend=False
                                ))    
    buffer_traces.append(go.Scatter(x=zaxis,
                                y=[1.0] * len(zaxis),
                                mode='lines',
                                line=dict(color=color, width=0),
                                legendgroup='buffer',
                                showlegend=False))

    
    fig.add_traces((traces + uncertainty_traces + buffer_traces)[::-1])

    fig.update_layout(
        title='Component Volume Occupancy with Uncertainty',
        template = 'plotly_white',
        xaxis_title=dict(text='z (\u212b)'),
        yaxis_title=dict(text='volume occupancy'),
        legend=dict(yanchor='top',
                    y=0.98,
                    xanchor='right',
                    x=0.99),
        yaxis_range=[0, 1]
    )

    return CustomWebviewPlot(fig_type='plotly',
                             plotdata=fig,
                             exportdata=exportdata_to_csv(export_data))

# ============= Results table =============
# parallelization code adapted from refl1d.errors

def _worker_eval_table_point(point):
    return _calc_stats(_shared_problem, _model_index, _layer_name, point)

def _calc_stats(problem: FitProblem | None, model_index: int, layer_name: str, pt: np.ndarray | list) -> dict:

    problem.setp(pt)
    model: Experiment = list(problem.models)[model_index]
    model.update()
    model.nllf()
    if hasattr(model, 'parts'):
        layer = None
        for p in model.parts:
            if hasattr(p.sample, 'molgroups_layer'):
                if p.sample.molgroups_layer.name == layer_name:
                    layer: MolgroupsLayer = p.sample.molgroups_layer
                    break
        if layer is None:
            raise ValueError(f'No molgroups layer found with name {layer_name}')
    else:
        layer: MolgroupsLayer = model.sample.molgroups_layer
    iresults = {'parameters': {k: v for k, v in zip(problem.labels(), pt)}}
    for group in [layer.base_group] + layer.add_groups + layer.overlay_groups:
        iresults = group._molgroup.fnWriteResults2Dict(iresults, group.name)
        iresults[group.name].update(group._stored_profile['referencepoints'])

    return iresults

def results_table(layer: MolgroupsLayer, model: Experiment | None = None, problem: FitProblem | None = None, state: MCMCDraw | None = None, n_samples: int = 50, report_delta=False):

    if state is None:
        return CustomWebviewPlot(fig_type='table',
                                 plotdata="")

    # TODO: Consider whether to include *all* fields (relative or absolute) in the same table. Would require calculating the table only once
    # and give the same values regardless.

    if report_delta:
        FIELDNAMES = ['origin', 'property', 'delta lower 95% CI', 'delta lower 68% CI', 'median', 'delta upper 68% CI', 'delta upper 95% CI']
    else:
        FIELDNAMES = ['origin', 'property', 'lower 95% CI', 'lower 68% CI', 'median', 'upper 68% CI', 'upper 95% CI']

    def combine_results(data: list[dict]) -> dict:
        # combines a list of identical dicts to a dict of lists having the same structure
        def combine_identical_dicts(d: dict, combined_dict: dict) -> dict:
            for k, v in d.items():
                if isinstance(v, dict):
                    if k not in combined_dict.keys():
                        combined_dict[k] = {}
                    combine_identical_dicts(v, combined_dict[k])
                elif isinstance(v, list):
                    if k in combined_dict.keys():
                        combined_dict[k] += v
                    else:
                        combined_dict[k] = v
                else:
                    if k in combined_dict.keys():
                        combined_dict[k] += [v]
                    else:
                        combined_dict[k] = [v]

        cd = {}
        for d in data:
            combine_identical_dicts(d, cd)

        return cd

    # condition the points draw (adapted from bumps.errplot.calc_errors_from_state)
    points = state.draw().points
    if points.shape[0] < n_samples:
        n_samples = points.shape[0]
    points = points[np.random.permutation(len(points) - 1)]
    points = points[-n_samples:-1]

    print('Starting statistical analysis...')
    init_time = time.time()

    # set up a parallel calculation
    model_index = list(problem.models).index(model)

    with multiprocessing.Manager() as manager:
        shared_serialized_problem = manager.Array("B", dill.dumps(problem))
        #args = [(shared_serialized_problem, point) for point in points]

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=None, initializer=_initialize_worker, initargs=(shared_serialized_problem, model_index, layer.name)
        ) as executor:
            results = executor.map(_worker_eval_table_point, points)

    # walk through keys and combine into lists
    combined_results = combine_results(results)

    # walk through outer levels of keys and calculate confidence intervals, writing to csv
    with io.StringIO() as f:

        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for origin, v in combined_results.items():
            for property, vlist in v.items():
                #print(origin, property, np.array(vlist).shape)
                (median, _), (onesigmam, onesigmap), (twosigmam, twosigmap) = credible_interval(np.squeeze(vlist), (0, 0.68, 0.95))
                value_delta = median if report_delta else 0.0
                median, onesigmam, onesigmap, twosigmam, twosigmap = (f'{v - value_delta: 0.4g}' for v in (median + value_delta, onesigmam, onesigmap, twosigmam, twosigmap))
                writer.writerow(dict([(fname, value) for fname, value in zip(FIELDNAMES, [origin, property, twosigmam, onesigmam, median, onesigmap, twosigmap])]))

        csv_result = f.getvalue()

    print(f'Statistical analysis done after {time.time() - init_time} seconds')

    return CustomWebviewPlot(fig_type='table',
                             plotdata=csv_result,
                             exportdata=csv_result)
