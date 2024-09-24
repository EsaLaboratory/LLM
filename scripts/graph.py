import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
from react import success
import pandas as pd

path = "../../test/"

models = ['OpenHermes-2.5-Mistral-7B',
          'Mistral-7B-Instruct-v0.2', 'Mistral-7B-Instruct-v0.3']
difficulty_levels = ["easy", "medium", "hard"]
agent_type = ['', '_noreact', '_noreact_example']
n_params = 8
n_models = len(models)
n_types = len(agent_type)
n_tests = 20

scores = {}
results = np.zeros((n_models, n_types, n_params, len(difficulty_levels)))
for k, mode in enumerate(agent_type):
    for i, model in enumerate(models):
        scores[model + mode] = []
        for j, level in enumerate(difficulty_levels):
            mean = 0
            for p in range(n_tests):
                test, s, result = success(f"{path}{model}_{level}{mode}_solution_{p}.json",
                                          f"{path}{model}_{level}{mode}_output_{p}.json")
                mean += s
                results[i, k, :, j] += result
            results[i, k, :, j] = np.round(
                100 * results[i, k, :, j] / n_tests, 1)
        scores[model + mode] = np.mean(results[i, k], axis=0)


for j, agent in enumerate(agent_type):
    f, ax = plt.subplots(1, n_models + 1, sharey=False,
                         gridspec_kw={'width_ratios': [0.9 for i in models] + [0.08]})
    labels = ['date start', 'date end', 'EV', 'city',
              'EV start time', 'EV end time', 'Tmin', 'Tmax']
    pos = [0, 0.5, 1]
    levels = ['E', 'M', 'H']
    Models = ['V1', 'V2', 'V3']
    for k, model in enumerate(models):
        df = pd.DataFrame(results[k][j])
        df = df.rename(columns={i: levels[i]
                                for i in range(len(levels))})
        df = df.rename(index={i: labels[i] for i in range(len(labels))})
        h = sns.heatmap(df, cmap="Greens", cbar=(
            k == n_models-1), ax=ax[k], annot=False, cbar_ax=ax[-1], linewidths=0.1, linecolor='black')
        ax[k].title.set_text(f'{Models[k]}')
        ax[k].title.set_fontsize(18)
        if k != 0:
            h.set(ylabel="")
            h.set_yticks([])
        else:
            h.set_yticklabels(labels, fontsize=15)
        h.set_xticklabels(levels, fontsize=12)
    cbar = ax[k].collections[0].colorbar
    cbar.set_label('accuracy (%)', rotation=270, fontsize=15, labelpad=16)
    plt.savefig(f"../img/results_test_params{agent}.pdf",
                format='pdf', bbox_inches='tight')

params_list = ['duration', 'calls', 'iteration']
titles = ["Pipeline duration (s)",
          "Number of user questions",
          "Number of pipeline iteration"]
params = np.zeros((n_models, n_types, 3, 3, n_tests))
for p in range(n_models):
    for k in range(n_types):
        for i in range(n_tests):
            for j, level in enumerate(difficulty_levels):
                path_dict = f"{path}{models[p]}_{level}{agent_type[k]}_output_{i}.json"
                with open(path_dict, 'r', encoding="utf8") as file:
                    dict = json.load(file)
                    file.close()
                params[p, k, :, j, i] = dict['duration']/n_params, np.sum(
                    dict['calls'])/n_params, dict['iteration']/n_params

x = ['E' for i in range(
    n_tests)] + ['M' for i in range(n_tests)] + ['H' for i in range(n_tests)]
yaxis = [
    'Parametrization duration (s)', 'Number of questions', 'Number of iterations']
markers = ["diamond-open", "star", "circle"]
for p in range(n_models):
    for k in range(len(params_list)):
        fig = go.Figure()
        fig.update_layout(
            yaxis_range=[np.min(params[:, :, k]), np.max(params[:, :, k])])
        for j in range(n_types):
            fig.add_trace(go.Box(
                y=params[p][j, k].flatten(),
                x=x,
                boxpoints='outliers',
                name=['ReAct+example', 'Act', 'Act+example'][j],
                marker={'size': 8, 'symbol': markers[j]}
            ))
        fig.update_layout(
            showlegend=False,
            legend={
                'yanchor': "top",
                'y': 0.99,
                'xanchor': "right",
                'x': 1.01,
                'font': {'size': 14},
                'borderwidth': 1},
            autosize=False,
            width=400,
            height=300,
            yaxis_title=yaxis[k],
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
            font={'size': 16},
            boxmode='group'  # group together boxes of the different traces for each value of x
        )
        if p != n_models-1:
            fig.update_xaxes(showticklabels=False)
        fig.write_image(
            f"../img/results_test_{params_list[k]}_{models[p]}.pdf")
        fig.show()
