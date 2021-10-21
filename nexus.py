#!/usr/bin/env python3

import algorithms
from main import RunnerForPool

import argparse
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os.path
import random
import sys
import time
import matplotlib.colors as mcolors

from multiprocessing import Pool

def EvalHelmbold(datasets, num_runs, output_basename):
  ALGORITHMS_ONLINE = (
    algorithms.OPT_multiple,
    algorithms.RandomOnline_multiple_prudent,
  )
  ALGORITHMS_PRED= (
    algorithms.FTP_multiple,
    algorithms.RobustRhoMu_multiple_prudent,
    algorithms.RobustKumar_multiple_prudent,
    algorithms.RobustAngelo_multiple,
    algorithms.RobustFTP_multiple_prudent,
  )
  ALGORITHMS = ALGORITHMS_ONLINE + ALGORITHMS_PRED

  json_fname = output_basename + '.json'
  try:
    with open(json_fname) as f:
      competitive_ratios = np.array(json.load(f))
    datasets = datasets + datasets
  except:
    datasets_requests = []
    for dataset in datasets:
      with open(dataset) as f:
        requests = [float(x) for x in f]
        datasets_requests.append(requests)
    datasets_predictions = [
        algorithms.PredHelmbold(requests) for requests in datasets_requests
    ] + [
        algorithms.PredHelmbold(requests, ETA=-4.0) for requests in datasets_requests
    ]
    datasets = datasets + datasets
    datasets_requests = datasets_requests + datasets_requests
    costs = np.zeros((len(datasets), len(ALGORITHMS), num_runs))
    for i, requests in enumerate(datasets_requests):
      for j, algorithm in enumerate(ALGORITHMS_ONLINE):
        output = algorithm(requests)
        cost = algorithms.Cost(output, requests, algorithm.__name__)
        for run in range(num_runs):  # these algorithms behave exactly the same across all runs
          costs[i][j][run] = cost
    with Pool() as pool:
      grid = list(itertools.product(range(len(datasets)), range(len(ALGORITHMS_PRED)), range(num_runs)))
      args = [(
        ALGORITHMS_PRED[algorithm_idx],
        datasets_requests[dataset_idx],
        datasets_predictions[dataset_idx]) for dataset_idx, algorithm_idx, _ in grid]
      runs = pool.starmap(RunnerForPool, args)
      for (dataset_idx, algorithm_idx, run_idx), cost in zip(grid, runs):
        costs[dataset_idx][len(ALGORITHMS_ONLINE) + algorithm_idx][run_idx] = cost
    costs = np.mean(costs, axis=2)
    competitive_ratios = costs / costs[:,0].reshape(-1, 1)

    with open(json_fname, 'w') as f:
      json.dump(competitive_ratios.tolist(), f)

  LABELS = [algorithm.__name__ for algorithm in ALGORITHMS]
  for i, dataset in enumerate(datasets):
    for j, label in enumerate(LABELS):
      print(dataset, label, competitive_ratios[i][j])

  TRANSLATE = {
    "RandomOnline_multiple_prudent" : "$(\\frac{e}{e-1})$-competitive",
    "FTP_multiple": "Lemma 3 + FTP",
    "RobustFTP_multiple_prudent": "Lemma 3 + Thm 28 + FTP",
    "RobustKumar_multiple_prudent": "Lemma 3 + Thm 28 + PSK",
    "RobustAngelo_multiple": "Lemma 3 + Thm 28 + ADJKR",
    "RobustRhoMu_multiple_prudent": "Our algorithm",
  }

  GENERAL_FONTSIZE, LABEL_FONTSIZE, LEGEND_FONTSIZE, DATASET_FONTSIZE = 11, 16, 11, 9
  plt.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.size': GENERAL_FONTSIZE,
    'pgf.rcfonts': False,
  })

  barwidth = 0.75 / (len(ALGORITHMS) - 1)
  for j, algorithm in enumerate(ALGORITHMS[1:]):
    plt.bar(
      np.arange(len(datasets)) + j * barwidth,
      competitive_ratios[:,j + 1],
      width=0.8 * barwidth,
      label=TRANSLATE[LABELS[j + 1]],
      )
  plt.xticks(
    np.arange(len(datasets)) + 0.75,
    [
      os.path.splitext(os.path.basename(dataset))[0].replace('nexus_log', '')
      for dataset in datasets
    ],
    rotation=30, ha='right', fontsize=DATASET_FONTSIZE,
    )
  plt.gca().tick_params('x', length=0)

  plt.legend(loc='best', fontsize=LEGEND_FONTSIZE)
  plt.ylim(1, 3)
  plt.ylabel('Competitive ratio', fontsize=LABEL_FONTSIZE)
  
  n = len(datasets)
  import matplotlib.ticker as ticker
  ax = plt.gca().twiny()
  ax.spines['bottom'].set_position(('axes', -0.50))
  ax.tick_params('both', length=0, width=0, which='minor', labelsize=LABEL_FONTSIZE)
  ax.tick_params('both', direction='in', which='major')
  ax.xaxis.set_ticks_position('bottom')
  ax.xaxis.set_label_position('bottom')
  ax.set_xticks([0, 0.5 * n, n])
  ax.xaxis.set_major_formatter(ticker.NullFormatter())
  ax.xaxis.set_minor_locator(ticker.FixedLocator([0.25 * n, 0.75 * n]))
  ax.xaxis.set_minor_formatter(ticker.FixedFormatter(['Good predictions', 'Bad predictions']))

  plt.gcf().set_size_inches(w=12, h=4.5)
  plt.tight_layout()
  plt.savefig(output_basename + '.pdf')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--num_runs', type=int, default=1, help='number of runs')
  parser.add_argument('-o', '--output_basename', type=str)
  parser.add_argument('DATASETS', type=str, nargs='+')
  args = parser.parse_args()
  EvalHelmbold(args.DATASETS, args.num_runs, args.output_basename)

if __name__ == '__main__':
  main()
