#!/usr/bin/env python3

import algorithms

import argparse
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import time
import matplotlib.colors as mcolors

from multiprocessing import Pool


# OPT must be the first algorithm in this list
# OPT_multiple must be the second
ALGORITHMS_ONLINE = (
  algorithms.OPT,
  algorithms.OPT_multiple,
  # algorithms.Online_multiple,
  algorithms.RandomOnline_multiple,
  algorithms.RandomOnline_multiple_prudent,
  # algorithms.ClassicDet,
  algorithms.ClassicRandom,
  # algorithms.Rent,
)

ALGORITHMS_PRED= (
  algorithms.FTP,
  algorithms.FTP_multiple,
  # algorithms.RhoMu_paretomu_1_05,
  algorithms.RhoMu_paretomu_1_1,
  algorithms.RhoMu_paretomu_1_1596,
  algorithms.RhoMu_paretomu_1_216,
  algorithms.RhoMu_paretomu_1_3,
  algorithms.RhoMu_paretomu_1_4,
  algorithms.RhoMu_paretomu_1_5,
  # algorithms.RhoMu_multiple_1_1596,
  # algorithms.RhoMu_multiple_1_216,
  algorithms.RobustRhoMu,
  algorithms.RobustRhoMu_multiple,
  algorithms.RobustRhoMu_multiple_prudent,
  algorithms.KumarRandom_1_1596,
  algorithms.KumarRandom_1_216,
  # algorithms.Kumar_multiple_1_1596,
  # algorithms.Kumar_multiple_1_216,
  algorithms.RobustKumar,
  algorithms.RobustKumar_multiple,
  algorithms.RobustKumar_multiple_prudent,
  algorithms.AngelopoulosDet_1_1596,
  algorithms.AngelopoulosDet_1_216,
  algorithms.RobustAngelo,
  algorithms.RobustAngelo_multiple,
  algorithms.RobustFTP,
  algorithms.RobustFTP_multiple_prudent,
)


def RunnerForPool(algorithm, requests, predictions):
  return algorithms.Cost(algorithm(requests, predictions), requests, algorithm.__name__)

def PlotPredRandom(datasets, num_runs, output_basename=None, load_json=None, keep='rhos'):
  ALGORITHMS = ALGORITHMS_ONLINE + ALGORITHMS_PRED
  with open(datasets[0]) as f:
    MAX_REQUEST = np.ceil(max(float(line) for line in f))
  SIGMAS = [min(MAX_REQUEST, 8) * i / 8 for i in range(10)]
  print('Sigmas:', SIGMAS)
  
  if load_json:
    with open(load_json) as f:
      costs = np.array(json.load(f))
  else:
    costs = np.zeros((len(ALGORITHMS), len(SIGMAS), num_runs))  
    for dataset in datasets:
      with open(dataset) as f:
        requests = tuple(f)
      assert(len(requests) > 0)
      
      requests = [float(x) for x in requests]

      for i, algorithm in enumerate(ALGORITHMS_ONLINE):
          output = algorithm(requests)
          cost = algorithms.Cost(output,requests, algorithm.__name__)
          for j in range(len(SIGMAS)):
            for run in range(num_runs):  # these algorithms behave exactly the same across all runs
              costs[i][j][run] += cost 

      predictor = lambda sigma: algorithms.PredRandom(requests, sigma)

      predictions = [[predictor(sigma) for _ in range(num_runs)] for sigma in SIGMAS]

      with Pool() as pool:
        grid = list(itertools.product(range(len(ALGORITHMS_PRED)), range(len(SIGMAS)), range(num_runs)))
        args = [(ALGORITHMS_PRED[algorithm_idx], requests, predictions[sigma_idx][run_idx])
                for algorithm_idx, sigma_idx, run_idx in grid]
        runs = pool.starmap(RunnerForPool, args)
        for (algorithm_idx, sigma_idx, run_idx), cost in zip(grid, runs):
          costs[len(ALGORITHMS_ONLINE) + algorithm_idx][sigma_idx][run_idx] += cost

  if output_basename:
    with open(output_basename + '.json', 'w') as f:
      json.dump(costs.tolist(), f)

  competitive_ratios = costs / costs[0]
  print('Maximum std dev among runs: %0.5f' % np.max(np.std(competitive_ratios, axis=2)))
  competitive_ratios = np.mean(competitive_ratios, axis=2)

  LABELS = [algorithm.__name__ for algorithm in ALGORITHMS]
  if (keep=='rhos'):
    OPT_LABEL = 0
    KEEP = {
      "ClassicRandom" : "$(\\frac{e}{e-1})$-competitive", # "Classical (randomized)",
      "KumarRandom_1_1596" : "PSK ($\\rho$=1.1596)",
      "KumarRandom_1_216" : "PSK ($\\rho$=1.216)",
      "RhoMu_paretomu_1_1596" : "Our alg. ($\\rho$=1.1596)",
      "RhoMu_paretomu_1_216" : "Our alg. ($\\rho$=1.216)",
      "AngelopoulosDet_1_1596" : "ADJKR ($\\rho$=1.1596)",
      "AngelopoulosDet_1_216" : "ADJKR ($\\rho$=1.216)",
      "FTP" : "FTP",  
    }
  elif (keep=='ouralg_manyrhos'):
    OPT_LABEL = 0
    KEEP = {
      "ClassicRandom" : "$(\\frac{e}{e-1})$-competitive", # "Classical (randomized)",
      "RhoMu_paretomu_1_1" : "Our alg. ($\\rho$=1.1)",
      "RhoMu_paretomu_1_1596" : "Our alg. ($\\rho$=1.1596)",
      "RhoMu_paretomu_1_3" : "Our alg. ($\\rho$=1.3)",
      "RhoMu_paretomu_1_4" : "Our alg. ($\\rho$=1.4)",
      "RhoMu_paretomu_1_5" : "Our alg. ($\\rho$=1.5)",
      "FTP" : "FTP",   
    }
  elif (keep=='best'):
    OPT_LABEL = 0
    KEEP = {
      "ClassicRandom" : "$(\\frac{e}{e-1})$-competitive", # "Classical (randomized)",
      "FTP" : "FTP",  
      "RobustFTP" : "Thm 28 + FTP",
      "RobustKumar" : "Thm 28 + PSK",
      "RobustAngelo" : "Thm 28 + ADJKR",
      "RobustRhoMu" : "Our algorithm",
    }
  elif (keep=='prudent'):
    OPT_LABEL = 1
    KEEP = {
      "RandomOnline_multiple_prudent" : "$(\\frac{e}{e-1})$-competitive (prudent)",
      "RandomOnline_multiple" : "$(\\frac{e}{e-1})$-competitive (not prudent)",
      "RobustKumar_multiple_prudent": "Lemma 3 + Thm 28 + PSK (prudent)",
      "RobustKumar_multiple": "Lemma 3 + Thm 28 + PSK (not prudent)",
      "RobustAngelo_multiple": "Lemma 3 + Thm 28 + ADJKR",
      "RobustRhoMu_multiple_prudent": "Our algorithm (prudent)",
      "RobustRhoMu_multiple": "Our algorithm (not prudent)",
    }
  else:  # keep == 'bestmult'
    OPT_LABEL = 1
    KEEP = { 
      "RandomOnline_multiple_prudent" : "$(\\frac{e}{e-1})$-competitive", # "Classical (randomized)",
      "FTP_multiple": "Lemma 3 + FTP",
      "RobustFTP_multiple_prudent": "Lemma 3 + Thm 28 + FTP",
      "RobustKumar_multiple_prudent": "Lemma 3 + Thm 28 + PSK",
      "RobustAngelo_multiple": "Lemma 3 + Thm 28 + ADJKR",
      "RobustRhoMu_multiple_prudent": "Our algorithm",
    }
  

  # GENERAL_FONTSIZE, LABEL_FONTSIZE, LEGEND_FONTSIZE = 9, 12, 10
  GENERAL_FONTSIZE, LABEL_FONTSIZE, LEGEND_FONTSIZE = 11, 16, 11
  
  plt.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.size': GENERAL_FONTSIZE,
    'pgf.rcfonts': False,
  })
  LINE = iter(itertools.cycle(itertools.product(['', '.', '+', 'x'], ['--', '-.', '-', ':'])))

  if keep == 'ouralg_manyrhos':
    LINE = iter(itertools.cycle((('','--'),)))
  elif keep == 'best':
    LINE = iter([('','--'), ('d', (0, (1, 2))), ('', '-'), ('x', (1, (1, 2))), ('.', '--'), ('.', (2, (1, 2)))])
  
  COLORS = iter(mcolors.TABLEAU_COLORS)
  
  oldlabel = ""
  oldcolor = ""
  start_mk, start_ls = "X","X"
  for label, ratios in zip(LABELS, competitive_ratios):
    if label not in KEEP:
      continue
    label = KEEP[label]
    if (label.split('[')[0] == oldlabel.split('[')[0]):
      color = oldcolor
      mk, ls = next(LINE)
    elif "rho=" in label and start_mk != "X":
      while (start_ls != ls):
        mk,ls = next(LINE)
    elif "rho=" in label and start_mk == "X":
      mk, ls = next(LINE)
      start_mk = mk
      start_ls = ls
    else:
      mk, ls = next(LINE)

    if (label.split('[')[0] != oldlabel.split('[')[0]):
      color = next(COLORS)
    
    oldlabel = label
    oldcolor = color
    data = [x/y for x,y in zip(ratios,competitive_ratios[OPT_LABEL])]
    plt.plot(SIGMAS, data, label=label, markersize=5, marker=mk, ls=ls, color=color)[0]
    if keep == 'ouralg_manyrhos':
      if label == 'FTP':
        rho = 1.0
      elif 'rho' in label:
        rho = float(label.split('=')[1].rstrip(')'))
      else:
        rho = 1.58
      plt.text(0, data[0], "$\\rho$=%0.2f" % rho, ha='right', va='center', in_layout=True, fontsize=LEGEND_FONTSIZE-1)

  if keep == 'ouralg_manyrhos':
    plt.xlim(left=-(max(SIGMAS)/6))
    
  xlabel = plt.xlabel('Noise parameter $\sigma$ of the synthetic predictor', fontsize=LABEL_FONTSIZE)
  ylabel = plt.ylabel('Competitive ratio', fontsize=LABEL_FONTSIZE)
  
  plt.gcf().set_size_inches(w=6, h=4.5)
  plt.tight_layout()
  if keep == 'ouralg_manyrhos':
    pass
  else:
    plt.legend(loc='best', ncol=(1 if len(KEEP) < 8 else 2), fontsize=LEGEND_FONTSIZE)

  if output_basename:
    plt.savefig(output_basename + '.pdf')
  else:
    plt.show()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--num_runs', type=int, default=1, help='number of runs')
  parser.add_argument('-o', '--output_basename', type=str)
  parser.add_argument('-l', '--load_json', type=str)
  parser.add_argument('-k', '--keep', type=str, choices=['rhos', 'best', 'bestmult', 'ouralg_manyrhos', 'prudent'], default='rhos')
  parser.add_argument('DATASETS', type=str, nargs='+')
  args = parser.parse_args()

  PlotPredRandom(args.DATASETS, args.num_runs, args.output_basename, args.load_json, args.keep)

if __name__ == '__main__':
  main()
