import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
import seaborn as sns
# sns.set_theme()
import math
import time
import argparse
from utils import sim_MC, plot_results, plot_multi_results, const_num_alpha_only_sell

matplotlib.use("pgf")
plt.rcParams['pgf.texsystem'] = "pdflatex"
plt.rcParams['savefig.format'] = 'pgf'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['font.size'] = 20
plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = "Helvetica"
n_colors = 10
plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.viridis(np.linspace(0,0.8,n_colors)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_MC", default=100_000, type = int)        # number of MC simulations to run
    parser.add_argument("--seed", default=11, type=int)              # rng seed
    parser.add_argument("--file_name", default="plot.png", type=str)        # where to save plot
    parser.add_argument("--dpi", default=300, type=int)              # dpi
    parser.add_argument("--start", type = int)
    parser.add_argument("--stop", type = int)
    parser.add_argument("--num", type = int)
    
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    fpath = args.file_name

    d = {}
    params = {
        'init_q': 1000,
        'target_q': 0,
        'c': 0.5,
        'lamb': 0.01,
        'tau': 10,
        'theta': 5,
        'gamma': 0.02,
        'T': 100,
        'sigma': 0.1,
        'n_MC': args.n_MC
    }

    p_vals = np.linspace(args.start, args.stop, num = args.num)
    print(f"Number of p_vals: {len(p_vals)}")
    print(f"n_MC: {params['n_MC']}")
    for i, frac in enumerate(p_vals):
        params['seed'] = i
        d[frac] = sim_MC(const_num_alpha_only_sell, key_args = {'p': frac}, **params)
        print(f"p = {frac}")
        print(f"Final Y: {d[frac]['Y'].iloc[-1]}")

    final_Y = np.array([d[x]['Y'].iloc[-1] for x in d.keys()])
    opt_p = list(d.keys())[np.argmax(final_Y)]
    print(f"opt_p: {opt_p}")
    fig, axes = plt.subplots(figsize=(8, 6))
    axes.plot(d.keys(), final_Y, '.')
    axes.set_title(fr"\textbf{{Final P\&L for Varying $\mathbf{{p,\lambda={params['lamb']}}}$,\#sims={params['n_MC']:.2E}}}")
    axes.set_xlabel(r"$\mathbf{{p}}$")
    axes.set_ylabel(r"\textbf{P\&L (\$)}")
    axes.ticklabel_format(style='sci',axis='both')
    # axes.tick_params(which='minor', bottom = False)
    fig.savefig(fpath, bbox_inches='tight', dpi = args.dpi)

# d = {}
# params = {
#     'init_q': 1000,
#     'target_q': 0,
#     'c': 0.5,
#     'lamb': 0.01,
#     'tau': 10,
#     'theta': 5,
#     'gamma': 0.02,
#     'T': 100,
#     'sigma': 0.1,
#     'n_MC': 1_0,
# }

# for i in range(5):
#     frac = 750 + 500/9 * i
#     params['seed'] = rng.integers(10000)
#     d[frac] = sim_MC(const_num_alpha_only_sell, key_args = {'p': frac}, **params)

# plot_multi_results(rf"$\mathbf{{x_t=\frac{{Q_T-Q_t}}{{T-t}}+p*\alpha_t,c={params['c']},\lambda={params['lamb']},\tau={params['tau']},\theta={params['theta']},\gamma={params['gamma']},\sigma={params['sigma']},T={params['T']},\#sims={params['n_MC']:.2E}}}$", d, save_fig=True, fpath = args.file_name)
