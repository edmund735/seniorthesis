import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("pgf")
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
from utils import sim_MC, const_num_alpha_only_sell
import pickle
import os

plt.rcParams['pgf.texsystem'] = "pdflatex"
plt.rcParams['savefig.format'] = 'pgf'
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr12'
plt.rcParams['font.size'] = 12
plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['figure.figsize'] = [4, 2]
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [
plt.rcParams['pgf.preamble'] = "\n".join([
    r"\usepackage{amsmath}"
    ])

n_colors = 5
plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.viridis(np.linspace(0,0.8,n_colors)))


def plot_multi_results(f_title, df, ncols = 2, fpath="", dpi = 300):
    '''
    Plot averages of multidimensional MC sim
    data is a dict
    '''

    # Create a figure and axis
    N = 7               
    ncols = 2
    fig, axes = plt.subplots(ncols = ncols, nrows = math.ceil(N/ncols), layout='constrained', figsize=(6, 6 * math.ceil(N/ncols)))

    # Columns to plot
    cols = ['Q', 'x', 'alpha', 'J', 'Y', 'I', 'S', 'P']
    ylabs = [r"\# of Shares", r"\# of Shares", r"Price Signal", r"Cum. Order Flow", r"P\&L (\$)", r"Price Impact", r"Price (\$)"]
    titles = [r"Position ($\mathbf{Q_t}$)", r"Shares Traded ($\mathbf{x_t}$)", r"Forecast ($\mathbf{\alpha_t}$)", r"EWMA of Cumulative Order Flow ($\mathbf{J_t}$)", r"P\&L ($\mathbf{Y_t}$)", r"Price Impact ($\mathbf{I_t}$)", r"Asset Price"]

    # if want to plot only P and not S
    # for i, c in enumerate(ylabs):
    #     leg_labels = []
    #     for p in df.keys():
    #         axes[i//2, i%2].plot(df[p][cols[i]], ".-", label = rf"$p={p:.2f}$")

    #         # Calculate final value
    #         final_val = df[p][cols[i]].iloc[len(df[p])-1]
    #         leg_labels.append(rf"$p={p:.2f}$, {final_val:.2f}")

    for i, j in enumerate(ylabs):
        r = i//ncols
        c = i%ncols
        lines = []
        leg_labels = []
        for p in df.keys():
            if i < 6:
                l, = axes[r,c].plot(df[p][cols[i]], ".-", label = rf"$p={p:.2f}$")

                # Calculate final value
                final_val = df[p][cols[i]].iloc[df[p][cols[i]].last_valid_index()]
            else:
                axes[r,c].plot(df[p][cols[i]], ".-", label = rf"$p={p:.2f}$")
                l, = axes[r,c].plot(df[p][cols[i+1]], "v-", c = axes[r, c].lines[-1].get_c(), label = rf"$p={p:.2f}$")

                # Calculate final value of P
                final_val = df[p][cols[i+1]].iloc[df[p][cols[i+1]].last_valid_index()]

            lines.append(l)
            leg_labels.append(rf"$p={p:.2f}$, {final_val:.3f}")

        axes[r,c].set_xlabel("Time")
        axes[r,c].set_ylabel(j)
        axes[r,c].set_title(titles[i])
        axes[r,c].ticklabel_format(style='sci',axis='y')
        # if i < 5: axes[r,c].legend(lines, leg_labels, ncols = 2, prop = {'size': 12})
        # elif i == 6:
        if i ==6:
            # leg1 = axes[r,c].legend(lines, leg_labels, ncols = 2, prop = {'size': 12}, loc = 'lower left')
            # axes[r,c].add_artist(leg1)
            handles = [Line2D([], [], marker=marker) for marker in ['.', 'v']]
            axes[r,c].legend(handles = handles, labels = [r'$S_t$', r'$P_t$'], prop = {'size': 15})
            
    # first create a dummy legend, so fig.tight_layout() makes enough space
    axes[0, 0].legend(handles=axes[0, 0].lines[:1], bbox_to_anchor=(0, 1.12), loc='lower left')
    fig.tight_layout(pad=3.0)

    # now create the real legend
    axes[0, 0].legend(handles=axes[0, 0].lines, ncols = len(df.keys()), bbox_to_anchor=(1.03, 1.12), loc='lower center', fontsize=18)

    final_Y = np.array([df[x]['Y'].iloc[-1] for x in df.keys()])
    axes[(i+1)//ncols, (i+1)%ncols].plot(df.keys(), final_Y, 'o-')
    axes[(i+1)//ncols, (i+1)%ncols].set_title(r"Final P\&L for Varying $p$ values")
    axes[(i+1)//ncols, (i+1)%ncols].set_xlabel(r"$\mathbf{{p}}$")
    axes[(i+1)//ncols, (i+1)%ncols].set_ylabel(r"P\&L")

    fig.suptitle(f"{f_title}", fontsize=24, fontweight="heavy")

    # Adjust layout to prevent clipping of titles
    fig.tight_layout()

    # Adjust layout to prevent clipping of titles
    fig.subplots_adjust(top=0.9, wspace=0.3, hspace=0.5)

    # Save figure
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    fig.savefig(fpath, bbox_inches='tight', dpi = dpi)
    return

def plot_optimal_p(data, fpath = "plot.pdf", dpi = 300, **params):
    '''
    Plot p (coeff of alpha) vs Y
    '''

    final_Y = np.array([data[x]['Y'].iloc[-1] for x in d.keys()])
    # opt_p = list(d.keys())[np.argmax(final_Y)]

    fig, axes = plt.subplots()

    print(f"Size of plot: {fig.get_size_inches()}")

    axes.plot(data.keys(), final_Y, '.')
    axes.set_title(fr"\textbf{{Final P\&L for Varying $\mathbf{{p}}$, \#sims={params['n_MC']:.2E}}}")
    axes.set_xlabel(r"$\mathbf{{p}}$")
    axes.set_ylabel(r"\textbf{P\&L (\$)}")
    axes.ticklabel_format(style='sci',axis='both')
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    fig.savefig(fpath, bbox_inches='tight', dpi = dpi)
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_MC", default=100_000, type = int, help = "number of MC simulations to run")
    parser.add_argument("--plot", default = "all", type = str, help = "all or Y, for all graphs or only P&L")
    parser.add_argument("--from_file", action="store_true", default=False, help="Use data from file")
    parser.add_argument("--data_file", type = str, help = "file path of existing data to read from")
    parser.add_argument("--save_results", action="store_true", default=False, help="Store simulation results")
    parser.add_argument("--sim_fpath", default="sim", type=str, help = "where to save simulation data")
    parser.add_argument("--plot_fpath", default="plot.pdf", type=str, help = "where to save plot")
    parser.add_argument("--dpi", default=300, type=int, help = "dpi")
    parser.add_argument("--start", type = int, help = "start value of p")
    parser.add_argument("--stop", type = int, help = "end value of p")
    parser.add_argument("--num", type = int, help = "num of values in between")
    
    args = parser.parse_args()

    if args.from_file:
        with open(args.data_file, 'rb') as file:
            d = pickle.load(file)
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
    else:
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
        
        for i, p in enumerate(p_vals):
            params['seed'] = i
            d[p] = sim_MC(const_num_alpha_only_sell, key_args = {'p': p}, **params)

        if args.save_results:
            os.makedirs(os.path.dirname(args.sim_fpath), exist_ok=True)
            with open(args.sim_fpath, 'wb') as file:
                pickle.dump(d, file)

    if args.plot == "all":
        t = rf"$\mathbf{{x_t=\frac{{Q_T-Q_t}}{{T-t}}+p*\alpha_t,c={params['c']},\lambda={params['lamb']},\tau={params['tau']},\theta={params['theta']},\gamma={params['gamma']},\sigma={params['sigma']},T={params['T']},\#sims={params['n_MC']:.2E}}}$"
        plot_multi_results(t, d, ncols = 2, fpath=args.plot_fpath, dpi = args.dpi)
    else:
        plot_optimal_p(d, args.plot_fpath, dpi = args.dpi, **params)
