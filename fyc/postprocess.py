import desc.io
from desc.objectives import get_equilibrium_objective
import desc.plotting as dplot
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import numpy as np
import argparse
import pandas as pd
import seaborn as sns


def compile_pkl(pkls, out_pkl):
    """
    Precompile pkls into a single pkl, seems to be even slower than direct read
    """
    print("Compiling all pkls into a single pkl")
    res = []
    for pkl in tqdm(pkls):
        res.append(desc.io.load(pkl))

    pw = desc.io.PickleWriter(target=out_pkl)
    pw.write_obj(res)
    return


def extract_data(pkls, smoke_test=False):
    """
    Extract pressure profile coefficients and maximum force balance error

    Args:
        a list of pkl paths

    Return:
        pandas dataframe
    """

    fig, ax_dummy = plt.subplots()

    df_dict = {"max_F": [], "p_l": [], "success": []}

    if smoke_test:
        pkls = pkls[0:10]

    for pkl in tqdm(pkls):
        eq, res = desc.io.load(pkl)

        # Evaluate force
        _, _, plot_data = dplot.plot_section(
            eq, "|F|", norm_F=False, return_data=True, ax=ax_dummy
        )

        # Save data
        df_dict["success"].append(res.success)
        df_dict["max_F"].append(plot_data["|F|"].max())
        df_dict["p_l"].append(eq.p_l)

    plarray = np.array(df_dict.pop("p_l"))

    for i in range(plarray.shape[1]):
        df_dict["p_l_%d" % i] = plarray[:, i]

    df = pd.DataFrame(df_dict)
    plt.close(fig)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="results", type=str)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument(
        "--refresh-data-frame",
        action="store_true",
        help="regenerate the data frame regardless if it's already existing",
    )
    args = p.parse_args()

    pkls = glob(os.path.join(args.prefix, "eqs", "eq_*.pkl"))

    df_pkl = os.path.join(args.prefix, "df.pkl")

    if not os.path.isfile(df_pkl) or args.refresh_data_frame:
        df = extract_data(pkls, smoke_test=args.smoke_test)
        df.to_pickle(df_pkl)
    else:
        print(
            "[Info]Using existing data frame pickle, use --refresh-data-frame"
            " to force regeneration of the data frame pickle"
        )
        df = pd.read_pickle(df_pkl)

    fig, axs = plt.subplots(figsize=(9.0, 3.0), ncols=3)
    for i, N in enumerate([100, 500, 1000]):
        vals = df["max_F"].astype(float).values[:N]
        sns.histplot(vals, ax=axs[i], kde=True, stat="count")
        axs[i].set_xlabel("max |F|")
        axs[i].set_title(
            "sample size: %d\nmean: %.2f, std: %.2f" % (N, vals.mean(), vals.std())
        )
    fig.savefig(os.path.join(args.prefix, "maxF_distributions.pdf"))

    fig2, ax2 = plt.subplots()
    for i in range(6):
        sns.histplot(data=df["p_l_%d" % i].astype(float), ax=ax2, label="p_l_%d" % i)
    ax2.legend()
    fig2.savefig(os.path.join(args.prefix, "coeffs_distributions.pdf"))
