import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sampling import perturb_poly_coef, perturb_poly_gp
import desc.plotting as dplot
import seaborn as sns


class Net(torch.nn.Module):
    def __init__(self, nfeatures):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(nfeatures, 256)
        self.layer2 = torch.nn.Linear(256, 256)
        self.layer3 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        y = self.layer3(x)
        return y


class NNRegressor:
    def __init__(self, nfeatures, learning_rate=0.001) -> None:
        self.net = Net(nfeatures)

        # Define a loss function and an optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=learning_rate, weight_decay=0.001
        )

        return

    def train(self, maxit=1000):
        # Train the neural network
        loss_vals = []
        for i in tqdm(range(maxit)):
            running_loss = 0.0
            self.optimizer.zero_grad()
            outputs = self.net(torch.Tensor(X_train.values)).flatten()
            loss = self.criterion(outputs, torch.Tensor(Y_train.values))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            loss_vals.append(running_loss)

        return loss_vals


def plot_nn(prefix, X_train, X_test, Y_train, Y_test):
    Y_pred_test = np.zeros_like(Y_test)
    Y_pred_train = np.zeros_like(Y_train)
    for i, x in enumerate(X_test.values):
        Y_pred_test[i] = nnr.net(torch.Tensor(x))
    for i, x in enumerate(X_train.values):
        Y_pred_train[i] = nnr.net(torch.Tensor(x))

    print("R2 training:", r2_score(Y_pred_train, Y_train))
    print("R2 test    :", r2_score(Y_pred_test, Y_test))

    fig0, ax0 = plt.subplots(figsize=(4.0, 3.0))
    fig1, ax1 = plt.subplots(figsize=(4.0, 3.0))
    ax0.scatter(
        Y_pred_train,
        Y_train,
        color="blue",
        label="training set, R2=%.4f" % r2_score(Y_pred_train, Y_train),
        alpha=0.5,
        zorder=100,
    )
    ax0.scatter(
        Y_pred_test,
        Y_test,
        color="red",
        label="test set, R2=%.4f" % r2_score(Y_pred_test, Y_test),
        alpha=0.5,
        zorder=100,
    )
    ax0.legend()
    xlim = ax0.get_xlim()
    ax0.plot(xlim, xlim, color="grey", zorder=10, alpha=1.0, lw=1.0)
    ax0.set_xlim(xlim)
    ax0.set_ylim(xlim)
    ax0.set_xlabel("Prediction")
    ax0.set_ylabel("Actual")

    ax1.semilogy(loss_vals)
    ax1.set_ylabel("MSE loss")
    ax1.set_xlabel("Iteration")
    fig0.savefig(os.path.join(prefix, "nn_pred.pdf"))
    fig1.savefig(os.path.join(prefix, "nn_loss.pdf"))
    plt.close()
    return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="results_gp", type=str)
    p.add_argument("--plot-nn", action="store_true")
    p.add_argument("--lr", default=0.001, type=float)
    p.add_argument("--maxit", default=1000, type=int)
    p.add_argument("--nsamples", default=50000, type=int)
    p.add_argument("--gp-length-scale", type=float, default=0.1)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    # Save option values
    with open(os.path.join(args.prefix, "nn_options.txt"), "w") as f:
        f.write("Options:\n")
        for k, v in vars(args).items():
            f.write(f"{k:<20}{v}\n")

    if args.smoke_test:
        args.maxit = 10

    # Load data frame
    assert os.path.isdir(args.prefix)
    df_path = os.path.join(args.prefix, "df.pkl")
    df = pd.read_pickle(df_path)
    df = df[df["success"] == True].reset_index(drop=True)

    # Get input and output
    X = df.loc[:, ["p_l_%d" % i for i in range(6)]]
    Y = df["max_F"].astype(float)
    X_mean = X.mean()
    X_std = X.std()

    # Normalize X such that E(x) = 0, std(X) = 1.0
    Xn = (X - X_mean) / X_std

    # Split into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
        Xn, Y, test_size=0.2, random_state=0
    )

    # Train the NN
    nfeatures = X_train.values.shape[1]
    nnr = NNRegressor(nfeatures, learning_rate=args.lr)
    loss_vals = nnr.train(maxit=args.maxit)

    # Use the neural network to make predictions on new data
    nnr.net.eval()

    if args.plot_nn:
        plot_nn(args.prefix, X_train, X_test, Y_train, Y_test)

    # Perform Monte Carlo
    options = {}
    with open(os.path.join(args.prefix, "options.txt")) as f:
        for l in f.readlines():
            pair = l.strip().split()
            if len(pair) == 2:
                options[pair[0]] = pair[1]

    p_modes_bl = list(range(0, int(options["degree"]) + 1, 2))
    p_coefs_bl = np.zeros_like(p_modes_bl)
    p_coefs_bl[0] = 1000.0
    p_coefs_bl[1] = -1000.0

    if "direct" in args.prefix:
        dcoefs = perturb_poly_coef(
            nsamples=args.nsamples,
            modes=p_modes_bl,
            upper=float(options["scale"]),
            lower=-float(options["scale"]),
            seed=12345,
        )
    elif "gp" in args.prefix:
        dcoefs = perturb_poly_gp(
            nsamples=args.nsamples,
            modes=p_modes_bl,
            length_scale=float(options["gp_length_scale"]),
            stddev=float(options["scale"]),
            seed=12345,
        )
    else:
        raise ValueError("Can deduct method type from prefix")

    X_samples = (dcoefs + p_coefs_bl - X_mean.values) / X_std.values
    Y_samples = nnr.net(torch.Tensor(X_samples)).detach().numpy()

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    sns.histplot(Y_samples, kde=True, stat="count")
    ax.set_xlabel("max |F|")
    ax.set_title(
        "sample size: %d\nmean: %.2f, std: %.2f"
        % (args.nsamples, Y_samples.mean(), Y_samples.std())
    )
    ax.get_legend().remove()
    fig.savefig(os.path.join(args.prefix, "nn_maxF_distributions.pdf"))
