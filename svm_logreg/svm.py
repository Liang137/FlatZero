import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler

import argparse
import time
import wandb
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def compute_loss(w, X, y):
    margins = 1 - y * (X.dot(w))
    loss = np.mean(np.maximum(0, margins) ** 2)
    return loss


def compute_grad(w, X, y):
    margins = 1 - y * (X.dot(w))
    active = margins > 0
    grad_loss = -2 * (active * y * margins)
    grad = X.T.dot(grad_loss) / X.shape[0]
    return grad


def sam_grad(w, X, y, lam, eps=1e-12):
    grad = compute_grad(w, X, y)
    norm_g = np.linalg.norm(grad)
    e_w = lam * grad / (norm_g + eps)
    w_perturbed = w + e_w
    grad_sam = compute_grad(w_perturbed, X, y)
    return grad_sam


def zo_grad(w, X, y, lam):
    u = np.random.randn(*w.shape)

    f_plus = compute_loss(w + lam*u, X, y)
    f_mins = compute_loss(w - lam*u, X, y)
    fd = (f_plus - f_mins) / (2.0 * lam)

    return fd * u


def compute_hess(w, X, y):
    margins = 1 - y * (X.dot(w))
    active = margins > 0
    hess = 0.0
    if np.any(active):
        norm = np.sum(X[active] ** 2, axis=1)
        hess = (2 * np.sum(norm)) / X.shape[0]
    return hess


def compute_acc(w, X, y):
    decision = X.dot(w)
    preds = np.where(decision >= 0, 1, -1)
    acc = np.mean(preds == y)
    return acc


def run(args, file_name):
    logger.info(args)

    np.random.seed(args.seed)

    if args.data == "a5a":
        n_features = 123
    else:
        n_features = 300

    X_train, y_train = load_svmlight_file(f"data/{args.data}", n_features=n_features)
    X_test, y_test = load_svmlight_file(f"data/{args.data}.t", n_features=n_features)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    y_train = np.where(y_train <= 0, -1, 1)
    y_test  = np.where(y_test  <= 0, -1, 1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    W_random = np.random.randn(n_features, args.n_params)
    X_train = X_train @ W_random
    X_test = X_test @ W_random

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test  = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    n_features = X_train.shape[1]
    w = np.random.randn(n_features) * 0.1

    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []
    train_hess_list, test_hess_list = [], []
    start_time = time.time()

    for i in range(args.max_iter):
        train_loss = compute_loss(w, X_train, y_train)
        test_loss = compute_loss(w, X_test, y_test)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        
        train_acc = compute_acc(w, X_train, y_train)
        test_acc = compute_acc(w, X_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        train_hess = compute_hess(w, X_train, y_train)
        test_hess = compute_hess(w, X_test, y_test)
        train_hess_list.append(train_hess)
        test_hess_list.append(test_hess)

        if i % 100 == 0:
            logger.info(f"Iter {i+1}: Train Loss = {train_loss}, Test Acc = {test_acc}, Train Hessian Trace = {train_hess}, Time={time.time()-start_time}")

        if i % 10 == 0:
            wandb.log({"train/loss":train_loss, "train/acc":train_acc, "train/hess":train_hess, "test/loss":test_loss, "test/acc":test_acc, "test/hess":test_hess}, step=i)

        if args.algo == 'gd':
            grad = compute_grad(w, X_train, y_train)
        elif args.algo == 'zo':
            grad = zo_grad(w, X_train, y_train, args.lam)
        elif args.algo == 'sam':
            grad = sam_grad(w, X_train, y_train, args.lam)
        else:
            raise NotImplementedError

        w -= args.lr * grad

    res = {
        "train_loss": train_loss_list,
        "test_loss": test_loss_list,
        "train_acc": train_acc_list,
        "test_acc": test_acc_list,
        "train_hess": train_hess_list,
        "test_hess": test_hess_list
    }

    with open(f"./res_svm/{file_name}.json", "w") as f:
        json.dump(res, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="SVM"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="Learning rate for updates"
    )
    parser.add_argument(
        "--lam", type=float, default=0.01,
        help="Smoothing parameter"
    )
    parser.add_argument(
        "--algo", choices=["gd", "sam", "zo"], default="gd",
        help="Algorithm to be used"
    )
    parser.add_argument(
        "--data", choices=["a5a", "w5a"], default="a5a",
        help="Dataset to be used"
    )
    parser.add_argument(
        "--max_iter", type=int, default=10000,
        help="Maximum number of training iterations"
    )
    parser.add_argument(
        "--seed", type=int, default=29,
        help="Random seed"
    )
    parser.add_argument(
        "--n_params", type=int, default=10000,
        help="Overparameterized random dimension"
    )
    args = parser.parse_args()

    if args.algo == "gd":
        auto_name = f"{args.algo}_{args.data}_{args.lr}_{args.seed}"
    else:
        auto_name = f"{args.algo}_{args.data}_{args.lr}_{args.lam}_{args.seed}"

    wandb.init(project=f"svm-{args.data}", name=auto_name, reinit=True, config=vars(args))
    run(args, auto_name)
    wandb.finish()


if __name__ == "__main__":
    main()
