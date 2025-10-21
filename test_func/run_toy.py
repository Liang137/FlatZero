import numpy as np
import json
import time

def loss(x, y):
    r = x @ y - 1.0
    return 0.5 * (r**2)


def grad(x, y):
    r = x @ y - 1.0
    grad_x = r * y
    grad_y = r * x
    return grad_x, grad_y


def zo_grad(x, y, lam):
    u_x = np.random.randn(*x.shape)
    u_y = np.random.randn(*y.shape)

    f_plus = loss(x + lam*u_x, y + lam*u_y)
    f_mins = loss(x - lam*u_x, y - lam*u_y)
    fd = (f_plus - f_mins) / (2.0 * lam)

    grad_x = fd * u_x
    grad_y = fd * u_y
    return grad_x, grad_y


def zo_dd(x, y):
    u_x = np.random.randn(*x.shape)
    u_y = np.random.randn(*y.shape)

    r = x @ y - 1.0
    dd = u_x @ y + u_y @ x

    grad_x = r * dd * u_x
    grad_y = r * dd * u_y
    return grad_x, grad_y


def hessian_trace(x, y):
    return np.dot(x, x) + np.dot(y, y)


def run(T, x, y, lr, method, lam=None):

    loss_val = []
    trace_val = []
    num_iter = []
    start_time = time.time()

    for i in range(T):
        hess_tr = hessian_trace(x, y)
        L_val = loss(x, y)

        if method == 'zo':
            gx, gy = zo_grad(x, y, lam)
        elif method == 'gd':
            gx, gy = grad(x, y)
        elif method == 'dd':
            gx, gy = zo_dd(x, y)
        else:
            raise NotImplementedError

        x = x - lr * gx
        y = y - lr * gy

        num_iter.append(i+1)
        loss_val.append(L_val)
        trace_val.append(hess_tr)

    print(f"finish in {time.time() - start_time}s.")
    return num_iter, loss_val, trace_val


if __name__ == "__main__":
    T = 100000
    d = 100

    fo_lr = 0.01
    zo_lr = 0.001
    lam = 0.1

    for algo in ["GD", "ZO", "DD"]:

        seed = 17

        np.random.seed(seed)
        x = np.random.randn(d)
        y = np.random.randn(d)

        if algo == "GD":
            num_iter, one_loss, one_hess = run(T, x, y, fo_lr, 'gd')
        elif algo == "ZO":
            num_iter, one_loss, one_hess = run(T, x, y, zo_lr, 'zo', lam)
        else:
            num_iter, one_loss, one_hess = run(T, x, y, zo_lr, 'dd')

        data = {
            "num_iter": num_iter,
            "list_loss": one_loss,
            "list_hess": one_hess
        }

        with open(f"./res/{algo}-{seed}.json", "w") as f:
            json.dump(data, f, indent=2)


    result = {}
    for lam in [0.1, 0.05, 0.01]:

        seed = 17

        np.random.seed(seed)
        x = np.random.randn(d)
        y = np.random.randn(d)

        num_iter, one_loss, one_hess = run(T, x, y, zo_lr, 'zo', lam)

        data = {
            "num_iter": num_iter,
            "list_loss": one_loss,
            "list_hess": one_hess
        }

        result[str(lam)] = data

    with open(f"./res/lam-{seed}.json", "w") as f:
        json.dump(result, f, indent=2)