import numpy as np
import scipy.linalg as LA
import scipy.sparse as spr
import scipy.sparse.linalg as spr_LA
from time import perf_counter

def safe_division(x, y):
    """
    Computes safe division x/y for small positive values x and y
    """
    return np.exp(np.log(x) - np.log(y)) if y != 0 else 1e16


def ad_grad(J, df, x0, la_0=1e-6, numb_iter=100):
    """
    Minimize f(x) by adaptive gradient method.
    Takes J as some evaluation function for comparison.
    """
    begin = perf_counter()
    x_old = x0
    grad_old = df(x0)
    x = x0 - la_0 * grad_old
    la_old = 1
    th = 1e9
    steps_array = []
    values = [J(x0)]
    x_values = [x0]

    for i in range(numb_iter):
        grad = df(x)
        norm_x = LA.norm(x - x_old)
        norm_grad = LA.norm(grad - grad_old)
        #la = min(np.sqrt(1 + th) * la_old,  0.5 * norm_x / norm_grad)
        la = min(np.sqrt(1 + th) * la_old,  0.5 * safe_division(norm_x, norm_grad))
        th = la / la_old
        x_old = x.copy()
        x -= la * grad
        la_old = la
        grad_old = grad
        values.append(J(x))
        x_values.append(x)
        steps_array.append(la)
    end = perf_counter()

    print("Time execution of adaptive gradient descent:", end - begin)
    return np.array(values), np.array(x_values), x, steps_array

# GD #
def gd(f, df, x0, la=1, numb_iter=100):
    """
    Gradient descent for minimizing smooth f.
    """
    begin = perf_counter()
    x = x0.copy()
    values = [f(x0)]
    x_values = [x0]

    for i in range(numb_iter):
        grad = df(x)
        x -=  la * grad
        values.append(f(x))
        x_values.append((x))
    end = perf_counter()
    print("Time execution for GD:", end - begin)
    return np.array(values), np.array(x_values),x

