import numpy as np

def my_f(x):
    return np.divide(x, np.sum(x))

def softmaxvector(x,w):
    """This function uses padded x with 1s in order to not use weights and bias seperately"""
    z = x @ w
    a = np.exp(z)
    result = np.apply_along_axis(my_f, 1, a)
    return result

def crossentropy(x,w,y):
    """This function uses padded x with 1s in order to not use weights and bias seperately"""
    z = x @ w
    a = np.exp(z)
    result = np.apply_along_axis(my_f, 1, a)
    result1 = np.log(result)
    result2 = np.multiply(result1, y)
    result3 = np.sum(result2)
    return -result3
def crossentropygradient(x,w,y):
    """This function uses padded x with 1s in order to not use weights and bias seperately"""
    (m, n) = x.shape
    ws = w[0:-1]
    bs = w[-1]
    t1 = x.T
    t2 = t1[0:-1]
    x_real = t2.T

    grad_for_w = (1 / m) * np.dot(x_real.T, softmaxvector(x, w) - y)
    gradb = (softmaxvector(x, w) - y)
    grad_for_b = np.sum(np.array(gradb), axis=0)
    return (grad_for_w,grad_for_b)

def CrossEntropy(x,w,y):
    y_pred=softmaxvector(x,w)
    eps = np.finfo(float).eps
    cross_entropy = -np.sum(y * np.log(y_pred + eps))
    return cross_entropy
