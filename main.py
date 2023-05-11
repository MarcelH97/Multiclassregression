import numpy as np

import numpy.linalg as la

from trainer import Trainer


class Gd(Trainer):
    """
    Gradient descent with constant learning rate.

    Arguments:
        lr (float): an estimate of the inverse smoothness constant
    """

    def __init__(self, lr, *args, **kwargs):
        super(Gd, self).__init__(*args, **kwargs)
        self.lr = lr

    def step(self):
        return self.w - self.lr * self.grad

    def init_run(self, *args, **kwargs):
        super(Gd, self).init_run(*args, **kwargs)


class Adgd(Trainer):
    """
    Adaptive gradient descent based on the local smoothness constant

    Arguments:
        eps (float, optional): an estimate of 1 / L^2, where L is the global smoothness constant (default: 0)
    """

    def __init__(self, eps=0.0, lr0=None, *args, **kwargs):
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {}".format(eps))
        super(Adgd, self).__init__(*args, **kwargs)
        self.eps = eps
        self.lr0 = lr0

    def estimate_stepsize(self):
        L = la.norm(self.grad - self.grad_old) / la.norm(self.w - self.w_old)
        if np.isinf(self.theta):
            lr_new = 0.5 / L
        else:
            lr_new = min(np.sqrt(1 + self.theta) * self.lr, self.eps / self.lr + 0.5 / L)
        self.theta = lr_new / self.lr
        self.lr = lr_new

    def step(self):
        self.w_old = self.w.copy()
        self.grad_old = self.grad.copy()
        return self.w - self.lr * self.grad

    def init_run(self, *args, **kwargs):
        super(Adgd, self).init_run(*args, **kwargs)
        self.theta = np.inf
        grad = self.grad_func(self.w)
        if self.lr0 is None:
            self.lr0 = 1e-10
        self.lr = self.lr0
        self.lrs = [self.lr]
        self.w_old = self.w.copy()
        self.grad_old = grad
        self.w -= self.lr * grad
        self.save_checkpoint()

    def update_logs(self):
        super(Adgd, self).update_logs()
        self.lrs.append(self.lr)


class Bb(Trainer):
    """
    Barzilai-Borwein Adaptive gradient descent based on the local smoothness constant
    """

    def __init__(self, lr0=1, option='1', *args, **kwargs):
        if not 0.0 < lr0:
            raise ValueError("Invalid lr0: {}".format(lr0))
        super(Bb, self).__init__(*args, **kwargs)
        self.lr0 = lr0
        self.option = option

    def estimate_stepsize(self):
        if self.option is '1':
            L = (self.w - self.w_old) @ (self.grad - self.grad_old) / la.norm(self.w - self.w_old) ** 2
        else:
            L = la.norm(self.grad - self.grad_old) ** 2 / ((self.grad - self.grad_old) @ (self.w - self.w_old))
        self.lr = self.lr0 / L

    def step(self):
        self.grad = self.grad_func(self.w)
        self.estimate_stepsize()
        self.w_old = self.w.copy()
        self.grad_old = self.grad.copy()
        return self.w - self.lr * self.grad

    def init_run(self, *args, **kwargs):
        super(Bb, self).init_run(*args, **kwargs)
        self.lrs = []
        self.theta = np.inf
        grad = self.grad_func(self.w)
        # The first estimate is normalized gradient with a small coefficient
        self.lr = 1 / la.norm(grad)
        self.w_old = self.w.copy()
        self.grad_old = grad
        self.w -= self.lr * grad
        self.save_checkpoint()

    def update_logs(self):
        super(Bb, self).update_logs()
        self.lrs.append(self.lr)


class Nesterov(Trainer):
    """
    Nesterov's accelerated gradient descent with constant learning rate.

    Arguments:
        lr (float): an estimate of the inverse smoothness constant
        strongly_convex (boolean, optional): if true, uses the variant
            for strongly convex functions, which requires mu>0 (default: False)
    """

    def __init__(self, lr, strongly_convex=False, mu=0, *args, **kwargs):
        super(Nesterov, self).__init__(*args, **kwargs)
        self.lr = lr
        if mu < 0:
            raise ValueError("Invalid mu: {}".format(mu))
        if strongly_convex and mu == 0:
            raise ValueError("""Mu must be larger than 0 for strongly_convex=True,
                             invalid value: {}""".format(mu))
        if strongly_convex:
            self.mu = mu
            kappa = (1 / self.lr) / self.mu
            self.momentum = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
        self.strongly_convex = strongly_convex

    def step(self):
        if not self.strongly_convex:
            alpha_new = 0.5 * (1 + np.sqrt(1 + 4 * self.alpha ** 2))
            self.momentum = (self.alpha - 1) / alpha_new
            self.alpha = alpha_new
        self.w_nesterov_old = self.w_nesterov.copy()
        self.w_nesterov = self.w - self.lr * self.grad
        return self.w_nesterov + self.momentum * (self.w_nesterov - self.w_nesterov_old)

    def init_run(self, *args, **kwargs):
        super(Nesterov, self).init_run(*args, **kwargs)
        self.w_nesterov = self.w.copy()
        self.alpha = 1.


class Armijo(Trainer):
    """
    Adaptive gradient descent based on the local smoothness constant

    Arguments:
        eps (float): an estimate of 1 / L^2, where L is the global smoothness constant
    """

    def __init__(self, backtracking=0.5, armijo_const=0.5, lr0=None, *args, **kwargs):
        if lr0 < 0:
            raise ValueError("Invalid lr0: {}".format(lr0))
        super(Armijo, self).__init__(*args, **kwargs)
        self.lr = lr0
        self.backtracking = backtracking
        self.armijo_const = armijo_const

    def estimate_stepsize(self):
        f = self.loss_func(self.w)
        lr = self.lr / self.backtracking
        w_new = self.w - lr * self.grad
        f_new = self.loss_func(w_new)
        armijo_condition = f_new <= f - self.lr * self.armijo_const * la.norm(self.grad) ** 2
        while not armijo_condition:
            lr *= self.backtracking
            w_new = self.w - lr * self.grad
            f_new = self.loss_func(w_new)
            armijo_condition = f_new <= f - lr * self.armijo_const * la.norm(self.grad) ** 2
            self.it += 1

        self.lr = lr

    def step(self):
        self.grad = self.grad_func(self.w)
        self.estimate_stepsize()
        return self.w - self.lr * self.grad

    def init_run(self, *args, **kwargs):
        super(Armijo, self).init_run(*args, **kwargs)
        self.w_ave = self.w.copy()
        self.ws_ave = [self.w_ave.copy()]
        self.lr_sum = 0
        self.lrs = []

    def update_logs(self):
        super(Armijo, self).update_logs()
        self.lrs.append(self.lr)
        self.ws_ave.append(self.w_ave.copy())
