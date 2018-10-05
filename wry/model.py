from math import ceil
import numpy as np
from scipy import linalg


class Model():
    pass


class Lowess(Model):
    """
    """
    def fit(self, x, y, f=0.75, iter=1):
        """ Fit lowess model.

        Parameters
        ----------
        x : array_like
            Training data.
        y : array_like
            Target values.
        f : float
            Span.
        iter : int
            Number of iterations to run.

        Notes
        -----
        """
        n = len(x)
        r = int(ceil(f * n)) # range of points used in local regression
        h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
        w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
        w = (1 - w ** 3) ** 3
        yest = np.zeros(n)
        delta = np.ones(n)
        smear = np.zeros(n) # array of smearing coefficients 

        for iteration in range(iter):

            for i in range(n):

                weights = delta * w[:, i]
                b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
                A = np.array([[np.sum(weights), np.sum(weights * x)],
                              [np.sum(weights * x), np.sum(weights * x * x)]])
                beta = linalg.solve(A, b)
                yest[i] = beta[0] + beta[1] * x[i]
                yest_local = beta[0] + beta[1] * x
                smear[i] = self._weighted_duan_smearing_coef(weights, y-yest_local)

            residuals = y - yest
            s = np.median(np.abs(residuals))

            if np.sum(s) == 0:
                break

            else:
                delta = np.clip(residuals / (6.0 * s), -1, 1)
                delta = (1 - delta ** 2) ** 2

        self.resid = residuals
        self.yest = yest
        self.smear = smear


    @staticmethod
    def _weighted_duan_smearing_coef(weights, residuals):
        """
        Parameters
        ----------
        weights : array
        residuals : array
        """
        weighted_sum_of_residuals = (weights * np.exp(residuals)).sum()
        sum_of_weights = weights.sum()
        return weighted_sum_of_residuals / sum_of_weights


class Loadest(Lowess):
    """Fit Loadest model.

    Warning: Not yet implemented.
    """
    def __init__(self):
        raise NotImplementedError()


class WRTDS(Lowess):
    """Fit WRTDS model.

    Warning: Not yet implemented.
    """
    def __init__(self):
        raise NotImplementedError()


class WRTDSX(WRTDS):
    """Fit WRTDSX model.

    Warning: Not yet implemented.
    """
     def __init__(self):
        raise NotImplementedError()
