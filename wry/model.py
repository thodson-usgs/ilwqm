"""
wry: weighted regression for hydrology
"""
from math import ceil
import numpy as np

from sklearn.linear_model import LinearRegression


class Model():
    """
    """
    @staticmethod
    def _duan_smearing_coef(residuals, weights=1):
        """Weighted smearing coefficient.

        Calculates duans smearing coefficient used to correct for
        retransformation bias that occurs when using log-transformed dara in
        linear regression.

        Parameters
        ----------
        weights : array
        residuals : array

        References
        ----------
        ..[1] Hirsch, R. M., Moyer, D. L., & Archfield, S. A. (2010). Weighted
        regressions on time, discharge, and season (WRTDS), with an application
        to Chesapeake Bay river inputs 1. JAWRA Journal of the American Water
        Resources Association, 46(5), 857-880.
        """
        weighted_sum_of_residuals = (weights * np.exp(residuals)).sum()
        sum_of_weights = np.sum(weights)
        return weighted_sum_of_residuals / sum_of_weights


class LinearRegression(Model, LinearRegression):
	"""
    Ordinary least squares Linear Regression.

    Parameters
    ----------
    fit_intercept : boolean, optional, default True
        whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
        an estimator with ``normalize=False``.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This will only provide
        speedup for n_targets > 1 and sufficient large problems.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    intercept_ : array
        Independent term in the linear model.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.score(X, y)
    1.0
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_ # doctest: +ELLIPSIS
    3.0000...
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (scipy.linalg.lstsq) wrapped as a predictor object.
    """
	pass


class Lowess(LinearRegression):
    """
    Locally weighted regression, also known as LOWESS.

    Parameters
    ----------
    n_jobs : int
        The number of jobs to use for the computation. This will only
        provide speedup for n_targets > 1 and sufficient large problems.
        -1 uses all processors.


    """
    def fit(self, x, y, f=0.75, iter=1, n_jobs=None):
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
        w = np.clip(np.abs((x - x.transpose()) / h), 0.0, 1.0)
        w = (1 - w ** 3) ** 3
        yest = np.zeros(n)
        delta = np.ones(n).reshape(-1, 1)
        smear = np.zeros(n) # array of smearing coefficients 

        for iteration in range(iter):

            for i in range(n):
                weights = delta * w[:, i]
                super().fit(x, y, sample_weight=weights[:,0])
                # TODO: need to avoid indexing of weights
                yest_local = self.predict(x)
                yest[i] = yest_local[i]
                smear[i] = self._duan_smearing_coef(y-yest_local, weights)

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
