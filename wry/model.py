"""
wry: weighted regression for hydrology
"""
from math import ceil
import numpy as np

from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.metrics import r2_score


class LinearRegression(skLinearRegression):
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
    def fit(self, x, y, sample_weight=None):
        super().fit(x, y, sample_weight=sample_weight)
        self._n = x.shape[0]
        self._p =  x.shape[1]
        self.yest = self.predict(x)
        self.resid = y - self.yest

    @property
    def rss(self):
        """Residual Sum of Squares.
        """
        y = self.resid + self.yest
        return np.sum( (y-y.est)**2 )

    @property
    def tss(self):
        """Total Sum of Squares.
        """
        y = self.resid + self.yest
        return np.sum( (y-y.mean())**2 )

    @property
    def r2(self):
        """Coefficient of determination (R^2).
        """
        return 1 - self.rss/self.tss
        #return r2_score(self.resid + self.yest, self.yest)

    @property
    def adjusted_r2(self):
        """ Adjusted coefficient of determination.
        """
        return 1 - (1-self.r2)*(self._n-1)/(self._n-self._p-1)


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


class Lowess(LinearRegression):
    """
    Locally weighted regression, also known as LOWESS.

    Parameters
    ----------
    n_jobs : int
        The number of jobs to use for the computation. This will only
        provide speedup for n_targets > 1 and sufficient large problems.
        -1 uses all processors.

    References
    ----------
    ..[1] William S. Cleveland (1979) Robust Locally Weighted Regression and
    Smoothing Scatterplots, Journal of the American Statistical Association,
    74:368, 829-836
    """
    def predict(self):
        raise NotImplementedError('This feature may be implemented in future')

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
        h = [np.sort(np.abs(x - x[i]).flatten())[r] for i in range(n)]
        w = np.clip(np.abs((x - x.transpose()) / h), 0.0, 1.0)
        w = (1 - w ** 3) ** 3
        yest = np.zeros([n,1])
        delta = np.ones([n,1])
        smear = np.zeros([n,1]) # array of smearing coefficients 

        for iteration in range(iter):

            for i in range(n):

                weights = delta * w[:,i,None] #[:, i]
                super().fit(x, y, sample_weight=weights.flatten())
                yest_local = super().predict(x)
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


class WRTDS(Lowess):
    """Weighted Regression on Time Discharge and Season.

    Warning: Not yet implemented.
    """
    def __init__(self):
        raise NotImplementedError()


    def fit(self, dataframe,
            constituent,
            surrogates,
            time_step=1/365.5,
            surrogate_hw =3.0, #2.0 orig
            seasonal_hw = 1, #0.5 orig
            trend_hw = 7.0):
        """ Initialize a WRSST model

        Parameters
        ----------
        dataframe : DataFrame
        time_step : float
        surrogate_hw : float
        seasonal_hw : float
        trend_hw : float
        """
        self.seasonal_halfwidth = seasonal_hw
        self.surrogate_halfwidth = surrogate_hw
        self.trend_halfwidth = trend_hw
        self.time_step = time_step

        self._halfwidth_matrix = np.array([])


    def predict(self):
        pass


    def _sample_weights(self, design_vector, design_matrix):
        """Calculate sample weights for WRTDS

        Parameters
        ----------
        design_vector : array
            Array containing explantory variables at current observations

        design_matrix : array
            Array containing all

        Returns
        -------
        An array containing the weights of each sample relative to the observations
        """
        sample_distance = self.distance(design_matrix, design_vector)
        sample_weight = self.tricubic_weight(sample_distance,
                                            self.halfwidth_vector)

        return sample_weight


    @property
    def halfwidth_vector(self):
        """
        Halfway translated
        """
        if self._halfwidth_matrix.size > 0:
            halfwidth_matrix = self._halfwidth_matrix

        else:
            halfwidth_matrix = np.empty(self.input_data.surrogate_count + 3)
            halfwidth_matrix[:-3] = self.surrogate_halfwidth
            halfwidth_matrix[-1] = self.trend_halfwidth
            halfwidth_matrix[-2] = self.seasonal_halfwidth
            halfwidth_matrix[-3] = self.seasonal_halfwidth
            self._halfwidth_vector = halfwidth_vector

        return halfwidth_vector


    @staticmethod
    def distance(array_1, array_2):
        """ L1 distance between two arrays.
        """
        return np.abs(array_1 - array_2)


    @staticmethod
    def tricubic_weight(distance, halfwidth_window):
        """ Tricube weight function (Tukey, 1977).

        Parameters
        ----------
        distance : array

        halfwidth_window : array

        Returns
        -------
        An array of weights.
        """
        x = np.divide(distance, halfwidth_window,
                      out=np.zeros_like(distance),
                      where=halfwidth_window!=0)

        weights = (1 - x**3)**3

        weights =  np.where(distance <= halfwidth_window, weights, 0)

        return np.prod(weights, axis=1)
