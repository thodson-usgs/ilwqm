import pandas as pd
import numpy as np

from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#TODO import block bootstrap

class Model:
    """
    Assumes data are Missing at Random (MAR) -- missingness only depends on
    observed variables, not unobserved variables.

    Attributes
    ----------
    bcf : bool
        Whether to apply a bias correction factor to output
    """

    def __init__(self, df, target,
                 estimator=BayesianRidge(),
                 missing_values=np.nan,
                 max_iter=10,
                 tol=1e-3,
                 n_jobs=None,
                 seed = 23412,
                ):
        """
        Parameters
        ---------
        df : DataFrame
        target :  str

        """
        self.data = df
        self.target = target
        self.bcf = bcf

        self.n_jobs = n_jobs

        self._imputer_kwargs = {
            estimator : estimator,
            missing_values : missing_values,
            max_iter : max_iter,
            tol : tol,
            sample_posterior : True,
        }

    def fit(self, n=0):
        """
        Parameters
        ----------
        n : int
            Number of block bootstrap replicates
        """
        imp = IterativeImputer(**self.imp,
                               random_state=None)
        col = 1
        rows = self.data.shape[0]
        temp = imp(self.data)
        self.result = temp[:,col]

        self.bs_result = np.zeros([rows, n])

        for i in range(n):
            imp = IterativeImputer(**self.imp,
                                   random_state=
            # get block bootstrap sample

            # create temp dataset including sample

            # impute the result
            self.bs_result[:,n] = self.imp(bbs_sample[col])



    def bcf(self):
        """
        Attribute
        """
        pass

    def predict(self):
        # transform results
        return  0

    def plot(self):
        """
        """
        pass
