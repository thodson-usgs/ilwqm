import pandas as pd
import numpy as np

from sklearn.linear_model import LassoCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#TODO import block bootstrap

class LassoImputeBootstrap:
    """
    """

    def __init__(sur_df, con_df, bcf=True):
        """
        """
        # merge df

        # get only complete samples

        # begin lasso cross-validation
        reg = LassoCV(cv=5, random_state=0).fit(X,y)

        # save positive regression coefficients

        pass

    def fit():
        """
        """
        pass
        imp = IterativeImputer(max_iter=10, random_state=0)

    def transform():

    def boot_fit(n):
        """
        """
        pass

    def plot():
        """
        """
        pass
