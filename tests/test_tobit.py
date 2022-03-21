import pytest

__author__ = "geyerbisschoff"
__copyright__ = "geyerbisschoff"
__license__ = "MIT"


def test_tobit_model():
    """API Tests"""
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_regression
    from statsmodels_tobit import TobitModel, TobitResults

    rs = np.random.RandomState(seed=10)
    ns = 1000
    nf = 10
    noise = 0
    x, y_orig, coef = make_regression(n_samples=ns, n_features=nf, coef=True, noise=noise, random_state=rs)
    x = pd.DataFrame(x)
    y = pd.Series(y_orig)

    n_quantiles = 3 # two-thirds of the data is truncated
    quantile = 100 / float(n_quantiles)
    lower = np.percentile(y, quantile)
    upper = np.percentile(y, (n_quantiles - 1) * quantile)
    y = y.clip(upper=upper, lower=lower)

    tr = TobitModel(y, x, lower_bound=lower, upper_bound=upper).fit()
    assert isinstance(tr, TobitResults)
    assert np.all(np.round(tr.params[:-1], 4) == np.round(coef, 4))
    assert np.round(np.exp(tr.params[-1]), 4) == noise
    assert isinstance(tr.predict(which='all'), pd.DataFrame)
