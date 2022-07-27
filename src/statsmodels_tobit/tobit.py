import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import GenericLikelihoodModel, GenericLikelihoodModelResults, _LLRMixin
from statsmodels.api import OLS


class TobitModel(GenericLikelihoodModel):
    def __init__(self, endog, exog, w=None, lower_bound=-np.Inf, upper_bound=np.Inf, **kwds):
        exog = np.reshape(exog, (endog.shape[0], -1))
        super(TobitModel, self).__init__(endog, exog, **kwds)
        self.w = w if w else 1
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        n, p = exog.shape
        self.df_model = p + 1
        self.df_resid = n - self.df_model
        self.k_constant = 0

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of the Tobit regression model.

        Parameters
        ----------
        params : ndarray
            The parameters of the model, coefficients for linear predictors
            of the mean and of the precision function.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`.
        """

        return self._llobs(self.endog, self.exog, self.lower_bound, self.upper_bound, params)

    def _llobs(self, endog, exog, lower_bound, upper_bound, params):
        """
        Loglikelihood for observations with data arguments.

        Parameters
        ----------
        endog : ndarray
            1d array of endogenous variable.
        exog : ndarray
            2d array of explanatory variables.
        lower_bound : float
            lower censor value.
        upper_bound : float
            upper censor value.
        params : ndarray
            The parameters of the model, coefficients for linear predictors
            of the mean and of the precision function.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`.
        """

        y, X, lb, ub = endog, exog, lower_bound, upper_bound

        # convert log(sigma) into sigma. This transformation ensures that sigma > 0
        sigma = np.exp(params[-1])
        beta = params[:-1]

        i_lb = (y <= lb)
        i_ub = (y >= ub)
        i_m = np.logical_not(i_lb) & np.logical_not(i_ub)
        xb = np.dot(X, beta)
        ll = np.zeros_like(y)
        ll[i_lb] = norm.logcdf((lb - xb[i_lb]) / sigma)
        ll[i_ub] = norm.logcdf((xb[i_ub] - ub) / sigma)
        ll[i_m] = norm.logpdf((y[i_m] - xb[i_m]) / sigma) - np.log(sigma)

        return ll * self.w



    def score_obs(self, params, **kwds):
        """
        Returns the score vector of the log-likelihood.

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.
        Returns
        -------
        score : ndarray
            First derivative of loglikelihood function.
        """

        return self._score_obs(self.endog, self.exog, self.lower_bound, self.upper_bound, params)

    def _score_obs(self, endog, exog, lower_bound, upper_bound, params, **kwds):
        """
        Returns the score vector of the log-likelihood.

        Parameters
        ----------
        endog : ndarray
            1d array of endogenous variable.
        exog : ndarray
            2d array of explanatory variables.
        lower_bound : float
            lower censor value.
        upper_bound : float
            upper censor value.
        params : ndarray
            Parameter at which score is evaluated.
        Returns
        -------
        score : ndarray
            First derivative of loglikelihood function.
        """

        y, X, lb, ub = endog, exog, lower_bound, upper_bound

        # convert log(sigma) into sigma. This transformation ensures that sigma > 0
        sigma = np.exp(params[-1])
        beta = params[:-1]

        i_lb = (y <= lb)
        i_ub = (y >= ub)
        i_m = np.logical_not(i_lb) & np.logical_not(i_ub)
        xb = np.dot(X, beta)
        g = np.zeros((y.shape[0], params.shape[0]))

        # <= LB
        z = (lb - xb[i_lb]) / sigma
        l_pdf = norm.logpdf(z)
        l_cdf = norm.logcdf(z)
        f = np.exp(l_pdf - l_cdf)

        g[i_lb, :-1] = -f[:, np.newaxis] * X[i_lb] / sigma
        g[i_lb, -1] = -f * z / sigma

        # >= UB
        z = (xb[i_ub] - ub) / sigma
        l_pdf = norm.logpdf(z)
        l_cdf = norm.logcdf(z)
        f = np.exp(l_pdf - l_cdf)

        g[i_ub, :-1] = f[:, np.newaxis] * X[i_ub] / sigma
        g[i_ub, -1] = -f * z / sigma

        # Mid
        z = (y[i_m] - xb[i_m]) / sigma
        g[i_m, :-1] = z[:, np.newaxis] * X[i_m] / sigma
        g[i_m, -1] = (z ** 2 - 1) / sigma

        return g * self.w

    def _start_params(self):
        # Reasonable starting values based on OLS within bounds regression
        i_m = (self.lower_bound <= self.endog) & (self.endog <= self.upper_bound)
        ols_model = OLS(self.endog[i_m], self.exog[i_m]).fit()
        start_params = np.append(ols_model.params, np.log(ols_model.mse_resid ** 0.5))
        return start_params

    def fit(self, start_params=None, maxiter=10000, method='bfgs', **kwds):
        """
        Fit the model by maximum likelihood.

        Parameters
        ----------
        start_params : array-like
            A vector of starting values for the regression
            coefficients.  If None, a default is chosen.
        maxiter : integer
            The maximum number of iterations
        method : str
            The optimization method to use.
        kwds :
            Keyword arguments for the optimizer.

        Returns
        -------
        TobitResults instance.
        """

        # we have one additional parameter and we need to add it to the summary
        self.exog_names.append('log(sigma)')
        if start_params is None:
            start_params = self._start_params()

        mlfit = super(TobitModel, self).fit(
            start_params=start_params,
            maxiter=maxiter,
            method=method,
            **kwds
        )

        return TobitResults(self, mlfit)

    def predict(self, params, exog=None, which='mean', *args, **kwargs):
        if exog is None:
            exog = self.exog

        lb = self.lower_bound
        ub = self.upper_bound
        sigma = np.exp(params[-1])
        beta = params[:-1]
        xb = np.dot(exog, beta)

        if which == 'linear':
            return xb

        z_lb = (lb - xb) / sigma
        z_ub = (ub - xb) / sigma

        lb_cdf = norm.cdf(z_lb)
        lb_pdf = norm.pdf(z_lb)
        ub_cdf = norm.cdf(z_ub)
        ub_pdf = norm.pdf(z_ub)
        m_cdf = ub_cdf - lb_cdf
        m_pdf = lb_pdf - ub_pdf

        h = m_pdf / m_cdf
        i_lb = (lb_cdf == 1) & (ub_cdf == 1)
        i_ub = (lb_cdf == 0) & (ub_cdf == 0)

        e_y_conditional = np.clip(xb + sigma * h, lb, ub)
        e_y_conditional[i_lb] = lb
        e_y_conditional[i_ub] = ub
        e_y_unconditional = lb_cdf * lb + m_cdf * e_y_conditional + (1 - ub_cdf) * ub
        var_f = (1 + (z_lb * lb_pdf - z_ub * ub_pdf) / m_cdf - h ** 2)
        var = (sigma ** 2) * var_f

        if which == 'conditional':
            return e_y_conditional
        elif which == 'mean':
            return e_y_unconditional
        elif which == 'var':
            return var
        elif which == 'all':
            return pd.DataFrame({
                'score': xb,
                'z_lb': z_lb,
                'z_ub': z_ub,
                'lb_cdf': lb_cdf,
                'lb_pdf': lb_pdf,
                'ub_cdf': ub_cdf,
                'ub_pdf': ub_pdf,
                'hazard': h,
                'conditional': e_y_conditional,
                'unconditional': e_y_unconditional,
                'var_scaling_factor': var_f,
                'variance': var
            })
        else:
            raise ValueError(f'which = {which} is not available')


class TobitResults(GenericLikelihoodModelResults, _LLRMixin):
    """Results class for Tobit regression

    This class inherits from GenericLikelihoodModelResults and not all
    inherited methods might be appropriate in this case.
    """

    # GenericLikeihoodmodel doesn't define fitted_values, residuals and similar
    @cache_readonly
    def fitted_values(self):
        """In-sample predicted mean, conditional expectation."""
        return self.model.predict(self.params)

    @cache_readonly
    def fitted_linear(self):
        """In-sample predicted precision"""
        return self.model.predict(self.params, which="linear")

    @cache_readonly
    def resid(self):
        """Response residual"""
        return self.model.endog - self.fitted_values

    @cache_readonly
    def resid_pearson(self):
        """Pearson standardize residual"""
        std = np.sqrt(self.model.predict(self.params, which="var"))
        return self.resid / std

    @cache_readonly
    def prsquared(self):
        """Cox-Snell Likelihood-Ratio pseudo-R-squared.
        1 - exp((llnull - .llf) * (2 / nobs))
        """
        return self.pseudo_rsquared(kind="lr")

    def bootstrap(self, *args, **kwargs):
        raise NotImplementedError
