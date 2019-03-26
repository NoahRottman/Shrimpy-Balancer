from scipy.optimize import minimize
import numpy as np

np.seterr(divide='ignore', invalid='ignore') # Log of 0 may be encountered


class PortfolioManager(object):
    def __init__(self, coins):
        """
        Initialize instance of the PortfolioManager.

        Parameters
        ----------
        coins: list of strings
            The ticker symbols for coins to be managered in portfolio
        """
        self.coins = coins
        self.n_coins = len(coins)
        self.portfolio_allocation = None
        self.p = None
        self.r = None

    def optimize_portfolio(self):
        """
        Compute the ``allocation``, the optimal control action when given predicted
        return rates ``p`` and corresponding confidence values ``r``.

        Parameters
        ----------
        r: np.array of shape (1, n_coins)
            The predicted return rates for the next time period.

        p: np.array of shape (1, n_coins)
            The confidence level for each return rate prediction.

        Returns
        -------
        success: bool
            Weather or not a maximizing prediciton was found

        allocation: np.array of shape (1, n_coins)
            The optimal distribution of
        """
        # Fund allocation must sum to 1
        cons = ({'type': 'eq',
                 'fun' : lambda x: np.sum(x) - 1
        })
        n_coins = r.shape[1]
        # Initialize allocation to be uniform amungst n_coins
        x_0 = [1/n_coins]*n_coins
        # Call minimization module on negated function to obtain maximization
        res = minimize(fun=func, x0=x_0, constraints=cons)
        # Return success and optimal array
        return res['success'], res['x']


    def func(self, x):
        """
        Compute the (negative) expected reward given portfolio allocation ``x``.

        Parameters
        ----------
        x: np.array of shape (1, n_coins)
        The allocation of portfolio value.

        Returns
        -------
        y: float
        The negated expected reward.
        """
        return -np.sum(self.p*np.log(100*x*(1+self.r)))

    def set_predicted_return_rates(self, r, p):
        """
        Set parameters ``r``, the return rates, and ``p``, the confidence scores
        for each predicted return rate.

        Parameters
        ----------
        r: np.array of shape (1, n_coins)
        The predicted return rates for the next time period.

        p: np.array of shape (1, n_coins)
        The confidence level for each return rate prediction.
        """
        self.r = r
        self.p = p
