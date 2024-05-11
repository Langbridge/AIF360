from warnings import warn

import numpy as np
import pandas as pd

from aif360.algorithms import Transformer

from sklearn.neighbors import KernelDensity
import ot


class DistributionalRepair(Transformer):
    """
    """

    def __init__(self):
        super(DistributionalRepair, self).__init__()

    def fit(self, s_R, u_R, x_R, feat, n_q, u_val, continuous):
        """
        """
        if continuous:
            return self._continuous_fit(s_R, u_R, x_R, feat, n_q, u_val)
        else:
            return self._discrete_fit(s_R, u_R, x_R, feat, u_val)

    def _continuous_fit(self, s_R, u_R, x_R, feat, n_q, u_val):
        """
        """
        self.support = self._get_support(x_R, u_R, feat, u_val, n_q)
        pmf_0, pmf_1 = self._get_pmfs(x_R, s_R, u_R, feat, u_val)
        barycenter = self._get_barycenter(pmf_0, pmf_1)
        self.T_0, self.T_1 = self._get_transport_plans(pmf_0, pmf_1, barycenter)
        return self

    def _discrete_fit(self, s_R, u_R, x_R, feat, u_val):
        if self._is_valid_data(x_R, s_R, u_R, u_val):
            self.pmf_0, self.pmf_1 = self._get_discrete_pmfs(x_R, s_R, u_R, feat, u_val)
            self.T = self._get_discrete_transport_plan(self.pmf_0, self.pmf_1)
        else:
            self.pmf_0 = None
            self.pmf_1 = None
        return self

    def transform(self, s_R, u_R, x_R, feat, s_A, u_A, x_A, tilde_x_R, tilde_x_A, u_val, continuous):
        """
        """
        if continuous:
            return self._continuous_transform(s_R, u_R, x_R, feat, s_A, u_A, x_A, tilde_x_R, tilde_x_A, u_val)
        else:
            return self._discrete_transform(s_R, u_R, x_R, feat, s_A, u_A, x_A, tilde_x_R, tilde_x_A, u_val)

    def _continuous_transform(self, s_R, u_R, x_R, feat, s_A, u_A, x_A, tilde_x_R, tilde_x_A, u_val):
        tilde_x_R = self._repair_continuous_data(x_R, s_R, u_R, feat, tilde_x_R, u_val)
        tilde_x_A = self._repair_continuous_data(x_A, s_A, u_A, feat, tilde_x_A, u_val)
        return tilde_x_A, tilde_x_R

    def _discrete_transform(self, s_R, u_R, x_R, feat, s_A, u_A, x_A, tilde_x_R, tilde_x_A, u_val):
        if self.pmf_0 is None or self.pmf_1 is None:
            return tilde_x_A, tilde_x_R
        tilde_x_R = self._repair_discrete_data(x_R, s_R, u_R, feat, tilde_x_R, u_val)
        tilde_x_A = self._repair_discrete_data(x_A, s_A, u_A, feat, tilde_x_A, u_val)
        return tilde_x_A, tilde_x_R

    def fit_transform(self, dataset):
        """
        """
        return self.fit(dataset).transform(dataset)

    def _get_support(self, x_R, u_R, feat, u_val, n_q):
        min_val = np.min(x_R[(u_R == u_val)][feat]) - np.ptp(x_R[(u_R == u_val)][feat])*0.1
        max_val = np.max(x_R[(u_R == u_val)][feat]) + np.ptp(x_R[(u_R == u_val)][feat])*0.1
        return np.linspace(min_val, max_val, n_q).reshape(-1,1)

    def _get_pmfs(self, x_R, s_R, u_R, feat, u_val):
        kde_0 = KernelDensity(kernel='gaussian').fit(x_R[(u_R == u_val) & (s_R == 0.0)][feat].values.reshape(-1,1))
        pmf_0 = np.exp(kde_0.score_samples(self.support))
        kde_1 = KernelDensity(kernel='gaussian').fit(x_R[(u_R == u_val) & (s_R == 1.0)][feat].values.reshape(-1,1))
        pmf_1 = np.exp(kde_1.score_samples(self.support))
        pmf_0 /= np.sum(pmf_0)
        pmf_1 /= np.sum(pmf_1)
        if np.any(np.isnan(pmf_0)) or np.any(np.isnan(pmf_1)):
            raise ZeroDivisionError("One or more PMFs have sum zero")
        return pmf_0, pmf_1

    def _get_barycenter(self, pmf_0, pmf_1):
        M = ot.utils.dist(self.support, self.support)
        A = np.vstack([pmf_0, pmf_1]).T
        barycenter = ot.bregman.barycenter(A, M, 10)
        if np.any(np.isnan(pmf_0)) or np.any(np.isnan(pmf_1)):
            raise RuntimeError("No valid barycenter was found, try to increase reg")
        return barycenter

    def _get_transport_plans(self, pmf_0, pmf_1, barycenter):
        M = ot.utils.dist(self.support, self.support)
        T_0 = ot.emd(pmf_0, barycenter, M)
        T_1 = ot.emd(pmf_1, barycenter, M)
        return T_0, T_1

    def _is_valid_data(self, x_R, s_R, u_R, u_val):
        return (len(x_R[(u_R == u_val) & (s_R == 0)]) > 1) and (len(x_R[(u_R == u_val) & (s_R == 1)]) > 1)

    def _get_discrete_pmfs(self, x_R, s_R, u_R, feat, u_val):
        pmf_0 = x_R[(u_R == u_val) & (s_R == 0)][feat].value_counts()
        pmf_1 = x_R[(u_R == u_val) & (s_R == 1)][feat].value_counts()
        return pmf_0, pmf_1

    def _get_discrete_transport_plan(self, pmf_0, pmf_1):
        M = ot.dist(pmf_0.index.values.reshape(-1,1), pmf_1.index.values.reshape(-1,1))
        weights = [pmf_0.values / pmf_0.values.sum(), pmf_1.values / pmf_1.values.sum()]
        return ot.emd(weights[0], weights[1], M)

    def _repair_continuous_data(self, x, s, u, feat, tilde_x, u_val):
        for i, row in x[(u == u_val)].iterrows():
            if s[i] == 1:
                tilde_x.loc[i, feat] = self._repair_data(row[feat], self.support[:,0], self.support[:,0], self.T_1)
            else:
                tilde_x.loc[i, feat] = self._repair_data(row[feat], self.support[:,0], self.support[:,0], self.T_0)
        return tilde_x

    def _repair_discrete_data(self, x, s, u, feat, tilde_x, u_val):
        for i, row in x[(u == u_val)].iterrows():
            if s[i] == 1:
                tilde_x.loc[i, feat] = self._repair_data(row[feat], self.pmf_1.index.values, self.pmf_0.index.values, self.T.T, i_split=False, j_split=False)
            else:
                tilde_x.loc[i, feat] = self._repair_data(row[feat], self.pmf_0.index.values, self.pmf_1.index.values, self.T, i_split=False, j_split=False)
        return tilde_x

    def _repair_data(self, x, support_i, support_j, T, i_split=True, j_split=False):
        if i_split:
            idx = np.searchsorted(support_i, x, side='left')
            if idx == 0 or idx == len(support_i):
                i = min(idx, len(support_i)-1)
            else:
                interp = float(x - support_i[idx-1]) / np.diff(support_i)[0]
                if np.round(interp, 4) == 1.0:
                    i = idx
                else:
                    i = np.random.choice([idx-1, idx], p=[1-interp, interp])
        else:
            i = np.argwhere(support_i == x)[0,0]
        
        if not j_split:
            if np.sum(T[i]) > 0.0:
                j = np.random.choice(T.shape[1], p=(T[i] / np.sum(T[i]))) # stochastic choice of which marginal entry to transport to
            else:
                j = i
            x_repaired = support_j[j]
        else:
            row = T[i] / np.sum(T[i])
            x_repaired = 0.5*x + 0.5*row@support_j
        return x_repaired