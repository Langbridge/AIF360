import numpy as np
import pandas as pd
import ot
from aif360.algorithms import Transformer
from aif360.datasets import StandardDataset
from tqdm import tqdm


class PointwiseRepair(Transformer):
    """Pointwise Repair class for mitigating bias in datasets.

    This class implements the Pointwise Repair algorithm to mitigate bias
    in datasets by aligning the distributions of protected and unprotected groups
    for each feature conditioned on the outcome variable.

    Args:
        s (str): Name of the protected attribute.
        u (str): Name of the outcome variable.
        x (list): List of feature names to be repaired.
        y (str): Name of the target variable.
    """

    def __init__(self, s, u, x, y):
        super(PointwiseRepair, self).__init__()
        self.s = s
        self.u = u
        self.x = x
        self.y = y

    def fit_transform(self, dataset):
        """Fit and transform the dataset to mitigate bias.

        Args:
            dataset (StandardDataset): Dataset to be transformed.

        Returns:
            StandardDataset: Transformed dataset with bias mitigated.
        """
        dataframe = dataset.convert_to_dataframe()[0]
        s_D, u_D, x_D, y_D = self._split_dataframe(dataframe)
        tilde_x_D = x_D.copy()

        for u_val in tqdm(u_D.unique()):
            M = ot.dist(
                x_D[(u_D == u_val) & (s_D == 0)].values,
                x_D[(u_D == u_val) & (s_D == 1)].values
            )
            n_0 = int(len(x_D[(u_D == u_val) & (s_D == 0)]))
            n_1 = int(len(x_D[(u_D == u_val) & (s_D == 1)]))
            T = ot.emd(np.ones(n_0) / n_0, np.ones(n_1) / n_1, M, numItermax=10000)

            for i, (pd_idx, x) in enumerate(x_D[(u_D == u_val) & (s_D == 0)].iterrows()):
                row = T[i, :]
                epsilon = 1e-8
                row /= (np.sum(row) + epsilon)
                tilde_x_D.loc[pd_idx, :] = 0.5 * x.values + 0.5 * row @ x_D[(u_D == u_val) & (s_D == 1)].values

            for j, (pd_idx, x) in enumerate(x_D[(u_D == u_val) & (s_D == 1)].iterrows()):
                col = T[:, j]
                epsilon = 1e-8
                col /= (np.sum(col) + epsilon)
                tilde_x_D.loc[pd_idx, :] = 0.5 * x.values + 0.5 * col @ x_D[(u_D == u_val) & (s_D == 0)].values

        tilde_dataframe_D = pd.concat([tilde_x_D, dataframe.drop(columns=self.x)], axis=1)
        tilde_dataset_D = StandardDataset(
            df=tilde_dataframe_D,
            label_name=self.y,
            favorable_classes=[1],
            protected_attribute_names=[self.s],
            privileged_classes=[[1]]
        )

        return tilde_dataset_D

    def _split_dataframe(self, dataframe):
        """Split the dataframe into protected attribute, outcome, features, and target.

        Args:
            dataframe (DataFrame): Input dataframe.

        Returns:
            tuple: Protected attribute series, outcome series, feature dataframe, and target series.
        """
        s_D = dataframe[self.s]
        u_D = dataframe[self.u]
        x_D = dataframe[self.x]
        y_D = dataframe[self.y]
        return s_D, u_D, x_D, y_D