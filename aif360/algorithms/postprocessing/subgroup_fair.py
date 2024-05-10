from aif360.algorithms.postprocessing.lds_baseclass import BaseLDS
import numpy as np
from picos import RealVariable
class SubgroupFairness(BaseLDS):
    """Subgroup fairness post-processing technique.

    This technique optimizes subgroup fairness, built off the BaseLDS class.

    This class implements the creation of the optimisation problem used to implement the subgroup fairness post-processing technique.
    """

    def __init__(self, privileged_groups, unprivileged_groups, debug=False):
        super().__init__(privileged_groups, unprivileged_groups, debug)

    def _get_constraints(self, D_train, privileged_train, unprivileged_train, a, c, d, e, z, attr1, attr2, attr3):
        """
        Create the constraints for the optimization problem.

        Args:
            D_train (pandas.DataFrame): Training dataframe.
            privileged_train (pandas.Index): Indices of privileged group in training data.
            unprivileged_train (pandas.Index): Indices of unprivileged group in training data.
            a (picos.RealVariable): Optimization variable for attribute 1.
            c (picos.RealVariable): Optimization variable for attribute 2.
            d (picos.RealVariable): Optimization variable for attribute 3.
            e (picos.RealVariable): Optimization variable for intercept.
            z (picos.RealVariable): Optimization variable for fairness constraint.
            attr1 (str): Name of attribute 1.
            attr2 (str): Name of attribute 2.
            attr3 (str): Name of attribute 3.

        Returns:
            tuple: A tuple containing the list of constraints and the objective function.
        """
        self._debug_message(f"predict_col: {self.predict_col} \n attr1: {attr1} \n attr2: {attr2} \n attr3: {attr3} \n")

        self._debug_message(f"SubgroupFairness - attr1: {attr1}")
        self._debug_message(f"SubgroupFairness - attr2: {attr2}")
        self._debug_message(f"SubgroupFairness - attr3: {attr3}")

        constraints = []
        
        self._debug_message(f"Length of privileged_train: {len(privileged_train)}")
        self._debug_message(f"Length of unprivileged_train: {len(unprivileged_train)}")

        #show the head of the dataframe 
        self._debug_message(f"Dataframe Columns:\n{D_train.columns}\nDataframe Head:\n{D_train.head()}")

        for i in privileged_train:
            constraints.append(z[0] + D_train.loc[i, self.predict_col] - a[0] * D_train.loc[i, attr1] -
                            c[0] * D_train.loc[i, attr2] - d[0] * D_train.loc[i, attr3] + e[0] >= 0)
            constraints.append(z[0] - D_train.loc[i, self.predict_col] + a[0] * D_train.loc[i, attr1] +
                            c[0] * D_train.loc[i, attr2] + d[0] * D_train.loc[i, attr3] + e[0] >= 0)

        for i in unprivileged_train:
            constraints.append(z[0] + D_train.loc[i, self.predict_col] - a[1] * D_train.loc[i, attr1] -
                            c[1] * D_train.loc[i, attr2] - d[1] * D_train.loc[i, attr3] + e[1] >= 0)
            constraints.append(z[0] - D_train.loc[i, self.predict_col] + a[1] * D_train.loc[i, attr1] +
                            c[1] * D_train.loc[i, attr2] + d[1] * D_train.loc[i, attr3] + e[1] >= 0)

        constraints.append(z[1] >= sum((D_train.loc[i, self.predict_col] - a[0] * D_train.loc[i, attr1] -
                                        c[0] * D_train.loc[i, attr2] - d[0] * D_train.loc[i, attr3] + e[0])**2
                                    for i in privileged_train) / len(privileged_train))
        constraints.append(z[2] >= sum((D_train.loc[i, self.predict_col] - a[1] * D_train.loc[i, attr1] -
                                        c[1] * D_train.loc[i, attr2] - d[1] * D_train.loc[i, attr3] + e[1])**2
                                    for i in unprivileged_train) / len(unprivileged_train))

        obj_D = z[0] + z[1] + z[2] + 0.05*(e[0]**2 + e[1]**2) #0.05*(e[0]**2 + e[1]**2) 24/02/2023

        return constraints, obj_D


    def _generate_optimisation_variables(self):
        """
        Returns:
            tuple: A tuple containing the optimization variables.
        """
        self._debug_message("_generate_optimisation_variables started")
        a = RealVariable('a', (2, 1))
        c = RealVariable('c', (2, 1))
        d = RealVariable('d', (2, 1))
        e = RealVariable('e', (2, 1))
        z = RealVariable('z', (3, 1))
        return a, c, d, e, z