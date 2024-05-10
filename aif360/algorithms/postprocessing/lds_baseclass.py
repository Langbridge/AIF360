from aif360.algorithms import Transformer
import numpy as np
import cvxpy as cp
from picos import Problem, RealVariable, trace

class BaseLDS(Transformer):
    '''
    The base class for the SubgroupFairness and InstantaneousFairness classes, used for fairness remediation on Linear Dynamical Systems.

    Demo code can be found in the 'demo_subgroup_and_instantaneous.ipynb' notebook in the 'examples' folder.

    This code is based on the paper 'Fairness in Forecasting of Observations of Linear Dynamical Systems' paper by Quan Zhou, et al. 
    The paper can be found at https://www.jair.org/index.php/jair/article/view/14050.
    '''
    def __init__(self, privileged_groups, unprivileged_groups, debug=False):
        """
        Args:
            privileged_groups (list(dict)): Privileged groups.
            unprivileged_groups (list(dict)): Unprivileged groups.
            debug (bool, optional): Print debug statements. Defaults to False.
        """
        super().__init__(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            debug=debug
        )
        self.debug = debug
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.protected_attribute = list(privileged_groups[0].keys())[0]
        self._debug_message("=====================================================")
        self._debug_message(f"BaseLDS - privileged_groups: {privileged_groups}")
        self._debug_message(f"BaseLDS - unprivileged_groups: {unprivileged_groups}")
        self._debug_message(f"BaseLDS - protected_attribute: {self.protected_attribute}")


    def fit(self, D_train, level=1):
        """Fit the model."""
        self._debug_message("BaseLDS.fit started")
        constraints, obj_D, a, c, d, e, z = self._create_inequality(D_train, self.attr1, self.attr2, self.attr3)
        a1, a0, c1, c0, d1, d0, e1, e0 = self._solve_inequality(constraints, obj_D, a, c, d, e, z, level)
        self.a0 = a0
        self.a1 = a1
        self.c0 = c0        
        self.c1 = c1
        self.d0 = d0
        self.d1 = d1
        self.e0 = e0
        self.e1 = e1
        return self

    def predict(self, D_test):
        """Predict the labels with fairness optimization.

        Args:
            D_test (BinaryLabelDataset): Dataset containing the predicted scores.

        Returns:
            BinaryLabelDataset: Dataset with the fairness-optimized labels.
        """
        self._debug_message("BaseLDS.predict started")
        self._debug_message(f"protected_attribute: {self.protected_attribute}\nprivileged_groups: {self.privileged_groups}\nunprivileged_groups: {self.unprivileged_groups}")
        privileged_test = D_test[D_test[self.protected_attribute] == self.privileged_groups[0][self.protected_attribute]].index
        unprivileged_test = D_test[D_test[self.protected_attribute] == self.unprivileged_groups[0][self.protected_attribute]].index
        
        if self.a0 is None or self.c0 is None or self.d0 is None or self.e0 is None:
            raise ValueError("Optimization problem failed to find a feasible solution.")

        arr = self._generate_scores(D_test, privileged_test, unprivileged_test)
        scores = self._normalise_score(arr)
        
        # D_test.labels = (scores > 0.5).astype(int)
        D_test.labels = scores
        return D_test

    def fit_predict(self, D_train, D_test, level=1):
        """Fit the model and predict the labels.

        Args:
            D_train (BinaryLabelDataset): Dataset containing the true labels.
            D_test (BinaryLabelDataset): Dataset containing the predicted scores.
            level (int, optional): Optimization level. Defaults to 1.

        Returns:
            BinaryLabelDataset: Dataset with the fairness-optimized labels.
        """
        self._debug_message("BaseLDS.fit_predict started")
        return self.fit(D_train, D_test, level=level).predict(D_test)

    def set_optimisation_parameters(self, predict_col, attr1, attr2, attr3):
        """Set the optimization parameters.

        Args:
            predict_col (str): Name of the column containing the predicted scores.
            attr1 (dict): Dictionary containing the name, lower and upper bounds of the first attribute.
            attr2 (dict): Dictionary containing the name, lower and upper bounds of the second attribute.
            attr3 (dict): Dictionary containing the name, lower and upper bounds of the third attribute.
        """
        self._debug_message("BaseLDS.set_optimisation_parameters started")
        self.predict_col = predict_col
        self.attr1 = attr1
        self.attr2 = attr2
        self.attr3 = attr3
    
    def _debug_message(self, message):
        if self.debug:
            print(message)

    def _normalise_score(self, scores):
        self._debug_message("_normalise_score started")
        outlier_positions = self._detect_outliers(scores)
        scores_clean = np.delete(scores, outlier_positions)
        scores_min = np.min(scores_clean)
        scores_max = np.max(scores_clean)

        normalized_scores = np.array([round(float(x - scores_min) / (scores_max - scores_min), 1) for x in scores])
        normalized_scores[normalized_scores > 1] = 1
        normalized_scores[normalized_scores < 0] = 0
        return normalized_scores

    def _detect_outliers(self, data, outlier_threshold=3):
        self._debug_message("_detect_outliers started")
        mean = np.mean(data)
        std = np.std(data)
        outliers = []
        for i in range(len(data)):
            z_score = (data[i] - mean) / std
            if np.abs(z_score) > outlier_threshold:
                outliers.append(i)
        return outliers

    def _create_inequality(self, D_train, attr1, attr2, attr3):
        self._debug_message("BaseLDS._create_inequality started")
        dataframe_train = D_train
        privileged_train = dataframe_train[dataframe_train[self.protected_attribute] == 1].index
        unprivileged_train = dataframe_train[dataframe_train[self.protected_attribute] != 1].index

        a, c, d, e, z = self._generate_optimisation_variables()

        # self._debug_message(f"_create_inequality:\nDataframe Columns:\n{dataframe_train.columns}\nDataframe Head:\n{dataframe_train.head()}")
            
        constraints, obj_D = self._get_constraints(dataframe_train, privileged_train, unprivileged_train, a, c, d, e, z, attr1, attr2, attr3)
        return constraints, obj_D, a, c, d, e, z

    def _solve_inequality(self, constraints, obj_D, a, c, d, e, z, level):

        self._debug_message("BaseLDS._solve_inequality started")
        prob = Problem()
        prob.add_list_of_constraints(constraints)
        prob.set_objective('min', obj_D)

        # check the objective function is convex
        

        try:
            prob.solve(solver='cvxopt', verbose=0)
        except Exception as e:
            raise ValueError(f"Optimization problem failed to find a feasible solution. Error: {str(e)}")
        self._debug_message(f"Optimization problem status: {prob.status}")

        return [a[0].value, a[1].value, c[0].value, c[1].value, d[0].value, d[1].value, e[0].value, e[1].value]

    def _generate_scores(self, dataframe, privileged_test, unprivileged_test):
        """
        Generate the scores array for prediction.

        Args:
            dataframe (pandas.DataFrame): Test dataframe.
            privileged_test (pandas.Index): Indices of privileged group in test data.
            unprivileged_test (pandas.Index): Indices of unprivileged group in test data.

        Returns:
            list: List of scores.
        """
        self._debug_message("BaseLDS._generate_scores started")
        scores = []

        for i in privileged_test:
            scores.append(self.a0 * dataframe.loc[i, self.attr1] + self.c0 * dataframe.loc[i, self.attr2] + self.d0 * dataframe.loc[i, self.attr3] + self.e0)

        for i in unprivileged_test:
            scores.append(self.a1 * dataframe.loc[i, self.attr1] + self.c1 * dataframe.loc[i, self.attr2] + self.d1 * dataframe.loc[i, self.attr3] + self.e1)

        return np.array(scores)
    
    def _get_constraints(self, dataframe_train, privileged_train, unprivileged_train, a, c, d, e, z, attr1, attr2, attr3):
        '''
        Implemented within the SubgroupFairness and InstantaneousFairness classes.
        '''
        raise NotImplementedError("This method must be implemented in the subclass.")

    def _generate_optimisation_variables(self):
        '''
        Implemented within the SubgroupFairness and InstantaneousFairness classes.
        '''
        raise NotImplementedError("This method must be implemented in the subclass.")

        