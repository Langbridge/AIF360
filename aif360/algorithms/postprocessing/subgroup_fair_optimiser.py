from aif360.algorithms import Transformer
import numpy as np
import cvxpy as cp
from picos import Problem, RealVariable, trace
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class SubgroupFairOptimiser(Transformer):
    def __init__(self, S, X, Y_hat):
        super().__init__()
        self.S = S #sensitive attribute, string
        self.X = X #features, list of strings
        self.Y_hat = Y_hat #predicted labels, string

    def fit(self, dataset):
        # print("Converting dataset to dataframe")
        D = dataset.convert_to_dataframe()[0]
        # print("Splitting dataset into privileged and unprivileged groups")
        D_privileged, D_unprivileged = self._split_data(D)
        # print("Creating optimisation problem")
        constraints, obj_D, x_vars, e, z = self._create_optimisation(D_privileged, D_unprivileged)
        # print("Solving optimisation problem")
        self.solved_x_vars, self.solved_e, self.solved_z = self._solve_optimisation(constraints, obj_D, x_vars, e, z)
        # print("Optimisation problem solved")
        return self

    def transform(self, dataset, threshold=0.5):
        # print("Converting dataset to dataframe")
        D = dataset.convert_to_dataframe()[0]
        # print("Reweighing data")
        D_reweighted = self._reweigh_data(D)
        # print("Normalising data")
        D_normalised = self._normalise(D_reweighted)
        # print("Converting dataframe to dataset")
        D_classified = self._apply_threshold(D_normalised, threshold)
        dataset.df = D_classified
        # print("Dataset transformed")
        return dataset.df

    def _split_data(self, dataset):
        # Split the dataset into privileged and unprivileged groups based on the S attribute
        D_privileged = dataset[dataset[self.S] == 1]
        D_unprivileged = dataset[dataset[self.S] == 0]
        return D_privileged, D_unprivileged

    def _create_optimisation(self, D_privileged, D_unprivileged):
        x_vars, e, z = self._create_decision_variables()
        constraints, obj_D = self._get_constraints(D_privileged, D_unprivileged, x_vars, e, z)
        return constraints, obj_D, x_vars, e, z

    def _create_decision_variables(self):
        #for x in self.X, there is an 0 and 1 decision variable
        x_vars = [RealVariable(f"x{i}",(2,1)) for i in range(len(self.X))]
        #error decision variable
        e = RealVariable("e",(2,1))
        #z decision variable
        z = RealVariable("z",(3,1))
        return x_vars, e, z

    def _get_constraints(self, D_privileged, D_unprivileged, x_vars, e, z):
        constraints = []

        for i in range(len(D_unprivileged)):
            x_i = D_unprivileged.iloc[i][self.X]
            y_hat_i = D_unprivileged.iloc[i][self.Y_hat]

            constraint1 = z[0] + y_hat_i - sum(x_vars[j][0] * x_i[j] for j in range(len(self.X))) + e[0] >= 0
            constraint2 = z[0] - y_hat_i + sum(x_vars[j][0] * x_i[j] for j in range(len(self.X))) + e[0] >= 0

            constraints.extend([constraint1, constraint2])

        for i in range(len(D_privileged)):
            x_i = D_privileged.iloc[i][self.X]
            y_hat_i = D_privileged.iloc[i][self.Y_hat]

            constraint3 = z[0] + y_hat_i - sum(x_vars[j][1] * x_i[j] for j in range(len(self.X))) + e[1] >= 0
            constraint4 = z[0] - y_hat_i + sum(x_vars[j][1] * x_i[j] for j in range(len(self.X))) + e[1] >= 0

            constraints.extend([constraint3, constraint4])

        # Constraint for z1
        z1_sum = 0
        for i in range(len(D_unprivileged)):
            y_hat_i = D_unprivileged.iloc[i][self.Y_hat]
            x_sum = sum(x_vars[j][0, 0] * D_unprivileged.iloc[i][x] for j, x in enumerate(self.X))
            z1_sum += (y_hat_i - x_sum + e[0, 0]) ** 2
        z1_constraint = z[1] - (1 / len(D_unprivileged)) * z1_sum >= 0
        constraints.append(z1_constraint)

        # Constraint for z2
        z2_sum = 0
        for i in range(len(D_privileged)):
            y_hat_i = D_privileged.iloc[i][self.Y_hat]
            x_sum = sum(x_vars[j][1, 0] * D_privileged.iloc[i][x] for j, x in enumerate(self.X))
            z2_sum += (y_hat_i - x_sum + e[1, 0]) ** 2
        z2_constraint = z[2] - (1 / len(D_privileged)) * z2_sum >= 0
        constraints.append(z2_constraint)

        # Objective function
        obj_D = z[0] + z[1] + z[2] + 0.5 * (z[2] - z[1])

        return constraints, obj_D


    def _solve_optimisation(self, constraints, obj_D, x_vars, e, z):
        # Solve the optimisation problem
        prob = Problem()
        prob.add_list_of_constraints(constraints)
        prob.set_objective('min', obj_D)

        try:
            prob.solve(solver='cvxopt', verbose=0)
        except Exception as exception:
            raise Exception(f"Optimisation failed to find a feasible solution. Error: {str(exception)}")
    
        solved_x_vars = [x.value for x in x_vars]
        solved_e = e.value
        solved_z = z.value
        
        return solved_x_vars, solved_e, solved_z

    def _reweigh_data(self, D):
        D_reweighted = D.copy()
        for index, row in D_reweighted.iterrows():
            score = 0
            for x in self.X:
                Y_hat_index = int(row[self.Y_hat])  
                score += row[x] * self.solved_x_vars[self.X.index(x)][Y_hat_index]
            Y_hat_index = int(row[self.Y_hat])  
            score += self.solved_e[Y_hat_index]
            D_reweighted.at[index, self.Y_hat] = score
        return D_reweighted

    def _normalise(self, D):
        # normalise the Y_hat column
        D_normalised = D.copy()
        #get the Y_hat column
        # print(D_normalised[self.Y_hat])
        Y_hat_col = D_normalised[self.Y_hat]
        outliers = self._detect_outliers(Y_hat_col)
        # print(outliers)
        Y_hat_clean = np.delete(Y_hat_col, outliers)
        Y_hat_min = np.min(Y_hat_clean)
        Y_hat_max = np.max(Y_hat_clean)
        Y_hat_normalised = np.array([round(float(x - Y_hat_min) / (Y_hat_max - Y_hat_min), 1) for x in Y_hat_col])
        Y_hat_normalised[Y_hat_normalised > 1] = 1
        Y_hat_normalised[Y_hat_normalised < 0] = 0
        D_normalised[self.Y_hat] = Y_hat_normalised
        return D_normalised


    def _detect_outliers(self, data, outlier_threshold=3):
        mean = np.mean(data)
        std = np.std(data)
        outliers = []
        for i in range(len(data)):
            z_score = (data[i] - mean) / std
            if np.abs(z_score) > outlier_threshold:
                outliers.append(i)
        return outliers

    def _apply_threshold(self,D_normalised, threshold):
        # Y_hat is 1 when the normalised score is greater than the threshold, 0 otherwise
        D_classified = D_normalised.copy()
        D_classified[self.Y_hat] = D_classified[self.Y_hat].apply(lambda x: 1 if x > threshold else 0)
        return D_classified