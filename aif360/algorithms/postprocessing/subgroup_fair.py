from aif360.algorithms import Transformer
from aif360.metrics import utils
from ncpol2sdpa import SdpRelaxation, generate_operators, flatten
import numpy as np

class SubgroupFairness(Transformer):

    def __init__(self, privileged_groups, unprivileged_groups):
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.protected_attribute = list(privileged_groups[0].keys())[0]

    def set_optimisation_parameters(self, predictCol, attr1, attr2, attr3, solver=None, solver_parameters=None):
        self.predictCol = predictCol
        self.attr1 = attr1
        self.attr2 = attr2
        self.attr3 = attr3
        self.solver = solver
        self.solver_parameters = solver_parameters

    def fit(self,dataset_true,dataset_pred,level=1):
        inequality, obj_D, a, c, d, e, z = self._create_inequality(dataset_true, self.attr1, self.attr2, self.attr3)
        a1,a0,c1,c0,d1,d0,e1,e0 = self._solve_inequality(inequality, obj_D, self.solver, self.solver_parameters, a, c, d, e, z, level)
        self.a0 = a0
        self.a1 = a1
        self.c0 = c0
        self.c1 = c1
        self.d0 = d0
        self.d1 = d1
        self.e0 = e0
        self.e1 = e1
        return self

    def predict(self, dataset_pred):
            dataframe = dataset_pred.convert_to_dataframe()[0]
            I0_test=dataframe[dataframe[self.protected_attribute]==self.privileged_groups[0][self.protected_attribute]].index
            I1_test=dataframe[dataframe[self.protected_attribute]==self.unprivileged_groups[0][self.protected_attribute]].index
            arr = []
            for i in I0_test:
                arr+=[self.a0*dataframe.loc[i,self.attr1] + self.c0*dataframe.loc[i,self.attr2] + self.d0*dataframe.loc[i,self.attr3] + self.e0]
            for i in I1_test:
                arr+=[self.a1*dataframe.loc[i,self.attr1] + self.c1*dataframe.loc[i,self.attr2] + self.d1*dataframe.loc[i,self.attr3] + self.e1]
            self.scores = self._normalise_score(arr)
            return self

    def _normalise_score(self, arr, outlier_threshold=3):
        outlierPosition=self._detect_outliers(arr,outlier_threshold)
        arr_clean = np.delete(arr,outlierPosition)
        #arr_clean=arr
        arr_min=np.min(arr_clean)
        arr_max=np.max(arr_clean)

        normalized_arr = np.array([round(float(x - arr_min)/(arr_max - arr_min),1) for x in arr])
        normalized_arr[normalized_arr>10]=1
        normalized_arr[normalized_arr<0]=0
        return normalized_arr

    def _detect_outliers(self,data,threshold):
        mean_d = np.mean(data)
        std_d = np.std(data)
        outliers = []
        for i in range(len(data)):
            z_score= (data[i] - mean_d)/std_d 
            if np.abs(z_score) > threshold:
                outliers.append(i)
        return outliers

    def base_rate(self, dataframe, protected_attribute, dataset_pred):
        base0=(1-dataframe[(dataframe[protected_attribute]==1) & (dataframe[dataset_pred]==1)].shape[0]/dataframe[dataframe[protected_attribute]==1].shape[0])*100
        base1=(1-dataframe[(dataframe[protected_attribute]!=1) & (dataframe[dataset_pred]==1)].shape[0]/ dataframe[dataframe[protected_attribute]!=1].shape[0])*100
        return base0, base1

    def _create_inequality(self, dataset_true, attr1, attr2, attr3):
        Dtrain = dataset_true.convert_to_dataframe()[0]
        I0_train=Dtrain[Dtrain[self.protected_attribute]==1].index
        I1_train=Dtrain[Dtrain[self.protected_attribute]!=1].index

        a = generate_operators("a", n_vars=2, hermitian=True, commutative=False)
        c = generate_operators("c", n_vars=2, hermitian=True, commutative=False)
        d = generate_operators("d", n_vars=2, hermitian=True, commutative=False)
        e = generate_operators("e", n_vars=2, hermitian=True, commutative=False)
        z = generate_operators("z", n_vars=3, hermitian=True, commutative=True)

        # Constraints
        ine1 = [z[0]+Dtrain.loc[i,self.predictCol] - a[0]*int(Dtrain.loc[i,attr1]<25) - c[0]*Dtrain.loc[i,attr2] - d[0]*Dtrain.loc[i,attr3] + e[0] for i in I0_train]
        ine2 = [z[0]-Dtrain.loc[i,self.predictCol] + a[0]*int(Dtrain.loc[i,attr1]<25) + c[0]*Dtrain.loc[i,attr2] + d[0]*Dtrain.loc[i,attr3] + e[0] for i in I0_train]
        ine3 = [z[0]+Dtrain.loc[i,self.predictCol] - a[1]*int(Dtrain.loc[i,attr1]<25) - c[1]*Dtrain.loc[i,attr2] - d[1]*Dtrain.loc[i,attr3] + e[1] for i in I1_train]
        ine4 = [z[0]-Dtrain.loc[i,self.predictCol] + a[1]*int(Dtrain.loc[i,attr1]<25) + c[1]*Dtrain.loc[i,attr2] + d[1]*Dtrain.loc[i,attr3] + e[1] for i in I1_train]
        max1 =[z[1]-sum((Dtrain.loc[i,self.predictCol]-a[0]*int(Dtrain.loc[i,attr1]<25) - c[0]*Dtrain.loc[i,attr2] - d[0]*Dtrain.loc[i,attr3] + e[0])**2 for i in I0_train)/len(I0_train)]
        max2 =[z[2]-sum((Dtrain.loc[i,self.predictCol]-a[1]*int(Dtrain.loc[i,attr1]<25) - c[1]*Dtrain.loc[i,attr2] - d[1]*Dtrain.loc[i,attr3] + e[1])**2 for i in I1_train)/len(I1_train)]
        
        obj_D = z[0] + z[1] + z[2] + 0.5*(z[2]-z[1])

        inequality = ine1 + ine2 + ine3 + ine4 + max1 + max2

        return inequality, obj_D, a, c, d, e, z

    def _solve_inequality(self, inequality, obj_D, solver, solver_parameters, a, c, d, e, z, level):
        sdp_D = SdpRelaxation(variables = flatten([a,c,d,e,z]), verbose = 0)
        sdp_D.get_relaxation(level, objective=obj_D, inequalities=inequality)
        sdp_D.solve(solver=solver, solverparameters=solver_parameters)
        return [sdp_D[a[0]],sdp_D[a[1]],sdp_D[c[0]],sdp_D[c[1]],sdp_D[d[0]],sdp_D[d[1]],sdp_D[e[0]],sdp_D[e[1]]]

    def _filter_dataset(self, dataframe, attr1, attr2, attr3, attr1_filter, attr2_filter, attr3_filter):
        if attr1_filter is not None:
            if attr1_filter.type == 'int':
                #convert the row with the attr1 value as header to an integer datatype
                dataframe = dataframe[dataframe[attr1] == int(dataframe[attr1])]
                if attr1_filter.upper_bound is not None:
                    #remove all rows where the attr1 value is greater than the upper bound
                    dataframe = dataframe[dataframe[attr1] <= attr1_filter.upper_bound]
                elif attr1_filter.lower_bound is not None:
                    #remove all rows where the attr1 value is less than the lower bound
                    dataframe = dataframe[dataframe[attr1] >= attr1_filter.lower_bound]
            else:
                #log an error for unsupported limit datatype
                error = "Unsupported limit datatype: " + attr1_filter.type
                logging.error(error)
        if attr2_filter is not None:
            if attr2_filter.type == 'int':
                #convert the row with the attr2 value as header to an integer datatype
                dataframe = dataframe[dataframe[attr2] == int(dataframe[attr2])]
                if attr2_filter.upper_bound is not None:
                    #remove all rows where the attr2 value is greater than the upper bound
                    dataframe = dataframe[dataframe[attr2] <= attr2_filter.upper_bound]
                elif attr2_filter.lower_bound is not None:
                    #remove all rows where the attr2 value is less than the lower bound
                    dataframe = dataframe[dataframe[attr2] >= attr2_filter.lower_bound]
            else:
                #log an error for unsupported limit datatype
                error = "Unsupported limit datatype: " + attr2_filter.type
                logging.error(error)
        if attr3_filter is not None:
            if attr3_filter.type == 'int':
                #convert the row with the attr3 value as header to an integer datatype
                dataframe = dataframe[dataframe[attr3] == int(dataframe[attr3])]
                if attr3_filter.upper_bound is not None:
                    #remove all rows where the attr3 value is greater than the upper bound
                    dataframe = dataframe[dataframe[attr3] <= attr3_filter.upper_bound]
                elif attr3_filter.lower_bound is not None:
                    #remove all rows where the attr3 value is less than the lower bound
                    dataframe = dataframe[dataframe[attr3] >= attr3_filter.lower_bound]
            else:
                #log an error for unsupported limit datatype
                error = "Unsupported limit datatype: " + attr3_filter.type
                logging.error(error)

        return dataframe
