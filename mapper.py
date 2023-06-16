######################################
#####   MLpronto version 1.0.3   #####
######################################

textFiles = {'.csv':True, '.tsv':True, '.txt':True}
excelFiles = {'.xls':True, '.xlsx':True, '.xlsm':True, '.xlsb':True, '.ods':True}
classificationAlgorithms = {'logistic_regression':True, 'kNN_classifier':True, 'gradient_boosting_classifier':True, 'random_forest':True, 'gaussian_naive_bayes':True, 'quadratic_discriminant_analysis':True,'support_vector_classifier':True, 'NN_classifier':True}
regressionAlgorithms = {'linear_regression':True, 'kNN_regressor':True, 'gradient_boosting_regressor':True, 'lasso':True, 'bayesian_ridge':True, 'elastic_net':True, 'stochastic_gradient_descent':True, 'NN_regressor':True}

classification_metrics = [("Training accuracy", "train_accuracy", "accuracy_score(y_train, y_train_pred)"),
                          ("Training F1 score", "train_f1", "f1_score(y_train, y_train_pred, average=\"weighted\", zero_division=0)"),
                          ("Training precision", "train_precision", "precision_score(y_train, y_train_pred, average=\"weighted\", zero_division=0)"),
                          ("Training recall", "train_recall", "recall_score(y_train, y_train_pred, average=\"weighted\", zero_division=0)"),
                          ("Training area under ROC", "train_roc_auc", "roc_auc_score(y_train, y_train_pred_proba, average=\"weighted\", multi_class=\"ovr\")"),
                          ("Testing accuracy", "test_accuracy", "accuracy_score(y_test, y_test_pred)"),
                          ("Testing F1 score", "test_f1", "f1_score(y_test, y_test_pred, average=\"weighted\", zero_division=0)"),
                          ("Testing precision", "test_precision", "precision_score(y_test, y_test_pred, average=\"weighted\", zero_division=0)"),
                          ("Testing recall", "test_recall", "recall_score(y_test, y_test_pred, average=\"weighted\", zero_division=0)"),
                          ("Testing area under ROC", "test_roc_auc", "roc_auc_score(y_test, y_test_pred_proba, average=\"weighted\", multi_class=\"ovr\")")]

regression_metrics = [("Training R-squared score", "train_r2", "metrics.r2_score(y_train, y_train_pred)"),
                      ("Training R-squared adjusted", "train_r2_adjusted", "1.0 - ((1.0-train_r2)*(X_train.shape[0]-1.0)) / (X_train.shape[0]-X_train.shape[1]-1.0)"),
                      ("Training mean squared error", "train_mse", "metrics.mean_squared_error(y_train, y_train_pred)"),
                      ("Training mean absolute error", "train_mae", "metrics.mean_absolute_error(y_train, y_train_pred)"),
                      ("Testing R-squared score", "test_r2", "metrics.r2_score(y_test, y_test_pred)"),
                      ("Testing R-squared adjusted", "test_r2_adjusted", "1.0 - ((1.0-test_r2)*(X_test.shape[0]-1.0)) / (X_test.shape[0]-X_test.shape[1]-1.0)"),
                      ("Testing R-squared out of sample", "test_r2_oos", "1.0 - np.sum((y_test_pred-y_test)**2) / np.sum((y_test-np.mean(y_train))**2)"),
                      ("Testing mean squared error", "test_mse", "metrics.mean_squared_error(y_test, y_test_pred)"),
                      ("Testing mean absolute error", "test_mae", "metrics.mean_absolute_error(y_test, y_test_pred)")]


################################
#####   HELPER FUNCTIONS   #####
################################


def get_filename(params):
    filename = ''
    if ('filename_temp' in params): filename = params['filename_temp']
    if (filename == ''): filename = params['filename']
    if (filename != ''): filename = "'" + filename + "'"
    return filename


################################
#####   CODING FUNCTIONS   #####
################################


def header(region, params):
    if (region == 'header'):
        return '\n'.join(["",
                          "################################",
                          "##########   HEADER   ##########",
                          "################################"]) + '\n\n'
    else: return ''


def libraries(region, params):
    if (region == 'header'):
        return '\n'.join(["import sys",
                          "import numpy as np",
                          "import pandas as pd",
                          "import matplotlib.pyplot as plt"]) + '\n'
    else: return ''


def warnings(region, params):
    if (region == 'header'):
        if ('warnings' in params) and (params['warnings']):
            return "import warnings; warnings.filterwarnings(\"ignore\")" + '\n'
        else: return ''
    else: return ''


def random_seed(region, params):
    if (region == 'header'): return "seed=0; np.random.seed(seed)  # Random number seed" + '\n'
    else: return ''


def body(region, params):
    if (region == 'body'):
        return '\n'.join(["",
                          "",
                          "##############################",
                          "##########   BODY   ##########",
                          "##############################"]) + '\n\n'
    else: return ''


def read_data(region, params):
    extension = params['extension']
    if (extension.lower() in textFiles): return read_data_text(region, params)
    elif (extension.lower() in excelFiles): return read_data_excel(region, params)
    else: return ''


def read_data_text(region, params):
    if (region == 'body'):
        filename = get_filename(params)
        delimiter, header_row = '', 'None'
        if ('delimiter' in params): delimiter = params['delimiter']
        if (delimiter == '\t'): delimiter = '\\t'
        if ('header_row' in params) and (params['header_row']): header_row = 0
        if (delimiter != ''): delimiter = ", sep='" + delimiter + "'"
        header_row = ", header=" + str(header_row)
        return '\n'.join(["# READ IN DATA. STORE IN PANDAS DATAFRAME.",
                          "df = pd.read_csv(" + filename + delimiter + ", skipinitialspace=True" + header_row + ")"]) + '\n\n'
    else: return ''


def read_data_excel(region, params):
    if (region == 'body'):
        filename = get_filename(params)
        engine, header_row = '', 'None'
        if ('engine' in params): engine = params['engine']
        if ('header_row' in params) and (params['header_row']): header_row = 0
        if (engine != ''): engine = ", engine='" + engine + "'"
        header_row = ", header=" + str(header_row)
        return '\n'.join(["# READ IN DATA. STORE IN PANDAS DATAFRAME.",
                          "df = pd.read_excel(" + filename + engine + header_row + ")"]) + '\n\n'
    else: return ''


def visualize(region, params):
    if (region == 'header'):
        if ('visualize' in params) and (params['visualize'] > 0):
            code = ["from sklearn.decomposition import PCA"]
            if (params['visualize'] >= 3):  # 3D plotting
                code += ["from mpl_toolkits import mplot3d"]
            return '\n'.join(code) + '\n'
        else: return ''
    elif (region == 'body'):
        if ('visualize' in params) and (params['visualize'] > 0):
            filename = get_filename(params)
            extension_index = filename.rfind('.')
            if (extension_index >= 0): filename = filename[:extension_index]
            filename2d, filename3d = filename + "2d.png'", filename + "3d.png'"
            if (params['algorithm_type'] == 'classification'):
                n_components = min(3, params['visualize'])
                exp_var_components = 2
                y_train_coords2d, y_train_coords3d = "X_train_pca[:,1]", "X_train_pca[:,2]"
                y_test_coords2d, y_test_coords3d = "X_test_pca[:,1]", "X_test_pca[:,2]"
                y_label2d, y_label3d = "'Principal Component 2'", "'Principal Component 3'"
                clr_train, clr_test = ", c=y_train", ", c=y_test"
            elif (params['algorithm_type'] == 'regression'):
                n_components = min(2, params['visualize']-1)
                exp_var_components = 1
                y_train_coords2d, y_train_coords3d = "y_train", "y_train"
                y_test_coords2d, y_test_coords3d = "y_test", "y_test"
                y_label2d, y_label3d = "y_header", "y_header"
                clr_train, clr_test = ", c='indigo'", ", c='indigo'"
                if ('header_row' not in params) or (not params['header_row']): 
                    y_label2d = '"Column " + str(' + y_label2d + ')'
                    y_label3d = '"Column " + str(' + y_label3d + ')'
            else: return ''  # Case should not be reached
            code = ["# PLOT DATA IN 2 DIMENSIONS",
                    "pca = PCA(n_components=" + str(n_components) + ", random_state=seed)",
                    "X_train_pca = pca.fit_transform(X_train)",
                    "X_test_pca = pca.transform(X_test)",
                    "explained_variance = np.sum(pca.explained_variance_ratio_[:" + str(exp_var_components) + "])",
                    "plt.scatter(X_train_pca[:,0], " + y_train_coords2d + clr_train + ", s=10)",
                    "plt.scatter(X_test_pca[:,0], " + y_test_coords2d + clr_test + ", s=10)",
                    "plt.title('2D plot\\nExplained variance: ' + str(round(explained_variance*100.0)) + '%', fontsize=14)",
                    "plt.xlabel('Principal Component 1', fontsize=12)",
                    "plt.ylabel(" + y_label2d + ", fontsize=12)",
                    "plt.xticks([])",
                    "plt.yticks([])",
                    "plt.savefig(" + filename2d + ", dpi=300, transparent=True)"]

            if (params['visualize'] >= 3):  # 3D plotting
                code += ["",
                         "# PLOT DATA IN 3 DIMENSIONS",
                         "explained_variance = np.sum(pca.explained_variance_ratio_[:" + str(n_components) + "])",
                         "plt.clf()",
                         "ax = plt.axes(projection='3d')",
                         "ax.scatter(X_train_pca[:,0], X_train_pca[:,1], " + y_train_coords3d + clr_train + ", s=10)",
                         "ax.scatter(X_test_pca[:,0], X_test_pca[:,1], " + y_test_coords3d + clr_test + ", s=10)",
                         "ax.set_title('3D plot\\nExplained variance: ' + str(round(explained_variance*100.0)) + '%', fontsize=14)",
                         "ax.set_xlabel('Principal Component 1', fontsize=12)",
                         "ax.set_ylabel('Principal Component 2', fontsize=12)",
                         "ax.set_zlabel(" + y_label3d + ", fontsize=12)",
                         "ax.set_xticks([])",
                         "ax.set_yticks([])",
                         "ax.set_zticks([])",
                         "ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.25))",
                         "ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.25))",
                         "ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.25))",
                         "plt.savefig(" + filename3d + ", dpi=300, transparent=True)"]
            return '\n'.join(code) + '\n\n'
        else: return ''
    else: return ''


def feature_relationships(region, params):
    if ('feature_relationships' not in params) or (not params['feature_relationships']): return ''
    if (region == 'header'):
        if (params['algorithm_type'] == 'classification'):
            return "from sklearn.feature_selection import f_classif, mutual_info_classif" + '\n'
        elif (params['algorithm_type'] == 'regression'):
            return "from sklearn.feature_selection import f_regression, mutual_info_regression" + '\n'
        else: return ''
    elif (region == 'body'):
        code = ["# CALCULATE FEATURE RELATIONSHIPS",
                "X_train_test, y_train_test = np.vstack((X_train, X_test)), np.concatenate((y_train, y_test))"]
        if ('header_row' not in params) or (not params['header_row']): 
            code += ["names = ['Column ' + str(s) for s in list(df.columns)] + ['Label']"]
        else:
            code += ["names = list(df.columns) + [y_header]"]
        code += ["df_correlations = pd.DataFrame(np.hstack((X_train_test, y_train_test.reshape(-1,1))), columns=names)",
                 "sys.stdout.write('Correlations\\n' + df_correlations.corr().round(2).to_string() + '\\n\\n')"]
        if (params['algorithm_type'] == 'classification'):
            code += ["mutual_information = mutual_info_classif(X_train_test, y_train_test, random_state=seed)",
                     "f_statistics, p_values = f_classif(X_train_test, y_train_test)"]
        elif (params['algorithm_type'] == 'regression'):
            code += ["mutual_information = mutual_info_regression(X_train_test, y_train_test, random_state=seed)",
                     "f_statistics, p_values = f_regression(X_train_test, y_train_test)"]
        else: return '\n'.join(code) + '\n\n'
        code += ["df_relationships = pd.DataFrame(np.vstack((mutual_information, f_statistics, p_values)).T, columns=['Mutual Information', 'F-statistic', 'p-value'], index=names[:-1])",
                 "sys.stdout.write(df_relationships.to_string() + '\\n\\n')"]
        return '\n'.join(code) + '\n\n'
    else: return ''


def split_data(region, params):
    if (region == 'header'):
        return "from sklearn.model_selection import train_test_split" + '\n'
    elif (region == 'body'):
        percent_training = params['percent_training'] / 100.0
        return '\n'.join(["# SPLIT DATA INTO TRAINING DATA AND TESTING DATA",
                          "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=" + str(percent_training) + ", random_state=seed)"]) + '\n\n'
    else: return ''


def preprocessing(region, params):
    if (region == 'header'): 
        if ('feature_scaling' in params) or ('labels_non_numeric' in params) or ('encode_binary_features' in params):
            return "from sklearn import preprocessing" + '\n'
        else: return ''
    else: return ''


def feature_scaling(region, params):
    if (region == 'body'):
        if ('feature_scaling' in params) and (params['feature_scaling']):
            return '\n'.join(["# FEATURE SCALING",
                              "scaler = preprocessing.StandardScaler().fit(X_train)",
                              "X_train = scaler.transform(X_train)",
                              "X_test = scaler.transform(X_test)"]) + '\n\n'
        else: return ''
    else: return ''


def remove_rows(region, params):
    if (region == 'body'):
        contains_nan = False
        if ('is_missing_values' in params) and (params['is_missing_values']): contains_nan = True
        if ('missing_values' in params) and (params['missing_values'] == 'remove_rows') and (contains_nan):
            return '\n'.join(["# REMOVE ROWS THAT CONTAIN MISSING VALUES",
                              "df.dropna(axis=0, inplace=True)"]) + '\n\n'
        else: return ''
    else: return ''


def labels_contain_nan(region, params):
    if (region == 'body'):
        if ('labels_contain_nan' in params) and (params['labels_contain_nan']):
            output_column = str(params['output_column'])
            return '\n'.join(["# REMOVE ROWS WITH MISSING VALUES IN LABEL COLUMN",
                              "df = df.dropna(subset=[df.columns[" + output_column + "]])"]) + '\n\n'
        else: return ''
    else: return ''


def labels_to_y(region, params):
    if (region == 'body'):
        output_column = str(params['output_column'])
        return '\n'.join(["# CONVERT LABEL COLUMN (I.E., DEPENDENT VARIABLE) TO ARRAY, Y",
                          "y_header = df.columns[" + output_column + "]",
                          "y = df.pop(df.columns[" + output_column + "]).to_numpy()"]) + '\n\n'
    else: return ''


def labels_non_numeric(region, params):
    if (region == 'body'):
        if ('labels_non_numeric' in params) and (params['labels_non_numeric']):
            return '\n'.join(["# ENCODE LABELS AS NUMBERS",
                              "y = preprocessing.LabelEncoder().fit_transform(y)"]) + '\n\n'
        else: return ''
    else: return ''


def remove_cols(region, params):
    if (region == 'body'):
        if ('missing_values' in params) and (params['missing_values'] == 'remove_cols') and ('features_contain_nan' in params) and (params['features_contain_nan']):
            return '\n'.join(["# REMOVE COLUMNS THAT CONTAIN MISSING VALUES",
                              "df.dropna(axis=1, inplace=True)"]) + '\n\n'
        else: return ''
    else: return ''


def encode_binary_features(region, params):
    if (region == 'body'):
        if ('encode_binary_features' in params) and (params['encode_binary_features']):
            return '\n'.join(["# ENCODE BINARY FEATURES AS NUMBERS",
                              "df2 = df.select_dtypes(exclude='number')",
                              "result = df2.apply(pd.Series.nunique)",
                              "result = result[result == 2]",
                              "binaryColumns = result.keys().tolist()",
                              "for col in binaryColumns:",
                              "\tif (not df[col].isnull().values.any()):",
                              "\t\tdf[col] = preprocessing.LabelEncoder().fit_transform(df[col])"]) + '\n\n'
        else: return ''
    else: return ''


def one_hot_encode_features(region, params):
    if (region == 'body'):
        if ('one_hot_encode_features' in params) and (params['one_hot_encode_features']):
            return '\n'.join(["# ONE-HOT ENCODE MULTI-CATEGORY FEATURES",
                              "df = pd.get_dummies(df)"]) + '\n\n'
        else: return ''
    else: return ''


def features_to_X(region, params):
    if (region == 'body'):
        return '\n'.join(["# CONVERT DATA FRAME TO NUMPY ARRAY, X",
                          "X = df.to_numpy()"]) + '\n\n'
    else: return ''


def univariate_imputation(region, params):
    if ('missing_values' in params) and (params['missing_values'] == 'univariate_imputation') and ('features_contain_nan' in params) and (params['features_contain_nan']):
        if (region == 'header'): return "from sklearn.impute import SimpleImputer" + '\n'
        elif (region == 'body'):
            return '\n'.join(["# UNIVARIATE IMPUTATION OF MISSING VALUES",
                              "imputer = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X_train)",
                              "X_train = imputer.transform(X_train)",
                              "X_test = imputer.transform(X_test)"]) + '\n\n'
        else: return ''
    else: return ''


def multivariate_imputation(region, params):
    if ('missing_values' in params) and (params['missing_values'] == 'multivariate_imputation') and ('features_contain_nan' in params) and (params['features_contain_nan']):
        if (region == 'header'):
            return '\n'.join(["from sklearn.experimental import enable_iterative_imputer",
                              "from sklearn.impute import IterativeImputer"]) + '\n'
        elif (region == 'body'):
            return '\n'.join(["# MULTIVARIATE IMPUTATION OF MISSING VALUES",
                              "imputer = IterativeImputer(max_iter=200, random_state=seed).fit(X_train)",
                              "X_train = imputer.transform(X_train)",
                              "X_test = imputer.transform(X_test)"]) + '\n\n'
        else: return ''
    else: return ''


def model(region, params):
    if (params['algorithm'] == 'logistic_regression'): return logistic_regression(region, params)
    elif (params['algorithm'] == 'kNN_classifier'): return kNN_classifier(region, params)
    elif (params['algorithm'] == 'gradient_boosting_classifier'): return gradient_boosting_classifier(region, params)
    elif (params['algorithm'] == 'random_forest'): return random_forest(region, params)
    elif (params['algorithm'] == 'gaussian_naive_bayes'): return gaussian_naive_bayes(region, params)
    elif (params['algorithm'] == 'quadratic_discriminant_analysis'): return quadratic_discriminant_analysis(region, params)
    elif (params['algorithm'] == 'support_vector_classifier'): return support_vector_classifier(region, params)
    elif (params['algorithm'] == 'NN_classifier'): return NN_classifier(region, params)
    elif (params['algorithm'] == 'linear_regression'): return linear_regression(region, params)
    elif (params['algorithm'] == 'kNN_regressor'): return kNN_regressor(region, params)
    elif (params['algorithm'] == 'gradient_boosting_regressor'): return gradient_boosting_regressor(region, params)
    elif (params['algorithm'] == 'lasso'): return lasso(region, params)
    elif (params['algorithm'] == 'bayesian_ridge'): return bayesian_ridge(region, params)
    elif (params['algorithm'] == 'elastic_net'): return elastic_net(region, params)
    elif (params['algorithm'] == 'stochastic_gradient_descent'): return stochastic_gradient_descent(region, params)
    elif (params['algorithm'] == 'NN_regressor'): return NN_regressor(region, params)
    else: return ''


def logistic_regression(region, params):
    if (region == 'header'): return "from sklearn.linear_model import LogisticRegression" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE LOGISTIC REGRESSION MODEL (CLASSIFICATION)",
                          "model = LogisticRegression(max_iter=1200, random_state=seed)"]) + '\n\n'
    else: return ''


def kNN_classifier(region, params):
    if (region == 'header'): return "from sklearn.neighbors import KNeighborsClassifier" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE K NEAREST NEIGHBORS MODEL (CLASSIFICATION)",
                          "model = KNeighborsClassifier()"]) + '\n\n'
    else: return ''


def gradient_boosting_classifier(region, params):
    if (region == 'header'): return "from sklearn.ensemble import GradientBoostingClassifier" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE GRADIENT BOOSTING MODEL (CLASSIFICATION)",
                          "model = GradientBoostingClassifier(random_state=seed)"]) + '\n\n'
    else: return ''


def random_forest(region, params):
    if (region == 'header'): return "from sklearn.ensemble import RandomForestClassifier" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE RANDOM FOREST MODEL (CLASSIFICATION)",
                          "model = RandomForestClassifier(random_state=seed)"]) + '\n\n'
    else: return ''


def gaussian_naive_bayes(region, params):
    if (region == 'header'): return "from sklearn.naive_bayes import GaussianNB" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE GAUSSIAN NAIVE BAYES MODEL (CLASSIFICATION)",
                          "model = GaussianNB()"]) + '\n\n'
    else: return ''


def quadratic_discriminant_analysis(region, params):
    if (region == 'header'): return "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE QUADRATIC DISCRIMINANT ANALYSIS MODEL (CLASSIFICATION)",
                          "model = QuadraticDiscriminantAnalysis()"]) + '\n\n'
    else: return ''


def support_vector_classifier(region, params):
    if (region == 'header'): return "from sklearn.svm import SVC" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE SUPPORT VECTOR MACHINE MODEL (CLASSIFICATION)",
                          "model = SVC(probability=True, random_state=seed)"]) + '\n\n'
    else: return ''


def NN_classifier(region, params):
    if (region == 'header'): return "from sklearn.neural_network import MLPClassifier" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE NEURAL NETWORK MODEL (CLASSIFICATION)",
                          "model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=seed)"]) + '\n\n'
    else: return ''


def linear_regression(region, params):
    if (region == 'header'): return "from sklearn.linear_model import LinearRegression" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE LINEAR REGRESSION MODEL (REGRESSION)",
                          "model = LinearRegression()"]) + '\n\n'
    else: return ''


def kNN_regressor(region, params):
    if (region == 'header'): return "from sklearn.neighbors import KNeighborsRegressor" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE K NEAREST NIGHBORS MODEL (REGRESSION)",
                          "model = KNeighborsRegressor()"]) + '\n\n'
    else: return ''


def gradient_boosting_regressor(region, params):
    if (region == 'header'): return "from sklearn.ensemble import GradientBoostingRegressor" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE GRADIENT BOOSTING MODEL (REGRESSION)",
                          "model = GradientBoostingRegressor(random_state=seed)"]) + '\n\n'
    else: return ''


def lasso(region, params):
    if (region == 'header'): return "from sklearn.linear_model import Lasso" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE LASSO MODEL (REGRESSION)",
                          "model = Lasso(random_state=seed)"]) + '\n\n'
    else: return ''


def bayesian_ridge(region, params):
    if (region == 'header'): return "from sklearn.linear_model import BayesianRidge" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE BAYESIAN RIDGE MODEL (REGRESSION)",
                          "model = BayesianRidge()"]) + '\n\n'
    else: return ''


def elastic_net(region, params):
    if (region == 'header'): return "from sklearn.linear_model import ElasticNet" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE ELASTIC NET MODEL (REGRESSION)",
                          "model = ElasticNet(random_state=seed)"]) + '\n\n'
    else: return ''


def stochastic_gradient_descent(region, params):
    if (region == 'header'): return "from sklearn.linear_model import SGDRegressor" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE STOCHASTIC GRADIENT DESCENT MODEL (REGRESSION)",
                          "model = SGDRegressor(random_state=seed)"]) + '\n\n'
    else: return ''


def NN_regressor(region, params):
    if (region == 'header'): return "from sklearn.neural_network import MLPRegressor" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE NEURAL NETWORK MODEL (REGRESSION)",
                          "model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=seed)"]) + '\n\n'
    else: return ''


def train_and_predict(region, params):
    if (region == 'body'):
        if (params['algorithm_type'] == 'classification'):  # Classification
            train_proba = "y_train_pred_proba = model.predict_proba(X_train)"
            test_proba = "y_test_pred_proba = model.predict_proba(X_test)"
            if ('labels_are_binary' in params) and (params['labels_are_binary']):
                train_proba += "[:,1]"
                test_proba += "[:,1]"
            return '\n'.join(["# TRAIN MODEL AND MAKE PREDICTIONS",
                              "model.fit(X_train, y_train)",
                              "y_train_pred = model.predict(X_train)",
                              train_proba,
                              "y_test_pred = model.predict(X_test)",
                              test_proba]) + '\n\n'
        elif (params['algorithm_type'] == 'regression'):  # Regression
            return '\n'.join(["# TRAIN MODEL AND MAKE PREDICTIONS",
                              "model.fit(X_train, y_train)",
                              "y_train_pred = model.predict(X_train)",
                              "y_test_pred = model.predict(X_test)"]) + '\n\n'
        else: return ''
    else: return ''


def evaluate(region, params):
    if (params['algorithm_type'] == 'classification'): return evaluate_classification(region, params)
    elif (params['algorithm_type'] == 'regression'): return evaluate_regression(region, params)
    else: return ''


def evaluate_classification(region, params):
    if (region == 'header'): return "from sklearn import metrics" + '\n'
    elif (region == 'body'):
        result = ["# EVALUATE MODEL PREDICTIONS"]
        for m in classification_metrics: result.append(m[1] + " = metrics." + m[2])
        return '\n'.join(result) + '\n\n'
    else: return ''


def evaluate_regression(region, params):
    if (region == 'header'): return "from sklearn import metrics" + '\n'
    elif (region == 'body'):
        result = ["# EVALUATE MODEL PREDICTIONS"]
        for m in regression_metrics: result.append(m[1] + " = " + m[2])
        return '\n'.join(result) + '\n\n'
    else: return ''


def output(region, params):
    result = ''
    if (region == 'body'):
        result = '\n'.join(["# OUTPUT RESULTS",
                            "sys.stdout.write('After processing, the data contain ' + str(X.shape[0]) + ' points and ' + str(X.shape[1]) + ' features' + '\\n')",
                            "sys.stdout.write('Of the ' + str(X.shape[0]) + ' points, ' + str(X_train.shape[0]) + ' are used for training and ' + str(X_test.shape[0]) + ' are used for testing' + '\\n')"]) + '\n'
    if (params['algorithm_type'] == 'classification'): return result + output_classification(region, params)
    elif (params['algorithm_type'] == 'regression'): return result + output_regression(region, params)
    else: return ''


def output_classification(region, params):
    if (region == 'header'): return confusion_and_classification(region, params) + roc_and_prc(region, params)
    elif (region == 'body'):
        result = []
        for m in classification_metrics: result.append("sys.stdout.write('" + m[0] + ":\\t' + str(" + m[1] + ") + '\\n')")
        return '\n'.join(result) + '\n\n' + confusion_and_classification(region, params) + roc_and_prc(region, params)
    else: return ''


def confusion_and_classification(region, params):
    if (region == 'header'): return "import itertools" + '\n'
    elif (region == 'body'):
        filename = get_filename(params)
        extension_index = filename.rfind('.')
        if (extension_index >= 0): filename = filename[:extension_index]
        filename_cm, filename_cr = filename + "_cm.png'", filename + "_cr.png'"
        return '\n'.join(["# CONFUSION MATRIX",
                          "plt.clf()",
                          "plt.rcParams.update({'font.size':16})",
                          "cm = metrics.confusion_matrix(y_test, y_test_pred, labels=model.classes_)",
                          "disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)",
                          "disp.plot()",
                          "plt.title('Confusion Matrix')",
                          "plt.tight_layout()",
                          "plt.savefig(" + filename_cm + ", dpi=300, transparent=True)",
                          "",
                          "# CLASSIFICATION REPORT",
                          "plt.clf()",
                          "prfs = np.array(metrics.precision_recall_fscore_support(y_test, y_test_pred, average=None, labels=model.classes_, zero_division=0))",
                          "prf, support = prfs[:-1,:].T, prfs[-1,:].astype(int)",
                          "xticklabels = ['Precision', 'Recall', 'F1-score']",
                          "yticklabels = ['{0} ({1})'.format(model.classes_[idx], sup) for idx, sup in enumerate(support)]",
                          "plt.imshow(prf, interpolation='nearest', cmap='RdBu_r', aspect='auto', vmin=0.0, vmax=1.0)",
                          "plt.title('Classification Report'); plt.colorbar()",
                          "plt.xticks(np.arange(3), xticklabels)",
                          "plt.yticks(np.arange(len(model.classes_)), yticklabels)",
                          "for i, j in itertools.product(range(prf.shape[0]), range(prf.shape[1])):",
                          "\tplt.text(j, i, format(prf[i, j], '.2f'), horizontalalignment='center',",
                          "\t\tcolor='white' if (prf[i, j] >= 0.8 or prf[i, j] <= 0.2) else 'black')",
                          "plt.ylabel('Classes')",
                          "plt.xlabel(' ')",
                          "plt.tight_layout()",
                          "plt.savefig(" + filename_cr + ", dpi=300, transparent=True)"]) + '\n\n'
    else: return ''


def roc_and_prc(region, params):
    if (region == 'header'):
        if ('feature_scaling' in params) or ('labels_non_numeric' in params) or ('encode_binary_features' in params): return ''
        else: return "from sklearn import preprocessing" + '\n'
    elif (region == 'body'):
        filename = get_filename(params)
        extension_index = filename.rfind('.')
        if (extension_index >= 0): filename = filename[:extension_index]
        filename_roc, filename_prc = filename + "_roc.png'", filename + "_prc.png'"
        return '\n'.join(["# ROC CURVE",
                          "y_classes = preprocessing.label_binarize(y_test, classes=model.classes_)",
                          "fpr, tpr, _ = metrics.roc_curve(y_classes.ravel(), y_test_pred_proba.ravel())",
                          "plt.clf()",
                          "plt.plot(fpr, tpr, label='ROC', color='indigo', lw=4)",
                          "plt.plot([0,1], [0,1], label='Random', linestyle='--', color='goldenrod', lw=4)",
                          "plt.axis('square'); plt.legend(frameon=False, loc='lower right')",
                          "plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')",
                          "plt.title('Receiver Operating Characteristic')",
                          "plt.tight_layout()",
                          "plt.savefig(" + filename_roc + ", dpi=300, transparent=True)",
                          "",
                          "# PRECISION RECALL CURVE",
                          "precision, recall, _ = metrics.precision_recall_curve(y_classes.ravel(), y_test_pred_proba.ravel())",
                          "plt.clf()",
                          "plt.plot(recall, precision, color='indigo', lw=4)",
                          "plt.axis('square'); plt.legend('', frameon=False)",
                          "plt.xlim(0,1.05); plt.ylim(0,1.05)",
                          "plt.xlabel('Recall'); plt.ylabel('Precision')",
                          "plt.title('Precision Recall Curve')",
                          "plt.tight_layout()",
                          "plt.savefig(" + filename_prc + ", dpi=300, transparent=True)"]) + '\n\n'
    else: return ''


def output_regression(region, params):
    if (region == 'body'):
        result = []
        for m in regression_metrics: result.append("sys.stdout.write('" + m[0] + ":\\t' + str(" + m[1] + ") + '\\n')")
        return '\n'.join(result) + '\n\n' + actual_vs_predicted(region, params) + residuals(region, params)
    else: return ''


def actual_vs_predicted(region, params):
    if (region == 'body'):
        filename = get_filename(params)
        extension_index = filename.rfind('.')
        if (extension_index >= 0): filename = filename[:extension_index]
        filename_ap = filename + "_ap.png'"
        return '\n'.join(["# PLOT ACTUAL VS PREDICTED",
                          "plt.clf()",
                          "plt.rcParams.update({'font.size':14})",
                          "plt.scatter(y_train_pred, y_train, label='Training data', c='indigo', alpha=0.3)",
                          "plt.scatter(y_test_pred, y_test, label='Testing data', c='goldenrod', alpha=0.3)",
                          "plt.xlabel('Predicted value')",
                          "plt.ylabel('Actual value')",
                          "maxie = min(np.max(y_train), np.max(y_train_pred), np.max(y_test), np.max(y_test_pred))",
                          "plt.plot([0, maxie], [0, maxie], c='k')",
                          "plt.xlim([0, maxie])",
                          "plt.ylim([0, maxie])",
                          "plt.legend(framealpha=0.0)",
                          "plt.tight_layout()",
                          "plt.savefig(" + filename_ap + ", dpi=300, transparent=True)"]) + '\n\n'
    else: return ''


def residuals(region, params):
    if (region == 'body'):
        filename = get_filename(params)
        extension_index = filename.rfind('.')
        if (extension_index >= 0): filename = filename[:extension_index]
        filename_resid1, filename_resid2 = filename + "_r1.png'", filename + "_r2.png'"
        return '\n'.join(["# PLOT RESIDUALS (SCATTER PLOT)",
                          "plt.clf()",
                          "plt.scatter(y_train_pred, y_train - y_train_pred, c='indigo', alpha=0.3)",
                          "plt.scatter(y_test_pred, y_test - y_test_pred, c='goldenrod', alpha=0.3)",
                          "plt.title('Residuals vs Fits')",
                          "plt.xlabel('Fitted Value')",
                          "plt.ylabel('Residual')",
                          "plt.legend(['Training data', 'Testing data'], framealpha=0.0)",
                          "plt.tight_layout()",
                          "plt.savefig(" + filename_resid1 + ", dpi=300, transparent=True)",
                          "",
                          "# PLOT RESIDUALS (HISTOGRAM)",
                          "plt.clf()",
                          "plt.hist(y_train - y_train_pred, color='indigo', ec='goldenrod')",
                          "plt.hist(y_test - y_test_pred, color='goldenrod', ec='indigo')",
                          "plt.title('Histogram')",
                          "plt.xlabel('Residual')",
                          "plt.ylabel('Frequency')",
                          "plt.legend(['Training data', 'Testing data'], framealpha=0.0)",
                          "plt.tight_layout()",
                          "plt.savefig(" + filename_resid2 + ", dpi=300, transparent=True)"]) + '\n\n'
    else: return ''

