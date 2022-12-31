
textFiles = {'.csv':True, '.tsv':True, '.txt':True}
excelFiles = {'.xls':True, '.xlsx':True, '.xlsm':True, '.xlsb':True, '.ods':True}
classificationAlgorithms = {'logistic_regression':True, 'support_vector_machine':True, 'random_forest':True}
regressionAlgorithms = {'linear_regression':True}

classification_metrics = [("Training accuracy", "train_accuracy", "accuracy_score(y_train, y_train_pred)"),
                          ("Training F1 score", "train_f1", "f1_score(y_train, y_train_pred, average=\"weighted\")"),
                          ("Training precision", "train_precision", "precision_score(y_train, y_train_pred, average=\"weighted\")"),
                          ("Training recall", "train_recall", "recall_score(y_train, y_train_pred, average=\"weighted\")"),
                          ("Training area under ROC", "train_roc_auc", "roc_auc_score(y_train, y_train_pred_proba, average=\"weighted\", multi_class=\"ovr\")"),
                          ("Testing accuracy", "test_accuracy", "accuracy_score(y_test, y_test_pred)"),
                          ("Testing F1 score", "test_f1", "f1_score(y_test, y_test_pred, average=\"weighted\")"),
                          ("Testing precision", "test_precision", "precision_score(y_test, y_test_pred, average=\"weighted\")"),
                          ("Testing recall", "test_recall", "recall_score(y_test, y_test_pred, average=\"weighted\")"),
                          ("Testing area under ROC", "test_roc_auc", "roc_auc_score(y_test, y_test_pred_proba, average=\"weighted\", multi_class=\"ovr\")")]

regression_metrics = [("Training R-squared score", "train_r2", "r2_score(y_train, y_train_pred)"),
                      ("Testing R-squared score", "test_r2", "r2_score(y_test, y_train_pred)")]



#########################
#####   FUNCTIONS   #####
#########################


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
                          "import pandas as pd"]) + '\n'
    else: return ''


def random_seed(region, params):
    if (region == 'header'): return "np.random.seed(0)  # Random number seed" + '\n'
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
        filename, delimiter, header_row = '', '', ''
        if ('filename_temp' in params): filename = params['filename_temp']
        if ('delimiter' in params): delimiter = params['delimiter']
        if (delimiter == '\t'): delimiter = '\\t'
        if ('header_row' in params) and (params['header_row']): header_row = 0
        if (filename != ''): filename = "'" + filename + "'"
        if (delimiter != ''): delimiter = ", sep='" + delimiter + "'"
        if (header_row != ''): header_row = ", header=" + str(header_row)
        return '\n'.join(["# READ IN DATA. STORE IN PANDAS DATAFRAME.",
                          "df = pd.read_csv(" + filename + delimiter + ", skipinitialspace=True" + header_row + ")"]) + '\n\n'
    else: return ''


def read_data_excel(region, params):
    if (region == 'body'):
        filename, engine, header_row = '', '', ''
        if ('filename_temp' in params): filename = params['filename_temp']
        if ('engine' in params): engine = params['engine']
        if ('header_row' in params) and (params['header_row']): header_row = 0
        if (filename != ''): filename = "'" + filename + "'"
        if (engine != ''): engine = ", engine='" + engine + "'"
        if (header_row != ''): header_row = ", header=" + str(header_row)
        return '\n'.join(["# READ IN DATA. STORE IN PANDAS DATAFRAME.",
                          "df = pd.read_excel(" + filename + engine + header_row + ")"]) + '\n\n'
    else: return ''


def split_data(region, params):
    if (region == 'header'):
        return "from sklearn.model_selection import train_test_split" + '\n'
    elif (region == 'body'):
        percent_training = params['percent_training'] / 100.0
        return '\n'.join(["# SPLIT DATA INTO TRAINING DATA AND TESTING DATA",
                          "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=" + str(percent_training) + ", random_state=0)"]) + '\n\n'
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
            output_column = (params['output_column'])
            return '\n'.join(["# REMOVE ROWS WITH MISSING VALUES IN LABEL COLUMN",
                              "df = df.dropna(subset=[df.columns[" + output_column + "]])"]) + '\n\n'
        else: return ''
    else: return ''


def labels_to_y(region, params):
    if (region == 'body'):
        output_column = str(params['output_column'])
        return '\n'.join(["# CONVERT LABEL COLUMN (I.E., DEPENDENT VARIABLE) TO ARRAY, Y",
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
                              "X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)"]) + '\n\n'
        else: return ''
    else: return ''


def multivariate_imputation(region, params):
    if ('missing_values' in params) and (params['missing_values'] == 'multivariate_imputation') and ('features_contain_nan' in params) and (params['features_contain_nan']):
        if (region == 'header'):
            return '\n'.join(["from sklearn.experimental import enable_iterative_imputer",
                              "from sklearn.impute import IterativeImputer"]) + '\n'
        elif (region == 'body'):
            return '\n'.join(["# MULTIVARIATE IMPUTATION OF MISSING VALUES",
                              "X = IterativeImputer(random_state=0).fit_transform(X)"]) + '\n\n'
        else: return ''
    else: return ''


def model(region, params):
    if (params['algorithm'] == 'logistic_regression'): return logistic_regression(region, params)
    elif (params['algorithm'] == 'support_vector_machine'): return support_vector_machine(region, params)
    elif (params['algorithm'] == 'random_forest'): return random_forest(region, params)
    elif (params['algorithm'] == 'linear_regression'): return linear_regression(region_params)
    else: return ''


def logistic_regression(region, params):
    if (region == 'header'): return "from sklearn.linear_model import LogisticRegression" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE LOGISTIC REGRESSION MODEL (CLASSIFICATION)",
                          "model = LogisticRegression(random_state=0)"]) + '\n\n'
    else: return ''


def support_vector_machine(region, params):
    if (region == 'header'): return "from sklearn.svm import SVC" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE SUPPORT VECTOR MACHINE MODEL (CLASSIFICATION)",
                          "model = SVC(probability=True, random_state=0)"]) + '\n\n'
    else: return ''

def random_forest(region, params):
    if (region == 'header'): return "from sklearn.ensemble import RandomForestClassifier" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE RANDOM FOREST MODEL (CLASSIFICATION)",
                          "model = RandomForestClassifier(random_state=0)"]) + '\n\n'
    else: return ''


def linear_regression(region, params):
    if (region == 'header'): return "from sklearn.linear_model import LinearRegression" + '\n'
    elif (region == 'body'):
        return '\n'.join(["# CREATE LINEAR REGRESSION MODEL (REGRESSION)",
                          "model = LinearRegression()"]) + '\n\n'
    else: return ''


def train_and_predict(region, params):
    if (region == 'body'):
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
    else: return ''


def evaluate(region, params):
    if (params['algorithm'] in classificationAlgorithms): return evaluate_classification(region, params)
    elif (params['algorithm'] in regressionAlgorithms): return evaluate_regression(region, params)
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
        for m in regression_metrics: result.append(m[1] + " = metrics." + m[2])
        return '\n'.join(result) + '\n\n'
    else: return ''


def output(region, params):
    result = ''
    if (region == 'body'):
        result = '\n'.join(["# OUTPUT RESULTS",
                            "sys.stdout.write('After processing, the data contain ' + str(X.shape[0]) + ' points and ' + str(X.shape[1]) + ' features' + '\\n')",
                            "sys.stdout.write('Of the ' + str(X.shape[0]) + ' points, ' + str(X_train.shape[0]) + ' are used for training and ' + str(X_test.shape[0]) + ' are used for testing' + '\\n')"]) + '\n'
    if (params['algorithm'] in classificationAlgorithms): return result + output_classification(region, params)
    elif (params['algorithm'] in regressionAlgorithms): return result + output_regression(region, params)
    else: return ''


def output_classification(region, params):
    if (region == 'body'):
        result = []
        for m in classification_metrics: result.append("sys.stdout.write('" + m[0] + ":\\t' + str(" + m[1] + ") + '\\n')")
        return '\n'.join(result) + '\n\n'
    else: return ''


def output_regression(region, params):
    if (region == 'body'):
        result = []
        for m in regression_metrics: result.append("sys.stdout.write('" + m[0] + ":\\t' + str(" + m[1] + ") + '\\n')")
        return '\n'.join(result) + '\n\n'
    else: return ''

