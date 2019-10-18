import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

def process_data(ctr_X, clk_y, categorical_col= None, numeric_col = None):
    """
    to preprocess sample data
    
    parameters:
        ctr_X: input dataframe, all features used to predict the click through
        clk_y: target, either 1: click or 0: not click
        categorical_col: list of categorical feature names we want to retain
        numeric_col: list of numeric feature names we want to retain
        
    return:
        ctr_X_com: processed predictors dataframe
        ctr_y_com: corresponding targets

    """

    #After explore the dataset, the click through entries without user feature information is around 6% of the 
    #total data set. We drop the entries without user feature infomation considering the sample size is big enough
    drop_index = ctr_X[ctr_X['cms_segid'].isnull()].index
    ctr_X_complete = ctr_X.drop(drop_index, axis = 0)
    clk_y_com = clk_y.drop(drop_index, axis = 0)

    #simply fill the missing value with mode data
    ctr_X_complete = ctr_X_complete.fillna(ctr_X_complete.mode().iloc[0])
  
   
    # convert the categorical columns data type to object, then create dummy variables for each feature values
    ctr_X_cate = ctr_X_complete[categorical_col].astype(str)
    ctr_X_cate1 = pd.get_dummies(ctr_X_cate)

    # normalize the numeric columns
    ss = StandardScaler()
    ctr_X_num = ss.fit_transform(ctr_X_complete[numeric_col]) #to update if more numeric features are added

    # combine categorical and numeric features
    for i in numeric_col:
        ctr_X_cate1[i] = ctr_X_num
    ctr_X_com = ctr_X_cate1

    return ctr_X_com, clk_y_com


def score_model(model, X_train, y_train, X_test, y_test, scorer = None):
    """
    plot ROC curve for each classifier model
    """

    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)

    tpr, fpr, thresholds = roc_curve(y_test, y_pred[:,0])
    score = roc_auc_score(y_test, y_pred[:,0])
    plt.plot(fpr, tpr, lw = 3, label = model.__class__.__name__)

    x = np.linspace(0,1,100)
    plt.plot(x,x, ls = '--', lw = 2, alpha = 0.9, color = 'grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC' + model.__class__.__name__)
    plt.legend()
    plt.show

    return score

# Function for EDA
def dist_explore(series, name, bin_num, zoom_in, x_lim, figsize = (18, 6)):
    """
    to plot the distribution and cummulative distribution of features
    
    input: a series of feature
    output: a distribution plot and a cummulative distribution plot
    
    """
    fig, ax = plt.subplots(1, 3, figsize =figsize)
    ax[0].hist(series, bins=bin_num)
#     ax[0].hist(series, bins=bin_num, density=True, histtype='step', label='Empirical')
    ax[0].set_title('Distribution of %s' %name)
    ax[0].set_xlabel('%s'%name)
    
    ax[1].hist(series, bins=zoom_in)
    ax[1].set_xlim(0, zoom_in)
#     ax[1].hist(series, bins=bin_num, density=True, histtype='step', label='Empirical')
    ax[1].set_title('Distribution of %s - Zoom In'%name)
    ax[1].set_xlabel('%s'%name)
    
    ax[2].hist(series, bins=bin_num, density=True, histtype='step', cumulative=True, label='Empirical')
    ax[2].set_xlim(0, x_lim)
    ax[2].set_xlabel('%s'%name)
    ax[2].set_ylabel('Cummulative Percentage')
    ax[2].set_title('Cum Distribution of %s'%name)

def get_train_test(df, target, fea_list, test_size = 0.3):
    """
    To train and test model with given features list
    input:
        df: DataFrame - complete dataset
        target: string = the label column name
        model: a given model object
        fea_list: list -  a list of selected features
    output:
        model score
    """
    #fill missing categorical values as 'unknown'
    # df.fillna(value = 'Unknown', inplace=True) #use later for new data

    #get dummies for all selected categorical features
    df1 = pd.get_dummies(df[fea_list])
    price_col = ['price_0', 'price_1', 'price_2', 'price_3', 'price_4', 'price_5', 'price_6', 'price_7']

    for i in price_col:
        df1[i] = df[i]

    x_train, x_test, y_train, y_test = train_test_split(df1, df[target], test_size = test_size)

    return x_train, x_test, y_train, y_test

def Model_Test(df, target, model, fea_list, test_size = 0.3):
    """
    To train and test model with given features list
    input:
        df: DataFrame - complete dataset
        target: string = the label column name
        model: a given model object
        fea_list: list -  a list of selected features
    output:
        model score
    """
    #fill missing categorical values as 'unknown'
    # df.fillna(value = 'Unknown', inplace=True) #use later for new data

    #get dummies for all selected categorical features
    df1 = pd.get_dummies(df[fea_list])
    price_col = ['price_0', 'price_1', 'price_2', 'price_3', 'price_4', 'price_5', 'price_6', 'price_7']

    for i in price_col:
        df1[i] = df[i]

    x_train, x_test, y_train, y_test = train_test_split(df1, df[target], test_size = test_size)
    model.fit(x_train, y_train)

    y_train_pred = model.predict_proba(x_train)

    y_pred = model.predict(x_test)
    y_test_pred = model.predict_proba(x_test)

    print('See the change in train auc and test auc for overfitting/underfitting')
    print('train auc', roc_auc_score(y_train, y_train_pred[:, 1]))
    print('test auc', roc_auc_score(y_test, y_test_pred[:, 1]))

    print(classification_report(y_test,y_pred))

    p, r, _ = precision_recall_curve(y_test, y_test_pred[:, 1])
    tpr, fpr, _ = roc_curve(y_test, y_test_pred[:, 1])
    
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].plot(r, p, label = 'precision_recall')
    axs[1].plot(tpr, fpr, label = 'ROC')

    plt.show()

def Model_Test1(df, target, model, fea_list, test_size = 0.3):
    """
    To train and test model with given features list
    input:
        df: DataFrame - complete dataset
        target: string = the label column name
        model: a given model object
        fea_list: list -  a list of selected features
    output:
        model score
    """
    #fill missing categorical values as 'unknown'
    # df.fillna(value = 'Unknown', inplace=True) #use later for new data

    #get dummies for all selected categorical features
    df1 = pd.get_dummies(df[fea_list])
    price_col = ['price_0', 'price_1', 'price_2', 'price_3', 'price_4', 'price_5', 'price_6', 'price_7']

    for i in price_col:
        df1[i] = df[i]

    x_train, x_test, y_train, y_test = train_test_split(df1, df[target], test_size = test_size)
    
    #concatenate back
    X = pd.concat([x_train, y_train], axis =1)

    #separate minority and majority classes
    not_clk = X[X.clk == 0]
    clk = X[X.clk == 1]

    #upsample minority
    clk_upsampled = resample(clk, replace = True, n_samples=len(not_clk), random_state=27)

    #combine majority and upsampled minority
    upsampled = pd.concat([not_clk, clk_upsampled])

    y_upsampled = upsampled.pop('clk')
    x_upsampled = upsampled

    model.fit(x_upsampled, y_upsampled)

    y_pred = model.predict(x_test)
    y_test_pred = model.predict_proba(x_test)

    print('test auc', roc_auc_score(y_test, y_test_pred[:, 1]))
    print(classification_report(y_test,y_pred))

    p, r, _ = precision_recall_curve(y_test, y_test_pred[:, 1])
    tpr, fpr, _ = roc_curve(y_test, y_test_pred[:, 1])
    
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].plot(r, p, label = 'precision_recall')
    axs[1].plot(fpr, tpr, label = 'ROC')
   
    axs[0].set_title('Precision Recall Curve')
    axs[0].set_xlabel('Recall Rate')
    axs[0].set_ylabel('Precision Rate')

    
    axs[1].set_title('ROC Curve')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')

    plt.show()
 
def Model_Test2(df, target, model, fea_list, test_size = 0.3):
    """
    To train and test model with given features list
    input:
        df: DataFrame - complete dataset
        target: string = the label column name
        model: a given model object
        fea_list: list -  a list of selected features
    output:
        model score
    """
    #fill missing categorical values as 'unknown'
    # df.fillna(value = 'Unknown', inplace=True) #use later for new data

    #get dummies for all selected categorical features
    df1 = pd.get_dummies(df[fea_list])
    price_col = ['price_0', 'price_1', 'price_2', 'price_3', 'price_4', 'price_5', 'price_6', 'price_7']

    for i in price_col:
        df1[i] = df[i]

    x_train, x_test, y_train, y_test = train_test_split(df1, df[target], test_size = test_size)
    
    #use SMOTE to generate more minority data
    sm = SMOTE(random_state=27, ratio=1.0)
    x_train, y_train = sm.fit_sample(x_train, y_train)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_test_pred = model.predict_proba(x_test)

    print('test auc', roc_auc_score(y_test, y_test_pred[:, 1]))
    print(classification_report(y_test,y_pred))

    p, r, _ = precision_recall_curve(y_test, y_test_pred[:, 1])
    tpr, fpr, _ = roc_curve(y_test, y_test_pred[:, 1])
    
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].plot(r, p, label = 'precision_recall')
    axs[0].set_title('Precision Recall Curve')
    axs[0].set_xlabel('Recall Rate')
    axs[0].set_ylabel('Precision Rate')

    axs[1].plot(fpr, tpr, label = 'ROC')
    axs[1].set_title('ROC Curve')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')

    plt.show()

    feat_scores = pd.DataFrame({'Fraction of Samples Affected' : model.feature_importances_},
                           index=x_test.columns)
    feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected', ascending=False)
    feat_scores[:50].plot(kind='barh', figsize=(18,18))