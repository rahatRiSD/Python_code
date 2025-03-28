# 1. Data Analysis & Manipulation (Pandas, NumPy)

# 1.1 Load and inspect data
import pandas as pd
import numpy as np

def load_and_inspect(filepath):
    """Loads a CSV and returns head, shape, and info."""
    df = pd.read_csv(filepath)
    print("Head:\n", df.head())
    print("\nShape:", df.shape)
    print("\nInfo:\n", df.info())
    return df

# Example usage (replace with your file path)
# df = load_and_inspect('data.csv')

# 1.2 Data cleaning: handling missing values
def handle_missing_values(df, strategy='mean'):
    """Handles missing values using mean, median, or drop."""
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Invalid strategy. Use 'mean', 'median', or 'drop'.")

# cleaned_df = handle_missing_values(df)

# 1.3 Data transformation: normalization
def normalize_data(df, columns):
    """Normalizes specified columns using min-max scaling."""
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

# normalized_df = normalize_data(df, ['column1', 'column2'])

# 1.4 Grouping and aggregation
def group_and_aggregate(df, group_col, agg_dict):
    """Groups data and applies aggregations."""
    return df.groupby(group_col).agg(agg_dict)

# aggregated_df = group_and_aggregate(df, 'category', {'value': 'mean'})

# 1.5 Correlation matrix
def correlation_matrix(df):
    """Calculates and returns the correlation matrix."""
    return df.corr()

# corr_matrix = correlation_matrix(df)

# 1.6 Time series analysis (simple rolling mean)
def rolling_mean(series, window):
    """Calculates the rolling mean of a time series."""
    return series.rolling(window=window).mean()

# rolling_avg = rolling_mean(df['time_series_column'], 7)

# 1.7 Feature Engineering: Polynomial features
from sklearn.preprocessing import PolynomialFeatures

def create_polynomial_features(df, columns, degree):
  """Creates polynomial features."""
  poly = PolynomialFeatures(degree=degree)
  poly_features = poly.fit_transform(df[columns])
  poly_df = pd.DataFrame(poly_features, columns=[f"poly_{i}" for i in range(poly_features.shape[1])])
  return pd.concat([df, poly_df], axis=1)

# poly_df = create_polynomial_features(df, ['column1'], 2)

# 2. Data Visualization (Matplotlib, Seaborn)

# 2.1 Basic line plot
import matplotlib.pyplot as plt
import seaborn as sns

def plot_line(x, y, xlabel, ylabel, title):
    """Creates a line plot."""
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# plot_line(df['x'], df['y'], 'X-axis', 'Y-axis', 'Line Plot')

# 2.2 Scatter plot
def plot_scatter(x, y, xlabel, ylabel, title):
    """Creates a scatter plot."""
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# plot_scatter(df['x'], df['y'], 'X', 'Y', 'Scatter Plot')

# 2.3 Histogram
def plot_histogram(data, bins, xlabel, ylabel, title):
    """Creates a histogram."""
    plt.hist(data, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# plot_histogram(df['column'], 10, 'Value', 'Frequency', 'Histogram')

# 2.4 Box plot
def plot_box(df, column, title):
    """Creates a box plot."""
    sns.boxplot(x=df[column])
    plt.title(title)
    plt.show()

# plot_box(df, 'column', 'Box Plot')

# 2.5 Heatmap (correlation matrix)
def plot_heatmap(corr_matrix, title):
    """Creates a heatmap of a correlation matrix."""
    sns.heatmap(corr_matrix, annot=True)
    plt.title(title)
    plt.show()

# plot_heatmap(corr_matrix, 'Correlation Heatmap')

# 2.6 Pair plot
def plot_pairplot(df, vars):
  """Creates a pairplot."""
  sns.pairplot(df[vars])
  plt.show()

# plot_pairplot(df, ['column1','column2','column3'])

# 2.7 Violin plot
def plot_violinplot(df, x, y, title):
    """Creates a violin plot."""
    sns.violinplot(x=x, y=y, data=df)
    plt.title(title)
    plt.show()

# plot_violinplot(df, 'category', 'value', 'Violin Plot')

# 3. Machine Learning (Scikit-learn)

# 3.1 Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def linear_regression(X, y):
    """Trains and evaluates a linear regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    return model

# model = linear_regression(df[['feature1']], df['target'])

# 3.2 Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def logistic_regression(X, y):
    """Trains and evaluates a logistic regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return model

# model = logistic_regression(df[['feature1', 'feature2']], df['target'])

# 3.3 Decision Tree
from sklearn.tree import DecisionTreeClassifier

def decision_tree(X, y):
    """Trains and evaluates a decision tree classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return model

# model = decision_tree(df[['feature1', 'feature2']], df['target'])

# 3.4 Random Forest
from sklearn.ensemble import RandomForestClassifier

def random_forest(X, y):
    """Trains and evaluates a random forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return model

# model = random_forest(df[['feature1', 'feature2']], df['target'])

# 3.5 K-Means Clustering
from sklearn.cluster import KMeans

def k_means_clustering(X, n_clusters):
    """Performs K-means clustering."""
    model = KMeans(n_clusters=n_clusters)
    model.fit(X)
    labels = model.predict(X)
    return labels, model

# labels, model = k_means_clustering(df[['feature1', 'feature2']], 3)

# 3.6 Support Vector Machine (SVM)
from sklearn.svm import SVC

def support_vector_machine(X,y):
  """Trains and evaluates an SVM classifier."""
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = SVC()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print("Accuracy:", accuracy)
  return model

# model = support_vector_machine(df[['feature1', 'feature2']], df['target'])

# 3.7 Naive Bayes
from sklearn.naive_bayes import GaussianNB
def naive_bayes(X,y):
  """Trains and evaluates a Naive Bayes classifier"""
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = GaussianNB()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print("Accuracy:", accuracy)
  return model

# model = naive_bayes(df[['feature1', 'feature2']], df['target'])

# 3.8 Model evaluation: cross-validation
from sklearn.model_selection import cross_val_score

def cross_validation(model, X, y, cv=5):
    """Performs cross-validation and prints the scores."""
    scores = cross_val_score(model, X, y, cv=cv)
    print("Cross-validation scores:", scores)
    print("Mean score:", scores.mean())
    return scores

# cross_validation(model, df[['feature1', 'feature2']], df['target'])

# 3.9 Model evaluation: ROC AUC
from sklearn.metrics import roc_auc_score

def roc_auc(model, X_test, y_test):
  """Calculates and prints the ROC AUC score."""
  y_pred_proba = model.predict_proba(X_test)[:, 1]
  roc_auc = roc_auc_score(y_test, y_pred_proba)
  print("ROC AUC:", roc_auc)
  return roc_auc

# roc_auc(model, X_test, y_test)

# 3.10 Model tuning: Grid Search
from sklearn.model_selection import GridSearchCV

def grid_search(model, param_grid, X_train, y_train):
    """Performs grid search for hyperparameter tuning."""
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
# best_model = grid_search(SVC(), param_grid, X_train, y_train)
# ...and many more using libraries like tensorflow, pytorch, nltk, spacy, etc.
# ... covering deep learning, NLP, and more advanced research topics.
# ... Further expansion can be done with topic modelling, recommender systems, time series forecasting, and more.
# ... each of the above functions can be further modified with more parameters and error handling.
# ... the examples are basic and can be expanded for more complex research tasks.
