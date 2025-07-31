#%% #List of Modules

# Standard libraries
import os
import random
import shutil
import warnings
import import_ipynb
import sys
import joblib

# Data processing libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import scipy.stats as stats
import pyforest
from wordcloud import WordCloud

# Terminal formatting
from colorama import Fore, Style
from termcolor import colored

# Scikit-learn utilities
from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold, StratifiedKFold, KFold,
    cross_val_predict, cross_val_score, cross_validate, GridSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, PolynomialFeatures, OneHotEncoder, PowerTransformer,
    MinMaxScaler, LabelEncoder, RobustScaler, label_binarize
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, r2_score,
    mean_absolute_error, mean_squared_error, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc,
    make_scorer, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
)
from sklearn.linear_model import (
    LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostClassifier
)
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, f_classif, f_regression, mutual_info_regression
)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Imbalanced data handling
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.over_sampling import BorderlineSMOTE


# Optimization and tuning
from skopt import BayesSearchCV
from keras_tuner import BayesianOptimization
from keras_tuner.tuners import BayesianOptimization as KerasBayesianOptimization

# XGBoost
from xgboost import XGBClassifier, XGBRegressor, plot_importance

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Conv1D, GlobalMaxPooling1D, Embedding, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

#%% ##Function to check missing values
def check_missing_values(df):
    missing_values = df.isnull().agg(['sum', 'mean']).T.sort_values(by='sum', ascending=False)
    missing_values.columns = ['Missing_Number', 'Missing_Percent']
    return missing_values

#%% ##Function for insighting summary information about the column
def inspect_columns(df, cols=None):
    """
    Provides a detailed summary of specified columns or all columns in a DataFrame,
    and returns the results as a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to inspect.
        cols (list, optional): List of column names to inspect.
                               If None, inspects all columns in the DataFrame.

    Returns:
        pd.DataFrame: Summary DataFrame containing statistics for each column.
    """
    # If no columns are provided, use all columns in the DataFrame
    cols = cols if cols is not None else df.columns

    summary_data = []

    for col in cols:
        summary = {
            "Column Name": col,
            "Percentage Nulls (%)": round(df[col].isnull().mean() * 100, 2),
            "Number of Nulls": df[col].isnull().sum(),
            "Number of Unique": df[col].nunique(),
            "Value Counts": df[col].value_counts(dropna=False).to_dict()
        }
        summary_data.append(summary)

    # Create and return a summary DataFrame
    return pd.DataFrame(summary_data)

#%% ##Function for plotting wordclouds
def plot_wordcloud(df, spec_col, text_col):
    labels = df[spec_col].unique()
    num_labels = len(labels)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))  # 2 rows, 3 columns

    for ax, label in zip(axes.flat, labels):
        text = " ".join(df[df[spec_col] == label][text_col])
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)

        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f"WordCloud for {label.capitalize()} Sentiment")
        ax.axis('off')

    for ax in axes.flat[num_labels:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

# %% ##=== CONFIG ===

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# === CONFIG ===
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
SAMPLE_SIZE = 30000
MAX_FEATURES = 2000
SVD_COMPONENTS = 100
BAYES_ITERATIONS = 5
CV_FOLDS = 2

# === GLOBALS ===
num_classes = None
label_list = None
X_train = X_test = y_train = y_test = X_smote = y_smote = None

# %% ##=== LOAD & PREPARE DATA ===
def prep_ml_data(df, x_col, y_col, save_path, skip_svd=False):
    global num_classes, label_list, X_train, X_test, y_train, y_test, X_smote, y_smote

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=SEED)

    label_list = sorted(df[y_col].unique().tolist())
    label_to_index = {label: idx for idx, label in enumerate(label_list)}
    df['label'] = df[y_col].map(label_to_index)

    label_list = list(label_to_index.keys())
    num_classes = len(label_list)

    vec = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english', ngram_range=(1, 2))
    X_tfidf = vec.fit_transform(df[x_col].values)

    # Save vectorizer if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(vec, save_path)

    if skip_svd:
        X_processed = X_tfidf.toarray()
    else:
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)
        X_processed = svd.fit_transform(X_tfidf)

    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, stratify=y, random_state=SEED)

    smote = BorderlineSMOTE(random_state=SEED)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    # return df, label_list

# %% ##=== MODEL DEFINITIONS ===
def get_model_instance(name):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1),
        "naive_bayes": MultinomialNB(),
        "svm": SVC(probability=True, random_state=SEED),
        "random_forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE and name == "xgboost":
        models["xgboost"] = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=SEED, n_jobs=-1)
    return models.get(name)

# %% ##=== MODEL DEFINITIONS ===

def get_search_space(name):
    spaces = {
        "logistic_regression": {'C': (1e-2, 1e2, 'log-uniform')},
        "naive_bayes": {'alpha': (1e-2, 1.0, 'log-uniform')},
        "svm": {'C': (1e-1, 1e2, 'log-uniform'), 'gamma': (1e-3, 1e-1, 'log-uniform')},
        "random_forest": {'n_estimators': (50, 200), 'max_depth': (3, 15)},
    }
    if XGBOOST_AVAILABLE:
        spaces["xgboost"] = {
            'n_estimators': (50, 200),
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'max_depth': (3, 8)
        }
    return spaces.get(name, {})

#%% ##=== EVALUATION ===
def evaluate_model(clf, X, y_true):
    try:
        y_prob = clf.predict_proba(X)
    except:
        y_prob = None

    try:
        y_pred = clf.predict(X)
    except:
        y_pred = np.argmax(y_prob, axis=1) if y_prob is not None else None

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted')
    }

    try:
        if y_prob is not None:
            metrics['ROC AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr') if num_classes > 2 else roc_auc_score(
                y_true, y_prob[:, 1])
        else:
            metrics['ROC AUC'] = np.nan
    except:
        metrics['ROC AUC'] = np.nan

    return metrics, y_true, y_pred, y_prob

#%% #=== WORKFLOW ===
def run_ml_workflow(df, x_col, y_col, model_name, save_path):
    global X_train, X_test, y_train, y_test, X_smote, y_smote

    print(f"\n=== {model_name} ===")
    results, preds = {}, {}

    skip_svd = (model_name == "naive_bayes")
    prep_ml_data(df, x_col, y_col, save_path, skip_svd=skip_svd)

    base_model = get_model_instance(model_name)
    base_model.fit(X_train, y_train)
    res_base, yt, yp, yp_prob = evaluate_model(base_model, X_test, y_test)
    results[f"Base {model_name}"] = res_base
    preds["base"] = (yt, yp, yp_prob)

    smote_model = get_model_instance(model_name)
    smote_model.fit(X_smote, y_smote)
    res_smote, yt, yp, yp_prob = evaluate_model(smote_model, X_test, y_test)
    results[f"SMOTE {model_name}"] = res_smote
    preds["smote"] = (yt, yp, yp_prob)

    space = get_search_space(model_name)
    if space:
        opt_model = get_model_instance(model_name)
        opt = BayesSearchCV(opt_model, space, n_iter=BAYES_ITERATIONS, cv=CV_FOLDS, random_state=SEED, n_jobs=-1,
                            scoring='f1_weighted')
        opt.fit(X_smote, y_smote)
        best_model = opt.best_estimator_
        print(f"Best Params: {opt.best_params_}")
    else:
        best_model = smote_model

    res_opt, yt, yp, yp_prob = evaluate_model(best_model, X_test, y_test)
    results[f"Optimized {model_name}"] = res_opt
    preds["opt"] = (yt, yp, yp_prob)

    return results, preds, best_model

#%% #=== CONFUSION MATRIX ===
def plot_confusion_matrices(preds, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    titles = [f"Base {model_name}", f"SMOTE {model_name}", f"Optimized {model_name}"]

    for ax, (key, (yt, yp, _)), title in zip(axes, preds.items(), titles):
        cm = confusion_matrix(yt, yp)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            # xticklabels=range(num_classes),  # ← numeric x-axis
            # yticklabels=range(num_classes),  # ← numeric y-axis
            xticklabels=label_list,
            yticklabels=label_list,
            cbar=True
        )

        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        ax.tick_params(axis='x', labelrotation=0)
        ax.tick_params(axis='y', labelrotation=0)
        ax.yaxis.set_label_coords(-0.15, 0.5)

    plt.tight_layout()
    plt.show()

#%% #=== ROC & PR CURVE ===
def plot_training_metrics(preds, model_name):
    """Plot ROC and PR curves for optimized model (multi-class one-vs-rest)"""
    global num_classes, label_list

    if 'opt' not in preds or len(preds['opt']) < 3:
        print("Insufficient data for plotting training metrics")
        return

    y_true, y_pred, y_prob = preds['opt']

    if y_prob is None:
        print("No probability predictions available for plotting")
        return

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # ==================== ROC Curve ====================
    axs[0].set_title(f"{model_name} - ROC Curve")
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")

    if num_classes > 2:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        for i in range(num_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                axs[0].plot(fpr, tpr, label=f"{label_list[i]} (AUC={roc_auc:.2f})")
            except:
                continue
    else:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        axs[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

    axs[0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # ==================== PR Curve ====================
    axs[1].set_title(f"{model_name} - Precision-Recall Curve")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")

    if num_classes > 2:
        for i in range(num_classes):
            try:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                pr_auc = auc(recall, precision)
                axs[1].plot(recall, precision, label=f"{label_list[i]} (AUC={pr_auc:.2f})")
            except:
                continue
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        pr_auc = auc(recall, precision)
        axs[1].plot(recall, precision, label=f"AUC = {pr_auc:.2f}")

    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

#%% ##=== RESULT ===

optimized_models_summary = []

def append_optimized_metrics(results, model_type, model_object):
    global optimized_models_summary

    opt_key = f"Optimized {model_type}"

    if opt_key in results:
        metrics = results[opt_key]
        optimized_models_summary.append({
            "Model": model_type,
            "Accuracy": round(metrics["Accuracy"], 4),
            "F1-Score": round(metrics["F1-Score"], 4),
            "Recall": round(metrics["Recall"], 4),
            "ROC AUC": round(metrics["ROC AUC"], 4),
            "Object": model_object
        })

#%% ##========================== DEEP LEARNING MODEL ===========================
# Set seed
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Sample size configuration
SAMPLE_SIZE = 30000
USE_SAMPLE = True

#%% #=== prep_dl_data ===
def prep_dl_data(df, x_col, y_col, save_path, use_sample=True):
    # global num_classes
    global num_classes, label_list  # ✅ make label_list global

    if use_sample and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=SEED)

    X_raw = df[x_col].values
    y_raw, label_list = pd.factorize(df[y_col])

    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_raw)
    X_seq = tokenizer.texts_to_sequences(X_raw)
    X_pad = pad_sequences(X_seq, maxlen=100)

    # Save vectorizer if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(tokenizer, save_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y_raw, test_size=0.3, stratify=y_raw, random_state=SEED
    )

    # SMOTE expects 2D flat inputs
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_smote, y_smote = SMOTE(random_state=SEED).fit_resample(X_train_flat, y_train)
    X_smote = X_smote.reshape((X_smote.shape[0], 100))

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    y_smote_cat = to_categorical(y_smote)
    num_classes = y_train_cat.shape[1]

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_cat': y_train_cat,
        'y_test_cat': y_test_cat,
        'X_smote': X_smote,
        'y_smote_cat': y_smote_cat,
        'tokenizer': tokenizer,
        'label_list': label_list
    }

#%% #=== evaluate_model ===

def evaluate_dl_model(model, X, y_true_cat):
    y_prob = model.predict(X)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_true_cat, axis=1)
    try:
        if num_classes > 2:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        else:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    except:
        roc_auc = np.nan
    return {
        'F1-Score': f1_score(y_true, y_pred, average='weighted'),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'ROC AUC': roc_auc
    }, y_true, y_pred, y_prob

#%% #=== build_base_model ===

def build_base_model(model_type):
    model = Sequential()
    model.add(Embedding(20000, 128, input_length=100))
    if model_type == 'LSTM':
        model.add(LSTM(64))
    elif model_type == 'GRU':
        model.add(GRU(64))
    elif model_type == 'CNN':
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#%% #=== run_dl_workflow ===

def run_dl_workflow(df, x_col, y_col, model_name, save_path):
    global num_classes

    # === Load and prepare data ===
    data = prep_dl_data(df, x_col, y_col, save_path, use_sample=True)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train_cat = data['y_train_cat']
    y_test_cat = data['y_test_cat']
    X_smote = data['X_smote']
    y_smote_cat = data['y_smote_cat']

    num_classes = y_train_cat.shape[1]

    model_variants = {}

    # === Base Model ===
    base_model = build_base_model(model_name)
    base_model.fit(X_train, y_train_cat, validation_split=0.2, epochs=5, callbacks=[EarlyStopping(patience=2)], verbose=0)
    metrics_base, y_true_base, y_pred_base, _ = evaluate_dl_model(base_model, X_test, y_test_cat)
    model_variants[f"Base {model_name}"] = metrics_base

    # === SMOTE Model ===
    smote_model = build_base_model(model_name)
    smote_model.fit(X_smote, y_smote_cat, validation_split=0.2, epochs=5, callbacks=[EarlyStopping(patience=2)], verbose=0)
    metrics_smote, y_true_smote, y_pred_smote, _ = evaluate_dl_model(smote_model, X_test, y_test_cat)
    model_variants[f"SMOTE {model_name}"] = metrics_smote

    # === Optimized Model ===
    def build_tuned_model(hp):
        model = Sequential()
        model.add(Embedding(20000, hp.Int("embed_dim", 64, 128, step=32), input_length=100))
        if model_name == 'LSTM':
            model.add(LSTM(hp.Int("units", 32, 128, step=32)))
        elif model_name == 'GRU':
            model.add(GRU(hp.Int("units", 32, 128, step=32)))
        elif model_name == 'CNN':
            model.add(Conv1D(filters=hp.Int("filters", 64, 128, step=32), kernel_size=5, activation='relu'))
            model.add(GlobalMaxPooling1D())
        model.add(Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1)))
        model.add(Dense(hp.Int("dense", 32, 128, step=32), activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling='LOG')),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    tune_dir = f"tune_{model_name.lower()}/opt_{model_name.lower()}"
    if os.path.exists(tune_dir):
        shutil.rmtree(tune_dir)

    tuner = BayesianOptimization(
        build_tuned_model,
        objective='val_accuracy',
        max_trials=5,
        directory=f"tune_{model_name.lower()}",
        project_name=f"opt_{model_name.lower()}"
    )
    tuner.search(X_smote, y_smote_cat, validation_split=0.2, epochs=5, callbacks=[EarlyStopping(patience=2)], verbose=0)
    best_model = tuner.get_best_models(1)[0]
    history_opt = best_model.fit(X_smote, y_smote_cat, validation_split=0.2, epochs=5, verbose=0)
    metrics_opt, y_true_opt, y_pred_opt, y_pred_prob_opt = evaluate_dl_model(best_model, X_test, y_test_cat)
    model_variants[f"Optimized {model_name}"] = metrics_opt

    return model_variants, {
        "base": (y_true_base, y_pred_base),
        "smote": (y_true_smote, y_pred_smote),
        "opt": (y_true_opt, y_pred_opt, y_pred_prob_opt)
    }, history_opt

#%% #=== plot_dl_confusion_matrices ===
# def plot_dl_confusion_matrices(preds, model_name):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     titles = [f"Base {model_name}", f"SMOTE {model_name}", f"Optimized {model_name}"]
#     for ax, (key, data), title in zip(axes, preds.items(), titles):
#         ConfusionMatrixDisplay.from_predictions(data[0], data[1], ax=ax, cmap='Blues', colorbar=False)
#         ax.set_title(title)
#     plt.tight_layout()
#     plt.show()

def plot_dl_confusion_matrices(preds, model_name):
    global label_list  # ✅ Access the global variable
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [f"Base {model_name}", f"SMOTE {model_name}", f"Optimized {model_name}"]

    for ax, (key, data), title in zip(axes, preds.items(), titles):
        ConfusionMatrixDisplay.from_predictions(
            data[0], data[1], display_labels=label_list, ax=ax, cmap='Blues', colorbar=False
        )
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


#%% #=== plot_dl_training_metrics ===
def plot_dl_training_metrics(history, preds, model_name):
    _, _, y_pred_prob = preds['opt']
    y_true = preds['opt'][0]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(history.history['loss'], label='Training Loss')
    axs[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title(f"{model_name} - Loss")
    axs[0, 0].legend()

    axs[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axs[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0, 1].set_title(f"{model_name} - Accuracy")
    axs[0, 1].legend()

    if num_classes > 2:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            axs[1, 0].plot(fpr, tpr, label=f'Class {i} AUC = {roc_auc:.2f}')
    else:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        axs[1, 0].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')

    axs[1, 0].set_title(f"{model_name} - ROC Curve")
    axs[1, 0].legend()

    if num_classes > 2:
        precision, recall, _ = precision_recall_curve(y_true_bin[:, 1], y_pred_prob[:, 1])
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob[:, 1])

    axs[1, 1].plot(recall, precision)
    axs[1, 1].set_title(f"{model_name} - Precision-Recall Curve")

    plt.tight_layout()
    plt.show()

#%% #=== plot_model_comparison ===
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
def plot_model_comparison(df):
    custom_palette = {
        'logistic_regression': '#4C72B0',
        'naive_bayes': '#DD8452',
        'random_forest': '#55A868',
        'xgboost': '#C44E52',
        'LSTM': '#8172B3',
        'GRU': '#937860',
        'CNN': '#64B5CD'
    }

    # Drop unwanted column if present
    df_drop = df.drop(columns=['Object'], errors='ignore')

    # Melt DataFrame for Seaborn plotting
    comparison_melted = df_drop.melt(id_vars="Model", var_name="Metric", value_name="Value")

    # Set plot style
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create barplot
    sns.barplot(
        x="Metric", y="Value", hue="Model", data=comparison_melted,
        palette=custom_palette, edgecolor="black", ax=ax
    )

    # Annotate bars with values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=9)

    # Format axes
    ax.set_title("Comparison of Model Metrics", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Value", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.legend_.remove()

    # Prepare table data
    table_data = df_drop.set_index("Model").round(3)
    row_colours = [custom_palette.get(model, '#FFFFFF') for model in table_data.index]

    # Add table below plot
    table_ax = plt.table(
        cellText=table_data.values,
        rowLabels=table_data.index,
        colLabels=table_data.columns,
        rowColours=row_colours,
        cellLoc='center',
        loc='bottom',
        bbox=[0, -0.55, 1, 0.45]
    )

    table_ax.auto_set_font_size(False)
    table_ax.set_fontsize(10)

    # Adjust layout
    plt.subplots_adjust(left=0.1, bottom=0.3)
    plt.show()
