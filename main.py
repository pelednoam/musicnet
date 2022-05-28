import numpy as np
import traceback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
import librosa as lbr
from librosa.display import specshow
import pandas as pd
import os.path as op
import glob
from collections import defaultdict, Counter
from operator import itemgetter
from tqdm import tqdm
import copy
from pprint import pprint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import utils
import songs_utils

PROJECT_FOL = [d for d in [
    'C:\\Users\\peled\\projects\\musicnet',
    '/autofs/space/mmvtclin_001/thibault/1/users/npeled/projects/musicnet'] if op.isdir(d)][0]
TRAINING_FOL = op.join(PROJECT_FOL, 'PS1_wav')
TESTING_FOL = op.join(PROJECT_FOL, 'PS2_wav')


def read_midi_training_files(songs_fol, buffer_len=30, buffer_shift=10, songs_type='wav', n_jobs=4):
    # https://github.com/mdeff/fma
    # https://github.com/mdeff/fma/blob/master/features.py
    is_training = songs_fol == TRAINING_FOL
    songs_composers_fname = op.join(PROJECT_FOL, 'songs_composers.pkl')
    songs_features_fname = op.join(PROJECT_FOL, 'songs_features_{}.csv'.format(
        'training' if is_training else 'testing'))
    songs_fnames = []
    songs_composers = {}
    all_features = pd.DataFrame(columns=songs_utils.columns(), dtype=np.float32)
    for song_fname, composer in songs_iterator(songs_fol, songs_type):
        file_name = op.basename(song_fname)
        songs_fnames.append(song_fname)
        if is_training:
            songs_composers[file_name] = composer
    # Separate the data into n_jobs chunks
    indices = np.array_split(np.arange(len(songs_fnames)), n_jobs)
    chunks = [([songs_fnames[info_ind] for info_ind in indices_chunk], buffer_len, buffer_shift)
              for indices_chunk in indices]
    results = utils.run_parallel(calc_songs_features, chunks, n_jobs)
    # Go over the results from the different chunks and concatenate them
    for chunk_features in results:
        all_features = pd.concat([all_features, chunk_features])
    print('Writing songs in {}'.format(songs_features_fname))
    all_features.to_csv(songs_features_fname)
    if is_training:
        utils.save(songs_composers, songs_composers_fname)


def songs_iterator(songs_fol, songs_type='wav'):
    if songs_fol == TRAINING_FOL:
        composers = [op.basename(d) for d in glob.glob(op.join(songs_fol, '*')) if op.isdir(d)]
        for composer in composers:
            for song_fname in glob.glob(op.join(songs_fol, composer, '*.{}'.format(songs_type))):
                yield song_fname, composer
    elif songs_fol == TESTING_FOL:
        for song_fname in glob.glob(op.join(songs_fol, '*.{}'.format(songs_type))):
            yield song_fname, None


def calc_songs_features(params):
    songs_fnames, buffer_len, buffer_shift = params
    chunk_features = pd.DataFrame(columns=songs_utils.columns(), dtype=np.float32)
    for song_fname in tqdm(songs_fnames):
        all_features = songs_utils.compute_features(song_fname, buffer_len, buffer_shift)
        # Go over the features from the different buffers (windows) and add them to the DataFrame
        for buffer_ind, features in enumerate(all_features):
            key = '{}_{}'.format(op.basename(song_fname), buffer_ind)
            if features is not None:
                chunk_features.loc[key] = features
    return chunk_features


def get_song_name(buffer_name, with_ext=False):
    if with_ext:
        return buffer_name[:buffer_name.rfind('_')]
    else:
        return buffer_name[:buffer_name.rfind('.')]


def train(features_selection_num=50, overwrite=False):
    songs_composers_fname = op.join(PROJECT_FOL, 'songs_composers.pkl')
    songs_train_features_fname = op.join(PROJECT_FOL, 'songs_features_training.csv')
    figures_fol = utils.make_dir(op.join(PROJECT_FOL, 'figures_train'))
    models_fol = utils.make_dir(op.join(PROJECT_FOL, 'models'))

    train_features = pd.read_csv(songs_train_features_fname, index_col=0, header=[0, 1, 2])
    songs_names = get_songs_names(train_features)
    songs_composers = utils.load(songs_composers_fname)
    labels = [songs_composers[get_song_name(buffer_name, True)] for buffer_name in train_features.index]

    X = standardize_features(train_features)
    y = encode_labels(labels)

    # Features selection
    features_selection_model = SelectKBest(mutual_info_classif, k=features_selection_num).fit(X, y)
    utils.save(features_selection_model, op.join(PROJECT_FOL, 'features_selection_model.pkl'))
    X = features_selection_model.transform(X)

    classifiers = get_classifiers()
    all_scores, conf_mats = calc_training_scores(X, y, classifiers, figures_fol, songs_names, overwrite)
    fit_and_save_classifiers(classifiers, X, y, all_scores, conf_mats, models_fol)


def standardize_features(features):
    X = features.values
    # Remove nans
    X[np.where(np.isnan(X))] = 0
    # Standardize features by removing the mean and scaling to unit variance.
    scaler = StandardScaler(copy=False)
    X = scaler.fit_transform(X)
    return X


def encode_labels(labels):
    # Encode the labels
    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    return y


def fit_and_save_classifiers(classifiers, X, y, all_scores, conf_mats, models_fol):
    for clf_name, clf in classifiers.items():
        # Scores: (score, max_prob_score,  var_prob_score, roc_auc_ovo, roc_auc_ovr)
        score, max_probs_score, var_prob_score, roc_auc_ovo, roc_auc_ovr = calc_mean_scores(all_scores, clf_name)
        print_scores(clf_name, score, max_probs_score, var_prob_score, roc_auc_ovo, conf_mats)
        # Create 10 random equally selected train sets and save the classifiers
        all_data_split = split_equally(y, n_split=10, test_size=0)
        for split_ind, (train, _) in enumerate(all_data_split):
            clf.fit(X[train], y[train])
            utils.save(clf, op.join(models_fol, '{}_{}.pkl'.format(clf_name, split_ind)))


def print_scores(clf_name, score, max_probs_score, var_prob_score, roc_auc_ovo, conf_mats):
    print('Classifier: {} <br/>'.format(clf_name))
    print('score: {:.3f}, max_probs: {:.3f}, var_probs: {:.3f}, roc_auc: {:.3f} <br/>'.format(
        score, max_probs_score, var_prob_score, roc_auc_ovo))
    print('Confusion matrix: <br/>')
    for l in conf_mats[clf_name]:
        print('{} <br/>'.format(str([int(_l) for _l in l]).replace('[', '').replace(']', '')))


def calc_training_scores(X, y, classifiers, figures_fol, songs_names, overwrite=False):
    scores_fname = op.join(PROJECT_FOL, 'all_train_scores.pkl')
    # If we've already ran this function, and don't want to overwrite, we can just load the scores
    if op.isfile(scores_fname) and not overwrite:
        all_scores, conf_mats = utils.load(scores_fname)
        return all_scores, conf_mats

    all_scores, all_probs_scores, conf_mats = defaultdict(list), defaultdict(list), {}
    labels_num = len(set(y))
    # Split equally (same number of items per label) using 10 splits and 10% for testing
    data_split = split_equally(y, n_split=10, test_size=0.1)
    for clf_name, clf in tqdm(classifiers.items()):
        conf_mat = np.zeros((labels_num, labels_num))
        for split_ind, (train, test) in enumerate(data_split):
            clf.fit(X[train], y[train])
            pred_model = clf.predict(X[test])
            probs = clf.predict_proba(X[test])
            # Calculate the different scores
            conf_mat += confusion_matrix(pred_model, y[test])
            score = clf.score(X[test], y[test])
            # Calc how much the classifer is "sure" about the decision
            max_prob_score = np.mean(np.max(probs, axis=1))
            # Calc how much the classifer is "uncertain" about the decision
            var_prob_score = np.mean(np.var(probs, axis=1))
            roc_auc_ovo, roc_auc_ovr = calc_auc(y[test], probs)
            all_scores[clf_name].append((score, max_prob_score,  var_prob_score, roc_auc_ovo, roc_auc_ovr))
            calc_probs(clf, clf_name, X[test], figures_fol, songs_names, figure_ind=split_ind, y=y[test])
        conf_mats[clf_name] = conf_mat
    utils.save((all_scores, conf_mats), scores_fname)
    return all_scores, conf_mats


def calc_mean_scores(all_scores, clf_name):
    score = np.mean([scores[0] for scores in all_scores[clf_name]])
    max_probs_score = np.mean([scores[1] for scores in all_scores[clf_name]])
    var_prob_score = np.mean([scores[2] for scores in all_scores[clf_name]])
    roc_auc_ovo = np.mean([scores[3] for scores in all_scores[clf_name]])
    roc_auc_ovr = np.mean([scores[4] for scores in all_scores[clf_name]])
    return score, max_probs_score, var_prob_score, roc_auc_ovo, roc_auc_ovr


def split_equally(y, n_split=5, test_size=0.2):
    y_count = Counter(y)
    n_per_class = min(y_count.items(), key=itemgetter(1))[1]
    inds_all = []
    for _ in range(n_split):
        inds_train, inds_test = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        for label in set(y):
            select_inds = np.random.choice(np.where(y == label)[0], n_per_class, replace=False)
            if test_size > 0:
                label_inds_train, label_inds_test = train_test_split(select_inds, test_size=test_size)
            else:
                # No test, only train
                label_inds_train, label_inds_test = select_inds, []
            inds_train = np.concatenate((inds_train, label_inds_train))
            inds_test = np.concatenate((inds_test, label_inds_test))
        inds_all.append((inds_train, inds_test))
    return inds_all


def calc_auc(y_test, y_prob):
    from sklearn.metrics import roc_auc_score
    roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro") # One-vs-one
    roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro") # One-vs-rest
    return roc_auc_ovo, roc_auc_ovr


def multi_dimensional_scaling(X):
    from sklearn import manifold
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9)
    X_transformed = mds.fit_transform(X)
    plt.figure()
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
    plt.show()
    print('asdf')


def get_songs_names(features):
    songs_names = features.index
    songs_names = [get_song_name(s) for s in songs_names] # Shorten the names
    return songs_names


def test():
    songs_features_test_fname = op.join(PROJECT_FOL, 'songs_features_testing.csv')
    figures_fol = utils.make_dir(op.join(PROJECT_FOL, 'figures_test_var'))
    all_scores, conf_mats = utils.load(op.join(PROJECT_FOL, 'all_train_scores.pkl')) # load classifiers scores

    features_test = pd.read_csv(songs_features_test_fname, index_col=0, header=[0, 1, 2])
    songs_names = get_songs_names(features_test)
    X = standardize_features(features_test)

    # Features selection
    features_selection_model = utils.load(op.join(PROJECT_FOL, 'features_selection_model.pkl'))
    X = features_selection_model.transform(X)

    models_probs_var, models_probs_max = defaultdict(list), defaultdict(list)
    models = glob.glob(op.join(PROJECT_FOL, 'models', '*.pkl'))
    for model_fname in tqdm(models):
        clf_name = op.basename(model_fname).split('_')[0]
        clf = utils.load(model_fname)
        probs = clf.predict_proba(X)
        probs_var = np.var(probs, axis=1)
        if np.all(probs_var == probs_var[0]):
            continue
        probs_max = np.max(probs, axis=1)
        models_probs_var[clf_name].append(probs_var)
        models_probs_max[clf_name].append(probs_max)

    for clf_name, model_probs_var in models_probs_var.items():
        score, max_probs_score, var_prob_score, roc_auc_ovo, roc_auc_ovr = calc_mean_scores(all_scores, clf_name)
        # if np.isnan(score) or score * max_probs_score < 0.8:
        #     continue
        if clf_name not in ['MLP2']:
            continue
        model_mean_probs_var = np.array(model_probs_var).mean(0)
        model_mean_probs_max = np.array(models_probs_max[clf_name]).mean(0)

        songs_probs_max, songs_probs_var = defaultdict(list), defaultdict(list)
        for ind in range(len(songs_names)):
            songs_probs_max[songs_names[ind]].append(model_mean_probs_max[ind])
            songs_probs_var[songs_names[ind]].append(model_mean_probs_var[ind])

        songs_probs_max_mean = [np.mean(probs) for song_name, probs in songs_probs_max.items()]
        songs_probs_var_mean = [np.mean(probs) for song_name, probs in songs_probs_var.items()]

        songs_scores = sorted([(np.mean(probs), song_name) for song_name, probs in songs_probs_max.items()
                              if (np.mean(probs) < 0.85)])
        print('Classifier: {}'.format(clf_name))
        print(songs_scores)

        # plt.figure()
        # plt.hist([score for (score, _ ) in songs_scores], bins=30)
        # plt.show()
        # print('asdf')

        plot_certainty_scatter_plot(clf_name, songs_probs_max_mean, songs_probs_var_mean, figures_fol)
    print('asdf')


def plot_certainty_scatter_plot(clf_name, model_mean_probs_max, model_mean_probs_var, figures_fol):
    plt.figure()
    plt.scatter(model_mean_probs_max, model_mean_probs_var)
    plt.title(clf_name)
    plt.xlabel('max(p)')
    plt.ylabel('var(p)')
    plt.savefig(op.join(figures_fol, '{}.jpg'.format(clf_name)))
    plt.close()


def calc_probs(clf, clf_name, X, figures_fol, songs_names, figure_ind=0, y=None, add_annotations=False):
    probs = clf.predict_proba(X)
    if y is not None:
        predict_labels = clf.predict(X)
    probs_sorted_indices = np.argsort(probs, axis=1)
    probs_big_small, songs_probs = [], {}
    for ind, (p_indices, p) in enumerate(zip(probs_sorted_indices, probs)):
        probs_big_small.append((p[p_indices[-1]], p[p_indices[-2]]))  # max, one after max
        songs_probs[songs_names[ind]] = p[p_indices[-1]]
    probs_big_small = np.array(probs_big_small)
    ax = plt.figure().add_subplot(111)
    if y is not None:
        colors = ['green' if label==pred_label else 'red' for label, pred_label in zip(y, predict_labels)]
    else:
        colors = ['blue' for _ in range(probs_big_small.shape[0])]
    plt.scatter(probs_big_small[:, 0], probs_big_small[:, 1], c=colors)
    plt.xlabel('max p')
    plt.ylabel('second biggest p')
    if add_annotations:
        for _x, _y, name in zip(probs_big_small[:, 0], probs_big_small[:, 1], songs_names):
            utils.add_annotation(name, _x, _y)
    if y is not None:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mpatches.Patch(color='green', label='correct'))
        handles.append(mpatches.Patch(color='red', label='wrong'))
        ax.legend(handles=handles)

    plt.savefig(op.join(figures_fol, '{}_{}.jpg'.format(clf_name, figure_ind)))
    plt.close()
    return songs_probs


def get_classifiers():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV
    from sklearn.ensemble import ExtraTreesClassifier

    # Identify the possible hyper params for rbf SVC
    gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    Cs = [1, 10, 100, 1e3, 1e4, 1e5]
    svc_rbf_params_grid = {'gamma': gammas, 'C': Cs}
    scv_rbf_clf = SVC(kernel='rbf', probability=True)

    # Identify the possible hyper params for poly SVC
    svc_poly_params_grid = {'degree': np.arange(1, 11)}
    scv_poly_clf = SVC(kernel='poly', probability=True)

    # Identify the possible hyper params for  Multi-layer Perceptron
    mlp_params_grid = {
        'hidden_layer_sizes': [
            (500, 400, 300, 200, 100), (400, 400, 400, 400, 400),
            (300, 300, 300, 300, 300), (200, 200, 200, 200, 200)],
         # 'activation': ['logistic', 'tanh', 'relu'],
         # 'alpha': [0.0001, 0.001, 0.005],
         }
    mlp_clf = MLPClassifier(max_iter=2000)

    classifiers = {
        'LR1': LogisticRegression(penalty='l1', max_iter=2000, solver='liblinear'),
        'LR2': LogisticRegression(penalty='l2', max_iter=2000),
        'kNN': KNeighborsClassifier(),
        'SCVrbf': HalvingGridSearchCV(estimator=scv_rbf_clf, param_grid=svc_rbf_params_grid),
        'SVCpoly': HalvingGridSearchCV(estimator=scv_poly_clf, param_grid=svc_poly_params_grid),
        'linSVC': SVC(kernel='linear', probability=True),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100),
        'MLP1': MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000),
        'MLP2': MLPClassifier(hidden_layer_sizes=(200, 50), max_iter=2000),
        # 'MLPgrid': HalvingGridSearchCV(estimator=mlp_clf, param_grid=mlp_params_grid),
    }
    return classifiers


if __name__ == '__main__':
    # A nice tutorial for analysing audio:
    # https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html
    # The features exatraction part is based on this github:
    # https://github.com/mdeff/fma

    songs_type, n_jobs = 'wav', 1
    # read_midi_training_files(TRAINING_FOL, buffer_len=30, buffer_shift=30, songs_type=songs_type, n_jobs=n_jobs)
    # read_midi_training_files(TESTING_FOL, buffer_len=30, buffer_shift=30, songs_type=songs_type, n_jobs=n_jobs)
    train(features_selection_num=50, overwrite=True)
    # test()


