import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess as sp
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from matplotlib import style; style.use('ggplot')


# Consistent with Tokyo 2018/19 plots
NUM_LABELS  = np.array([50, 100, 200, 400, 600, 800, 1600])
TRAIN_SIZES = NUM_LABELS + 500
TEST_SIZE   = 1000
NUM_TRIALS  = len(NUM_LABELS)
NUM_REPEATS = 10

# # To compare with Tokyo 2018 table
# NUM_LABELS  = np.array([500])
# TRAIN_SIZES = NUM_LABELS + 500
# TEST_SIZE   = 1000
# NUM_TRIALS  = len(NUM_LABELS)
# NUM_REPEATS = 10


def cal_uncertainty(y, W):
  """
  Computes uncertaintity matrix epsilon
  Step 6
  Equation 7
  """
  # Returns uncertaintity matrix \xi
  pair_dist = pairwise_distance(y)
  weighted_distance = [W_k * pair_dist for W_k in W]
  return np.sum(weighted_distance, axis=0)
  # normalized_weighted_distance = [wd/np.sum(wd) for wd in weighted_distance]
  # return np.sum(normalized_weighted_distance, 0)


def pairwise_distance(y):
  """ Helper function used in cal_uncertainty """
  # Calculate the pairwise exponential distance matrix of instances
  # pairwise_distance_{ij} = \exp(y_j - y_i)
  y = y / (np.max(y) + 1e-12)
  if len(y.shape) == 1:
    y = np.expand_dims(y, 1)
  if y.shape[0] == 1:
    y = y.T
  return np.exp(-y).dot(np.exp(y).T)


def cal_weights(xi):
  """
  Computes importance (weight) of each instance, wi
  Step 7
  Equation 8
  """
  return np.sum(xi - xi.T, axis=1)


def cal_alpha(y, xi):
  """
  Calculates the weight of classifier i, alpha
  Step 12
  """
  y_p = (y > 0).astype(float).reshape(-1, 1)
  y_n = (y < 0).astype(float).reshape(-1, 1)
  I_1 = xi * y_p.dot(y_n.T)
  I_2 = xi * y_n.dot(y_p.T)
  return 0.5 * np.log(np.sum(I_1) / np.sum(I_2))


################################################################################
# Neil's Code
################################################################################
def generate_W(y_train):
    """ Generates confidence matrix using only the original labels (no ranking) """

    m = y_train.shape[0]
    W = np.full(shape=(m, m), fill_value=np.nan)

    for i in range(m):
        for j in range(m):
            if y_train[i] == y_train[j]:
                W[i, j] = 0.5
            elif y_train[i] > y_train[j]:
                W[i, j] = 1.0
            elif y_train[i] < y_train[j]:
                W[i, j] = 0.0

    return W


def clear_W(W, m):
    """ Replace entries in W with 0.5 to get m total labels """

    # Get indices of upper triangle of nxn matrix
    n = W.shape[0]
    idx = np.triu_indices(n, k=1)
    idx = np.hstack((idx[0].reshape(-1, 1), idx[1].reshape(-1, 1)))

    # Get number of elements (above diagonal, since symmetric) to clear
    n_rel = (n * (n - 1) / 2)
    m_rel = (m * (m - 1) / 2)
    d_rel = int(n_rel - m_rel)

    # Randomly select some index pairs
    rm_row = np.random.choice(idx.shape[0], size=d_rel, replace=False)
    rm_idx = idx[rm_row]

    # Clear out selected elements symmetrically across diagonal
    for (i, j) in rm_idx:
        W[i, j] = 0.5
        W[j, i] = 0.5

    return W


def generate_rank_data(X_train, y_train, m):
    """ Generates data to train a ranking model with m true labels """

    """ Set m instead of m_rel """
    # m_rel = int(m * (m - 1) / 2)

    # Get indices of upper triangle of nxn matrix
    n = y_train.size
    idx = np.triu_indices(n, k=1)
    idx = np.hstack((idx[0].reshape(-1, 1), idx[1].reshape(-1, 1)))

    # Randomly select m index pairs
    save_row = np.random.choice(idx.shape[0], size=m, replace=False)
    save_idx = idx[save_row]

    # Fill in new randomly sampled features and label differences
    X_new = np.full(shape=(m, 2 * X_train.shape[1]), fill_value=np.NaN)
    y_new = np.full(shape=m, fill_value=np.NaN)
    k = 0

    for (i,j) in save_idx:
        X_new[k] = np.hstack((X_train[i], X_train[j]))
        if y_train[i] == y_train[j]:
            y_new[k] = 0.5
        elif y_train[i] > y_train[j]:
            y_new[k] = 1.0
        elif y_train[i] < y_train[j]:
            y_new[k] = 0.0
        k += 1

    return X_new, y_new


def generate_rank_W(X_train, y_train, m):
    """ Generates pairwise comparison matrix W using a learned ranker trained on m true comparisons """

    # Get m randomly sampled pairs in W to train with
    X_new, y_new = generate_rank_data(X_train, y_train, m)

    """ CHANGED TO XGBOOST """
    svr = XGBRegressor(max_depth=3, n_estimators=100, objective='reg:squarederror')
    # svr = LinearSVR()
    svr.fit(X_new, y_new)

    # Fill in predicted values of W
    n = y_train.size
    X_pair = np.full(shape=(n * n, 2 * X_train.shape[1]), fill_value=np.NaN)
    k = 0

    """ MAY ONLY WANT TO STOP HALFWAY """
    for i in range(n):
        for j in range(n):
            X_pair[k] = np.hstack((X_train[i], X_train[j]))
            k += 1

    y_pair = svr.predict(X_pair)
    y_pair = minmax_scale(y_pair, feature_range=(0, 1))
    W = y_pair.reshape(n, n)

    return W


def svm_lambdaboost(X_train, y_train, X_test, y_test, W, T=10, sample_prop=2, random_seed=None, verbose=True):
    """ Generates linear svm lambdaboost classifier """

    if verbose:
        print('LINEAR SVM SUBMODELS')
        print('Using %d classifiers and sample proportion of %d'
              % (T, sample_prop))
        if random_seed:
            print('Random seed %d', random_seed)

    # Constants
    m = X_train.shape[0]
    n = X_train.shape[1]
    np.random.seed(random_seed)

    # Initialize model parameters
    f_intercept = 0
    f_coefficient = np.zeros(n)

    # Initialize counters
    t = 1
    alpha = [0.0]

    # Training
    while t <= T and alpha[-1] >= 0:

        # Step 6: compute epsilon
        curr_pred = np.dot(X_train, f_coefficient) + f_intercept
        # Scale predictions, works well empirically
        curr_pred = minmax_scale(curr_pred, feature_range=(-1, 1))
        # Remember that W is passed as a list of arrays!
        epsilon = cal_uncertainty(curr_pred, [W])

        # Step 7: compute weights
        weight = cal_weights(epsilon)

        # Step 8: extract labels
        y = np.sign(weight)

        # Step 9: create training (sample) data by sampling based on weights
        p_weight = np.abs(weight)
        p_weight /= (np.sum(p_weight) + 1e-12)
        sample = np.random.choice(m, size=m*sample_prop, replace=True, p=p_weight)
        X_sample = X_train[sample]
        y_sample = y[sample]

        # Step 10: learn binary classifier on training (sample) data
        clf = LinearSVC(max_iter=1000)
        clf.fit(X_sample, y_sample)

        # Step 11: predict labels using current classifier
        y_pred = clf.predict(X_train)

        # Step 12: compute weight of current classifier
        alpha_t = cal_alpha(y_pred, epsilon)

        # Step 13: update final classifier
        f_coefficient += alpha_t * clf.coef_.ravel()
        f_intercept += alpha_t * clf.intercept_

        # Update loop
        alpha.append(alpha_t)
        t += 1

        # Evaluation
        y_train_pred = np.dot(X_train, f_coefficient) + f_intercept
        y_test_pred = np.dot(X_test, f_coefficient) + f_intercept
        y_train_pred = np.sign(y_train_pred)
        y_test_pred = np.sign(y_test_pred)

        if verbose:
            if t == 2:
                print('t\tTrain\t\tTest')
            print('%d\t%.3f\t\t%.3f' %(t - 1,
                                       accuracy_score(y_train, y_train_pred),
                                       accuracy_score(y_test, y_test_pred)))
            if alpha_t < 0:
                print('Alpha %.3f, terminated' % alpha_t)

    if verbose:
        print('Done!')

    # To skip initialized 0 in alpha list
    alpha = alpha[1:]

    """ CHANGED """
    # # Return final classifier parameters, weight of each submodel
    # return f_coefficient, f_intercept, alpha
    return 1 - accuracy_score(y_train, y_train_pred), 1 - accuracy_score(y_test, y_test_pred)


def tree_lambdaboost(X_train, y_train, X_test, y_test, W, T=10, max_depth=5, sample_prop=2, random_seed=None,
                    verbose=True):
    """ Generates decision tree lambdaboost classifier """

    if verbose:
        print('DECISION TREE SUBMODELS')
        print('Using %d classifiers and sample proportion of %d'
              % (T, sample_prop))
        if random_seed:
            print('Random seed %d', random_seed)

    # Constants
    m = X_train.shape[0]
    n = X_train.shape[1]
    np.random.seed(random_seed)

    # Initialize counters
    t = 1
    alpha_t = 0

    # Instantiate models and weights
    f = []
    alpha = []

    # Training
    while t <= T and alpha_t >= 0:

        # Step 6: compute epsilon
        if t == 1:
            curr_pred = np.zeros(y_train.shape)
        else:
            curr_pred = sum([alpha[i] * f[i].predict(X_train) for i in range(t - 1)])
        # Scale predictions, works well empirically for SVM...
        #curr_pred = minmax_scale(curr_pred, feature_range=(-1, 1))
        # Remember that W is passed as a list of arrays!
        epsilon = cal_uncertainty(curr_pred, [W])

        # Step 7: compute weights
        weight = cal_weights(epsilon)

        # Step 8: extract labels
        y = np.sign(weight)

        # Step 9: create training (sample) data by sampling based on weights
        p_weight = np.abs(weight)
        p_weight /= (np.sum(p_weight) + 1e-12)
        sample = np.random.choice(m, size=m*sample_prop, replace=True, p=p_weight)
        X_sample = X_train[sample]
        y_sample = y[sample]

        # Step 10: learn binary classifier on training (sample) data
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X_sample, y_sample)

        # Step 11: predict labels using current classifier
        y_pred = clf.predict(X_train)

        # Step 12: compute weight of current classifier
        alpha_t = cal_alpha(y_pred, epsilon)

        # Step 13: update final classifier
        f.append(clf)

        # Update loop
        alpha.append(alpha_t)
        t += 1

        # Evaluation
        y_train_pred = sum([alpha[i] * f[i].predict(X_train) for i in range(t - 1)])
        y_test_pred = sum([alpha[i] * f[i].predict(X_test) for i in range(t - 1)])
        y_train_pred = np.sign(y_train_pred)
        y_test_pred = np.sign(y_test_pred)

        if verbose:
            if t == 2:
                print('t\tTrain\t\tTest')
            print('%d\t%.3f\t\t%.3f' %(t - 1,
                                       accuracy_score(y_train, y_train_pred),
                                       accuracy_score(y_test, y_test_pred)))
            if alpha_t < 0:
                print('Alpha %.3f, terminated' % alpha_t)

    if verbose:
        print('Done!')

    """ CHANGED """
    # # Return subtrees and weights for each
    # return f, alpha
    return 1 - accuracy_score(y_train, y_train_pred), 1 - accuracy_score(y_test, y_test_pred)


def data_size_experiment(X, y, rank, random_state=None, verbose=False):
    """ Run experiments modifying training data size and number of labels """

    svm_arr  = np.full(shape=(NUM_TRIALS, NUM_REPEATS, 2), fill_value=np.NaN)
    tree_arr = np.full(shape=(NUM_TRIALS, NUM_REPEATS, 2), fill_value=np.NaN)

    # Ensure train and test sets disjoint
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.5,
                                                                stratify=y, random_state=None)

    # Ensure data is sampled from same set for all trials
    X_train_all = X_train_all[:max(TRAIN_SIZES)]
    y_train_all = y_train_all[:max(TRAIN_SIZES)]
    X_test  = X_test[:TEST_SIZE]
    y_test  = y_test[:TEST_SIZE]

    print('Progress:')

    for i in range(NUM_TRIALS):

        # Each new trial is trained on all the data (and more) from the prev trial
        X_train = X_train_all[:TRAIN_SIZES[i]]
        y_train = y_train_all[:TRAIN_SIZES[i]]

        if rank:
            # Generate W from ranker
            W = generate_rank_W(X_train, y_train, NUM_LABELS[i])
        else:
            # Use partially filled in W pairwise comparison matrix
            W = generate_W(y_train)
            W = clear_W(W, NUM_LABELS[i])

        for j in range(NUM_REPEATS):

            # Fill in results as (train error, test error)
            svm_arr[i, j, 0], svm_arr[i, j, 1]   = svm_lambdaboost(X_train, y_train,
                                                                   X_test, y_test,
                                                                   W, T=20, verbose=verbose)
            tree_arr[i, j, 0], tree_arr[i, j, 1] = tree_lambdaboost(X_train, y_train,
                                                                    X_test, y_test,
                                                                    W, T=20, verbose=verbose)

            print(NUM_REPEATS * i + j + 1)

    return svm_arr, tree_arr


def plot_err(arr, dname, mname, save=False, group='median'):
    """ Plots performance when modifying training data size and number of labels"""

    x = NUM_LABELS
    if group == 'median':
        y = np.median(arr, axis=1)
    elif group == 'mean':
        y = np.mean(arr, axis=1)
    elif group == 'min':
        y = np.min(arr, axis=1)

    plt.rcParams['figure.figsize'] = [8, 4]
    title = dname + ' Dataset, ' + mname + ' Model'

    plt.plot(x, y[:, 1])
    plt.plot(x, y[:, 0])
    plt.scatter(x, y[:, 1], label='Test')
    plt.scatter(x, y[:, 0], label='Train')
    plt.legend()
    plt.title(title)
    plt.xlabel('m: Number of Labeled Points')
    plt.ylabel('Classification Error')

    if save:
        plt.savefig('plots/' + title + '.png', bbox_inches='tight')

    return None


def baselines(X_train, y_train, X_test, y_test):
    """ Performance baselines """

    print('BASELINES ACCURACIES')

    # Guessing majority
    maj = np.sum(y_test == 1) / y_test.size
    maj = max(maj, 1 - maj)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Linear model
    svm = LinearSVC(max_iter=1000)
    svm.fit(X_train, y_train)

    # Random forest
    rf = RandomForestClassifier(n_estimators=25, max_depth=3)
    rf.fit(X_train, y_train)

    # XGBoost
    xgb = XGBClassifier(n_estimators=25, max_depth=3)
    xgb.fit(X_train, y_train)

    print('Guess Majority:\t %.3f' % maj)
    print('KNN:\t\t %.3f\nLinear SVM:\t %.3f\nRandom Forest:\t %.3f\nXGBoost:\t %.3f'
          % (accuracy_score(y_test, knn.predict(X_test)),
             accuracy_score(y_test, svm.predict(X_test)),
             accuracy_score(y_test, rf.predict(X_test)),
             accuracy_score(y_test, xgb.predict(X_test))))

    return None
