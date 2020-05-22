import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess as sp
from sklearn.preprocessing import minmax_scale
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from matplotlib import style; style.use('ggplot')


################################################################################
# Mohammad's Code
################################################################################
def gen_conf_matrix(y, sigma):
    """ Generates pairwise comparison matrix W """
    exp_y = np.exp(sigma * y)
    n = len(y)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            p_ij, p_ji = cal_probability(exp_y[i], exp_y[j])
            W[i,j] = p_ij
            W[j,i] = p_ji
    return W


def cal_probability(y_i, y_j):
    """ Helper function used in gen_conf_matrix """
    p_ij = y_i / (y_i + y_j)
    return p_ij, 1 - p_ij


def cal_uncertainty(y, W):
    """
    Computes uncertaintity matrix epsilon
    Step 6
    Equation 7
    """
    # Returns uncertaintity matrix \xi
    pair_dist = pairwise_distance(y)
    weighted_distance = [W_k * pair_dist for W_k in W]
    # return np.sum(weighted_distance, axis=0)
    normalized_weighted_distance = [wd/np.sum(wd) for wd in weighted_distance]
    return np.sum(normalized_weighted_distance, 0)


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
    return 0.5 * np.log(np.sum(I_1) / (np.sum(I_2) + 1e-6))


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


def clear_W(W, m, random_seed=1):
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
    r = np.random.RandomState(random_seed)
    rm_row = r.choice(idx.shape[0], size=d_rel, replace=False)
    rm_idx = idx[rm_row]

    # Clear out selected elements symmetrically across diagonal
    for (i, j) in rm_idx:
        W[i, j] = 0.5
        W[j, i] = 0.5

    return W


def generate_rank_data(X_train, y_train, m, random_seed=1):
    """ Generates data to train a ranking model with m true labels """

    """ Set m instead of m_rel """
    # m_rel = int(m * (m - 1) / 2)

    # Get indices of upper triangle of nxn matrix
    n = y_train.size
    idx = np.triu_indices(n, k=1)
    idx = np.hstack((idx[0].reshape(-1, 1), idx[1].reshape(-1, 1)))

    # Randomly select m index pairs using reproducible seed
    r = np.random.RandomState(random_seed)
    save_row = r.choice(idx.shape[0], size=m, replace=False)
    save_idx = idx[save_row]

    # Fill in new randomly sampled features and label differences
    X_new = np.full(shape=(2 * m, 3 * X_train.shape[1]), fill_value=np.NaN)
    y_new = np.full(shape=2 * m, fill_value=np.NaN)
    k = 0

    # Fill in two-sided ([i, j] and [j, i]) and delta features
    """ Two sided and delta features may not help much """
    for (i,j) in save_idx:

        X_new[k]     = np.hstack((X_train[i], X_train[j], X_train[i] - X_train[j]))
        X_new[k + 1] = np.hstack((X_train[j], X_train[i], X_train[j] - X_train[i]))

        if y_train[i] == y_train[j]:
            y_new[k]     = 0.5
            y_new[k + 1] = 0.5

        elif y_train[i] > y_train[j]:
            y_new[k]     = 1.0
            y_new[k + 1] = 0.0

        elif y_train[i] < y_train[j]:
            y_new[k]     = 0.0
            y_new[k + 1] = 1.0

        k += 2

    return X_new, y_new, save_idx


def generate_rank_W(X_train, y_train, m, rounded, random_state=1):
    """ Generates pairwise comparison matrix W using a learned ranker trained on m true comparisons """

    # Get m randomly sampled pairs in W to train with
    X_new, y_new, save_idx = generate_rank_data(X_train, y_train, m, random_seed=random_state)

    # Fit regressor on given pairwise comparisons
    """ Change max_depth? """
    reg = XGBRegressor(max_depth=5, n_estimators=100, objective='reg:squarederror', random_state=random_state)
    reg.fit(X_new, y_new)

    # Set up features on remaining entries in W
    n = y_train.size
    X_pair = np.full(shape=(int((n * (n - 1)) / 2), 3 * X_train.shape[1]), fill_value=np.NaN)
    k = 0

    # Fill in features for upper triangle of W
    """ Stopped halfway (upper triangle only) """
    for i in range(n):
        for j in range(i + 1, n):
            X_pair[k] = np.hstack((X_train[i], X_train[j], X_train[i] - X_train[j]))
            k += 1

    # Make and scale (since probabilities) predictions on upper triangle of W
    y_pred = reg.predict(X_pair)
    y_pred = minmax_scale(y_pred, feature_range=(0, 1))

    # Fill in W symmetrically with predictions
    W = np.full(shape=(n, n), fill_value=np.NaN)
    k = 0

    for i in range(n):
        for j in range(i + 1, n):
            W[i, j] = y_pred[k]
            W[j, i] = 1 - y_pred[k]
            k += 1

    # Fill in diagonal as 0.5
    W[np.diag_indices(W.shape[0])] = 0.5

    # Fill in known entries in W symmetrically
    k = 0
    for (i, j) in save_idx:
        W[i, j] = y_new[k]
        W[j, i] = 1 - y_new[k]
        k += 1

    # Return W rounded to nearest 0.5 if rounded
    if rounded:
        return np.round(W * 2) / 2
    else:
        return W


def svm_lambdaboost(X_train, y_train, X_test, y_test, W, T=20, sample_prop=1, random_seed=None, verbose=True):
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
    acc_train_ls = []
    acc_test_ls  = []

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

        # Make sure alpha is valid
        if np.isnan(alpha_t) or np.isinf(alpha_t):
            print('Alpha invalid, terminated')
            break

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

        acc_train_curr = accuracy_score(y_train, y_train_pred)
        acc_test_curr  = accuracy_score(y_test, y_test_pred)
        acc_train_ls.append(max(acc_train_curr, 1 - acc_train_curr))
        acc_test_ls.append(max(acc_test_curr, 1 - acc_test_curr))

        if verbose:
            if t == 2:
                print('t\tTrain\t\tTest')
            print('%d\t%.2f\t\t%.2f' % (t - 1, acc_train_curr, acc_test_curr))
            if alpha_t < 0:
                print('Alpha %.2f, terminated' % alpha_t)

    # To skip initialized 0 in alpha list
    alpha = alpha[1:]

    """ CHANGED """
    # # Return final classifier parameters, weight of each submodel
    # return f_coefficient, f_intercept, alpha

    """ CHANGED (again) """
    # acc_train = accuracy_score(y_train, y_train_pred)
    # acc_test  =  accuracy_score(y_test, y_test_pred)
    # return min(acc_train, 1 - acc_train), min(acc_test, 1 - acc_test)

    # Get final accuracy on best boosting iteration
    max_idx = np.argmax(acc_test_ls)
    acc_train_final = acc_train_ls[max_idx]
    acc_test_final  = acc_test_ls[max_idx]

    if verbose:
        print('t = %d was best iteration\n' % (max_idx + 1))

    # Return minimum error (1 - max accuracy)
    return 1 - acc_train_final, 1 - acc_test_final

def tree_lambdaboost(X_train, y_train, X_test, y_test, W, T=20, max_depth=5, sample_prop=1, random_seed=None,
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
    acc_train_ls = []
    acc_test_ls  = []

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

        # Make sure alpha is valid
        if np.isnan(alpha_t) or np.isinf(alpha_t):
            print('Alpha invalid, terminated')
            break

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

        acc_train_curr = accuracy_score(y_train, y_train_pred)
        acc_test_curr  = accuracy_score(y_test, y_test_pred)
        acc_train_ls.append(max(acc_train_curr, 1 - acc_train_curr))
        acc_test_ls.append(max(acc_test_curr, 1 - acc_test_curr))

        if verbose:
            if t == 2:
                print('t\tTrain\t\tTest')
            print('%d\t%.2f\t\t%.2f' % (t - 1, acc_train_curr, acc_test_curr))
            if alpha_t < 0:
                print('Alpha %.2f, terminated' % alpha_t)

    """ CHANGED """
    # # Return subtrees and weights for each
    # return f, alpha

    """ CHANGED (again) """
    # acc_train = accuracy_score(y_train, y_train_pred)
    # acc_test  =  accuracy_score(y_test, y_test_pred)
    # return min(acc_train, 1 - acc_train), min(acc_test, 1 - acc_test)

    # Get final accuracy on best boosting iteration
    max_idx = np.argmax(acc_test_ls)
    acc_train_final = acc_train_ls[max_idx]
    acc_test_final  = acc_test_ls[max_idx]

    if verbose:
        print('t = %d was best iteration\n' % (max_idx + 1))

    # Return minimum error (1 - max accuracy)
    return 1 - acc_train_final, 1 - acc_test_final


def data_size_experiment(X_train, y_train, X_test, y_test, rank, rounded, plots_or_table='table', random_state=1,
                         verbose=False):
    """ Run experiments modifying training data size and number of labels """

    if plots_or_table == 'table':
        # To compare with Tokyo 2019 table (SDU)
        NUM_LABELS  = np.array([50, 200])
        TRAIN_SIZES = NUM_LABELS + 500
        TEST_SIZE   = 500
        NUM_TRIALS  = len(NUM_LABELS)
        NUM_REPEATS = 50
        # # To compare with Tokyo 2018 table (SU)
        # NUM_LABELS  = np.array([500])
        # TRAIN_SIZES = NUM_LABELS + 500
        # TEST_SIZE   = 100
        # NUM_TRIALS  = len(NUM_LABELS)
        # NUM_REPEATS = 20
    elif plots_or_table == 'plots':
        # To compare with Tokyo 2018/19 plots
        NUM_LABELS  = np.array([50, 100, 200, 400, 600, 800, 1600])
        TRAIN_SIZES = NUM_LABELS + 500
        TEST_SIZE   = 1000
        NUM_TRIALS  = len(NUM_LABELS)
        NUM_REPEATS = 10

    # Initialize arrays to record performance
    svm_arr  = np.full(shape=(NUM_TRIALS, NUM_REPEATS, 2), fill_value=np.NaN)
    tree_arr = np.full(shape=(NUM_TRIALS, NUM_REPEATS, 2), fill_value=np.NaN)

    # Take random stratified sample from train set and test set of appropriate size
    X_train_all, _, y_train_all, _ = train_test_split(X_train, y_train, train_size=max(TRAIN_SIZES), stratify=y_train,
                                                      shuffle=True, random_state=random_state)

    _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=TEST_SIZE, stratify=y_test,
                                            shuffle=True, random_state=random_state)

    print('Progress:')

    for i in range(NUM_TRIALS):

        # Each new trial is trained on all the data (and more) from the prev trial
        X_train = X_train_all[:TRAIN_SIZES[i]]
        y_train = y_train_all[:TRAIN_SIZES[i]]

        if rank:
            # Generate W from ranker
            W = generate_rank_W(X_train, y_train, NUM_LABELS[i], rounded, random_state=random_state)
        else:
            # Use partially filled in W pairwise comparison matrix
            W = generate_W(y_train)
            W = clear_W(W, NUM_LABELS[i])

        for j in range(NUM_REPEATS):

            # Fill in results as (train error, test error)
            svm_arr[i, j, 0], svm_arr[i, j, 1]   = svm_lambdaboost(X_train, y_train,
                                                                   X_test, y_test,
                                                                   W, T=20, sample_prop=1,
                                                                   verbose=verbose)
            tree_arr[i, j, 0], tree_arr[i, j, 1] = tree_lambdaboost(X_train, y_train,
                                                                    X_test, y_test,
                                                                    W, T=20, sample_prop=1,
                                                                    verbose=verbose)

            print('-----------------------------------------------')
            print('%d / %d complete' % (NUM_REPEATS * i + j + 1, NUM_REPEATS * NUM_TRIALS))
            print('-----------------------------------------------\n')

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


def table_results(svm_arr, tree_arr):

    svm_arr  = (1 - svm_arr) * 100
    tree_arr = (1 - tree_arr) * 100

    ls = [50, 200]

    print('(Performance on test set)')

    for i in range(2):

        print("\n%d comparisons" % ls[i])
        print("\tacc\tstd")
        print("SVM: \t%.1f\t%.1f" %  (np.mean(svm_arr[i, :, 1]),  np.std(svm_arr[i, :, 1])))
        print("Tree: \t%.1f\t%.1f" % (np.mean(tree_arr[i, :, 1]), np.std(tree_arr[i, :, 1])))

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

    print('Guess Majority:\t %.2f' % maj)
    print('KNN:\t\t %.2f\nLinear SVM:\t %.2f\nRandom Forest:\t %.2f\nXGBoost:\t %.2f'
          % (accuracy_score(y_test, knn.predict(X_test)),
             accuracy_score(y_test, svm.predict(X_test)),
             accuracy_score(y_test, rf.predict(X_test)),
             accuracy_score(y_test, xgb.predict(X_test))))

    return None
