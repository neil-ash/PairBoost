import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


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
    normalized_weighted_distance = [wd / np.sum(wd) for wd in weighted_distance]
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
    return 0.5 * np.log((np.sum(I_1) + 1e-6) / (np.sum(I_2) + 1e-6))


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


def generate_svm_rank_W(X_train, y_train, m, kernel='linear', random_seed=1):
    """ Generates pairwise comparison matrix W using Mohammad's method: learned linear SVM ranker trained on m pairs"""

    np.random.seed(random_seed)

    pos_idx = np.squeeze(np.argwhere(y_train == 1))
    neg_idx = np.squeeze(np.argwhere(y_train == -1))
    pos_idx = pos_idx[np.random.permutation(pos_idx.shape[0])[:m]]
    neg_idx = neg_idx[np.random.permutation(neg_idx.shape[0])[:m]]

    cut = min(pos_idx.size, neg_idx.size)
    if cut != m:
        print("Cut at %d rather than expected %d" % (cut, m))

    rnd_c = np.random.choice([1, -1], [pos_idx.shape[0], 1])[:cut]
    X_pair = (X_train[pos_idx,:][:cut] - X_train[neg_idx,:][:cut]) * rnd_c
    y_pair = (y_train[pos_idx][:cut] - y_train[neg_idx][:cut]) * np.squeeze(rnd_c) / 2

    clf = SVC(C=1.0, kernel=kernel)
    clf.fit(X_pair, y_pair)

    print("Training accuracy of rank SVM: %.2f" % clf.score(X_pair, y_pair))

    rank = clf.decision_function(X_train)
    sigma = 1
    W = gen_conf_matrix(rank, sigma)

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
        p_weight /= np.sum(p_weight)
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

    # Get final accuracy on best boosting iteration on train set
    # Do not record best iteration on test set -- would train hyperparameter on test
    max_idx = np.argmax(acc_train_ls)
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
        p_weight /= np.sum(p_weight)
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

    # Get final accuracy on best boosting iteration on train set
    # Do not record best iteration on test set -- would train hyperparameter on test
    max_idx = np.argmax(acc_train_ls)
    acc_train_final = acc_train_ls[max_idx]
    acc_test_final  = acc_test_ls[max_idx]

    if verbose:
        print('t = %d was best iteration\n' % (max_idx + 1))

    # Return minimum error (1 - max accuracy)
    return 1 - acc_train_final, 1 - acc_test_final


def data_size_experiment(X_train, y_train, X_test, y_test, rank, rounded, plots_or_table='table', kernel='rbf',
                         random_state=1, verbose=False):
    """ Run experiments modifying training data size and number of labels """

    if plots_or_table == 'table':
        # To compare with Tokyo 2019 table (SDU)
        NUM_LABELS  = np.array([50, 200])
        TRAIN_SIZES = NUM_LABELS + 500
        TEST_SIZE   = 500
        NUM_TRIALS  = len(NUM_LABELS)
        NUM_REPEATS = 20    # CHANGE BACK TO 50 LATER!!
        # # To compare with Tokyo 2018 table (SU)
        # NUM_LABELS  = np.array([500])
        # TRAIN_SIZES = NUM_LABELS + 500
        # TEST_SIZE   = 100
        # NUM_TRIALS  = len(NUM_LABELS)
        # NUM_REPEATS = 20
    elif plots_or_table == 'plots':
        # To compare with Tokyo 2018/19 nSD plots
        NUM_LABELS  = np.array([50, 100, 200, 400, 600, 800, 1600])
        TRAIN_SIZES = NUM_LABELS + 500
        TEST_SIZE   = 500
        NUM_TRIALS  = len(NUM_LABELS)
        NUM_REPEATS = 10
    elif plots_or_table == 'unlabeled':
        # To compare with Tokyo 2018/19 nU plots
        NUM_LABELS  = np.array([50, 50, 50, 50, 50, 50])
        TRAIN_SIZES = NUM_LABELS + np.array([100, 200, 800, 1200, 1600, 2000])
        TEST_SIZE   = 500
        NUM_TRIALS  = len(TRAIN_SIZES)
        NUM_REPEATS = 50
    elif plots_or_table == 'labeled':
        NUM_LABELS  = np.array([50, 100, 200, 400, 600, 800, 1600])
        TRAIN_SIZES = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000])
        TEST_SIZE   = 500
        NUM_TRIALS  = len(TRAIN_SIZES)
        NUM_REPEATS = 10
    elif plots_or_table == 'ratio':
        NUM_LABELS  = np.array([50, 100, 200, 400, 600, 800, 1600])
        TRAIN_SIZES = 2 * NUM_LABELS
        TEST_SIZE   = 500
        NUM_TRIALS  = len(TRAIN_SIZES)
        NUM_REPEATS = 10
        # Record intermediate lin SVM acc
        ls_base = []

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
            """ CHANGED TO SVM """
            # # Generate W from ranker
            # W = generate_rank_W(X_train, y_train, NUM_LABELS[i], rounded, random_state=random_state)
            W = generate_svm_rank_W(X_train, y_train, NUM_LABELS[i], kernel=kernel, random_seed=random_state)
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

        if plots_or_table == 'ratio':
            clf = LinearSVC()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print('Linear SVM Accuracy for n = %d: %.3f\n' % (y_train.size, acc))
            ls_base.append(acc)

    if plots_or_table == 'labeled':
        clf = LinearSVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print('\nLinear SVM Accuracy: %.3f' % accuracy_score(y_test, y_pred))

    if plots_or_table == 'ratio':
        return svm_arr, tree_arr, 1 - np.array(ls_base)

    return svm_arr, tree_arr


def plot_err(arr, dname, mname, save=False, SD_or_U='SD', scale=1, pltstyle='seaborn-whitegrid'):
    """ Plots performance when modifying training data size and number of labels"""

    plt.style.use(pltstyle)
    plt.rcParams["font.family"] = "Times New Roman"

    # (std / sqrt(n)) to calculate standard error, n = 10 trials
    NUM_REPEATS = 10

    if SD_or_U == 'SD':
        x = np.array([50, 100, 200, 400, 600, 800, 1600])
    elif SD_or_U == 'U':
        x = np.array([100, 200, 800, 1200, 1600, 2000])
    y = np.mean(arr, axis=1)

    sderr = np.std(arr, axis=1)
    sderr = sderr / np.sqrt(NUM_REPEATS)

    plt.rcParams['figure.figsize'] = [5, 3]
    title = dname + ' Dataset, ' + mname + ' Model'

    plt.plot(x, y[:, 1], c='C1')
    plt.fill_between(x, y[:, 1] - (sderr[:, 1] * scale), y[:, 1] + (sderr[:, 1] * scale),
                 facecolors='C1', alpha=0.5)
    plt.scatter(x, y[:, 1], label='Test', c='C1')

    plt.plot(x, y[:, 0], c='C0')
    plt.fill_between(x, y[:, 0] - (sderr[:, 0] * scale), y[:, 0] + (sderr[:, 0] * scale),
                 facecolors='C0', alpha=0.5)
    plt.scatter(x, y[:, 0], label='Train', c='C0')

    plt.legend(frameon=True ,shadow=True,fancybox=True, framealpha=1.0)
    # plt.title(title)
    if SD_or_U == 'SD':
        # plt.xlabel('m: Number of Pairwise Comparisons')
        plt.xlabel('m')
    elif SD_or_U == 'U':
        plt.xlabel('n: Number of unlabeled points')
    plt.ylabel('Classification Error')

    if save:
        plt.savefig('final_plots/' + SD_or_U + '/' + title + '.pdf', bbox_inches='tight')

    return None


def table_results(svm_arr, tree_arr):
    """ Nicely print out results from data_size_experiment vs Tokyo plots """

    # (std / sqrt(n)) to calculate standard error, n = 50 trials (NOT 50 or 200 labels)
    NUM_REPEATS = 20    # CHANGE BACK TO 50 LATER!!

    # Want accuracy not error
    svm_arr  = (1 - svm_arr) * 100
    tree_arr = (1 - tree_arr) * 100

    ls = [50, 200]

    print('(Performance on test set)')

    for i in range(2):

        print("\n%d comparisons" % ls[i])
        print("\tacc\tsderr")
        print("SVM: \t%.1f\t%.1f" %  (np.mean(svm_arr[i, :, 1]),  np.std(svm_arr[i, :, 1]) / np.sqrt(NUM_REPEATS)))
        print("Tree: \t%.1f\t%.1f" % (np.mean(tree_arr[i, :, 1]), np.std(tree_arr[i, :, 1]) / np.sqrt(NUM_REPEATS)))

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
