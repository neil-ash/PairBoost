import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess as sp
from sklearn.preprocessing import minmax_scale
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# def gen_conf_matrix1(y,sigma):
#   """
#   Computes confidence matrix W (faster, approximate version)
#   Step 3
#   """
#   exp_y = np.exp(sigma * y)
#   n = len(y)
#   W = np.zeros((n,n))
#   expand_y = np.tile(y,(n,1)).T
#   exp_sum_y = np.exp(exp_y).reshape(-1,1).dot(np.exp(exp_y).reshape(1,-1))
#   log_exp_sum_y = np.log(exp_sum_y)
#   return expand_y / log_exp_sum_y
#
# def gen_conf_matrix(y, sigma):
#   """
#   Computes confidence matrix W
#   Step 3
#   Assumes labels are (-1, +1) (?)
#   """
#   exp_y = np.exp(sigma * y)
#   n = len(y)
#   W = np.zeros((n,n))
#   for i in range(n):
#     for j in range(i+1):
#       p_ij, p_ji = cal_probability(exp_y[i],exp_y[j])
#       W[i,j] = p_ij
#       W[j,i] = p_ji
#   return W
#
# def cal_probability(y_i, y_j):
#   """
#   Helper function for gen_conf_matrix
#   Assumes labels are (-1, +1) (?)
#   """
#   p_ij = y_i / (y_i + y_j)
#   return p_ij, 1 - p_ij

def cal_uncertainty(y, W):
  """
  Computes uncertaintity matrix epsilon
  Step 6
  Equation 7
  """
  # Returns uncertaintity matrix \xi
  pair_dist = pairwise_distance(y)
  weighted_distance = [W_k * pair_dist for W_k in W]
  normalized_weighted_distance = [wd/np.sum(wd) for wd in weighted_distance]
  return np.sum(normalized_weighted_distance, 0)

def pairwise_distance(y):
  """ Helper function used in cal_uncertainty """
  # Calculate the pairwise exponential distance matrix of instances
  # pairwise_distance_{ij} = \exp(y_j - y_i)
  y = y / (np.max(y) + 1e-12)
  if len(y.shape) == 1:
    y = np.expand_dims(y,1)
  if y.shape[0] == 1:
    y = y.T
  return np.exp(-y).dot(np.exp(y).T)

def cal_weights(xi):
  """
  Computes importance (weight) of each instance, wi
  Step 7
  Equation 8
  """
  return np.sum(xi - xi.T, 1)

def cal_alpha(y, xi):
  """
  Calculates the weight of classifier i, alpha
  Step 12
  """
  y_p = (y>0).astype(float).reshape(-1,1)
  y_n = (y<0).astype(float).reshape(-1,1)
  I_1 = xi * y_p.dot(y_n.T)
  I_2 = xi * y_n.dot(y_p.T)
  return 0.5 * np.log(np.sum(I_1)/np.sum(I_2))

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

def svm_lambdaboost(X_train, y_train, X_test, y_test, W, T=20, SAMPLE_PROP=4, RANDOM_SEED=1):
    """ Generates linear svm lambdaboost classifier """

    print('LINEAR SVM SUBMODELS')

    # Constants
    m = X_train.shape[0]
    n = X_train.shape[1]
    np.random.seed(RANDOM_SEED)
    print('Using %d classifiers, sample proportion of %d, and random seed %d'
          % (T, SAMPLE_PROP, RANDOM_SEED))

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
        weights = cal_weights(epsilon)

        # Step 8: extract labels
        y = np.sign(weights)

        # Step 9: create training (sample) data by sampling based on weights
        p_weights = np.abs(weights)
        p_weights /= np.sum(p_weights)
        sample = np.random.choice(m, size=m*SAMPLE_PROP, replace=True, p=p_weights)
        X_sample = X_train[sample]
        y_sample = y[sample]

        # Step 10: learn binary classifier on training (sample) data
        clf = LinearSVC(max_iter=10000)
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
        if t == 2: print('t\tTrain\t\tTest')
        print('%d\t%.3f\t\t%.3f' %(t - 1,
                                   accuracy_score(y_train, y_train_pred),
                                   accuracy_score(y_test, y_test_pred)))
        if alpha_t < 0: print('Alpha %.3f, terminated' % alpha_t)

    print('Done!')

    # To skip initialized 0 in alpha list
    alpha = alpha[1:]

    # Return final classifier parameters, weight of each submodel
    return f_coefficient, f_intercept, alpha

def tree_lambdaboost(X_train, y_train, X_test, y_test, W, T=20, SAMPLE_PROP=4, RANDOM_SEED=1):
    """ Generates decision tree lambdaboost classifier """

    print('DECISION TREE SUBMODELS')

    # Constants
    m = X_train.shape[0]
    n = X_train.shape[1]
    np.random.seed(RANDOM_SEED)
    print('Using %d classifiers, sample proportion of %d, and random seed %d'
          % (T, SAMPLE_PROP, RANDOM_SEED))

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
        p_weight /= np.sum(p_weight)
        sample = np.random.choice(m, size=m*SAMPLE_PROP, replace=True, p=p_weight)
        X_sample = X_train[sample]
        y_sample = y[sample]

        # Step 10: learn binary classifier on training (sample) data
        clf = DecisionTreeClassifier(max_depth=5)
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
        if t == 2: print('t\tTrain\t\tTest')
        print('%d\t%.3f\t\t%.3f' %(t - 1,
                                   accuracy_score(y_train, y_train_pred),
                                   accuracy_score(y_test, y_test_pred)))
        if alpha_t < 0: print('Alpha %.3f, terminated' % alpha_t)

    print('Done!')

    # Return subtrees and weights for each
    return f, alpha

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
    svm = LinearSVC(max_iter=10000)
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

# def get_data(file_loc):
#   f = open(file_loc, 'r')
#   data = []
#   for line in f:
#     new_arr = []
#     arr = line.split(' #')[0].split()
#     ''' Get the score and query id '''
#     score = arr[0]
#     q_id = arr[1].split(':')[1]
#     new_arr.append(int(score))
#     new_arr.append(int(q_id))
#     arr = arr[2:]
#     ''' Extract each feature from the feature vector '''
#     for el in arr:
#       new_arr.append(float(el.split(':')[1]))
#     data.append(new_arr)
#   f.close()
#   return np.array(data)
#
# def clean_data(source_path, dest_path):
#   source_file = open(source_path, 'r')
#   dest_file = open(dest_path, 'a')
#   for line in source_file:
#     arr = line.split(' #')[0]
#     if arr[0]=='2':
#       continue
#     arr += ' \n'
#     dest_file.write(arr)
#   source_file.close()
#   dest_file.close()
#   return
#
# def plot_svc_decision_function(clf, ax=None):
#   """Plot the decision function for a 2D SVC"""
#   if ax is None:
#     ax = plt.gca()
#   x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
#   y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
#   Y, X = np.meshgrid(y, x)
#   P = np.zeros_like(X)
#   for i, xi in enumerate(x):
#     for j, yj in enumerate(y):
#       P[i, j] = clf.decision_function([xi, yj])
#   # plot the margins
#   ax.contour(X, Y, P, colors='k',
#               levels=[-1, 0, 1], alpha=0.5,
#               linestyles=['--', '-', '--'])
#
#
# def get_rank_svmrank(data_path,
#                      model_path='svm_rank_model.dat',
#                      rank_path='svm_rank_score.dat',
#                      c=3,
#                      remove_files=True,
#                      generate_label=False):
#   #Learn Ranks using SVM-rank
#   run_score_learning = ['./SVMrank/svm_rank_learn','-c', '2', data_path, model_path]
#   run_score_cal = ['./SVMrank/svm_rank_classify', data_path, model_path, rank_path]
#   sp.call(run_score_learning)
#   sp.call(run_score_cal)
#
#   #Load ranks
#   ranks=[]
#   with open(rank_path, 'r') as f:
#       for line in f:
#           ranks.append(float(line))
#   ranks=[np.array(ranks)]
#
#   # Remove generated files
#   if remove_files:
#     os.remove(model_path)
#     os.remove(rank_path)
#
#   # Load Data
#   data = get_data(data_path)
#   x = data[:,2:]
#   if generate_label:
#     f_arb = np.random.rand(x.shape[1],1).astype('float32')
#     y_arb = x.dot(f_arb)
#     label_train = np.sign(y_arb - np.mean(y_arb))
#   else:
#     label_train = (data[:,0] - 0.5)*2
#   return x, label_train, ranks
#
#
# def save_data_rank_format(data_path, data, label, max_qid=2, delete_exist=True):
#   if os.path.exists(data_path):
#     if delete_exist:
#       os.remove(data_path)
#     else:
#       print('Data exists!')
#       return
#   qids = np.sort(np.random.randint(1,max_qid,len(label)))
#   with open(data_path, 'a') as f:
#     for i in range(len(label)):
#         qid = np.random.randint(1,max_qid)
#         row = [str(label[i]),'qid:%1d' % qids[i]] + ['%1d:%05f' % (i+1,val) for i,val in enumerate(data[i])] + ['\n']
#         f.write(' '.join(row))
#   return
