# Example found here: http://scikit-learn.org/0.11/_downloads/document_classification_20newsgroups2.py

# Edited by Ariel Marcus for use in Kaggle Competition as a part of STATW4242

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: Simplified BSD

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import pylab as pl

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.datasets.base import Bunch


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print __doc__
op.print_help()
print


###############################################################################
# Load some categories from the training set
if opts.all_categories:
    categories = ['1','2','3','4','5','6','7','8','9','10','11','12']
else:
    categories = ['1','2','3']

print "Loading 20 newsgroups dataset for categories:"
print categories

data = load_files('../data/container', categories=categories,
                               shuffle=True, random_state=42)

train_size = int(len(data.data)*0.8)
test_size = train_size + 1

data_train = Bunch(data=data.data[0:train_size],
                   filenames=data.filenames[0:train_size],
                   target_names=data.target_names,
                   target=data.target[0:train_size],
                   DESCR=data.DESCR)

data_test = Bunch(data=data.data[test_size:len(data.data)],
                   filenames=data.filenames[test_size:len(data.data)],
                   target_names=data.target_names,
                   target=data.target[test_size:len(data.data)],
                   DESCR=data.DESCR)

print 'data loaded'

categories = data_train.target_names    # for case categories == None

print "%d documents (training set)" % len(data_train.data)
print "%d documents (testing set)" % len(data_test.data)
print "%d categories" % len(categories)
print

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % X_train.shape
print

print "Extracting features from the test dataset using the same vectorizer"
t0 = time()
X_test = vectorizer.transform(data_test.data)
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % X_test.shape
print

if opts.select_chi2:
    print ("Extracting %d best features by a chi-squared test" %
           opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    print "done in %fs" % (time() - t0)
    print


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# mapping from integer feature name to original token string
feature_names = vectorizer.get_feature_names()


###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print 80 * '_'
    print "Training: "
    print clf
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print "train time: %0.3fs" % train_time

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print "test time:  %0.3fs" % test_time

    score = metrics.f1_score(y_test, pred)
    print "f1-score:   %0.3f" % score

    if hasattr(clf, 'coef_'):
        print "dimensionality: %d" % clf.coef_.shape[1]
        print "density: %f" % density(clf.coef_)

        if opts.print_top10:
            print "top 10 keywords per class:"
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print trim("%s: %s" % (
                    category, " ".join(feature_names[top10])))
        print

    if opts.print_report:
        print "classification report:"
        print metrics.classification_report(y_test, pred,
                                            target_names=categories)

    if opts.print_cm:
        print "confusion matrix:"
        print metrics.confusion_matrix(y_test, pred)

    print
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in ((Perceptron(n_iter=50), "Perceptron"),
                  (KNeighborsClassifier(n_neighbors=10), "kNN")):
    print 80 * '='
    print name
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print 80 * '='
    print "%s penalty" % penalty.upper()
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                          penalty=penalty)))

# Train SGD with Elastic Net penalty
print 80 * '='
print "Elastic-Net penalty"
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                      penalty="elasticnet")))

# Train NearestCentroid without threshold
print 80 * '='
print "NearestCentroid (aka Rocchio classifier)"
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print 80 * '='
print "Naive Bayes"
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))


class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

print 80 * '='
print "LinearSVC with L1-based feature selection"
results.append(benchmark(L1LinearSVC()))


# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in xrange(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)


print "Test Time: %s" % test_time
print "Training Time: %s" % training_time
print "CLF Names: %s" % clf_names
print "Score: %s" % score

#pl.title("Score")
#pl.barh(indices, score, .2, label="score", color='r')
#pl.barh(indices + .3, training_time, .2, label="training time", color='g')
#pl.barh(indices + .6, test_time, .2, label="test time", color='b')
#pl.yticks(())
#pl.legend(loc='best')
#pl.subplots_adjust(left=.25)

#for i, c in  zip(indices, clf_names):
#    pl.text(-.3, i, c)

#pl.show()
