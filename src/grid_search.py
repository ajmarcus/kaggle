#!/usr/bin/python

print __doc__

# Example found here: http://scikit-learn.org/0.12/_downloads/grid_search_text_feature_extraction.py

# Edited by Ariel Marcus for use in Kaggle Competition as a part of STATW4242

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: Simplified BSD


from pprint import pprint
from time import time
import logging

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import SparsePCA
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


###############################################################################
# Load some categories from the training set
#categories = ['1','2','3']

# Uncomment the following to do the analysis on all the categories
categories = None

print "Loading essay data:"
print categories

data = load_files('../data/container', categories=categories, shuffle=True)

#train_size = int(len(data.data)*0.8)
#test_size = train_size + 1

#data_train = Bunch(data=data.data[0:train_size],
#                   filenames=data.filenames[0:train_size],
#                   target_names=data.target_names,
#                   target=data.target[0:train_size],
#                   DESCR=data.DESCR)

#data_test = Bunch(data=data.data[test_size:len(data.data)],
#                   filenames=data.filenames[test_size:len(data.data)],
#                   target_names=data.target_names,
#                   target=data.target[test_size:len(data.data)],
#                   DESCR=data.DESCR)

print 'data loaded'

categories = data.target_names    # for case categories == None


print "%d documents" % len(data.filenames)
print "%d categories" % len(data.target_names)
print

###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
#    ('pca', SparsePCA()),
    ('clf', SGDClassifier())
])

parameters = {
# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
    'vect__analyzer': ('word', 'char_wb'),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__max_n': (1, 2),  # words or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    'clf__n_iter': (10, 50, 80)
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=1)

    print "Performing grid search..."
    print "pipeline:", [name for name, _ in pipeline.steps]
    print "parameters:"
    pprint(parameters)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print "done in %0.3fs" % (time() - t0)
    print

    print "Best score: %0.3f" % grid_search.best_score_
    print "Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print "\t%s: %r" % (param_name, best_parameters[param_name])
