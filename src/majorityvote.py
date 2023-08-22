from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """A majority vote ensemble classifier
    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
    Different classifiers for the ensemble
    vote : str, {'classlabel', 'probability'}
    Default: 'classlabel'
    If 'classlabel' the prediction is based on
    the argmax of class labels. Else if
    'probability', the argmax of the sum of
    probabilities is used to predict the class label
    (recommended for calibrated classifiers).
    weights : array-like, shape = [n_classifiers]
    Optional, default: None
    If a list of 'int' or 'float' values are
    provided, the classifiers are weighted by
    importance; Uses uniform weights if 'weights=None'.
    """

    def __init__(self, classifiers, prob_cutoff=0.5, vote="classlabel", weights=None):
        self.classifiers = classifiers
        self.vote = vote
        self.weights = weights
        self.prob_cutoff = prob_cutoff

    def fit(self, X, y):
        """Fit classifiers.
        Parameters
        ----------
        X : {array-like, sparse matrix},
        shape = [n_examples, n_features]
        Matrix of training examples.
        y : array-like, shape = [n_examples]
        Vector of target class labels.
        Returns
        -------
        self : object
        """
        if self.vote not in ("probability", "classlabel"):
            raise ValueError(
                "vote must be 'probability'"
                "or 'classlabel'; got (vote=%r)" % self.vote
            )
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(
                "Number of classifiers and weights"
                "must be equal; got %d weights,"
                "%d classifiers" % (len(self.weights), len(self.classifiers))
            )
        self.classes_ = np.unique(y)
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, y)
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix},
        Shape = [n_examples, n_features]
        Matrix of training examples.
        Returns
        ----------
        maj_vote : array-like, shape = [n_examples]
        Predicted class labels.
        """
        predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
        maj_vote = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, weights=self.weights)),
            axis=1,
            arr=predictions,
        )
        return maj_vote


def build_majorityvote() -> MajorityVoteClassifier:
    clf1 = LogisticRegression(penalty="l2", C=0.001, solver="lbfgs", random_state=42)
    clf2 = DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=42)
    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric="minkowski")
    clfs = [clf1, clf2, clf3]
    majority_vote = MajorityVoteClassifier(clfs)
    return majority_vote
