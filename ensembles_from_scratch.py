##############################################################################
#  BAGGING and BOOSTING written from scratch.
#
#  The only thing borrowed from sklearn is a single plain decision tree used
#  as the "weak learner". The ensemble logic -- the bootstrap loop, the
#  majority vote, the AdaBoost weights and alphas -- is all hand-written here
#  so you can see exactly what happens.
#
#  Run it:  python ensembles_from_scratch.py
##############################################################################

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ===========================================================================
# BAGGING (Bootstrap AGGregating), by hand
# ===========================================================================
# The recipe:
#   1. Make N bootstrap samples of the training data (sample WITH replacement).
#   2. Train one tree on each sample.
#   3. To predict, let all N trees vote and take the majority.
class MyBagging:
    def __init__(self, n_estimators=25, random_state=0):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        self.trees = []
        for i in range(self.n_estimators):
            # --- Step 1: draw a bootstrap sample (with replacement) ---------
            idx = rng.randint(0, n, size=n)      # some rows repeat, some dropped
            X_boot, y_boot = X[idx], y[idx]

            # --- Step 2: train one tree on that sample ----------------------
            tree = DecisionTreeClassifier(random_state=self.random_state)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, X):
        # --- Step 3: every tree votes, take the majority --------------------
        # Collect a (n_trees x n_samples) grid of 0/1 votes...
        votes = np.array([tree.predict(X) for tree in self.trees])
        # ...then average down the columns and round: >0.5 means most said 1.
        return (votes.mean(axis=0) >= 0.5).astype(int)


# ===========================================================================
# BOOSTING (AdaBoost), by hand
# ===========================================================================
# The idea: train weak learners one at a time. After each one, give the rows
# it got WRONG more weight, so the next learner focuses on them. At the end,
# each learner votes -- but stronger learners get a louder vote (alpha).
#
# AdaBoost works with labels of -1 and +1 (not 0 and 1), so we convert.
class MyAdaBoost:
    def __init__(self, n_estimators=50, random_state=0):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        y = np.where(y == 0, -1, 1)          # convert {0,1} -> {-1,+1}
        n = len(X)
        w = np.full(n, 1.0 / n)              # every row starts equally important
        self.stumps, self.alphas = [], []

        for t in range(self.n_estimators):
            # --- Train a weak learner (a depth-1 "stump") on weighted rows --
            stump = DecisionTreeClassifier(max_depth=1,
                                           random_state=self.random_state)
            stump.fit(X, y, sample_weight=w)
            pred = stump.predict(X)

            # --- Weighted error: how much weight sits on the wrong rows? ----
            miss = (pred != y)
            err = np.sum(w[miss])
            err = min(max(err, 1e-10), 1 - 1e-10)   # keep out of log(0)

            # --- alpha = this stump's voting power. Low error -> big alpha. -
            alpha = 0.5 * np.log((1 - err) / err)

            # --- Reweight: blow up the wrong rows, shrink the right ones ----
            w = w * np.exp(-alpha * y * pred)
            w = w / np.sum(w)                # renormalize so weights sum to 1

            self.stumps.append(stump)
            self.alphas.append(alpha)
        return self

    def predict(self, X):
        # Weighted vote of every stump, then take the sign (+1 / -1)...
        agg = sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))
        # ...and convert -1/+1 back to 0/1.
        return np.where(np.sign(agg) == -1, 0, 1)


# ===========================================================================
# Try them out
# ===========================================================================
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, random_state=42
)

# Baseline: one plain tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
print(f"Single tree        : {accuracy_score(y_test, tree.predict(X_test)):.4f}")

# Your hand-built bagging
bag = MyBagging(n_estimators=25, random_state=42).fit(X_train, y_train)
print(f"MyBagging (25)     : {accuracy_score(y_test, bag.predict(X_test)):.4f}")

# Your hand-built AdaBoost
boost = MyAdaBoost(n_estimators=50, random_state=42).fit(X_train, y_train)
print(f"MyAdaBoost (50)    : {accuracy_score(y_test, boost.predict(X_test)):.4f}")


##############################################################################
#  TRY THIS:
#   1. In MyAdaBoost, print `err` and `alpha` each round to watch a stump's
#      voting power shrink as the easy rows get used up.
#   2. Set MyBagging(n_estimators=1) -- it becomes a single tree again.
#   3. Give MyAdaBoost stumps max_depth=2 instead of 1 and see what changes.
##############################################################################
