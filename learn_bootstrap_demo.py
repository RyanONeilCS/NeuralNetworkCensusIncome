##############################################################################
#  A tiny hands-on demo of BOOTSTRAPPING, BAGGING, and BOOSTING.
#
#  Goal: understand the ideas, not chase accuracy. It runs in a couple of
#  seconds and needs no internet (uses a small built-in dataset).
#
#  Run it:   python learn_bootstrap_demo.py
#  Then scroll to the "TRY THIS" notes at the bottom and change things.
##############################################################################

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# PART 1: What a bootstrap sample actually is
# ---------------------------------------------------------------------------
# Imagine a training set of just 8 rows. A "bootstrap sample" draws 8 rows
# back out WITH REPLACEMENT -- so some rows get picked more than once and
# some don't get picked at all. Let's watch it happen.

rows = ["apple", "banana", "cherry", "date",
        "elderberry", "fig", "grape", "honeydew"]

print("=" * 60)
print("PART 1: one bootstrap sample of 8 rows")
print("=" * 60)
print("Original rows:", rows)

rng = np.random.RandomState(0)          # change the 0 to get a different draw
picked_positions = rng.randint(0, len(rows), size=len(rows))
picked = [rows[i] for i in picked_positions]

print("\nBootstrap sample (drawn with replacement):")
print("  ", picked)

# Which rows got left out? Those are the "out-of-bag" (OOB) rows -- free
# leftover data you never trained on.
left_out = [r for r in rows if r not in picked]
print("\nLeft out (out-of-bag) this time:", left_out)


# ---------------------------------------------------------------------------
# The famous ~63.2% fact, measured instead of just quoted
# ---------------------------------------------------------------------------
# On average, a bootstrap sample contains about 63.2% of the UNIQUE original
# rows. Let's prove it by doing it 5000 times and averaging.

n = 1000
trials = 5000
rng = np.random.RandomState(1)
unique_fractions = []
for _ in range(trials):
    sample = rng.randint(0, n, size=n)
    unique_fractions.append(len(np.unique(sample)) / n)

print(f"\nOver {trials} bootstrap samples of {n} rows:")
print(f"  average unique rows = {np.mean(unique_fractions):.1%}  (theory: 63.2%)")


# ---------------------------------------------------------------------------
# PART 2: Why we bother -- bagging and boosting on a real (small) dataset
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2: single model vs bagging vs boosting")
print("=" * 60)

data = load_breast_cancer()             # 569 rows, predicts benign vs malignant
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, random_state=42
)

# One plain decision tree (our baseline)
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_acc = accuracy_score(y_test, tree.predict(X_test))

# BAGGING = bootstrapping + averaging.
# 25 trees, each trained on its OWN bootstrap sample, then they vote.
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=25,
    bootstrap=True,     # <-- this is the bootstrapping switch
    oob_score=True,     # <-- score using the ~37% left-out rows, for free
    random_state=42,
)
bag.fit(X_train, y_train)
bag_acc = accuracy_score(y_test, bag.predict(X_test))

# BOOSTING = trees built one after another, each fixing the last one's mistakes
boost = GradientBoostingClassifier(random_state=42)
boost.fit(X_train, y_train)
boost_acc = accuracy_score(y_test, boost.predict(X_test))

print(f"Single tree      : {tree_acc:.4f}")
print(f"Bagging (25)     : {bag_acc:.4f}   (out-of-bag estimate: {bag.oob_score_:.4f})")
print(f"Boosting         : {boost_acc:.4f}")


##############################################################################
#  TRY THIS (change one thing, rerun, see what happens):
#
#  1. In Part 1, change RandomState(0) to RandomState(3). Different rows get
#     picked and left out. Bootstrapping is random every time.
#
#  2. In Part 2, drop n_estimators from 25 down to 1. Bagging with a single
#     tree is basically just one tree again -- the magic is in the averaging.
#
#  3. Compare "Bagging" accuracy to its "out-of-bag estimate". They're close!
#     That means you can judge a bagged model without a separate test set.
#
#  4. Swap load_breast_cancer() for load_wine() or load_digits() (both from
#     sklearn.datasets) and see whether bagging/boosting still help.
##############################################################################
