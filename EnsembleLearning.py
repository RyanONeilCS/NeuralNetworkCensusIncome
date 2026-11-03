#####################################################################################################################
#   PERSONAL LEARNING FILE (not an assignment)
#   Trees and ensembles, using the same Adult / Census Income dataset as the HW2 neural net.
#
#   The goal of this file is to LEARN, so it is heavily commented. Each method builds on
#   the last:
#       1. one Decision Tree            (the building block)
#       2. Bagging  -> Random Forest    (many independent trees that VOTE)
#       3. Boosting -> AdaBoost         (trees in sequence, each fixing the last's mistakes)
#       4. Boosting -> Gradient Boost   (the usual strongest one)
#   At the end it prints a comparison table so you can see them side by side.
#
#   Run it with:  python EnsembleLearning.py
#####################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# The four models we are learning. They all live in sklearn and share the same
# .fit() / .predict() interface, which is a big part of why sklearn is nice.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,      # bagging of trees + random feature subsets
    AdaBoostClassifier,          # boosting (re-weights the hard examples)
    GradientBoostingClassifier,  # boosting (each tree fits the previous errors)
    BaggingClassifier,           # lets us turn bootstrapping ON/OFF to compare
)
# The neural net from the HW, so we can put its best models on the same chart.
from sklearn.neural_network import MLPClassifier


class TreeLearner:
    def __init__(self, dataset_id=2):
        # Same data source as the HW. Pulled from UCI at runtime, no local file.
        self.raw_input = fetch_ucirepo(id=dataset_id)

    # -----------------------------------------------------------------------------
    #  PRE-PROCESSING
    #  This is basically copied from your HW2 preprocess(), with ONE important
    #  difference: there is NO StandardScaler here.
    #
    #  WHY NO SCALING?
    #  Neural nets multiply inputs by weights, so a huge feature (capital-gain in
    #  the tens of thousands) dominates the gradient -> you must scale.
    #  Trees do NOT multiply anything. A tree just asks "is capital-gain > 5000?".
    #  That yes/no split works exactly the same whether the number is 5000 or 5.0,
    #  so the scale of a feature does not matter to a tree at all. This is one of
    #  the nicest practical differences between trees and neural nets.
    # -----------------------------------------------------------------------------
    def preprocess(self):
        X = self.raw_input.data.features
        y = self.raw_input.data.targets

        # 1. strip the trailing period on the test-file labels (">50K." -> ">50K")
        y = y['income'].str.strip('.')

        # 2. glue X and y together so dropping bad rows keeps them aligned
        data = pd.concat([X, y], axis=1)

        # 3. the missing values are literal "?" strings -> make them real NaN, then drop
        data = data.replace('?', np.nan)
        data = data.dropna()

        # 4. split back apart
        X = data.drop(columns='income')
        y = data['income']

        # 5. one-hot encode the text columns into numbers
        X = pd.get_dummies(X)

        # 6. encode the target 0/1
        y = y.map({'<=50K': 0, '>50K': 1})

        self.X = X
        self.y = y
        print(f"Data ready: {X.shape[0]} rows, {X.shape[1]} feature columns")

    # -----------------------------------------------------------------------------
    #  A tiny helper so every model is trained and scored the exact same way.
    #  It takes a model, fits it, and returns train + test accuracy.
    #  Doing this in one place means the comparison at the bottom is fair.
    # -----------------------------------------------------------------------------
    def _train_and_score(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        return train_acc, test_acc

    # -----------------------------------------------------------------------------
    #  BOOTSTRAP EXPERIMENT  --  does resampling actually help?
    #
    #  We build the SAME 50-tree ensemble two ways, changing only one thing:
    #     bootstrap=True   -> each tree trains on a different resample (real bagging)
    #     bootstrap=False  -> every tree trains on the identical full dataset
    #
    #  To make it a FAIR test of bootstrapping alone, we force every tree to use ALL
    #  features (max_features=1.0). That removes the "random feature" source of
    #  diversity, so bootstrapping becomes the ONLY thing that can make the trees
    #  differ. Prediction:
    #     - bootstrap=False -> all trees come out identical -> the ensemble is no
    #       better than a single tree (averaging identical things does nothing)
    #     - bootstrap=True  -> the trees differ -> their errors cancel -> better test
    # -----------------------------------------------------------------------------
    def bootstrap_experiment(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=42
        )

        # one plain tree, for reference
        base_tree = DecisionTreeClassifier(random_state=42)
        tr0, te0 = self._train_and_score(base_tree, X_train, X_test, y_train, y_test)

        results = [("1 single Decision Tree", tr0, te0)]

        for use_bootstrap in [False, True]:
            ensemble = BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42),
                n_estimators=50,
                bootstrap=use_bootstrap,   # <-- the ONLY thing we change
                max_features=1.0,          # every tree sees all features (fair test)
                random_state=42,
                n_jobs=-1,
            )
            tr, te = self._train_and_score(ensemble, X_train, X_test, y_train, y_test)
            label = f"50 trees, bootstrap={use_bootstrap}"
            results.append((label, tr, te))

        print("\n" + "=" * 62)
        print("BOOTSTRAP EXPERIMENT (only 'bootstrap' changes between rows)")
        print("=" * 62)
        print(f"{'Setup':<34}{'Train Acc':>12}{'Test Acc':>12}")
        print("-" * 62)
        for name, tr, te in results:
            print(f"{name:<34}{tr:>12.4f}{te:>12.4f}")
        print("=" * 62)
        print("Read it like this:")
        print("  bootstrap=False -> trees are identical -> same as one tree")
        print("  bootstrap=True  -> trees differ -> errors cancel -> better test acc")

    def run(self):
        # Same split style as the HW. Trees don't need scaling, so this is all
        # the setup we need.
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=42
        )

        results = []

        # =============================================================================
        #  1. ONE DECISION TREE  --  the building block everything else is made of
        #
        #  A decision tree learns a flowchart of yes/no questions ("is age > 30?",
        #  "is education Bachelors?") that split the people into smaller and smaller
        #  groups until each group is mostly one class.
        #
        #  KEY KNOB: max_depth = how many questions deep the flowchart can go.
        #    - deep tree  -> can carve out tiny groups -> memorizes -> OVERFITS
        #      (you'll see this as a big gap between train_acc and test_acc)
        #    - shallow tree -> only a few questions -> might UNDERFIT
        #  This is the same overfitting idea you saw with 100 neurons in the HW,
        #  just controlled by depth instead of neuron count.
        # =============================================================================
        # Try changing max_depth to None (unlimited) and watch train_acc shoot to
        # ~1.0 while test_acc drops -- that is a single tree overfitting.
        tree = DecisionTreeClassifier(max_depth=8, random_state=42)
        tr, te = self._train_and_score(tree, X_train, X_test, y_train, y_test)
        results.append(("Decision Tree (depth 8)", tr, te))

        # =============================================================================
        #  2. BAGGING -> RANDOM FOREST  --  many trees that VOTE
        #
        #  "Bagging" = Bootstrap AGGregating:
        #    - make N new training sets by sampling the rows WITH replacement
        #      (each tree sees a slightly different, randomly resampled dataset)
        #    - train one tree on each
        #    - to predict, let all N trees vote and take the majority
        #
        #  A Random Forest is bagging PLUS one extra trick: at each split, each tree
        #  is only allowed to look at a random SUBSET of the features. That keeps the
        #  trees from all looking alike, so their mistakes are less correlated.
        #
        #  WHY IT HELPS: one deep tree is high-variance (its answer changes a lot if
        #  you nudge the data). Averaging many independent noisy trees cancels out
        #  that noise -> less overfitting, without making the trees themselves shallow.
        #
        #  KEY KNOB: n_estimators = how many trees are in the forest (more = more
        #  stable, but slower). n_jobs=-1 just uses all CPU cores.
        # =============================================================================
        forest = RandomForestClassifier(
            n_estimators=100,   # 100 trees voting
            random_state=42,
            n_jobs=-1,
        )
        tr, te = self._train_and_score(forest, X_train, X_test, y_train, y_test)
        results.append(("Random Forest (100 trees, bagging)", tr, te))

        # =============================================================================
        #  3. BOOSTING -> ADABOOST  --  trees in SEQUENCE, each fixing the last's misses
        #
        #  Boosting is the opposite philosophy from bagging. Instead of many
        #  independent trees trained in parallel, boosting trains them ONE AT A TIME:
        #    - train a weak tree
        #    - look at which rows it got WRONG, and increase their weight
        #    - train the next tree, which now focuses harder on those hard rows
        #    - repeat, then combine all the little trees into one strong prediction
        #
        #  The trees here are deliberately WEAK (shallow "stumps"), because boosting
        #  gets its power from combining many weak learners, not from each one being
        #  strong. Bagging wanted strong independent trees; boosting wants weak
        #  sequential ones. That contrast is the whole point.
        # =============================================================================
        ada = AdaBoostClassifier(
            n_estimators=100,       # 100 weak trees, added one after another
            random_state=42,
        )
        tr, te = self._train_and_score(ada, X_train, X_test, y_train, y_test)
        results.append(("AdaBoost (100 stumps, boosting)", tr, te))

        # =============================================================================
        #  4. BOOSTING -> GRADIENT BOOSTING  --  usually the strongest of these
        #
        #  Same sequential idea as AdaBoost, but instead of re-weighting the hard
        #  rows, each new tree is trained to predict the RESIDUAL ERRORS (how far off
        #  the current combined model still is). In other words each tree nudges the
        #  prediction in the direction that reduces the loss -- this is literally
        #  gradient descent (the thing you derived in HW1/HW2), but each "step" is a
        #  whole little tree instead of a weight update. That is why it's called
        #  GRADIENT boosting.
        #
        #  KEY KNOBS:
        #    n_estimators  = number of trees / boosting rounds
        #    learning_rate = how big a step each tree takes (SAME tradeoff you saw
        #                    with the neural net: too big overshoots, too small is
        #                    slow -- small learning_rate + more trees is the classic
        #                    strong combo)
        #    max_depth     = keep each tree shallow (3 is typical for boosting)
        # =============================================================================
        gboost = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        tr, te = self._train_and_score(gboost, X_train, X_test, y_train, y_test)
        results.append(("Gradient Boosting (boosting)", tr, te))

        # =============================================================================
        #  5. THE NEURAL NET FROM THE HW  --  best model per activation
        #
        #  We put the neural net on the same chart so you can see how trees stack up
        #  against it. Two important details:
        #    - Neural nets DO need scaled features (unlike trees), so we make a scaled
        #      copy of the SAME split here. Fit the scaler on train only (no leakage).
        #    - For each activation we try a few configs and keep the BEST test score,
        #      so the neural net gets a fair shot. We already know from the HW that
        #      lr=0.01 and 2 layers tend to win, so we search around there.
        # =============================================================================
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)   # scaled -- for the neural nets only
        X_test_s = scaler.transform(X_test)

        for activation in ['logistic', 'tanh', 'relu']:
            best_te, best_tr = -1, -1
            # small search: 2 learning rates x 2 layer counts, 200 epochs
            for lr in [0.01, 0.1]:
                for num_layers in [2, 3]:
                    nn = MLPClassifier(
                        activation=activation,
                        learning_rate_init=lr,
                        max_iter=200,
                        hidden_layer_sizes=tuple([15] * num_layers),
                        random_state=42,
                    )
                    tr, te = self._train_and_score(nn, X_train_s, X_test_s, y_train, y_test)
                    if te > best_te:      # keep this activation's best config
                        best_te, best_tr = te, tr
            results.append((f"Neural Net ({activation})", best_tr, best_te))

        # =============================================================================
        #  COMPARISON  --  print the table and draw a bar chart of test accuracy
        #  Watch two things:
        #    (a) test accuracy: where do the trees land vs the neural nets?
        #    (b) the gap between train and test: a single deep tree / a random forest
        #        overfit the most; gradient boosting keeps the gap small.
        # =============================================================================
        print("\n" + "=" * 70)
        print(f"{'Model':<40}{'Train Acc':>12}{'Test Acc':>12}")
        print("-" * 70)
        for name, tr, te in results:
            print(f"{name:<40}{tr:>12.4f}{te:>12.4f}")
        print("=" * 70)
        # Baseline reminder: ~75% of people are <=50K, so anything near 0.75 has
        # basically not learned. Everything here should be comfortably above that.

        # ---- bar chart of TEST accuracy, coloured by model family ----
        # Colour each bar by which family it belongs to, using the keyword in its name.
        def family_colour(name):
            if "Neural Net" in name:      return "#2a78d6"   # blue  = neural nets
            if "Forest" in name:          return "#1baf7a"   # green = bagging
            if "Boost" in name:           return "#eda100"   # amber = boosting
            return "#8a8a8a"                                 # grey  = single tree

        names = [r[0] for r in results]
        test_accs = [r[2] for r in results]
        colours = [family_colour(n) for n in names]

        plt.figure(figsize=(11, 6))
        bars = plt.bar(names, test_accs, color=colours)
        plt.axhline(0.752, color="red", linestyle="--", linewidth=1,
                    label="baseline (guess <=50K) = 0.752")
        plt.ylim(0.70, 0.90)                 # zoom in so the differences are visible
        plt.ylabel("Test Accuracy")
        plt.title("Test accuracy: trees & ensembles vs the neural net")
        plt.xticks(rotation=30, ha="right")
        # print the exact number on top of each bar
        for bar, acc in zip(bars, test_accs):
            plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.002,
                     f"{acc:.3f}", ha="center", fontsize=8)
        plt.legend()
        plt.tight_layout()
        plt.savefig("model_comparison.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    learner = TreeLearner(dataset_id=2)
    learner.preprocess()
    learner.run()
