#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = fetch_ucirepo(id=2)
        
        
    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        #Gets the raw data and splits it into X and y to check the inatial data
        X = self.raw_input.data.features
        y = self.raw_input.data.targets
        
        print("Start of X", X.head()) #First 5 rows of x
        print("Start of y", y.head()) # First 5 rows of y
        print("Amount of Rows and Columns", X.shape) #Checks how many rows and columns
        print("How many nulls", X.isnull().sum()) #Checks for any missing values so I can filter them out
        print("Print Out all Y values:", y.value_counts())
        print("Print Out all X values:", X.value_counts())
        
        
        #Strip the trailing period from the target that some of the y's had
        y = y['income'].str.strip('.')

        #Merge X and y together so whenever we drop rows with nulls, we drop the corresponding target value as well
        data = pd.concat([X, y], axis=1)

        print("Re Print Out all Y values:", y.value_counts())
        
        #Turn the leftover literal '?' strings into real nulls
        data = data.replace('?', np.nan)

        #Drop the rows with nulls
        data = data.dropna()

        #Split the data back apart
        X = data.drop(columns='income')
        y = data['income']

        #Encode features to make them numeric
        X = pd.get_dummies(X)

        print("RE Print Out all X values:", X.shape)
        
        
        #Encode target 0/1
        y = y.map({'<=50K': 0, '>50K': 1})

        print("Re Print Out all Y values:", y.value_counts())

        #Reassemble income
        self.processed_data = pd.concat([X, y], axis=1)

        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)    #figure out the mean and std, then scale
        X_test  = scaler.transform(X_test)         #use the mean and std you already learned


        # Below are the hyperparameters that you need to use for model evaluation
        # You can assume any fixed number of neurons for each hidden layer. 
        
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        histories = []  # List to store the history of each model
        results = []     # List to store the results of each model

        # Plot encoding: colour = activation, line style = learning rate,
        # opacity = hidden layer count. Colours are distinguishable under
        # red-green colour-vision deficiency (red/green would not be).
        COLORS = {'logistic': '#2a78d6', 'tanh': '#eda100', 'relu': '#1baf7a'}
        STYLES = {0.01: '-', 0.1: '--'}
        ALPHAS = {2: 1.0, 3: 0.45}

        # Create the neural network and be sure to keep track of the performance
        #   metrics
        
        for activation in activations:
            for lr in learning_rate:
                for max_iter in max_iterations:
                    for num_layers in num_hidden_layers:
                        # Create the model with the specified hyperparameters
                        hidden_layer_sizes = tuple([15] * num_layers)  # 15 neurons per layer
                        model = MLPClassifier(
                            activation=activation,
                            learning_rate_init=lr,
                            max_iter=max_iter,
                            hidden_layer_sizes=hidden_layer_sizes,
                            random_state=42
                        )
                            
                        # Fit the model to the training data
                        model.fit(X_train, y_train)
                            
                        # Make predictions on the training and test sets
                        train_pred = model.predict(X_train)
                        test_pred = model.predict(X_test)
                        
                        train_mse = mean_squared_error(y_train, train_pred)
                        test_mse = mean_squared_error(y_test, test_pred)
                        
                        # Calculate accuracy scores
                        train_acc = accuracy_score(y_train, train_pred)
                        test_acc = accuracy_score(y_test, test_pred)
                        
                        results.append({
                            'activation': activation,
                            'learning_rate': lr,
                            'epochs': max_iter,
                            'hidden_layers': num_layers,
                            'train_acc': train_acc,
                            'test_acc': test_acc,
                            'train_mse': train_mse,
                            'test_mse': test_mse,
                            'epochs_ran': len(model.loss_curve_),
                        })

                        histories.append({
                            'label': f"{activation} | lr={lr} | {max_iter}ep | {num_layers}L",
                            'curve': model.loss_curve_,
                            'color': COLORS[activation],
                            'style': STYLES[lr],
                            'alpha': ALPHAS[num_layers],
                        })
                        
                        # Print out the results
                        print(f"Activation: {activation}, Learning Rate: {lr}, Max Iterations: {max_iter}, Hidden Layers: {num_layers}")
                        print(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")
                        print(f"Epochs Ran: {len(model.loss_curve_)}\n")

        results_df = pd.DataFrame(results)
        print(results_df.sort_values('test_acc', ascending=False).to_string())
        results_df.to_csv('results.csv', index=False)

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.
        
        plt.figure(figsize=(14, 8))
        for h in histories:
            plt.plot(range(len(h['curve'])), h['curve'],
                     label=h['label'],
                     color=h['color'],
                     linestyle=h['style'],
                     alpha=h['alpha'],
                     linewidth=1.8)

        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Model History - colour = activation, solid = lr 0.01, dashed = lr 0.1')
        plt.legend(fontsize=7, ncol=2)
        plt.grid(alpha=0.2)
        plt.savefig('history_all.png', dpi=150, bbox_inches='tight')
        plt.show()

        return 0

    # ------------------------------------------------------------------
    # EXTRA EXPLORATION (not required by the assignment):
    # Compare one plain neural net against two ensemble ideas on the
    # SAME train/test split so the accuracy numbers are comparable.
    #   1. Bagging  - many neural nets, each trained on its own bootstrap
    #                 resample of the training rows (this is bootstrapping)
    #   2. Boosting - shallow trees built in sequence, each one focusing
    #                 on the rows the previous trees got wrong
    # ------------------------------------------------------------------
    def bag_and_boost(self):
        from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier

        # Same split + scaling as train_evaluate so results line up
        ncols = len(self.processed_data.columns)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols - 1)]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)   # learn mean/std on train only
        X_test = scaler.transform(X_test)         # reuse them on test

        # --- Prove the "63.2%" bootstrap fact on our own data ----------
        # A bootstrap sample draws len(X_train) rows WITH replacement.
        # On average only ~63.2% of the original rows appear at least once
        # (the rest are left out and become the "out-of-bag" rows).
        n = len(X_train)
        boot_idx = np.random.RandomState(42).randint(0, n, size=n)
        unique_frac = len(np.unique(boot_idx)) / n
        print(f"\nBootstrap sample: {unique_frac:.1%} of the original rows are unique "
              f"(theory says ~63.2%)\n")

        # --- Baseline: one plain neural network ------------------------
        base_mlp = MLPClassifier(
            activation='relu', learning_rate_init=0.01,
            max_iter=200, hidden_layer_sizes=(15, 15), random_state=42
        )
        base_mlp.fit(X_train, y_train)
        base_acc = accuracy_score(y_test, base_mlp.predict(X_test))

        # --- Bagging: 10 neural nets, each on its own bootstrap sample --
        bag = BaggingClassifier(
            estimator=MLPClassifier(
                activation='relu', learning_rate_init=0.01,
                max_iter=200, hidden_layer_sizes=(15, 15), random_state=42
            ),
            n_estimators=10, bootstrap=True, random_state=42, n_jobs=-1
        )
        bag.fit(X_train, y_train)
        bag_acc = accuracy_score(y_test, bag.predict(X_test))

        # --- Boosting: trees built in sequence -------------------------
        # Note: boosting is normally used with shallow trees, not neural
        # nets, so this model is tree-based on purpose.
        boost = HistGradientBoostingClassifier(random_state=42)
        boost.fit(X_train, y_train)
        boost_acc = accuracy_score(y_test, boost.predict(X_test))

        print("Test accuracy comparison")
        print(f"  Single neural net : {base_acc:.4f}")
        print(f"  Bagging (10 nets) : {bag_acc:.4f}")
        print(f"  Boosting (trees)  : {boost_acc:.4f}")
        return 0


if __name__ == "__main__":
    neural_network = NeuralNet("train.csv") # put in path to your file
    neural_network.preprocess()
    # neural_network.train_evaluate()   # original 24-model sweep (slow)
    neural_network.bag_and_boost()      # extra: bootstrapping + boosting


    
