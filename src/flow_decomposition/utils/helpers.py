import numpy as np
from sklearn.model_selection import KFold
import copy
from FastCCM import PairwiseCCM
from scipy.stats import pearsonr

class ModelCV:
    def __init__(self, model, n_splits=5, shuffle=False, random_state=None):
        """
        Cross-validation class that works with any model that has
        a `fit` and `evaluate_loss` method.

        Args:
            model: An instance of your model (e.g. FlowRegression or FlowDecomposition).
            n_splits (int): Number of folds.
            shuffle (bool): Whether to shuffle the data before splitting.
            random_state (int): Seed for reproducibility.
        """
        self.model = model
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.fold_models = []  # Store a clone of the model from each fold
        self.scores = []       # Store the evaluation scores (loss) from each fold

    def cross_validate(self, X, fit_params: dict, eval_params: dict = None, y=None):
        """
        Performs k-fold cross-validation.

        Args:
            X (array-like): Feature data.
            y (array-like, optional): Target data. If not provided, will default to None.
            fit_params (dict): Keyword arguments to pass to the model's `fit()` method.
            eval_params (dict, optional): Keyword arguments to pass to the model's `evaluate_loss()` method.
                                          If None, `fit_params` will be used.
        
        Returns:
            scores (list): A list of evaluation loss scores for each fold.
        """
        if eval_params is None:
            eval_params = fit_params

        supervised = y is not None

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        fold = 1

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            if supervised:
                Y_train, Y_val = y[train_idx], y[val_idx]

            # Create a fresh clone of the model for this fold.
            model_clone = copy.deepcopy(self.model)

            # Train the model on the training data.
            if supervised:
                model_clone.fit(X_train, Y_train, **fit_params)
            else:
                model_clone.fit(X_train, **fit_params)

            # Evaluate the model on the validation set.
            if supervised:
                val_loss = model_clone.evaluate_loss(X_val, Y_val, **eval_params)
            else:
                val_loss = model_clone.evaluate_loss(X_val, **eval_params)

            print(f"Fold {fold} validation loss: {val_loss:.4f}")
            self.fold_models.append(model_clone)
            self.scores.append(val_loss)
            fold += 1

        return self.scores
    
    def cross_validate_manual(self, X, fit_params: dict, eval_params: dict = None, y=None, scorer=lambda a,b: np.mean([pearsonr(a,b)[0] for i in range(a.shape[1])])):
        """
        Performs k-fold cross-validation, using FastCCM to evalute the model.

        Args:
            X (array-like): Feature data.
            y (array-like, optional): Target data. If not provided, will default to None.
            fit_params (dict): Keyword arguments to pass to the model's `fit()` method.
            eval_params (dict, optional): Keyword arguments to pass to the model's `evaluate_loss()` method.
                                          If None, `fit_params` will be used.
        
        Returns:
            scores (list): A list of CCM rho values for each fold.
        """
        if eval_params is None:
            eval_params = fit_params

        supervised = y is not None

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        fold = 1

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            if supervised:
                Y_train, Y_val = y[train_idx], y[val_idx]

            # Create a fresh clone of the model for this fold.
            model_clone = copy.deepcopy(self.model)

            # Train the model on the training data.
            if supervised:
                model_clone.fit(X_train, Y_train, **fit_params)
            else:
                model_clone.fit(X_train, **fit_params)

            # Evaluate the model on the validation set.
            if supervised:
                X_train_z = model_clone.transform(X_train)
                X_val_z = model_clone.transform(X_val)

                if eval_params["optim_policy"] == "range":
                    tp_range = np.linspace(0, eval_params["time_intv"] + (1 - 1e-5), eval_params["num_rand_samples"]).astype(int)
                elif eval_params["optim_policy"] == "fixed":
                    tp_range = np.linspace(eval_params["time_intv"], eval_params["time_intv"] + (1 - 1e-5), eval_params["num_rand_samples"]).astype(int)
                else:
                    raise ValueError("Invalid optimization policy")
                val_loss = []

                for tp in tp_range:
                    rand_i = np.random.permutation(X_val_z.shape[0])[:eval_params["sample_size"]]
                    X_val_z = X_val_z[rand_i]
                    Y_val = Y_val[rand_i]

                    Y_pred = np.array([
                            PairwiseCCM("cpu").predict(X_train_z[None], 
                                                        Y_train[None],
                                                        X_val_z[None],
                                                            subset_size=eval_params["library_size"],
                                                            exclusion_rad=eval_params["exclusion_rad"],tp=tp,
                                                            method="simplex" if eval_params["method"] == "knn" else "smap",
                                                            theta=eval_params["theta"], nrst_num = eval_params["nbrs_num"])[:,:,0,0]
                            for i in range(5)
                        ]).mean(axis=0)
                    
                    val_loss += [scorer(Y_val[-Y_pred.shape[0]:, :], Y_pred[:, :])]

                val_loss = np.mean(val_loss)
            else:
                #TODO: implement manual evaluation for unsupervised models
                pass

            print(f"Fold {fold} validation loss: {val_loss:.4f}")
            self.fold_models.append(model_clone)
            self.scores.append(val_loss)
            fold += 1

        return self.scores