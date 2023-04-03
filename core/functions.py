import numpy as np
import optuna
import pandas as pd
from lightgbm import early_stopping
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score


def continuous_to_binary(y_array: np.array, threshold: float = None) -> list:
    """Convert an np.array with continuous values to boolean according to the threshold specified. The threshold is defined according to the **define_best_threshold** method.

    Args:
        y_array (np.array): np.array with continuous values

        threshold (str): default threshold

    Returns:
        list: list of boolean values
    """

    if threshold is None:
        threshold = 0.5
    y_array = [1 if i > threshold else 0 for i in y_array]

    return y_array


def lgbm_class_hyperparameter_tuning_pipeline(
    train: pd.DataFrame,
    train_y: pd.Series,
    base_params: dict,
    parameters_dict: dict,
    WEIGHT_FOR_METRIC_OPTUNA: float,
    OPTIMIZE_THRESHOLD_OPTUNA: bool,
    N_TRIALS: 50,
) -> optuna.study:
    """Performs hyperparameter tuning optimization for the LightGBM model. This pipeline was designed for a classification problem.

    The model parameters are described in the target_variable configuration file.

    The objective function of this problem is given by the weighting between precision and recall. **WEIGHT_FOR_METRIC_OPTUNA** is used to give a little more importance to precision.

    Precision and recall are calculated by summing the given test set metric with the difference between the metric in the training and testing set to help control overfitting.

    The purpose of the function is to maximize the score described above.

    For better explanations and details, see the answer to question 5.

    Returns:
        optuna.study: object of hyperparameter optimization performed or None, if "run_hyperparameter_tuning" is False
    """
    parameters = {}

    def objective(trial):
        for param in parameters_dict:
            if parameters_dict[param]["type"] == "float":
                parameters[param] = trial.suggest_float(
                    param,
                    parameters_dict[param]["min"],
                    parameters_dict[param]["max"],
                    log=True,
                )
            if parameters_dict[param]["type"] == "int":
                parameters[param] = trial.suggest_int(
                    param,
                    parameters_dict[param]["min"],
                    parameters_dict[param]["max"],
                    parameters_dict[param]["step"],
                )
            if parameters_dict[param]["type"] == "cat":
                parameters[param] = trial.suggest_categorical(
                    param, parameters_dict[param]["values"]
                )

        parameters.update(base_params)

        # This is because callback cannot be used with dart mode
        if "boosting_type" in parameters.keys():
            if parameters["boosting_type"] == "dart":
                callback = None
            else:
                callback = [early_stopping(10)]
        else:
            callback = [early_stopping(10)]

        # Should optimize the threshold?
        if OPTIMIZE_THRESHOLD_OPTUNA:
            threshold = trial.suggest_float("threshold", 0.45, 0.57, step=0.01)
        else:
            threshold = 0.5

        scores = []

        # define stratified k-fold cross-validation
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        #  the data just to shuffle it
        # Cross Validation
        for train_index, test_index in kf.split(train, train_y):

            X_train, X_test = train.iloc[train_index], train.iloc[test_index]
            y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]

            train_set = lgbm.Dataset(
                X_train,
                label=y_train,
            )

            dval = lgbm.Dataset(
                X_test,
                y_test,
                reference=train_set,
            )

            model = lgbm.train(
                parameters,
                train_set,
                valid_sets=dval,
                callbacks=callback,
            )

            y_pred_test = model.predict(
                X_test,
                num_iteration=model.best_iteration,
            )

            y_pred_train = model.predict(
                X_train,
                num_iteration=model.best_iteration,
            )

            # Convert to binary
            y_pred_train = continuous_to_binary(y_pred_train, threshold)
            y_pred_test = continuous_to_binary(y_pred_test, threshold)

            current_score_test_recall = recall_score(
                y_test, y_pred_test, zero_division=0
            )
            current_score_train_recall = recall_score(
                y_train, y_pred_train, zero_division=0
            )

            current_score_test_precision = precision_score(
                y_test, y_pred_test, zero_division=0
            )
            current_score_train_precision = precision_score(
                y_train, y_pred_train, zero_division=0
            )

            precision = current_score_test_precision - abs(
                current_score_train_precision - current_score_test_precision
            )
            recall = current_score_test_recall - abs(
                current_score_train_recall - current_score_test_recall
            )
            print(
                f"test_recall: {current_score_test_recall:.2%} | train_recall: {current_score_train_recall:.2%} \n"
            )
            print(
                f"test_recall: {current_score_test_precision:.2%} | train_precision: {current_score_train_precision:.2%}"
            )

            if (precision + recall) == 0:
                metric_otm = 0
            else:
                metric_otm = (
                    2
                    * (precision * recall)
                    / (precision + recall * WEIGHT_FOR_METRIC_OPTUNA)
                )
            scores += [metric_otm]

        score = np.mean(scores)

        return score

    study = optuna.create_study(
        study_name="study_delay_15",
        direction="maximize",
        pruner=optuna.pruners.PercentilePruner(
            25.0, n_startup_trials=5, n_warmup_steps=30, interval_steps=10
        ),
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=True,
    )

    return study
