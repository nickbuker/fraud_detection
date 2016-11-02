from sklearn.metrics import confusion_matrix, log_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def grid_score(y, y_pred, y_pred_proba):
    # Scoring functions that returns False Prediction Rates and log loss
    mat = confusion_matrix(y, y_pred)
    FNR = float(mat[1][0]) / (mat[1][1] + mat[1][0])
    FPR = float(mat[0][1]) / (mat[0][0] + mat[0][1])
    lg_loss = log_loss(y, y_pred_proba)
    return FNR, FPR, lg_loss


def forrest_search():
    # Test all combinations of hyperparameters in for loops
    results = {}
    for depth in xrange(1,21):
        for feat in xrange(1,21):
            for bool_ in [True, False]:
                for crit in ['gini', 'entropy']:
                    # Build and fit model with training data
                    model_forrest = RandomForestClassifier(n_jobs=-1, max_features=feat,
                                                           max_depth=depth, bootstrap=bool_,
                                                           criterion=crit, n_estimators=60)
                    model_forrest.fit(X_train, y_train)
                    # Make predictions on test data
                    y_test_pred = model_forrest.predict(X_test)
                    y_test_prob = model_forrest.predict_proba(X_test)
                    # Score predictions using grid_score function
                    FN, FP, LL = grid_score(y_test, y_test_pred, y_test_prob)
                    # Combine hyperparameters to string for key
                    key = str(depth) + ", " + str(feat) + ", " + str(bool_) + ", " + crit
                    results[key] =  LL
    # Find min log loss value and key
    min_key = min(results, key=results.get)
    return min_key, results[min_key], results


def boosting_search():
    # Test all combinations of hyperparameters in for loops
    results = {}
    for depth in xrange(1,21):
        for feat in xrange(1,21):
            for sub in [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]:
                for loss in ['deviance', 'exponential']:
                    # Build and fit model with training data
                    model_boosting = GradientBoostingClassifier(max_features=feat, max_depth=depth,
                                                                loss=loss, subsample=sub,
                                                                learning_rate=0.1, n_estimators=100)
                    model_boosting.fit(X_train, y_train)
                    # Make predictions on test data
                    y_test_pred = model_boosting.predict(X_test)
                    y_test_prob = model_boosting.predict_proba(X_test)
                    # Score predictions using grid_score function
                    FN, FP, LL = grid_score(y_test, y_test_pred, y_test_prob)
                    # Combine hyperparameters to string for key
                    key = str(depth) + ", " + str(feat) + ", " + str(sub) + ", " + loss
                    results[key] =  LL
    # Find min log loss value and key
    min_key = min(results, key=results.get)
    return min_key, results[min_key], results
