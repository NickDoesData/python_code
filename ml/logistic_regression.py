def logistic_regression(X, y):

    from sklearn.linear_model import LogisticRegression
    from sklearn import cross_validation as cv
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    # splits data into training and test sets. 70% train, 30% test.
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.3, random_state=0)

    stdsc      = StandardScaler()

    # z-scores training set data, then applies z-score formula from training set to test set.
    # this ensures the training data does not learn anything about the test data.
    X_train    = stdsc.fit_transform(X_train)
    X_test     = stdsc.transform(X_test)

    model      = LogisticRegression()
    model.fit(X_train, y_train)

    expected   = y_test
    predicted  = model.predict(X_test)
    probs      = model.predict_proba(X_test)

    print ('Accuracy Score: %.2f%%' % (accuracy_score(expected, predicted) * 100))
    print (classification_report(expected, predicted))
    
    return X_train, y_train, X_test, y_test,  expected, predicted, probs,  model


def create_coef_importance(model, predictors):
    
    coef = pd.DataFrame(model.coef_, columns=[predictors]).T
    importance = pd.DataFrame(model.coef_, columns=[predictors]).T.abs()
    coef_and_importance = pd.concat([coef, importance], axis=1)
    coef_and_importance = coef_and_importance.reset_index()
    coef_and_importance.columns = ['Feature', 'Coef', 'Importance']

    return coef_and_importance.sort_values('Importance', ascending=False).reset_index(drop=True)

