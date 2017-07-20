def random_forest(X, y, n_estimators=100, n_jobs=-1):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn import cross_validation as cv
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.3, random_state=1)

    stdsc = StandardScaler()
    X_train = stdsc.fit_transform(X_train)
    X_test = stdsc.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
    model.fit(X_train, y_train)
  
    expected   = y_test
    predicted  = model.predict(X_test)

    print ('Random Forest \n')
    print ('Accuracy Score: %s \n' % accuracy_score(expected, predicted))
    print (classification_report(expected, predicted))
    print ('Confusion Matrix: \n %s \n' % confusion_matrix(expected, predicted))
    feature_importance = pd.DataFrame({'importance':model.feature_importances_, 'feature':X.columns})


    print ('Feature Importances: \n %s' % feature_importance.sort_values('importance', ascending=False))
    
    return model