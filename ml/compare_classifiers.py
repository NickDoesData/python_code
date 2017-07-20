def classifiers(strModel, X_train, X_test, y_train, y_test):

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn import cross_validation as cv
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import GradientBoostingClassifier

    models = {'Random_Forest':RandomForestClassifier(), 'Logistic_Regression':LogisticRegression(), 
              'LDA':LinearDiscriminantAnalysis(), 'KNN':KNeighborsClassifier(), 'Naive_Bayes':GaussianNB(), 
              'Decision_Tree':DecisionTreeClassifier(), 'Gradient_Boost': GradientBoostingClassifier()}

    model = models[strModel]
    model.fit(X_train, y_train)
  
    expected   = y_test
    predicted  = model.predict(X_test)

    print (strModel)
    print ('Accuracy Score: %s' % accuracy_score(expected, predicted))
#    print classification_report(expected, predicted)
    
    return model, strModel, accuracy_score(expected, predicted)

def evaluate_classifiers(X,y):
    
    from sklearn import cross_validation as cv
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.3, random_state=1)
    
    stdsc = StandardScaler()
    X_train = stdsc.fit_transform(X_train)
    X_test = stdsc.transform(X_test)
    
    
    results = {}
    
    for classifier in ['Random_Forest', 'Logistic_Regression', 'LDA', 'KNN', 'Naive_Bayes', 'Decision_Tree', 'Gradient_Boost']:
        model, strModel, accuracy_score = classifiers(strModel=classifier, X_train=X_train, 
                                                    X_test=X_test, y_train=y_train, y_test=y_test)
        results[strModel]= accuracy_score
        
    return results