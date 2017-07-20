def svm(X,y):

    from sklearn.svm import SVC
    from sklearn import cross_validation as cv
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    kernels = ['linear', 'poly', 'rbf']

    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.3, random_state=1)

    stdsc = StandardScaler()
    X_train = stdsc.fit_transform(X_train)
    X_test = stdsc.transform(X_test)

    for kernel in kernels:
        if kernel != 'poly':
            model      = SVC(kernel=kernel)
        else:
            model      = SVC(kernel=kernel, degree=3)

        model.fit(X_train, y_train)
        expected   = y_test
        predicted  = model.predict(X_test)
        
        print(kernel)
        print (classification_report(expected, predicted))