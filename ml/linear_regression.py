def linear_regression(X,y, normalize=True):
    
    from sklearn import linear_model
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    ols = linear_model.LinearRegression()

    if isinstance(X, pd.DataFrame) == False:
        X = X.reshape(-1,1)
    
    if normalize == True:
        stdsc = StandardScaler()
        X_train = stdsc.fit_transform(X_train)
        X_test = stdsc.transform(X_test)
    
    model = ols.fit(X_train, y_train)
    
    print ('B0 = %r' % ols.intercept_)    
    
    if isinstance(X, pd.DataFrame) == False:
        print ('B1 = %r' % ols.coef_[0])
    else:
        for num, col in enumerate(X.columns):
            print ('B%r = %r' % (num + 1, ols.coef_[num]))
    
    print ('R2 = %r' % ols.score(X_train, y_train))
    print ('test score %r' % ols.score(X_test, y_test))
    
    return model