#GIVEN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);


# NOTE: Uncomment the function below to visualize the data, but be sure
# to **re-comment it before submitting this assignment to the autograder**.
#part1_scatter()


#Question 1
def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Your code here
    degrees = [1, 3, 6, 9]
    pred = np.linspace(0,10,100).reshape(100, 1)
    out = np.zeros((4, 100))

    for i, deg in enumerate(degrees):
        poly = PolynomialFeatures(degree = deg)
        X_train_poly = poly.fit_transform(X_train.reshape(11, 1))

        linreg = LinearRegression().fit(X_train_poly, y_train)
        y = linreg.predict(poly.fit_transform(pred))

        out[i, :] = y

    # Return your answer
    return out


#GIVEN
# feel free to use the function plot_one() to replicate the figure
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

#plot_one(answer_one())


#Question 2
def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    # Your code here
    r2_train = []
    r2_test = []

    for i in range(10):
        poly = PolynomialFeatures(degree = i)
        X_train_poly = poly.fit_transform(X_train.reshape(11, 1))
        X_test_poly = poly.fit_transform(X_test.reshape(4, 1))

        linreg = LinearRegression().fit(X_train_poly, y_train)
        r2_train.append(linreg.score(X_train_poly, y_train))
        r2_test.append(linreg.score(X_test_poly, y_test))

    # Your answer here
    return (r2_train, r2_test)


#Question 3
def answer_three():
    # Your code here
    #import matplotlib.pyplot as plt
    #%matplotlib notebook

    #r2_scores = answer_two()
    #plt.figure()
    #plt.plot(r2_scores[0], c = 'red', label = 'training')
    #plt.plot(r2_scores[1], c = 'blue', label = 'test')
    #plt.legend()

    # Return your answer
    return (0, 9, 6)


#Question 4
def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    # Your code here
    poly = PolynomialFeatures(degree = 12)
    X_train_poly = poly.fit_transform(X_train.reshape(11, 1))
    X_test_poly = poly.fit_transform(X_test.reshape(4, 1))

    linreg = LinearRegression().fit(X_train_poly, y_train)
    lin_r2_test_score = linreg.score(X_test_poly, y_test)

    linlasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train_poly, y_train)
    lasso_r2_test_score = linlasso.score(X_test_poly, y_test)

    # Your answer here
    return (lin_r2_test_score, lasso_r2_test_score)


#GIVEN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2


#Question 5
def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    # Your code here
    clf = DecisionTreeClassifier(random_state = 0).fit(X_train2, y_train2)

    df = pd.DataFrame({'feature_name' : X_train2.columns.values, 'importance' : clf.feature_importances_})
    df = df.sort(['importance'], ascending = False)

    # Your answer here
    return df['feature_name'][:5].tolist()


#Question 6
def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    # Your code here
    param_range = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(SVC(kernel='rbf', C=1, random_state = 0),
                                                 X_subset, y_subset, param_name = 'gamma',
                                                 param_range = param_range)


    # Your answer here
    return (np.array(list(map(np.mean, train_scores))), np.array(list(map(np.mean, test_scores))))


#Question 7
def answer_seven():
    # Your code here
    #import matplotlib.pyplot as plt
    #%matplotlib notebook

    #scores = answer_six()
    #plt.figure()
    #plt.plot(param_range, scores[0], c = 'red', label = 'training')
    #plt.plot(param_range, scores[1], c = 'blue', label = 'test')
    #plt.xscale('log')
    #plt.legend()

    # Return your answer
    return (0.001, 10, 0.1)
