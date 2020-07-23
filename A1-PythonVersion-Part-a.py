from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
# print(diabetes_X)

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# print(df.head())


# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
# print("Data train dataset = " , diabetes_X_train)
print("Data test dataset = ", diabetes_X_test)

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
# print("Target train dataset = " , diabetes_y_train)
# print("Target test dataset = " , diabetes_y_test)

#### Approach 1 -  Using Scikit Learn Linear Regression function #######
# Instantiating a linear Regression object
lnr_regression = linear_model.LinearRegression()

# Fitting the training sets to the linear regression object (model)
lnr_regression.fit(diabetes_X_train, diabetes_y_train)

# defining a variable and assigning predicted values to it
diabetes_y_hat = lnr_regression.predict(diabetes_X_test)

# print('Details of the Regression \n  Coefficient (Slope) : \n', lnr_regression.coef_)
print(f'Details of the Regression (for approach #1) is presented below:\n  '
      f'Coefficient (Slope) : {lnr_regression.coef_}\n'
      f'Y-intercept : {lnr_regression.intercept_}\n'
      f'Mean squared error : {mean_squared_error(diabetes_y_test, diabetes_y_hat)}\n'
      f'Variance score: {r2_score(diabetes_y_test, diabetes_y_hat)}'
      )

# Plot outputs
plt.figure(figsize=(12, 12))
plt.title("Linear Regression for Diabetes Data Using Scikit Linear Regression", color="white")
plt.scatter(diabetes_X_test, diabetes_y_test, color='blue')
plt.plot(diabetes_X_test, diabetes_y_hat, color='red', linewidth=2)
plt.grid(True)
plt.show()

#### Approach 2 - Using Ordinary Least Square Method - Iterating through to find values for coefficient and y-intercept in y = mx + b  #######

# Calculating the mean of X and y
diabetes_X_train_mean = np.mean(diabetes_X_train)
diabetes_y_train_mean = np.mean(diabetes_y_train)
# print(diabetes_X_train_mean)
# print(diabetes_y_train_mean)

len_of_data = len(diabetes_X_train)

# Using the formula to calculate b1 and b2   https://mubaris.com/posts/linear-regression/
numer = 0
denom = 0
# Iterating to find the wights (in this case, Y-intercept and coefficient
for i in range(len_of_data):
    numer += (diabetes_X_train[i] - diabetes_X_train_mean) * (diabetes_y_train[i] - diabetes_y_train_mean)
    denom += (diabetes_X_train[i] - diabetes_X_train_mean) ** 2
b1 = numer / denom
b0 = diabetes_y_train_mean - (b1 * diabetes_X_train_mean)

# Print the details of the regression line
print(f'By Using Approach # 2, for the line y = b0 + b1x , the coefficient (slope) is {b1}, and the y-intercept is {b0}')

# Plotting Values and Regression Line
min_x = np.min(diabetes_X_train)
max_x = np.max(diabetes_X_train)

# Calculating line values x and y
x = np.linspace(min_x, max_x, 10)
y = b0 + b1 * x

plt.figure(figsize=(12, 12))
plt.title("Linear Regression for Diabetes Data Using Ordinary Least Square Method", color='white')
plt.scatter(diabetes_X_test, diabetes_y_test, color='blue')
plt.plot(x, y, color='red', linewidth=2)
plt.show()



# ---------------------------------------------------------------------------------------------------------
# Using Ordinary Least Square Method to calculate the coefficient and intercept
# formula:  https://mubaris.com/posts/linear-regression/


#
#
# #%% md
#
# [2 Marks]
# # b
#
# #%% md
#
# Load data on movie ratings, revenue, metadata etc. Split data into a relevant set for training, testing and classification. Explain your choice of split. It is ok if you decide to split into these subsets after part c -> if you do so, mention this at the end of your explanation.
#
# Explanation:
#
#
# #%%
#
# ### An example to load a csv file
# import pandas as pd
# import numpy as np
# from ast import literal_eval
# meta_data=pd.read_csv('movies_metadata.csv', low_memory=False) # You may wish to specify types, or process columns once read
# ratings_small=pd.read_csv('ratings_small.csv')
# import warnings; warnings.simplefilter('ignore')
#
# ##### YOUR CODE HERE #######
#
#
#
# #%% md
#
# [5 Marks]
# # c
#
# #%% md
#
# Organize the data into relevant features for predicting revenue. <br>
# i.  Explain your feature sets and organization. <br>
#
# YOUR EXPLANATION HERE
#
# ii. Plot movie revenue vs. rating as a scatter plot and discuss your findings. <br>
#
# YOUR EXPLANATION HERE
#
# iii. Visualize any other relationships you deem interesting and explain. <br>
#
# YOUR EXPLANATION HERE
#
#
# #%%
#
# meta_data.head()
# # The following line is one way of cleaning up the genres field - there are more verbose ways of doing this that are easier for a human to read
# #meta_data['genres'] = meta_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# #meta_data['year'] = pd.to_datetime(meta_data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
# #meta_data.head()
# # Consider how to columns look before and after this 'clean-up' - it is very common to have to massage the data to get the right features
#
# ##### YOUR CODE HERE #######
#
# #%%
#
#
#
# #%% md
#
# [3 Marks]
# # d
#
# #%% md
#
# Train a regression model to predict movie revenue. Plot predicted revenue vs. actual revenue on the test set. Quantify the error in your prediction. (You may use sklearn for this step)
#
# #%%
#
# # Regression model here, plot your fit to the revenue data versus the actual data from the test set as a scatter plot.
#
# ##### YOUR CODE HERE #######
#
# #%% md
#
# [4 Marks]
# # e
#
# Try a non-linear fit to the data, with and without regularization. Find your best fit and justify the choice of parameters, regularization constant and norm. Plot predicted revenue vs. actual revenue on the test set. In each case, quantify the error. (See e.g. Generalized linear models, Kernel Ridge regression, SVR and others from sklearn)
#
# #%%
#
# ##### YOUR CODE HERE WITHOUT REGULARIZATION #######
#
# #%%
#
# ##### YOUR CODE HERE WITH REGULARIZATION #######
#
# #%% md
#
# ## Part 2 [10 Marks]
#
# [4 Marks]
# # a
#
# Write a simple version of the basic algorithm for k-means clustering. Simple here means the core of the algorithm and not optimizations or extensions you might find in standard python libraries. Typically you might rely on a standard library for doing this, but it helps to see the core by manipulating the data and labels by hand as practice for numerical python and how to frame the algorithm.
#
# #%%
#
# # Import packages
# %matplotlib inline
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()  # for plot styling
# import numpy as np
#
#
# # Generate Samples
# from sklearn.datasets.samples_generator import make_blobs
# X, y_true = make_blobs(n_samples=300, centers=4,
#                        cluster_std=0.60, random_state=0)
# plt.scatter(X[:, 0], X[:, 1], s=50);
#
# ###############################################
# # YOUR CODE GOES HERE
# # Put some code to find clusters here
# # Assign the clusters and labels in your code
# ###############################################
#
#
#
#
# # Uncomment to display clusters and cluster centers
# #plt.scatter(X[:, 0], X[:, 1], c=labels,
# #            s=50, cmap='viridis');
# #plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
#
# #%% md
#
# <a id="data set"></a>
# [6 Marks]
# # b
#
# Load the mystery data below, and cluster the data (you don't need to use your code from part a). Visualize the data including assigned cluster labels.
#
# #%%
#
# # Load the mystery data here and cluster using k-means (now you can use libraries e.g. sklearn)
# mystery = np.load('mystery.npy')
# mystery.shape
#
# #%%
#
# # Find a way to visualize the data (e.g. in 2D or 3D), color datapoints based on assigned labels.
#
#
# #%% md
#
# Based on the results above and any other analysis you wish to include, discuss how many clusters you see in the data.
#
# YOUR EXPLANATION HERE
#
# (any additional code supporting your assertion on the number of clusters may be included below)
#
# #%%
#
# ##### YOUR (OPTIONAL) CODE HERE #######
#
# #%% md
#
# [2 Marks]
# # Bonus
#
# #%% md
#
# What is the mystery data in part 2? Show this in markdown and code below.
#
# #%% md
#
# EXPLANATION HERE, code goes below.
#
# #%%
#
# ##### YOUR (OPTIONAL) CODE HERE #######
#
# #%% md
#
# [10 Marks]
# # CP 8318 Questions
#
# #%% md
#
# Describe how you might implement a solution to recommend new movies to a user based on their existing preferences or ratings from Part 1.
#
# #%% md
#
# YOUR EXPLANATION HERE, provide an example for one user id below.
#
# #%%
#
# ##### YOUR CODE HERE #######
#
#
# #%%
#
#
