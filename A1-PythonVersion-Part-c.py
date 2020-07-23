# Load data on movie ratings, revenue, metadata etc. Split data into a relevant set for training, testing and classification.
# Explain your choice of split. It is ok if you decide to split into these subsets after part c -> if you do so, mention this at the end of your explanation.
##### YOUR CODE HERE #######
# Note: To make the execution of application slightly more efficient, I have followed the following steps: created a new data frame
# 1- Construct a new panda data frame from meta_data data set
# 2- Drop extra attributes that are not selected for prediction
# 3- add rating to the newly created panda data frame so all information we need is in one data frame
# 4- Divide data to three sets of "Training", "Validation", and "Testing" with 64%, 16%, and 20% respectively

import ipywidgets as widgets
from ipywidgets import Button, Layout
from IPython import display
widget1 = widgets.Output()
widget2 = widgets.Output()
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

movie_prediction_df = pd.read_csv('movies_metadata_budget_cleaned.csv',
                        low_memory=False)# ,dtype={'revenu': np.int32} )  # You

# print(movie_prediction_df)

# Remove missing values
movie_prediction_df.dropna(inplace = True)

# Data Cleaning
# Dropping extra attributes that will not be used for prediction
movie_prediction_df = movie_prediction_df.drop(["adult","belongs_to_collection","homepage","imdb_id","original_title","overview", "poster_path","release_date","runtime","spoken_languages","status","tagline","title", "video","vote_average", "vote_count"],axis=1)
print(f"Columns for movie_prediction_df dataset:  {movie_prediction_df.columns}")

# Attributes such as "genres", "production_companies", and "production_countries" have dictionary type
# content and we need to make them more flat (we only need names anyway).

movie_prediction_df['genres'] = movie_prediction_df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movie_prediction_df['production_companies'] = movie_prediction_df['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movie_prediction_df['production_countries'] = movie_prediction_df['production_countries'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# Numerical attributes "budget", and "revenue" should not have any 0 value. Hence, we need to remove rows
# that contain this issue. Moreover,I have exluded any movie with lower than 1000 dollar budget or revenue
# (what kind of movie has a 1000 budget - Most likey it's a data collection error or someone
# did not know how do to accounting! :)

movie_prediction_df= movie_prediction_df[movie_prediction_df['revenue'] > 1000]
movie_prediction_df= movie_prediction_df[movie_prediction_df['budget']  > 1000]


# movie_prediction_df['alphanumeric_budget'] = map(lambda x: x.isalpha(), movie_prediction_df['budget'])
# df['Quarters_isalphabetic'] = map(lambda x: x.isalpha(), df['Quarters'])


# movie_prediction_df= movie_prediction_df[movie_prediction_df['budget'] > 100]  I wish!!!!


# Checking for an zero values in the budget and revenue columns
# movie_prediction_df.apply(lambda budget:pd.to_numeric(budget, errors='coerce'))

# movie_prediction_df["revenue"] = pd.to_numeric(movie_prediction_df["revenue"])
#

# movie_prediction_df.info()

# movie_prediction_df = movie_prediction_df[movie_prediction_df.budget != 0]
# new_df = movie_prediction_df.query('budget' == 0)

# movie_prediction_df.loc[~(movie_prediction_df==0).all(axis=1)]
# print("------", movie_prediction_df[~(movie_prediction_df==0).all(axis=1)])
# print("Rows With Zero Values In The Budget Column:",movie_prediction_df[(movie_prediction_df['budget']==0)].shape[0])
# print("Rows With Zero Values In The Revenue Column:",reader[(reader['revenue']==0)].shape[0])

# print(movie_prediction_df.head())

# render in output widgets (for better table view of data)
with widget1:
     display.display(movie_prediction_df)
# create HBox
hbox = widgets.HBox([widget1])
hbox



### !!!!!!
# Renaming the column movieId  to id
# ratings_small.rename(columns={'movieId':'id'},inplace=True)
# print(f"Columns for ratings_small dataset:  {ratings_small.columns}")
# print(movie_prediction_df.dtypes)
# movie_prediction_df.id.astype('int', errors='ignore')
# print(movie_prediction_df.dtypes)
# print(ratings_small.dtypes)


# df_B.b_number.astype('int')

# print(ratings_small.dtypes)
# Join two panda data frames
# df_outer = pd.merge(movie_prediction_df, ratings_small, on='id', how='outer')
# print(df_outer)

# # The following line is one way of cleaning up the genres field - there are more verbose ways of doing this that are easier for a human to read
# meta_data_training['genres_cleaned'] = meta_data_training['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# meta_data_validation['genres_cleaned'] = meta_data_validation['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# meta_data_testing['genres_cleaned'] = meta_data_testing['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# #  df[column] = df[column].apply(lambda x: np.nan if pd.isnull(x) else ast.literal_eval(x))
# print("Here is the Genre attribute before Clearning: \n ")
# print(meta_data_training['genres_cleaned'].head())
# print(meta_data_validation['genres'].head())
# print(meta_data_testing['genres'].head())
# print("Here is the Genre attribute after Clearning: \n ")
# print(meta_data_training['genres_cleaned'].head())
# print(meta_data_validation['genres_cleaned'].head())
# print(meta_data_testing['genres_cleaned'].head())


# meta_data['year'] = pd.to_datetime(meta_data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
# Consider how to columns look before and after this 'clean-up' - it is very common to have to massage the data to get the right features



### An example to load a csv file
import pandas as pd
import numpy as np
from ast import literal_eval
import warnings  # `do not disturb` mode
from sklearn.model_selection import train_test_split
import ipywidgets as widgets
from IPython import display

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

# Settings for Panda dataframe displays
pd.options.display.width = 1200
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100

# Importing the data

meta_data = pd.read_csv('movies_metadata.csv',
                        low_memory=False)  # You may wish to specify types, or process columns once read
ratings_small = pd.read_csv('ratings_small.csv')

# Review the shape of data
print(meta_data.shape)
print(ratings_small.shape)

# Display Name of Columns for data sets
print(f"Columns for meta_data dataset:  {meta_data.columns}")
print(f"Columns for rating_small dataset:  {ratings_small.columns}")

# Display the head of data sets
print("Top 5 values for each dataset")
print(meta_data.head(5))
print(ratings_small.head(5))

# Display the missing values (null) for each attribute in data sets
print(f"Missing values for meta_data dataset: \n {meta_data.isnull().sum(0)} \n")
print(f"Missing values for rating_small dataset: \n  {ratings_small.isnull().sum()} \n")

# Splitting Data to Training and Test (80% for Training and 20% for Testing)
meta_data_training, meta_data_testing = train_test_split(meta_data, test_size=0.2)
ratings_small_training, ratings_small_test = train_test_split(ratings_small, test_size=0.2)
# Splitting Training Data to Training and Validation (80% for Training and 20% for validation)
meta_data_training, meta_data_validation = train_test_split(meta_data_training, test_size=0.2)
ratings_small_training, ratings_small_validation = train_test_split(ratings_small_training, test_size=0.2)

print(f"Training data for meta_data data set: {meta_data_training.shape}")
print(f"Validation data for meta_data data set: {meta_data_validation.shape}")
print(f"Testing data for meta_data data set: {meta_data_testing.shape}")

print(f"Training data for rating_small data set: {ratings_small_training.shape}")
print(f"Validation data for rating_small data set: {ratings_small_validation.shape}")
print(f"Testing data for rating_small data set: {ratings_small_test.shape}")



############ part c begins ################################################


# print('Before cleaning')
# print(meta_data.head())
# # The following line is one way of cleaning up the genres field - there are more verbose ways of doing this that are easier for a human to read
# meta_data['genres'] = meta_data['genres'].fillna('[]').apply(literal_eval).apply(
#     lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# meta_data['year'] = pd.to_datetime(meta_data['release_date'], errors='coerce').apply(
#     lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
# print('After cleaning')
# print(meta_data.head())

# create output widgets
widget1 = widgets.Output()
widget2 = widgets.Output()

# render in output widgets
with widget1:
    display.display(meta_data_training)
with widget2:
    display.display(ratings_small_training)

create HBox
hbox = widgets.HBox([widget1, widget2])



# plt.scatter((train[‘budget’]), (train[‘revenue’]))
# plt.title(‘Revenue vs Budget’)
# plt.xlabel(‘Budget’)
# plt.ylabel(‘Revenue’)
# plt.show()

### !!!!!!
# Renaming the column movieId  to id
# ratings_small.rename(columns={'movieId':'id'},inplace=True)
# print(f"Columns for ratings_small dataset:  {ratings_small.columns}")
# print(movie_prediction_df.dtypes)
# movie_prediction_df.id.astype('int', errors='ignore')
# print(movie_prediction_df.dtypes)
# print(ratings_small.dtypes)


# df_B.b_number.astype('int')

# print(ratings_small.dtypes)
# Join two panda data frames
# df_outer = pd.merge(movie_prediction_df, ratings_small, on='id', how='outer')
# print(df_outer)

# # The following line is one way of cleaning up the genres field - there are more verbose ways of doing this that are easier for a human to read
# meta_data_training['genres_cleaned'] = meta_data_training['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# meta_data_validation['genres_cleaned'] = meta_data_validation['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# meta_data_testing['genres_cleaned'] = meta_data_testing['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# #  df[column] = df[column].apply(lambda x: np.nan if pd.isnull(x) else ast.literal_eval(x))
# print("Here is the Genre attribute before Clearning: \n ")
# print(meta_data_training['genres_cleaned'].head())
# print(meta_data_validation['genres'].head())
# print(meta_data_testing['genres'].head())
# print("Here is the Genre attribute after Clearning: \n ")
# print(meta_data_training['genres_cleaned'].head())
# print(meta_data_validation['genres_cleaned'].head())
# print(meta_data_testing['genres_cleaned'].head())


# meta_data['year'] = pd.to_datetime(meta_data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
# Consider how to columns look before and after this 'clean-up' - it is very common to have to massage the data to get the right features


