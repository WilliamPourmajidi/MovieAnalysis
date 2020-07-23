# Load data on movie ratings, revenue, metadata etc. Split data into a relevant set for training, testing and classification.
# Explain your choice of split. It is ok if you decide to split into these subsets after part c -> if you do so, mention this at the end of your explanation.

### An example to load a csv file
import pandas as pd
import numpy as np
from ast import literal_eval
import warnings  # `do not disturb` mode
from sklearn.model_selection import train_test_split

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
import ipywidgets as widgets
from IPython import display
import pandas as pd
import numpy as np




# Settings for Panda dataframe displays
pd.options.display.width = 1200
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100

# Importing the data

meta_data = pd.read_csv('movies_metadata.csv',
                        low_memory=False)  # You may wish to specify types, or process columns once read
ratings_small = pd.read_csv('ratings_small.csv')

##### YOUR CODE HERE #######
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






