import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split

# read ratings
ratings = pd.read_csv("ml-latest-small/ratings.csv", usecols=[0,1,2])
# assumption: uid/mid is of string type
ratings = ratings.astype({"userId": "category", "movieId": "category"})

# n_user
n_user = ratings["userId"].nunique()
# n_item
n_item = ratings["movieId"].nunique()

# inner id -> raw id
user_index = dict(enumerate(ratings["userId"].unique()))
# raw_id -> inner_id
user_inverted_index = { v: k for k, v in user_index.items()}

# inner id -> raw id
item_index = dict(enumerate(ratings["movieId"].unique()))
# raw_id -> inner_id
item_inverted_index = { v: k for k, v in item_index.items()}

# add inner id
uids = [user_inverted_index[userId] for userId in ratings["userId"].values]
iids = [item_inverted_index[itemId] for itemId in ratings["movieId"].values]
ratings = ratings.assign(
    uid=uids, 
    iid=iids)
# drop unused columns
ratings.drop(columns=["userId", "movieId"], axis=1, inplace=True)

# split
ratings_train, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)
def stat(df):
    print("dataframe shape: ", df.shape)
    print(f"unique user count: {df['uid'].nunique()}, unique item count: {df['iid'].nunique()}")

print(ratings.dtypes)
stat(ratings)
stat(ratings_train)
stat(ratings_test)


sparse_ratings_train = scipy.sparse.csr_matrix((ratings_train.rating, (ratings_train.iid, ratings_train.uid)), shape=(n_item, n_user))

# training
from implicit.als import AlternatingLeastSquares
import time
import logging
model_name = "als"
log = logging.getLogger("implicit")

model = AlternatingLeastSquares(iterations=20, calculate_training_loss=True)
start = time.time()
model.fit(sparse_ratings_train)
log.debug("trained model '%s' in %s", model_name, time.time() - start)

# TODO: test eval
# recommendation item list
# real interaction item list
