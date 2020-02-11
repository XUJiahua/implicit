import logging
import time

import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

from helper import ItemsHelper, RatingsHelper, similar_items
from eval import evaluate
from model import get_model

ratings = pd.read_csv("ml-latest-small/ratings.csv", usecols=[0,1,2])
movies = pd.read_csv("ml-latest-small/movies.csv", usecols=[0, 1])

items_helper = ItemsHelper(movies)
ratings_helper = RatingsHelper()
ratings = ratings_helper.parse_and_transform(ratings)

def stat(df):
    print("dataframe shape: ", df.shape)
    print(f"unique user count: {df['uid'].nunique()}, unique item count: {df['iid'].nunique()}")

print(ratings.dtypes)
stat(ratings)

item_users = ratings_helper.to_sparse_item_users(ratings)

model_name = "als"
log = logging.getLogger("implicit")

model = get_model(model_name)
start = time.time()
model.fit(item_users)
log.debug("trained model '%s' in %s", model_name, time.time() - start)

print(similar_items(model, 588, ratings_helper, items_helper))