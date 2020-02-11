import logging
import time

import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from implicit.als import AlternatingLeastSquares
from helper import ItemsHelper, RatingsHelper, similar_items
from eval import evaluate
from model import get_model
from implicit.approximate_als import (AnnoyAlternatingLeastSquares, FaissAlternatingLeastSquares,
                                      NMSLibAlternatingLeastSquares)


# movielens_prefix = "ml-latest-small"
movielens_prefix = "/media/john/data/data/ml-20m"

ratings = pd.read_csv(movielens_prefix + "/ratings.csv", usecols=[0, 1, 2])
movies = pd.read_csv(movielens_prefix + "/movies.csv", usecols=[0, 1])

items_helper = ItemsHelper(movies)
ratings_helper = RatingsHelper()
ratings = ratings_helper.parse_and_transform(ratings)

item_users = ratings_helper.to_sparse_item_users(ratings)

log = logging.getLogger("implicit")

model = AlternatingLeastSquares()

model.fit(item_users)

user_id = 0
user_items = item_users.T.tocsr()
recommendations = model.recommend_all(user_items)
print(recommendations[user_id,:])

recommendation = model.recommend(user_id, user_items)
print([item[0] for item in recommendation])