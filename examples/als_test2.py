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


ratings = pd.read_csv("ml-latest-small/ratings.csv", usecols=[0, 1, 2])
movies = pd.read_csv("ml-latest-small/movies.csv", usecols=[0, 1])

items_helper = ItemsHelper(movies)
ratings_helper = RatingsHelper()
ratings = ratings_helper.parse_and_transform(ratings)

sparse_ratings = ratings_helper.to_sparse_item_users(ratings)

log = logging.getLogger("implicit")

model = AlternatingLeastSquares()

model.fit(sparse_ratings)

# print(similar_items(model, 588, ratings_helper, items_helper))

# recommendations = model.recommend_all(sparse_ratings.transpose(), filter_already_liked_items=True)
recommendations = model.recommend_all(sparse_ratings.transpose(), filter_already_liked_items=False)
print(recommendations.shape)
print(recommendations[0,:])

user_item_sparse_ratings = ratings_helper.to_sparse_user_items(ratings)
# recommendation = model.recommend(0, user_item_sparse_ratings)
recommendation = model.recommend(0, sparse_ratings.transpose(), filter_already_liked_items=False)
print([item[0] for item in recommendation])