import logging
import time

import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

from helper import ItemsHelper, RatingsHelper
from eval import evaluate, evaluate_avg
from model import get_model, ALS_MODELS

# movielens_prefix = "ml-latest-small"
movielens_prefix = "/media/john/data/data/ml-20m"

ratings = pd.read_csv(movielens_prefix + "/ratings.csv", usecols=[0, 1, 2])
movies = pd.read_csv(movielens_prefix + "/movies.csv", usecols=[0, 1])

items_helper = ItemsHelper(movies)
ratings_helper = RatingsHelper()
ratings = ratings_helper.parse_and_transform(ratings)

ratings_train, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)


def stat(df):
    print("dataframe shape: ", df.shape)
    print(f"unique user count: {df['uid'].nunique()}, unique item count: {df['iid'].nunique()}")


print(ratings.dtypes)
stat(ratings)
stat(ratings_train)
stat(ratings_test)

item_users = ratings_helper.to_sparse_item_users(ratings_train)
user_items = item_users.T.tocsr()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("implicit")
# TODO: bm25_weights

recommend_all_flag = True

metrics = []
for model_name in ALS_MODELS.keys():
    log.info(model_name)
    model = get_model(model_name)
    start = time.time()
    model.fit(item_users)
    log.info("trained model '%s' in %s", model_name, time.time() - start)

    start = time.time()
    s1 = evaluate_avg(model, user_items, ratings_test, K=10, recommend_all=recommend_all_flag)
    s2 = evaluate_avg(model, user_items, ratings_test, K=5, recommend_all=recommend_all_flag)
    log.info("evaluated model '%s' in %s", model_name, time.time() - start)
    s1.update(s2)
    s1["model"] = model_name
    metrics.append(s1)

df = pd.DataFrame(metrics)
df.to_csv("benchmark.csv")
