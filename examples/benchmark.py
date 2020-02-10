import logging
import time

import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

from helper import ItemsHelper, RatingsHelper
from eval import evaluate, evaluate_avg
from model import get_model, MODELS

ratings = pd.read_csv("ml-latest-small/ratings.csv", usecols=[0, 1, 2])
movies = pd.read_csv("ml-latest-small/movies.csv", usecols=[0, 1])

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
user_items = ratings_helper.to_sparse_user_items(ratings_train)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("implicit")

metrics = []
for model_name in MODELS.keys():
    log.info(model_name)
    model = get_model(model_name)
    start = time.time()
    model.fit(item_users)
    log.debug("trained model '%s' in %s", model_name, time.time() - start)

    s1 = evaluate_avg(model, user_items, ratings_test, K=10)
    s2 = evaluate_avg(model, user_items, ratings_test, K=5)
    s1.update(s2)
    s1["model"] = model_name
    metrics.append(s1)

df = pd.DataFrame(metrics)
df.to_csv("benchmark.csv")
