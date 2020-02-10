from helper import RatingsHelper, ItemsHelper, similar_items
from itemcf import ItemItemRecommender
import pandas as pd

ratings = pd.read_csv("ml-latest-small/ratings.csv", usecols=[0, 1, 2])
movies = pd.read_csv("ml-latest-small/movies.csv", usecols=[0, 1])

items_helper = ItemsHelper(movies)
ratings_helper = RatingsHelper()
ratings = ratings_helper.parse_and_transform(ratings)
model = ItemItemRecommender()
model.fit(ratings)

print(similar_items(model, 588, ratings_helper, items_helper))
