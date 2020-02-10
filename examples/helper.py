import pandas as pd
import scipy


class ItemsHelper:
    def __init__(self, items, col_item="movieId", col_item_detail="title"):
        """
        item_id: item_detail(for now, title)
        """
        self.item_details = dict(zip(items[col_item], items[col_item_detail]))

    def title_of(self, item_id):
        if item_id not in self.item_details:
            return "not found"

        return self.item_details[item_id]


class RatingsHelper:
    def parse_and_transform(self, ratings, implicit=True, min_rating=4,
                            col_user="userId", col_item="movieId", col_rating="rating"):
        """
        get ratings information and transform
        """
        # get ratings information
        # n_user
        self.n_user = ratings[col_user].nunique()
        # n_item
        self.n_item = ratings[col_item].nunique()

        # inner id -> raw id
        self.user_index = dict(enumerate(ratings[col_user].unique()))
        # raw_id -> inner_id
        self.user_inverted_index = {v: k for k, v in self.user_index.items()}
        # inner id -> raw id
        self.item_index = dict(enumerate(ratings[col_item].unique()))
        # raw_id -> inner_id
        self.item_inverted_index = {v: k for k, v in self.item_index.items()}

        # column names
        self.col_rating = col_rating
        self.col_item = "iid"
        self.col_user = "uid"

        # transform#1 add inner ids columns
        uids = [self.user_inverted_index[user_id] for user_id in ratings[col_user].values]
        iids = [self.item_inverted_index[item_id] for item_id in ratings[col_item].values]
        ratings = ratings.assign(
            uid=uids,
            iid=iids)
        # transform#2 drop unused columns
        ratings.drop(columns=[col_user, col_item], axis=1, inplace=True)
        # transform#3 implict dataset
        if implicit:
            ratings = self.to_implict_dataset(ratings, min_rating)
        # transform#4 dtype float32
        ratings = ratings.astype({"rating": "float32"})

        return ratings

    def to_implict_dataset(self, ratings, min_rating=4):
        # convert to implicit dataset
        ratings = ratings[ratings[self.col_rating] >= min_rating]
        ratings[self.col_rating] = 1
        return ratings

    def to_sparse_item_users(self, ratings):
        """
        pandas dataframe to sparse item-user rating matrix
        """
        return scipy.sparse.csr_matrix((ratings[self.col_rating],
                                        (ratings[self.col_item], ratings[self.col_user])),
                                       shape=(self.n_item, self.n_user))

    def to_sparse_user_items(self, ratings):
        """
        pandas dataframe to sparse item-user rating matrix
        """
        return scipy.sparse.csr_matrix((ratings[self.col_rating],
                                        (ratings[self.col_user], ratings[self.col_item])),
                                       shape=(self.n_user, self.n_item))
    def to_raw_uid(self, uid):
        """
        inner id -> raw id
        """
        return self.user_index[uid]

    def to_raw_iid(self, iid):
        """
        inner id -> raw id
        """
        return self.item_index[iid]

    def to_inner_uid(self, uid):
        """
        raw_id -> inner_id
        """
        return self.user_inverted_index[uid]

    def to_inner_iid(self, iid):
        """
        raw_id -> inner_id
        """
        return self.item_inverted_index[iid]


def similar_items(model, item_id, ratings_helper, items_helper):
    """
    item_id: raw item_id
    """

    inner_item_id = ratings_helper.to_inner_iid(item_id)
    inner_items = model.similar_items(inner_item_id)
    items = [(ratings_helper.to_raw_iid(item[0]), item[1]) for item in inner_items]
    return [(items_helper.title_of(item[0]), item[1]) for item in items]
