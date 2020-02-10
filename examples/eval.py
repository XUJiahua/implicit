import pandas as pd
import json


def evaluate(model, ratings_train, ratings_test, K=10, col_user='uid', col_item='iid'):
    recommendations = None
    if hasattr(model, 'recommend_all'):
        # recommendations matrix: n_user x K (recommend list length)
        recommendations = model.recommend_all(ratings_train,
                                              N=K,
                                              filter_already_liked_items=True)
    # uid as index
    interactions = ratings_test.groupby([col_user])[col_item].apply(list)

    # recall@k, precision@k, f1@k
    # https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
    records = []
    for index, interactionList in interactions.items():
        if recommendations is None:
            recommendationList = model.recommend(
                index,
                ratings_train,
                N=K
            )
            recommendationList = [item[0] for item in recommendationList]
        else:
            # every user should has recommendations
            recommendationList = recommendations[index]

        if len(recommendationList) == 0:
            continue

        intersection = set(recommendationList).intersection(set(interactionList))
        record = {
            f"n_intersection@{K}": len(intersection),
            f"n_interaction@{K}": len(interactionList),
            #             "n_recommendation": len(recommendationList),
            f"P@{K}": len(intersection) * 1.0 / len(recommendationList),
            f"R@{K}": len(intersection) * 1.0 / len(interactionList),
        }
        if record[f"n_intersection@{K}"] > 0:
            record[f"F1@{K}"] = 2 * record[f"P@{K}"] * \
                record[f"R@{K}"] / (record[f"P@{K}"] + record[f"R@{K}"])
        records.append(record)

    df = pd.DataFrame(records)
    return df


def evaluate_avg(model, ratings_train, ratings_test, K=10, col_user='uid', col_item='iid'):
    df = evaluate(model, ratings_train, ratings_test, K, col_user, col_item)
    return json.loads(df.describe().loc["mean", :].to_json())
