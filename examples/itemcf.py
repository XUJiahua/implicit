import math

class ItemItemRecommender:
    def _key1(self, k):
        return f"{k}"

    def _key2(self, k1, k2):
        return f"{k1}#{k2}"

    def fit(self, ratings, col_user="uid", col_item="iid"):
        self.items = ratings[col_item].unique()

        # co-occurance metrics
        freq2 = {}
        freq1 = {}

        preferences = ratings.groupby([col_user])[col_item].apply(list)
        for _, l in preferences.items():
            l = sorted(l)
            for e in l:
                k = self._key1(e)
                if k not in freq1:
                    freq1[k] = 1
                else:
                    freq1[k] += 1

            for i in range(len(l) - 1):
                for j in range(1, len(l)):
                    k = self._key2(l[i], l[j])
                    if k not in freq2:
                        freq2[k] = 1
                    else:
                        freq2[k] += 1
        
        self.freq2 = freq2
        self.freq1 = freq1

    def _similar_score(self, item_id1, item_id2):
        a = sorted([item_id1, item_id2])
        key =  self._key2(a[0], a[1])
        if key not in self.freq2:
            return 0
        return self.freq2[key] * 1.0 / math.sqrt(self.freq1[self._key1(item_id1)]* self.freq1[self._key1(item_id2)] )


    def similar_items(self, item_id, N=10):
        item_sims = [(item_id2, self._similar_score(item_id, item_id2)) for item_id2 in self.items]
        return sorted(item_sims, key=lambda x: -x[1])[:N]
        
