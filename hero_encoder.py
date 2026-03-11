import polars
from itertools import combinations
from scipy.sparse import csr_matrix, lil_matrix


class HeroesEncoder:
    def __init__(self, pop_value, ngram_range=(1, 1), min_df=1):
        self.pop_value = pop_value
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vocab = None
        self.vocab_list = None

    @staticmethod
    def _generate_ngrams(hero_list, min_n, max_n):
        heroes = sorted(hero_list)
        for n in range(min_n, max_n + 1):
            yield from combinations(heroes, n)

    def fit(self, column: polars.Series):
        min_n, max_n = self.ngram_range
        counts = {}
        for hero_list in column.to_list():
            if hero_list is not None:
                for ngram in self._generate_ngrams(hero_list, min_n, max_n):
                    counts[ngram] = counts.get(ngram, 0) + 1
        filtered = sorted(ng for ng, cnt in counts.items() if cnt >= self.min_df)
        self.vocab = {ng: i for i, ng in enumerate(filtered)}
        self.vocab_list = filtered

    def transform(self, column: polars.Series):
        n_rows = len(column)
        n_cols = len(self.vocab)
        min_n, max_n = self.ngram_range
        matrix = lil_matrix((n_rows, n_cols))
        for i, hero_list in enumerate(column.to_list()):
            if hero_list is not None:
                for ngram in self._generate_ngrams(hero_list, min_n, max_n):
                    idx = self.vocab.get(ngram)
                    if idx is not None:
                        matrix[i, idx] = self.pop_value
        return csr_matrix(matrix)

    def fit_transform(self, column):
        self.fit(column)
        return self.transform(column)

    def get_keys(self):
        return [
            str(ng[0]) if len(ng) == 1 else "_".join(str(h) for h in ng)
            for ng in self.vocab_list
        ]