import pathlib
import sys

p = pathlib.Path(__file__).resolve().parent.parent
sys.path.append('{}/'.format(p))

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TFIDF:

    def __init__(self,
                 analyzer=None,
                 ngram_range=None,
                 stop_words=None,
                 tokenizer=None):
        self.ngram_range = ngram_range
        self.analyzer = analyzer  #'char'
        self.ngram_range = ngram_range  #(1,1)
        self.stop_words = stop_words  #'english'
        self.tokenizer = tokenizer

    def set_args(self):
        if self.tokenizer != None:
            return {"tokenizer": self.tokenizer}
        else:
            args = dict()
            if self.analyzer != None:
                args.update({"analyzer": self.analyzer})

            if self.ngram_range != None:
                args.update({"ngram_range": self.ngram_range})

            if self.stop_words != None:
                args.update({"stop_words": self.stop_words})

            return args

    def get_tf(self, data):
        arg = self.set_args()
        counts = CountVectorizer(**arg).fit(data)
        t = counts.transform(data).toarray()

        return counts, counts.vocabulary_, t

    def get_tfidf(self, data):
        arg = self.set_args()
        tfidf = TfidfVectorizer(**arg)
        tfidf.fit(data)
        t = tfidf.fit_transform(data).todense()

        return tfidf, tfidf.vocabulary_, t

    def show(self, data, t):
        for i, j in zip(data, t):
            print(i)
            print(j)
            print('---')

    def extract_top_n(self, voc, t, n=5):
        scores = zip(voc, np.asarray(t.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for item in sorted_scores:
            print("{0:50}\tScore: {1}".format(item[0], item[1]))

        return sorted_scores[:n]
