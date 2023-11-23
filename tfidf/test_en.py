from tfidf import TFIDF

if __name__ == '__main__':
    data = ['an apple a day keeps doctors away.', 'apple is good to eat.']

    tfidf = TFIDF(ngram_range=(1,1), stop_words='english')
    obj, voc, t = tfidf.get_tfidf(data)
    tfidf.show(voc, t)
    top_n = tfidf.extract_top_n(voc, t)
    print(top_n)
