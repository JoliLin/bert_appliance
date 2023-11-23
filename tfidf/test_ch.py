from tfidf import TFIDF
from tokenizer import tokenizer

if __name__ == '__main__':
    data = [
        '韓國瑜參選總統', '韓國瑜伽老師', '蔡英文是民進黨主席', '英文屬於拉丁語系', '英文老師是美國人', '美國獨立宣佈獨立宣言',
        '向量線性獨立', '線性代數包含矩陣運算', '矩陣乘法需要三次方時間複雜度', '深度學習需要大量運算資源'
    ]
    tfidf = TFIDF(tokenizer=tokenizer)
    obj, voc, t = tfidf.get_tfidf(data)
    tfidf.show(voc, t)
    top_n = tfidf.extract_top_n(voc, t)
    print(top_n)
