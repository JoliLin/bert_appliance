from topic import TopicModel

if __name__ == '__main__':
    data = [
        '韓國瑜參選總統', '韓國瑜伽老師', '蔡英文是民進黨主席', '英文屬於拉丁語系', '英文老師是美國人', '美國獨立宣佈獨立宣言',
        '向量線性獨立', '線性代數包含矩陣運算', '矩陣乘法需要三次方時間複雜度', '深度學習需要大量運算資源'
    ]
    topic = TopicModel()
    topic.sent_embedding(data)
    #topic.whitening(embedding_size=10)
    topic_count, docs_per_topic = topic.generate_topic(data)
    topic.show( topic_count, docs_per_topic )


