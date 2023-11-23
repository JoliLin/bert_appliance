from segment import Segment

if __name__ == '__main__':
    seg = Segment()
    res = seg.segment('韓國瑜參選總統')
    print(res)
    res = seg.segment('韓國瑜伽老師')
    print(res)
    res = seg.segment('蔡英文是民進黨主席')
    print(res)
