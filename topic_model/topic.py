import pathlib
import sys

p = pathlib.Path(__file__).resolve().parent.parent
sys.path.append('{}/'.format(p))

import hdbscan
import numpy as np
import umap
import torch
import tqdm
from transformers import BertTokenizerFast, AutoModel
from sklearn.decomposition import PCA
from collections import defaultdict
from itertools import count



class TopicModel():

    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.model = AutoModel.from_pretrained('ckiplab/bert-base-chinese')

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else "cpu")

    def sent_embedding(self, data, batch_size=64):
        self.data = data
        print(f'# Data size:\t{len(data)}')

        sent = []

        def sum_with_mask(vectors, masks):
            masked = [i[0] for i in zip(vectors, masks) if i[1] == 1]
            return np.array(sum(masked) / len(masked))

        text = self.tokenizer(self.data,
                              max_length=300,
                              padding='max_length',
                              truncation=True)

        for i in tqdm.trange(0, len(self.data), batch_size, desc='Batch'):
            features = text['input_ids'][i:i + batch_size]
            masks = text['attention_mask'][i:i + batch_size]
            with torch.no_grad():
                _ = self.model(torch.tensor(features).to(self.device),
                               return_dict=True,
                               output_hidden_states=True)

                last = _.hidden_states[-1].detach().cpu()
                last_mx = np.array(list(map(sum_with_mask, last, masks)))
                fst = _.hidden_states[0].detach().cpu()
                fst_mx = np.array(list(map(sum_with_mask, fst, masks)))
                sent.append((last_mx + fst_mx) / 2.0)
        self.embeddings = np.concatenate(sent, axis=0)

    def whitening(self, embedding_size=35):
        self.model = PCA(n_components=embedding_size,
                         whiten=True).fit(self.embeddings)
        self.umap_embeddings = self.model.transform(self.embeddings)

    def generate_topic(self,
                       data,
                       n_neighbors=5,
                       n_components=5,
                       min_cluster_size=2):
        umap_embeddings = umap.UMAP(n_neighbors=n_neighbors,
                                    n_components=n_components,
                                    metric='cosine').fit_transform(
                                        self.embeddings)
        cluster = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom').fit(umap_embeddings)

        docs_per_topic = defaultdict(list)
        topic_count = defaultdict(list)
        for i in zip(count(0), data, cluster.labels_):
            docs_per_topic[i[2]].append(i[1])
            topic_count[i[2]].append(i[0])

        return topic_count, docs_per_topic

    def show(self, topic_count, docs_per_topic):
        print('topic\tcorpus_count')
        for i in topic_count.keys():
            print(f'{i}\t{len(topic_count[i])}')

        for i in docs_per_topic.keys():
            print(i)
            print(docs_per_topic[i])
