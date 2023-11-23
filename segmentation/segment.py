import pathlib
import sys

p = pathlib.Path(__file__).resolve().parent.parent
sys.path.append('{}/'.format(p))

import numpy as np
import torch
from transformers import BertTokenizerFast, AutoModel

from util import dist


class Segment():

    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.model = AutoModel.from_pretrained('ckiplab/bert-base-chinese')

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else "cpu")

    def segment(self, text, threshold=8):
        token_ids = self.tokenizer.encode(text)
        length = len(token_ids) - 2

        batch_token_ids = np.array([token_ids] * (2 * length - 1))

        mask = self.tokenizer.mask_token_id

        for i in range(length):
            if i > 0:
                batch_token_ids[2 * i - 1, i] = mask
                batch_token_ids[2 * i - 1, i + 1] = mask
                batch_token_ids[2 * i, i + 1] = mask

        vectors = self.model(torch.tensor(batch_token_ids))[0].detach().numpy()
        word_token_ids = [[token_ids[1]]]
        for i in range(1, length):
            d1 = dist(vectors[2 * i, i + 1], vectors[2 * i - 1, i + 1])
            d2 = dist(vectors[2 * i - 2, i], vectors[2 * i - 1, i])
            d = (d1 + d2) / 2

            if d >= threshold:
                word_token_ids[-1].append(token_ids[i + 1])
            else:
                word_token_ids.append([token_ids[i + 1]])

        words = [
            self.tokenizer.decode(ids).replace(' ', '')
            for ids in word_token_ids
        ]
        return words
