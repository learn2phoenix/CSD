import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))


class Metrics(object):
    def __init__(self):
        self.data = None

    @staticmethod
    def get_recall(preds, gts, topk=5):
        preds = preds[:, :topk]
        preds -= gts[:, None]
        found = np.where(np.amin(np.absolute(preds), axis=1) == 0)[0]
        return found.shape[0] / gts.shape[0]

    @staticmethod
    def get_mrr(preds, gts, topk=5):
        preds = preds[:, :topk]
        preds -= gts[:, None]
        rows, cols = np.where(preds == 0)
        _, unique_rows = np.unique(rows, return_index=True)
        valid_cols = cols[unique_rows]
        valid_cols += 1
        return np.mean(1/valid_cols)

    @staticmethod
    def get_map(preds, gts, topk=5):
        preds = preds[:, :topk]
        preds -= gts[:, None]
        rows, cols = np.where(preds == 0)
        _, unique_rows = np.unique(rows, return_index=True)
        row_cols = np.split(cols, unique_rows)[1:]
        row_cols = [np.hstack([x[0], np.diff(x), topk - x[-1]]) for x in row_cols]
        row_cols = [np.pad(x, (0, topk + 1 - x.shape[0]), 'constant', constant_values=(0, 0)) for x in row_cols]
        precision = np.asarray([np.repeat(np.arange(topk + 1), x) / np.arange(1, topk + 1) for x in row_cols])
        return np.sum(np.mean(precision, axis=1)) / preds.shape[0]
        # numpy increasing array according to bins

    @staticmethod
    def get_recall_bin(preds, topk=5):
        # preds is a binary matrix of size Q x K
        preds = preds[:, :topk]
        found = np.where(np.amax(preds, axis=1) == True)[0]
        return found.shape[0] / preds.shape[0]

    @staticmethod
    def get_mrr_bin(preds, topk=5):
        # preds is a binary matrix of size Q x K
        preds = preds[:, :topk]
        rows, cols = np.where(preds)
        _, unique_rows = np.unique(rows, return_index=True)
        valid_cols = cols[unique_rows]
        valid_cols += 1
        return np.mean(1/valid_cols)

    @staticmethod
    def get_map_bin(preds, topk=5):
        # preds is a binary matrix of size Q x K
        preds = preds[:, :topk]
        rows, cols = np.where(preds)
        _, unique_rows = np.unique(rows, return_index=True)
        row_cols = np.split(cols, unique_rows)[1:]
        row_cols = [np.hstack([x[0], np.diff(x), topk - x[-1]]) for x in row_cols]
        row_cols = [np.pad(x, (0, topk + 1 - x.shape[0]), 'constant', constant_values=(0, 0)) for x in row_cols]
        precision = np.asarray([np.repeat(np.arange(topk + 1), x) / np.arange(1, topk + 1) for x in row_cols])
        return np.sum(np.mean(precision, axis=1)) / preds.shape[0]

    @staticmethod
    def get_per_query_precision_bin(preds):
        return np.sum(preds, axis=1)/preds.shape[1]
