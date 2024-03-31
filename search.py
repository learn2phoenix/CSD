import argparse
import copy
import logging
import logging.handlers as handlers
import pathlib
import sys

import faiss
import numpy as np
import vaex as vx
import wandb

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from search.embeddings import Embeddings
from search.faiss_search import FaissIndex
from metrics import metrics
from data.wikiart import WikiArt

logger = logging.getLogger()


def get_parser():
    parser = argparse.ArgumentParser('dynamicDistances-NN Search Module')
    parser.add_argument('--dataset', default='wikiart', type=str, required=True)
    parser.add_argument('--topk', nargs='+', type=int, default=[5],
                        help='Number of NN to consider while calculating recall')
    parser.add_argument('--mode', type=str, required=True, choices=['artist', 'label'],
                        help='The type of matching to do')
    parser.add_argument('--method', type=str, default='IP', choices=['IP', 'L2'], help='The method to do NN search')
    parser.add_argument('--emb-dir', type=str, default=None,
                        help='The directory where per image embeddings are stored (NOT USED when chunked)')
    parser.add_argument('--query_count', default=-1, type=int,
                        help='Number of queries to consider. Works only for domainnet')
    parser.add_argument('--chunked', action='store_true', help='If I should read from chunked directory instead')
    parser.add_argument('--query-chunk-dir', type=str, required=True,
                        help='The directory where chunked query embeddings should be saved/are already saved')
    parser.add_argument('--database-chunk-dir', type=str, required=True,
                        help='The directory where chunked val embeddings should be saved/are already saved')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='The directory of concerned dataset. (HARD CODED LATER)')
    parser.add_argument('--multilabel', action='store_true', help='If the dataset is multilabel')

    return parser


def get_log_handlers(args):
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = handlers.RotatingFileHandler(f'search.log', maxBytes=int(1e6), backupCount=1000)
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    return c_handler, f_handler


def main():
    parser = get_parser()
    args = parser.parse_args()

    handlers = get_log_handlers(args)
    logger.addHandler(handlers[0])
    logger.addHandler(handlers[1])
    logger.setLevel(logging.DEBUG)

    if args.dataset == 'wikiart':
        dataset = WikiArt(args.data_dir)
    else:
        raise NotImplementedError

    query_embeddings = Embeddings(args.emb_dir, args.query_chunk_dir,
                                  files=list(map(lambda x: f'{x.split(".")[0]}.npy', dataset.query_images)),
                                  chunked=args.chunked,
                                  file_ext='.npy')
    val_embeddings = Embeddings(args.emb_dir, args.database_chunk_dir,
                                files=list(map(lambda x: f'{x.split(".")[0]}.npy', dataset.val_images)),
                                chunked=args.chunked,
                                file_ext='.npy')

    query_embeddings.filenames = list(query_embeddings.filenames)
    val_embeddings.filenames = list(val_embeddings.filenames)

    # Filtering the dataset based on the files which actually exist.
    dataset.query_db = dataset.query_db[
        dataset.query_db['name'].isin(query_embeddings.filenames)]
    dataset.val_db = dataset.val_db[
        dataset.val_db['name'].isin(val_embeddings.filenames)]

    # Using only the embeddings corresponding to images in the datasets
    temp = vx.from_arrays(filename=query_embeddings.filenames, index=np.arange(len(query_embeddings.filenames)))
    dataset.query_db = dataset.query_db.join(temp, left_on='name', right_on='filename', how='left')
    query_embeddings.embeddings = query_embeddings.embeddings[dataset.get_query_col('index')]
    try:
        b, h, w = query_embeddings.embeddings.shape
        query_embeddings.embeddings = query_embeddings.embeddings.reshape(b, 1, h * w)
    except ValueError:
        b, d = query_embeddings.embeddings.shape
        query_embeddings.embeddings = query_embeddings.embeddings.reshape(b, 1, d)
    query_embeddings.filenames = np.asarray(query_embeddings.filenames)[dataset.get_query_col('index')]

    temp = vx.from_arrays(filename=val_embeddings.filenames, index=np.arange(len(val_embeddings.filenames)))
    dataset.val_db = dataset.val_db.join(temp, left_on='name', right_on='filename', how='left')
    val_embeddings.embeddings = val_embeddings.embeddings[dataset.get_val_col('index')]
    try:
        b, h, w = val_embeddings.embeddings.shape
        val_embeddings.embeddings = val_embeddings.embeddings.reshape(b, 1, h * w)
    except ValueError:
        b, d = val_embeddings.embeddings.shape
        val_embeddings.embeddings = val_embeddings.embeddings.reshape(b, 1, d)
    val_embeddings.filenames = np.asarray(val_embeddings.filenames)[dataset.get_val_col('index')]

    # Building the faiss index
    embedding_size = query_embeddings.embeddings[0].shape[1]
    if args.method == 'IP':
        method = faiss.IndexFlatIP
    else:
        method = faiss.IndexFlatL2
    search_module = FaissIndex(embedding_size=embedding_size, index_func=method)
    queries = np.asarray(query_embeddings.embeddings).reshape(len(query_embeddings.embeddings), embedding_size)
    database = np.asarray(val_embeddings.embeddings).reshape(len(val_embeddings.embeddings), embedding_size)
    search_module.build_index(database)

    _, nns_all = search_module.search_nns(queries, max(args.topk))
    if args.multilabel:
        q_labels = dataset.query_db['multilabel'].values
        db_labels = dataset.val_db['multilabel'].values
        nns_all_pred = [q_labels[i] @ db_labels[nns_all[i]].T for i in range(len(nns_all))]
        nns_all_pred = np.array(nns_all_pred)
    else:
        nns_all_pred = nns_all
        classes = np.unique(dataset.get_val_col(args.mode))
        mode_to_index = {classname: i for i, classname in enumerate(classes)}
        try:
            gts = np.asarray(list(map(lambda x: mode_to_index[x], dataset.get_query_col(args.mode).tolist())))
        except KeyError:
            logger.error('Class not found in database. This query list cannot be evaluated')
            return

    evals = metrics.Metrics()

    for topk in args.topk:
        logger.info(f'Calculating recall@{topk}')
        nns_all_pred_topk = nns_all_pred[:, :topk]
        if args.multilabel:
            mode_recall = evals.get_recall_bin(copy.deepcopy(nns_all_pred_topk), topk)
            mode_mrr = evals.get_mrr_bin(copy.deepcopy(nns_all_pred_topk), topk)
            mode_map = evals.get_map_bin(copy.deepcopy(nns_all_pred_topk), topk)
        else:
            preds = dataset.get_val_col(args.mode)[nns_all_pred_topk.flatten()].reshape(len(queries), topk)
            preds = np.vectorize(mode_to_index.get)(preds)
            mode_recall = evals.get_recall(copy.deepcopy(preds), gts, topk)
            mode_mrr = evals.get_mrr(copy.deepcopy(preds), gts, topk)
            mode_map = evals.get_map(copy.deepcopy(preds), gts, topk)
        logger.info(f'Recall@{topk}: {mode_recall}')
        logger.info(f'MRR@{topk}: {mode_mrr}')
        logger.info(f'mAP@{topk}: {mode_map}')


if __name__ == '__main__':
    main()
