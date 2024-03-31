import concurrent.futures as concfut
import glob
import os

import pickle
import logging
import queue
import os.path as osp
import threading
from multiprocessing import Process
import math
import numpy as np

module_logger = logging.getLogger(__name__)


class Embeddings(object):
    """Class to read embeddings from the disk and store them in memory"""
    def __init__(self, data_dir, chunk_dir, file_ext='.pt', files=None, chunked=False, chunk_size=5000):
        if files is not None:
            self.embedding_files = list(map(lambda x: osp.join(data_dir, x), files))
        else:
            self.embedding_files = glob.glob(f'{data_dir}/*{file_ext}')
        self.embedding_queue = queue.Queue()
        self.embeddings = []
        self.filenames = []
        self.chunk_dir = chunk_dir
        self.chunk_size = chunk_size
        self.chunked = chunked
        if not self.chunked:
            threading.Thread(target=self.__result_consumer, daemon=True).start()
            self.__read_embeddings()
            self.embeddings, self.filenames = self.__remove_missing(self.embeddings, self.filenames)
        else:
            self.__read_embeddings_chunked()
        self.__sort_embeddings()

    def __result_consumer(self):
        """Consumes the results from the embedding queue and saves them to the disk"""
        processed = 0
        fnf = 0  # FileNotFound
        embedding_chunk = []
        filename_chunk = []
        chunk_cnt = 0
        while True:
            data = self.embedding_queue.get()
            if not isinstance(data, str):
                self.filenames.append(data['filename'])
                if data['embedding'] is not None:
                    self.embeddings.append(data['embedding'])
                    processed += 1
                    if processed % 1000 == 0:
                        module_logger.info(f'Read {processed}/{len(self.embedding_files)} embeddings')
                else:
                    fnf += 1
                    self.embeddings.append(None)
                if len(embedding_chunk) < self.chunk_size:
                    embedding_chunk.append(data['embedding'])
                    filename_chunk.append(data['filename'])
                else:
                    chunk_cnt += 1
                    embedding_chunk, filename_chunk = self.__remove_missing(embedding_chunk, filename_chunk)
                    Process(target=save_chunk, args=(embedding_chunk, filename_chunk, chunk_cnt, self.chunk_dir),
                            daemon=True).start()
                    embedding_chunk = []
                    filename_chunk = []
                self.embedding_queue.task_done()
            elif data == 'DONE':
                chunk_cnt += 1
                embedding_chunk, filename_chunk = self.__remove_missing(embedding_chunk, filename_chunk)
                save_chunk(embedding_chunk, filename_chunk, chunk_cnt, self.chunk_dir)
                module_logger.info(
                    f'Completed reading embeddings. There were {fnf} images for which embeddings were not found')
                self.embedding_queue.task_done()
                break

    def __sort_embeddings(self):
        """Sort embeddings and filenames by filename"""
        self.filenames = np.asarray(self.filenames)
        sort_order = np.argsort(self.filenames)
        self.embeddings = np.asarray(self.embeddings)[sort_order]
        self.filenames = self.filenames[sort_order]

    def __load_embedding(self, filename):
        """Loads an embedding from the disk and puts it in the embedding queue"""
        if osp.exists(filename):
            embedding = np.load(filename)
            data = {
                'embedding': embedding,
                'filename': filename.split('/')[-1],
            }
        else:
            data = {
                'filename': filename.split('/')[-1],
                'embedding': None
            }
        self.embedding_queue.put(data)

    def __read_embeddings(self):
        """Reads embeddings from the disk"""
        with concfut.ThreadPoolExecutor(max_workers=32) as executor:
            worker = self.__load_embedding
            executor.map(worker, self.embedding_files)
            executor.shutdown(wait=True, cancel_futures=False)
            self.embedding_queue.put('DONE')
            self.embedding_queue.join()

    def __read_embeddings_chunked(self):
        """Reads embeddings from the disk in chunks"""
        files = os.listdir(self.chunk_dir)
        cnt = 0
        with concfut.ProcessPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(load_chunk, osp.join(self.chunk_dir, filename)) for filename in files]
            for future in concfut.as_completed(futures):
                result = future.result()
                module_logger.info(f'Consuming {cnt}/{len(files)} chunks')
                self.embeddings.extend(list(map(lambda x: x.squeeze(), result['embeddings'])))
                self.filenames.extend(list(map(lambda x: '.'.join(x.split('/')[-1].split('.')[:-1]), result['filenames'])))
                cnt += 1
            module_logger.info('Finished reading chunks')

    @staticmethod
    def get_missing(x):
        """Returns the indices of missing embeddings"""
        indices = filter(lambda i_x: i_x[1] is None, enumerate(x))
        res = np.asarray([i for i, x in indices])
        return res

    def __remove_missing(self, embeddings, filenames):
        """Removes embeddings and filenames for which embeddings were not found"""
        missing_ids = self.get_missing(embeddings)
        embeddings = [ele for idx, ele in enumerate(embeddings) if idx not in missing_ids]
        filenames = [ele for idx, ele in enumerate(filenames) if idx not in missing_ids]
        return embeddings, filenames


def load_chunk(filename):
    """Loads a chunk file containing embeddings and filenames"""
    data = pickle.load(open(filename, 'rb'))
    return data


def save_chunk(embeddings, filenames, count, chunk_dir, chunk_size=50000):
    """Saves a chunk file containing embeddings and filenames. If the number of embeddings is less than chunk_size, it
    saves all embeddings and filenames in one file. Otherwise, it splits the embeddings and filenames into chunks of
    size chunk_size and saves each chunk in a separate file."""
    assert len(embeddings) == len(filenames)
    os.makedirs(chunk_dir, exist_ok=True)

    if len(embeddings) < chunk_size:
        data = {
            'embeddings': embeddings,
            'filenames': filenames,
        }
        pickle.dump(data, open(osp.join(chunk_dir, f'embeddings_{count}.pkl'), 'wb'))
    else:
        # Split into len(embeddings) / 50000 chunks
        for i in range(0, math.ceil(len(embeddings)/chunk_size)):
            data = {
                'embeddings': embeddings[i*chunk_size: min((i+1)*chunk_size, len(embeddings))],
                'filenames': filenames[i*chunk_size: min((i+1)*chunk_size, len(embeddings))],
            }
            with open(osp.join(chunk_dir, f'embeddings_{i}.pkl'), 'wb') as f:
                pickle.dump(data, f)