import logging
import faiss

module_logger = logging.getLogger(__name__)


class FaissIndex(object):
    def __init__(self, index_func=faiss.IndexFlatIP, embedding_size=512*512):
        self.index = index_func(embedding_size)
        # Enable GPU support
        # self.index_gpu = faiss.index_cpu_to_all_gpus(self.index)

    def build_index(self, nodes):
        self.index.add(nodes)
        # Enable GPU support
        # self.index_gpu.add(nodes)

    def search_nns(self, embeddings, n):
        # Enable GPU support
        # return self.index_gpu.search(embeddings, n)
        return self.index.search(embeddings, n)
