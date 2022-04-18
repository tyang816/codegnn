import torch
import yaml
import torch.nn as nn
import torch.nn.functional as f
from torchtext.legacy.data import Field, TabularDataset, Iterator
from torch_geometric.data import Data, InMemoryDataset
import javalang
import utils.data_tools as dt

bos = '<s>'
eos = '</s>'

# load config
config_path = './configs/config.yml'
config = yaml.load(open(config_path), Loader=yaml.FullLoader)
DATA_HOME = './data'

class Vocab(object):
    def __init__(self, config):
        self.config = config

    def build_raw_data(self, data_name, key):
        assert isinstance(key, str), "get raw data of `base` need to declare the key word, like `method`"
        lines_base = dt.load_base(path=DATA_HOME + data_name, key=key, is_json=True)
        token_lines_base = dt.tokenize_code(lines_base)
        if key == 'method':
            dt.save(token_lines_base, DATA_HOME + self.config['data']['raw_base_method'], is_json=True)
        elif key == 'summary':
            dt.save(token_lines_base, DATA_HOME + self.config['data']['raw_base_summary'], is_json=True)

    def build_vocab(self, data_name, key):
        """
        build vocab from the raw data
        """
        if key == 'method':
            METHOD = Field(sequential=True, lower=True, init_token=bos, eos_token=eos,
                           fix_length=self.config['model']['max_code_len'])
            # METHOD vocab can built by a list of files or a single file
            if isinstance(data_name, list):
                data = []
                for i in range(len(data_name)):
                    data = data + dt.load_raw(DATA_HOME + data_name[i])
            elif isinstance(data_name, str):
                data = dt.load_raw(DATA_HOME + data_name)
            METHOD.build_vocab(data)
            torch.save(METHOD, DATA_HOME + self.config['data']['field_method'])
        elif key == 'summary':
            SUMMARY = Field(sequential=True, lower=True, init_token=bos, eos_token=eos,
                            fix_length=self.config['model']['max_com_len'])
            if isinstance(data_name, list):
                data = []
                for i in range(len(data_name)):
                    data = data + dt.load_raw(DATA_HOME + data_name[i])
            elif isinstance(data_name, str):
                data = dt.load_raw(DATA_HOME + data_name)
            SUMMARY.build_vocab(data)
            torch.save(SUMMARY, DATA_HOME + self.config['data']['field_summary'])
        else:
            return

class GNNDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GNNDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def raw_file_names(self):
            return [config['data']['raw_method'].replace('/raw/', ''),
                    config['data']['raw_summary'].replace('/raw/', '')]

        @property
        def processed_file_names(self):
            return [config['data']['graph_dataset']]

        def class_graph(self, x, edge_index, y=None):
            data = Data(x=x, edge_index=edge_index, y=y)
            return data

        def process(self):
            method_path = self.raw_paths[0]
            summary_path = self.raw_paths[1]
            method_field = torch.load(DATA_HOME + config['data']['field_method'])
            summary_field = torch.load(DATA_HOME + config['data']['field_summary'])
            # [method0, method1, ...,]
            base_method_list = dt.load_raw(method_path, 'base')
            # [summary0, summary1, ...,]
            base_summary_list = dt.load_raw(summary_path, 'base')

