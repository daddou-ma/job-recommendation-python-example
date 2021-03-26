from configparser import SafeConfigParser

parser = SafeConfigParser()
parser.read('config.ini')

def get_config(obj, attr):
    return parser.get(obj, attr)

def get_raw_file_path(attr):
    return parser.get('raw_files', attr)

def get_data_file_path(attr):
    return parser.get('data_files', attr)

def get_pickle_file_path(attr):
    return parser.get('pickle_files', attr)

def get_result_file_path(attr):
    return parser.get('result_files', attr)