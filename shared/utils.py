import json
import pickle
import os

def isDir(dirname):
    """
    :param dirname: dirname
    :return: boolean
    """
    return os.path.isdir(dirname)

def make_dirs(path):
    """ Create directory at given path 
    
    :param path: path name
    """
    if not os.path.exists(path):
        os.makedirs(path)

def dump_to_json(data, filepath, indent=4, sort_keys=True):
    """ Dump dictionary to json file to a given filepath name 
    
    :param data: python dictionary
    :param filepath: filepath name
    :param indent: indent keys in json file
    :param sort_keys: boolean flag to sort keys
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)

def load_from_json(filepath):
    """ Load json file from filepath into a dictionary 
    
    :param filepath:
    :return: python dictionary
    """
    data = dict()
    with open(filepath) as data_file:    
        data = json.load(data_file)
    return data

def dump_to_txt(data, filepath):
    """ Dump data to txt file format to a given filepath name 
    
    :param filepath: filepath name
    """
    with open(filepath, "w") as file :
        file.write(data)

def dump_to_pickle(data, filepath):
    """ Dump data to pickle format to a given filepath name
    :param filepath: filepath name
    """ 
    with open(filepath, "wb") as file:
        pickle.dump(data, file)