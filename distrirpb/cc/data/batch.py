# -*- coding: utf-8 -*-
# Copied by wangjun
# from AIRCtRL designed by Zhang Xiaochuan

import torch
import numpy as np
import context


def list_dict_to_dict_list(data):
    """ Convert a list of dict to a dict of list."""
    def init_list_data(dict_data):
        ret = {}
        for k, v in dict_data.items():
            if isinstance(v, dict):
                ret.update({k: init_list_data(v)})
            else:
                ret.update({k: [v]})
        return ret

    def append(list_data, dict_item):
        for k, v in list_data.items():
            if isinstance(v, dict):
                append(v, dict_item[k])
            else:
                v.append(dict_item[k])

    result = init_list_data(data[0])
    for item in data[1:]:
        append(result, item)
    return result


def torch_list_to_numpy(data):
    """ Convert list of torch.Tensor data to numpy array. """

    ret = []
    for item in data:
        if isinstance(item, list):
            ret.append(torch_list_to_numpy(item))
        else:
            assert isinstance(item, torch.Tensor)
            ret.append(item.detach().cpu().numpy())
    return np.array(ret)


def list_to_numpy(data):
    """ Convert list (nested) data to numpy array. """

    def get_list_item(list_data):
        if len(list_data) == 0:
            return np.array(list_data, dtype=np.float32)
        item = list_data[0]
        if isinstance(item, list):
            return get_list_item(item)
        else:
            return item
    if len(data) == 0:
        return np.array(data, dtype=np.float32)
    item = get_list_item(data)
    if isinstance(item, (bool, np.bool)):
        data = np.array(data, dtype=np.bool)
    elif isinstance(item, (int, np.int8, np.int16, np.int32, np.int64)):
        data = np.array(data, dtype=np.int)
    elif isinstance(item, (float, np.float64, np.float32, np.float16)):
        data = np.array(data, dtype=np.float32)
    elif isinstance(item, torch.Tensor):
        data = torch_list_to_numpy(data)
    elif isinstance(item, np.ndarray):
        data = np.array(data)
    else:
        data = np.array(data, dtype=np.float32)
    return data


class Batch(object):
    """ This class store list like data or dict (nested) of list like data. """
    def __init__(self, data):
        if isinstance(data, list):
            if len(data) == 0:
                pass
            elif isinstance(data[0], dict):

                data = list_dict_to_dict_list(data)

        if isinstance(data, dict):
            new_data = {}
            for key in data.keys():
                new_data[key] = Batch(data[key])
            self._core = new_data
        else:
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            if isinstance(data, list):
                data = list_to_numpy(data)

            if data is not None:
                if not isinstance(data, np.ndarray):
                    raise ValueError
                if data.dtype == np.float64:
                    data = data.astype(np.float32)
            self._core = data

    def __getitem__(self, key):
        """
        Get item by the input `key`.

        Args:
            key: key for specific item(s).

        Returns:
            the corresponding item(s).

        Examples:
            data = Batch({'a': [1, 2, 3]})
            data['a'].to_list()  # [1.0, 2.0, 3.0]
            data[1]  # {'a': 2.0}
            data[2]['a']  # 3.0
        """
        if self._core is None:
            return None
        if isinstance(key, int) and isinstance(self._core, dict):
            data = {}
            for k in self._core.keys():
                data[k] = self._core[k][key]
            return data
        else:
            return self._core[key]

    def __len__(self):
        """ Return len(self). """
        if self._core is None:
            return 0
        elif isinstance(self._core, np.ndarray):
            return len(self._core)
        else:
            assert isinstance(self._core, dict)
            value = list(self._core.values())
            return max(len(value[idx]) for idx in range(len(value)))

    def to_tensor(self, device='cpu'):
        """ Convert batch data to torch tensor or dict of torch tensor """
        if self._core is None:
            return None
        elif isinstance(self._core, dict):
            tensor_data = {}
            for key in self._core.keys():
                tensor_data[key] = self[key].to_tensor(device)
            return tensor_data
        else:
            if context.get_context('backend') == 'torch':
                return torch.tensor(self._core).to(device)
            else:
                pass

    def to_list(self):
        """ Convert batch data to list or dict of list """
        if self._core is None:
            return None
        elif isinstance(self._core, dict):
            tensor_data = {}
            for key in self._core.keys():
                tensor_data[key] = self[key].to_list()
            return tensor_data
        else:
            return torch.tensor(self._core).tolist()

    def to_numpy(self):
        """ Convert batch data to numpy array or dict of numpy array """
        if self._core is None:
            return None
        elif isinstance(self._core, dict):
            numpy_data = {}
            for key in self._core.keys():
                numpy_data[key] = self[key].to_numpy()
            return numpy_data
        else:
            return self._core
