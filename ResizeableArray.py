from typing import Collection, List
import numpy as np

class ResizeableArray():
    def __init__(self, array = None, max_cache_size = 50, dtype = int) -> None:
        self.array = array
        self.temp_list = []
        self.max_cache_size = max_cache_size
    
    def append(self, toAppend):
        self.temp_list.append(toAppend)

        if len(self.temp_list) > self.max_cache_size:
            self.move_cache_to_main_array()
    
    def move_cache_to_main_array(self):
        self.append_to_array(self.temp_list)
        del self.temp_list
        self.temp_list = []

    def append_to_array(self, toAppend):
        if self.array is None:
            self.array = np.array(toAppend)
        else:
            self.array = np.append(self.array, toAppend)
    
    def sum(self):
        return np.sum(self.retrieve_array())

    def __str__(self) -> str:
        return str(self.array)
    
    def retrieve_array(self):
        if self.temp_list:
            self.move_cache_to_main_array()
        return self.array