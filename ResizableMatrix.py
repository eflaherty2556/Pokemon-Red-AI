from typing import Collection, List
import numpy as np
import copy

class ResizeableMatrix:
    def __init__(self, matrix = None, max_cache_size = 500, dtype = int) -> None:
        self.matrix = matrix
        self.temp_list = []
        self.max_cache_size = max_cache_size
    
    def append(self, toAppend):
        if isinstance(toAppend, int):
            self.temp_list.append([toAppend])
        else:
            self.temp_list.append(toAppend)
        
        if len(self.temp_list) > self.max_cache_size:
            self.move_cache_to_main_matrix()
    
    def move_cache_to_main_matrix(self):
        self.append_to_matrix(self.temp_list)
        del self.temp_list
        self.temp_list = []
    
    def append_to_matrix(self, toAppend):
        if self.matrix is None:
            self.matrix = np.asmatrix(toAppend)
        else:
            self.matrix = np.vstack((self.matrix, toAppend))

        #Old Version
        """return
        if not np.array_equal(self.matrix, np.matrix([[]])):
            temp_list = self.matrix.tolist()
            temp_list.append(toAppend)

        else:
            temp_list = [toAppend]
        
        
        self.matrix = np.matrix(temp_list)
        del temp_list"""

    def __getitem__(self, key:Collection):
        if self.matrix is None:
            raise IndexError("Matrix not initialized!")
        x,y = key
        return self.matrix[x, y]

    def __str__(self) -> str:
        return str(self.matrix)
        
    def retrieve_matrix(self):
        if self.temp_list:
            self.move_cache_to_main_matrix()
        return self.matrix