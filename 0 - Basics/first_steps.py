import numpy as np
from numpy.lib.shape_base import array_split

# Example of list in Python
my_list = [1, 2, 3]
print(type(my_list))

# Example of NumPy array
array_list = np.array(my_list)
print(type(array_list))

# Generating an array with values from 0 to 10
# in increments of 2
my_array = np.arange(0, 10, 2)
print(my_array)

# Generating matrixes in NumPy
matrix = np.zeros(shape=(5, 5))
print(matrix)

matrix2 = np.zeros(shape=(5, 10))
print(matrix2)

matrix3 = np.ones(shape=(3, 3))
print(matrix3)
