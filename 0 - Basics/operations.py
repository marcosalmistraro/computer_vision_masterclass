import numpy as np

# generating random values for np arrays
np.random.seed(101)
arr1 = np.random.randint(0, 100, 10)
print(arr1)
arr2 = np.random.randint(0, 100, 10)
print(arr2)

# displaying specific values for arr1
print(arr1.max())
print(arr1.argmax()) # returns the index for the maximum value
print(arr1.min())
print(arr1.argmin()) # returns the index for the minimum value
print(arr1.mean())
print(arr1.shape)

# reshaping arr1 into 2 rows, 5 columns
arr1 = arr1.reshape((2, 5))
print(arr1)

# generate array and reshape into 10x10 matrix
matrix = np.arange(0, 100).reshape(10, 10)
print(matrix)

# output specific values from the same matrix
row = 4
col = 6

print(matrix[row, col])
print(matrix[:, 1].reshape(10, 1))
print(matrix[0:3, 0:3])

# generate new matrix 
new_matrix = matrix.copy()
new_matrix[0:6, :] = 0 # substitute existing values with 0s
print(new_matrix)
