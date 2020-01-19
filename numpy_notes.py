import numpy as np

###################################################
# Create an array
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
# [[1 2 3]
#  [4 5 6]]

###################################################
# Less memory
import sys

s = range(1000)
print(sys.getsizeof(5) * len(s))  # 14000
d = np.arange(1000)
print(d.size * d.itemsize)  # 4000

###################################################
# Faster

import time
l1, l2 = range(1000000), range(1000000)
a1, a2 = np.arange(1000000), np.arange(1000000)

start = time.time()
result = [(x, y) for x, y in zip(l1, l2)]
print((time.time() - start) * 1000)  # 149.91450309753418

start = time.time()
result = a1 + a2
print((time.time() - start) * 1000)  # 54.51202392578125

###################################################
# Dim/Shape/size/Reshape
a = np.array([[1, 2, 3], ['a', 32, 3]])
print(a.ndim)  # 2
print(a.itemsize)  # 84
print(a.dtype)  # <U21
print(a.size)  # 6
print(a.shape)  # (2,3)
print(a.reshape(1, -1))  # [['1' '2' '3' 'a' '32' '3']]

###################################################
# Slicing
a = np.array([[1, 2, 3, 4, 5],
              ['a', 'b', 'c', 'd', 'e'],
              [11, 22, 33, 44, 55],
              ['A', ' B', 'C', 'D', 'E'],
              [6, 7, 8, 9, 0],
              ['aa', 'bb', 'cc', 'dd', 'ee']])
print(a[1:3, 2:4])
# [['c' 'd']
#  ['33' '44']]

###################################################
# Linspace/Max/Min
a = np.linspace(0, 10, 6)
print(a)  # [ 0.  2.  4.  6.  8. 10.]
print(a.max()) # 10.0
print(a.sum()) # 30.0

###################################################
# Sum on axis/sqrt
a = np.array([(1, 2, 3, 4, 5),
              (11, 22, 33, 44, 55),
              (6, 7, 8, 9, 0)])

print(a.sum(axis=0))  # [18 31 44 57 60]
print(a.sum(axis=1))  # [ 15 165  30]
print(np.sqrt(a))
#  [[1.         1.41421356 1.73205081 2.         2.23606798]
#   [3.31662479 4.69041576 5.74456265 6.63324958 7.41619849]
#   [2.44948974 2.64575131 2.82842712 3.         0.        ]]
print(np.std(a))  # 16.32993161855452

###################################################
# Stacking
a = np.array([(4, 4, 4),
              (3, 2, 1)])
b = np.array([(2, 2, 3),
              (3, 2, 1)])
print(a - b)
# [[2 2 1]
#  [0 0 0]]
print(np.vstack((a, b)))
# [[4 4 4]
#  [3 2 1]
#  [2 2 3]
#  [3 2 1]]
print(np.hstack((a, b)))
# [[4 4 4 2 2 3]
#  [3 2 1 3 2 1]]

###################################################
# Sin/Cos/Ravel
a = np.array([(4, 4, 4),
              (3, 2, 1)])
print(a.ravel()) #[4 4 4 3 2 1]
print(np.sin(a))
# [[-0.7568025  -0.7568025  -0.7568025 ]
#  [ 0.14112001  0.90929743  0.84147098]]
print(np.cos(a))
# [[-0.65364362 -0.65364362 -0.65364362]
#  [-0.9899925  -0.41614684  0.54030231]]

###################################################
# Exp/Log
ar = np.array([1, 2, 3])
print(np.exp(ar))  # [ 2.71828183  7.3890561  20.08553692]
print(np.log10(ar))  # [0.         0.30103    0.47712125]

###################################################
# Filtering
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr[arr % 2 == 1]) # [1 3 5 7 9]
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
arr[arr % 2 == 1] = -1
print(arr) # [-1  2 -1  4 -1  6 -1  8 -1]
a = np.arange(10)
b = np.where(a % 2 == 0, -1, a)
print(b) # [-1  1 -1  3 -1  5 -1  7 -1  9]
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
b = np.reshape(arr, (2, -1))
print(b)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]]

a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)
c = np.vstack([a, b])  # hstack
c = np.concatenate([a, b], axis=1)  # axis=0
c = np.r_[a, b]  # c_[a, b]
print(c)
# [[0 1 2 3 4]
#  [5 6 7 8 9]
#  [1 1 1 1 1]
#  [1 1 1 1 1]]

###################################################
# Repeat/Tile
a = np.array([1, 2, 3])
b = np.r_[np.repeat(a, 3), np.tile(a, 3)]
print(b)  # [1 1 1 2 2 2 3 3 3 1 2 3 1 2 3 1 2 3]

###################################################
# Difference/Intersection
a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
print(np.intersect1d(a, b)) # [2 4]
a = np.array([1, 2, 3, 3])
b = np.array([3, 4, 5])
print(np.setdiff1d(a, b))  # [1 2]
a = np.array([1, 2, 3, 3, 6, 7])
b = np.array([3, 4, 5, 3, 4, 7])
print(np.where(a == b))  # (array([3, 5]),)
a = np.array([2, 6, 1, 9, 10, 3, 27])
index = np.where((a >= 5) & (a <= 10))
print(a[index]) # [ 6  9 10]
print(a[(a >= 5) & (a <= 10)])# [ 6  9 10]

###################################################
# Vectorize
a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])

def ceva(x, y):
    return x * 10 + y

pair_max = np.vectorize(ceva)
print(pair_max(a, b)) # [56 73 94 88 69 47 51]

###################################################
# Reversing
a = np.arange(9).reshape((3,3))
print(a[:,[1,0,2]])
# [[1 0 2]
#  [4 3 5]
#  [7 6 8]]
print(a[[1,0,2], :])
# [[3 4 5]
#  [0 1 2]
#  [6 7 8]]
a = np.arange(9).reshape((3,3))
print(a[::-1])
# [[6 7 8]
#  [3 4 5]
#  [0 1 2]]
a = np.arange(9).reshape((3,3))
print(a[:, ::-1])
# [[2 1 0]
#  [5 4 3]
#  [8 7 6]]
