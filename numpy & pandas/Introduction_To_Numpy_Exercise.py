# 1: 2D Array (3x3)-------------------------------

import numpy as np
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print("Exercise 1: 2D Array (3x3)")
print(arr2d)

print("Number of dimensions:", arr2d.ndim)
print("Shape of array:", arr2d.shape)

print("Slicing (first 2 rows and 2 columns):")
print(arr2d[:2, :2])


# 2: 1D Array -------------------------------

import numpy as np
arr1d = np.array([10, 20, 30, 40, 50, 60])

print("\nExercise 2: 1D Array")
print(arr1d)

print("Number of dimensions:", arr1d.ndim)

print("Shape of array:", arr1d.shape)

reshaped = arr1d.reshape(2, 3)
print("Reshaped array (2x3):")
print(reshaped)