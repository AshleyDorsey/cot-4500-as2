import numpy as np
import decimal
from scipy.linalg import solve
import pandas as pd

# 1. Using Neville's method, find the 2nd degree interpolating value for f(3.7) for the 
#    following set of data: x - 3.6, 3.8, 3.9; f(x) - 1.675, 1.436, 1.318

def nevilles_method(x_points, y_points, approximated_x):
    size = len(x_points)
    matrix = np.zeros((size, size))

    for index, row in enumerate(matrix):
        row[0] = y_points[index]
    
    num_of_points = len(x_points)
    for i in range(1, num_of_points):
        for j in range(1, i + 1):
            first_multiplication = (approximated_x - x_points[i - j]) * matrix[i][j - 1]
            second_multiplication = (approximated_x - x_points[i]) * matrix[i - 1][j - 1]

            denominator = x_points[i] - x_points[i - j]

            coefficient = (first_multiplication - second_multiplication) / denominator

            matrix[i][j] = coefficient
    
    # print(matrix, "\n") - check, comment out later
    print(matrix[num_of_points - 1][num_of_points - 1], "\n")

# 2. Using Newton's forward method, print out the polynomial approximations for degrees 1, 2,
#    and 3 using the following set of data: 
#       (a) Hint, create the table first
#       (b) x - 7.2, 7.4, 7.5, 7.6; f(x) - 23.5492, 25.3913, 26.8224, 27.4589

def newton_method_and_approx():
    x0 = 7.2
    x1 = 7.4
    x2 = 7.5
    x3 = 7.6
    f_x0 = 23.5492
    f_x1 = 25.3913
    f_x2 = 26.8224
    f_x3 = 27.4589
    first_dd_1 = (f_x1 - f_x0) / (x1 - x0)
    first_dd_2 = (f_x2 - f_x1) / (x2 - x1)
    first_dd_3 = (f_x3 - f_x2) / (x3 - x2)
    second_dd_1 = (first_dd_2 - first_dd_1) / (x2 - x0)
    second_dd_2 = (first_dd_3 - first_dd_2) / (x3 - x1)
    third_dd = (second_dd_2 - second_dd_1) / (x3 - x0)
    print([first_dd_1, second_dd_1, third_dd], "\n")
    
    # start of three
    approx_x = 7.3
    p_x = f_x0 + first_dd_1 * (approx_x - x0) + second_dd_1 * (approx_x - x1) * (approx_x - x0)\
          + third_dd * (approx_x - x2) * (approx_x - x1) * (approx_x - x0)
    print(p_x, "\n")


# 4. Using the divided difference method, print out the Hermite polynomial approximation matrix
#       x = 3.6, 3.8, 3.9
#       f(x) = 1.675, 1.436, 1.318
#       f'(x) = -1.195, -1.188, -1.182

def apply_div_diff(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i + 2):

            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            # something get left and diagonal left
            left: float = matrix[i][j - 1]
            diagonal_left: float = matrix[i - 1][j - 1]
            numerator: float = left - diagonal_left
            
            # something get operation
            denominator = matrix[i][0] - matrix[i - j][0]

            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation
    return matrix

def hermite_interpolation():
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]

    slopes = [-1.195, -1.188, -1.182]
    
    size = len(x_points) 
    matrix = np.zeros((size * 2, size * 2))

    # populate x values
    index = 0
    for x in range(0, size*2, 2):
        matrix[x][0] = x_points[index]
        matrix[x + 1][0] = x_points[index]
        index += 1
        
    # prepopulate
    index = 0
    for y in range(0, size * 2, 2):
        matrix[y][1] = y_points[index]
        matrix[y + 1][1] = y_points[index]
        index += 1


    # prepopulate derivatives ( every other row )
    index = 0
    for x in range(1, size * 2 -1, 2):
        matrix[x][2] = slopes[index]
        matrix[x + 1][2] = slopes[index]
        index += 1

    # apply the divided differences
    filled_matrix = apply_div_diff(matrix)
    print(filled_matrix, "\n")

# 5. Using cubic spline interpolation, solve for the following using this set of data:
#    x - 2, 5, 8, 10; f(x) - 3, 5, 7, 9
#       (a) Find matrix A
#       (b) Vector b
#       (c) Vector c       

def cubic_spline_matrix(x, y):
    size = len(x)
    matrix: np.array = np.zeros((size, size))
    matrix[0][0] = 1
    matrix[1][0] = x[1] - x[0]
    matrix[1][1] = 2 * ((x[1] - x[0]) + (x[2] - x[1]))
    matrix[1][2] = x[2] - x[1]
    matrix[2][1] = x[2] - x[1]
    matrix[2][2] = 2 * ((x[3] - x[2]) + (x[2] - x[1]))
    matrix[2][3] = x[3] - x[2]
    matrix[3][3] = 1
    print(matrix, "\n")

    c0 = 0
    c1 = ((3 / (x[2] - x[1])) * (y[2] - y[1])) - ((3 / (x[1] - x[0])) * (y[1] - y[0]))
    c2 = ((3 / (x[3] - x[2])) * (y[3] - y[2])) - ((3 / (x[2] - x[1])) * (y[2] - y[1]))
    c3 = 0
    print([c0, c1, c2, c3], "\n")

    f = [[matrix]]
    g = [[c0], [c1], [c2], [c3]]

    h = solve(f, g)

    # h.T is a 1X3 matrix
    # we want to EXTRACT that row
    print(h.T[0])


if __name__ == "__main__":
    np.set_printoptions(precision = 7, suppress = True, linewidth = 100)
    
    # 1
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    approximated_x = 3.7 
    #nevilles_method(x_points, y_points, approximated_x)

    # 2 and 3
    #newton_method_and_approx()

    # 4
    hermite_interpolation()

    # 5
    x = [2, 5, 8, 10]
    y = [3, 5, 7, 9]
    cubic_spline_matrix(x, y)