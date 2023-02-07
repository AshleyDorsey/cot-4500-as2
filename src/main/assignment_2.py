import numpy as np
import decimal

# 1. Using Neville's method, find the 2nd degree interpolating value for f(3.7) for the 
#    following set of data: x - 3.6, 3.8, 3.9; f(x) - 1.675, 1.436, 1.318
def nevilles_method():
    x0 = 3.6
    x1 = 3.8
    x2 = 3.9
    x = 3.7
    f_x0 = 1.675
    f_x1 = 1.436
    f_x2 = 1.318
    Q_0_0 = f_x0
    Q_1_0 = f_x1
    Q_2_0 = f_x2
    Q_1_1 = ((1) / (x1 - x0)) * (((x - x0) * Q_1_0) - ((x - x1) * Q_0_0))
    # print(Q_1_1) # just to check the value (comment out later)
    Q_2_1 = ((1) / (x2 - x1)) * (((x - x1) * Q_2_0) - ((x - x2) * Q_1_0))
    # print(Q_2_1) # just to check the value (comment out later)
    Q_2_2 = ((1) / (x2 - x0)) * (((x - x0) * Q_2_1) - ((x - x2) * Q_1_1))
    print(Q_2_2, "\n")

# 2. Using Newton's forward method, print out the polynomial approximations for degrees 1, 2,
#    and 3 using the following set of data: 
#       (a) Hint, create the table first
#       (b) x - 7.2, 7.4, 7.5, 7.6; f(x) - 23.5492, 25.3913, 26.8224, 27.4589
def divided_difference(x_points, y_points, value):
    # set up the matrix
    size = len(x_points)
    matrix = np.zeros((size, size))

    # fill the matrix
    for index, row in enumerate(matrix):
        row[0] = y_points[index]

    for i in range(1, size):
        for j in range(1, i+1):
            numerator = y_points[j] - y_points[j - 1]
            denominator = x_points[i] - x_points[i - j]

            operation = numerator / denominator
            matrix[i][j] = (operation)

    print(matrix)
    print(matrix[1][1])
    print(matrix[2][2])
    print(matrix[3][3])
    
    # 3. Using the results from 3, approximate f(7.3)
    f_x0 = matrix[0][0]
    x = 7.3
    x0 = 7.2
    slope = matrix[1][1]
    
    P_1 = f_x0 + (slope * (x - x0))
    print(P_1)
 

# 4. Using the divided difference method, print out the Hermite polynomial approximation matrix
#       x = 3.6, 3.8, 3.9
#       f(x) = 1.675, 1.436, 1.318
#       f'(x) = -1.195, -1.188, -1.182
 

# 5. Using cubic spline interpolation, solve for the following using this set of data:
#    x - 2, 5, 8, 10; f(x) - 3, 5, 7, 9
#       (a) Find matrix A
#       (b) Vector b
#       (c) Vector c       
            

if __name__ == "__main__":
    nevilles_method()

    # 2
    x_points = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]

    approximating_value = 7.3
    divided_difference(x_points, y_points, approximating_value)