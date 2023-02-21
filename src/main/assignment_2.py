# import the libraries necessary: numpy and solve from scipy

import numpy as np
from scipy.linalg import solve

# Question solutions start here:

# 1. Using Neville's method, find the 2nd degree interpolating value for f(3.7) for the 
#    following set of data: x - 3.6, 3.8, 3.9; f(x) - 1.675, 1.436, 1.318

# create function nevilles_method, which takes the points given and the approximated value and produces 
# the last row and column value
def nevilles_method(x_points, y_points, approximated_x):
    # make the size equal to the index of x_points
    size = len(x_points)
    # create the matrix
    matrix = np.zeros((size, size))

    # using a for loop, prepopulate the first column to equal the y-values
    for index, row in enumerate(matrix):
        row[0] = y_points[index]
    
    # set the number of points equal to the size of the matrix
    num_of_points = len(x_points)
    # use a for loop to loop through the matrix
    for i in range(1, num_of_points):
        for j in range(1, i + 1):
            # set up the numerator:
            # the first part is the approximated value minus the span x-value multiplied by the matrix value in 
            # the column before     
            first_multiplication = (approximated_x - x_points[i - j]) * matrix[i][j - 1]
            # the second part is the approximate value minus the x-value multiplied by the matrix value in the
            # row and column before.
            second_multiplication = (approximated_x - x_points[i]) * matrix[i - 1][j - 1]
            # to get the numerator, subtract the second part from the first part
            numerator = first_multiplication - second_multiplication

            # the denominator is just the current x-value minus the span x-value
            denominator = x_points[i] - x_points[i - j]

            # the matrix value is the numerator divided by the denominator
            coefficient = numerator / denominator

            # set the matrix equal to the coefficient
            matrix[i][j] = coefficient
    
    # print(matrix, "\n") - check, comment out later
    # print the specific part of the matrix needed, add a newline
    print(matrix[num_of_points - 1][num_of_points - 1], "\n")

# 2. Using Newton's forward method, print out the polynomial approximations for degrees 1, 2,
#    and 3 using the following set of data: 
#       (a) Hint, create the table first
#       (b) x - 7.2, 7.4, 7.5, 7.6; f(x) - 23.5492, 25.3913, 26.8224, 27.4589

# create the function newton_method_and_approx, which will solve both question 2 and 3. The first part will
# use the x-values and y-values to then create the rest of the derivatives, in order to solve for approximations
def newton_method_and_approx():
    # input the x-values:
    x0 = 7.2
    x1 = 7.4
    x2 = 7.5
    x3 = 7.6

    # input the y-values:
    f_x0 = 23.5492
    f_x1 = 25.3913
    f_x2 = 26.8224
    f_x3 = 27.4589

    # start the calculations:
    # first derivative is the subtraction of y-values over the subtraction of x-values
    first_dd_1 = (f_x1 - f_x0) / (x1 - x0)
    first_dd_2 = (f_x2 - f_x1) / (x2 - x1)
    first_dd_3 = (f_x3 - f_x2) / (x3 - x2)

    # second derivative is the subtraction of the first derivatives over the subtraction of x-values
    second_dd_1 = (first_dd_2 - first_dd_1) / (x2 - x0)
    second_dd_2 = (first_dd_3 - first_dd_2) / (x3 - x1)

    # third derivative is the subtraction of the second derivatives over the subtraction of x-values
    third_dd = (second_dd_2 - second_dd_1) / (x3 - x0)

    # set d equal to what is being asked for
    d = [first_dd_1, second_dd_1, third_dd]
    # print d and add a newline
    print(d, "\n")
    
    # 3. Using the results from 2, approximate f(7.3)?

    # set the approximated value (7.3) equal to a reusable variable
    approx_x = 7.3

    # create the series of additions and multiplications that will give us the approximated value of 7.3
    # start with f_x0 and add on the first first derivative found, multiply that by the approximated value
    # minus x0, then add on the first second derivative, multiply that by the approximated value minus x1
    # and by the approximated value minus x0, then add the third derivative multiplied by what the last two
    # were multiplied by and the approximated value minus x2.
    p_x = f_x0 + first_dd_1 * (approx_x - x0) + second_dd_1 * (approx_x - x1) * (approx_x - x0)\
          + third_dd * (approx_x - x2) * (approx_x - x1) * (approx_x - x0)
    
    # print the calculation and add a newline
    print(p_x, "\n")


# 4. Using the divided difference method, print out the Hermite polynomial approximation matrix
#       x = 3.6, 3.8, 3.9
#       f(x) = 1.675, 1.436, 1.318
#       f'(x) = -1.195, -1.188, -1.182

# create the apply_div_diff function, which will create the rest of the matrix not given.
def apply_div_diff(matrix: np.array):
    # use a for loop to go through the matrix
    for i in range(2, len(matrix)):
        for j in range(2, i + 2):
            
            # if the column is inside the size, or it doesn't equal 0, continue
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            
            # set the numerator equal to the matrix value from the column before minus the matrix value
            # from the row and column before (diagonal)
            numerator = matrix[i][j - 1] - matrix[i - 1][j - 1]
            
            # set the denominator equal to the x-value of the current spot minus the span of x-values plus 1
            denominator = matrix[i][0] - matrix[i - j + 1][0]

            # set operation equal to numerator over denominator
            operation = numerator / denominator
            # set the spots of the matrix equal to the operation
            matrix[i][j] = operation
    # return the matrix
    return matrix

# create the hermite_interpolation function which has the x_points, y_points, and slopes and sets those into
# the matrix, as well as prints the final matrix
def hermite_interpolation():
    # set up the x_points:
    x_points = [3.6, 3.8, 3.9]

    # set up the y_points:
    y_points = [1.675, 1.436, 1.318]

    # set up the slopes:
    slopes = [-1.195, -1.188, -1.182]
    
    # set the size equal to the lens of the x_points
    size = len(x_points) 
    # set the matrix equal to the double of the size
    matrix = np.zeros((size * 2, size * 2))

    # populate x-values:
    # set the index equal to 0 (starts at x0)
    index = 0
    # use a for loop to go through the matrix
    for x in range(0, size * 2, 2):
        # for the current matrix spot, print the x-value 
        matrix[x][0] = x_points[index]
        # for the next matrix spot, print the x-value
        matrix[x + 1][0] = x_points[index]
        # add one to the index
        index += 1
        
    # populate the y-values:
    # set the index equal to 0
    index = 0
    # use a for loop to go through the matrix
    for y in range(0, size * 2, 2):
        # for the current matrix spot, print the y-value
        matrix[y][1] = y_points[index]
        # for the next matrix spot, print the y-value
        matrix[y + 1][1] = y_points[index]
        # add one to the index
        index += 1


    # populate the derivatives (every other row):
    # set the index equal to 0
    index = 0
    # use a for loop to go through the matrix
    for i in range(1, size * 2 - 1, 2):
        # for the current matrix spot, print the slope
        matrix[i][2] = slopes[index]
        # numerator is equal to the y-value at the current spot minus 2 spots ago
        numerator = y_points[index] - y_points[index - 2]
        # denominator is equal to the x-value at the current spot minus 2 spots ago
        denominator = x_points[index] - x_points[index - 2]
        # set the next spot in the matrix equal to the numerator divided by the denominator
        matrix[i + 1][2] = numerator / denominator
        # add one to the index
        index += 1
    
    # apply the divided differences to create the full matrix
    filled_matrix = apply_div_diff(matrix)

    # print the finished matrix and add a newline
    print(filled_matrix, "\n")

# 5. Using cubic spline interpolation, solve for the following using this set of data:
#    x - 2, 5, 8, 10; f(x) - 3, 5, 7, 9
#       (a) Find matrix A
#       (b) Vector b
#       (c) Vector c       

# create the cubic_spline function to create a matrix and two vectors from the give x- and y-values
def cubic_spline(x, y):
    # set the size equal to the lens of the x-values
    size = len(x)
    # set the matrix equal to the size
    matrix: np.array = np.zeros((size, size))

    # start creating the matrix:
    # the starting value is 1
    matrix[0][0] = 1
    # subtract the x-values
    matrix[1][0] = x[1] - x[0]
    # double the additions of two x-value subtractions
    matrix[1][1] = 2 * ((x[1] - x[0]) + (x[2] - x[1]))
    # subtract x-values
    matrix[1][2] = x[2] - x[1]
    matrix[2][1] = x[2] - x[1]
    # double the additions of two x-value subtractions
    matrix[2][2] = 2 * ((x[3] - x[2]) + (x[2] - x[1]))
    # subtract x-values
    matrix[2][3] = x[3] - x[2]
    # starts and ends the same way
    matrix[3][3] = 1
    # print the matrix and add a newline
    print(matrix, "\n")

    # start preparing for the first vector
    # c0 and c3 both equal 0
    c0 = 0
    # c1 and c2 are the subtraction of 3 divided by the subtraction of x-values multiplied by 
    # the subtraction of y-values
    c1 = ((3 / (x[2] - x[1])) * (y[2] - y[1])) - ((3 / (x[1] - x[0])) * (y[1] - y[0]))
    c2 = ((3 / (x[3] - x[2])) * (y[3] - y[2])) - ((3 / (x[2] - x[1])) * (y[2] - y[1]))
    c3 = 0
    # set c equal to an array of c0, c1, c2, and c3
    c = np.array([c0, c1, c2, c3])
    # print c and add a newline
    print(c, "\n")

    # start the second vector
    # set f equal to matrix A
    f = [[matrix]]
    # set g equal to the values c0, c1, c2, c3
    g = [[c0], [c1], [c2], [c3]]

    # use solve from scipy to solve the system of equation, and set equal to h
    h = solve(f, g)

    # print the transposed h, but only the first line, and add a newline
    print(h.T[0], "\n")


# main function:
if __name__ == "__main__":
    # requirement from assignment
    np.set_printoptions(precision = 7, suppress = True, linewidth = 100)
    
    # 1
    # x-values
    x_points = [3.6, 3.8, 3.9]
    # y-values
    y_points = [1.675, 1.436, 1.318]
    # approximated value
    approximated_x = 3.7 
    nevilles_method(x_points, y_points, approximated_x)

    # 2 and 3
    newton_method_and_approx()

    # 4
    hermite_interpolation()

    # 5
    # x-values
    x = [2, 5, 8, 10]
    # y-values
    y = [3, 5, 7, 9]
    cubic_spline(x, y)