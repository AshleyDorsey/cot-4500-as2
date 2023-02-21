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
    
    # print(matrix, "\n")
    # print the specific part of the matrix needed, add a newline
    print(matrix[num_of_points - 1][num_of_points - 1], "\n")

# 2. Using Newton's forward method, print out the polynomial approximations for degrees 1, 2,
#    and 3 using the following set of data: 
#       (a) Hint, create the table first
#       (b) x - 7.2, 7.4, 7.5, 7.6; f(x) - 23.5492, 25.3913, 26.8224, 27.4589

# create the function newton_method_and_approx, which will solve both question 2 and 3. The first part will
# use the x-values and y-values to then create the rest of the derivatives, in order to solve for approximations
def newton_method_and_approx(x_values, f_x):
    # start the calculations:
    # first derivative is the subtraction of y-values over the subtraction of x-values
    first_dd_1 = (f_x[1] - f_x[0]) / (x_values[1] - x_values[0])
    first_dd_2 = (f_x[2] - f_x[1]) / (x_values[2] - x_values[1])
    first_dd_3 = (f_x[3] - f_x[2]) / (x_values[3] - x_values[2])

    # second derivative is the subtraction of the first derivatives over the subtraction of x-values
    second_dd_1 = (first_dd_2 - first_dd_1) / (x_values[2] - x_values[0])
    second_dd_2 = (first_dd_3 - first_dd_2) / (x_values[3] - x_values[1])

    # third derivative is the subtraction of the second derivatives over the subtraction of x-values
    third_dd = (second_dd_2 - second_dd_1) / (x_values[3] - x_values[0])

    # set degree_approximation equal to what is being asked for
    degree_approximation = [first_dd_1, second_dd_1, third_dd]
    # print degree_approximation and add a newline
    print(degree_approximation, "\n")
    
    # 3. Using the results from 2, approximate f(7.3)?

    # set the approximated value (7.3) equal to a reusable variable
    approx_x = 7.3

    # create the series of additions and multiplications that will give us the approximated value of 7.3
    # start with f_x0 and add on the first first derivative found, multiply that by the approximated value
    # minus x0, then add on the first second derivative, multiply that by the approximated value minus x1
    # and by the approximated value minus x0, then add the third derivative multiplied by what the last two
    # were multiplied by and the approximated value minus x2.
    polynomial_approx = f_x[0] + first_dd_1 * (approx_x - x_values[0]) + second_dd_1 * (approx_x - x_values[1]) *\
                         (approx_x - x_values[0]) + third_dd * (approx_x - x_values[2]) *\
                         (approx_x - x_values[1]) * (approx_x - x_values[0])
    
    # print the calculation and add a newline
    print(polynomial_approx, "\n")


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
    for i in range(1, size * 2, 2):
        # for the current matrix spot, print the slope
        matrix[i][2] = slopes[index]
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
def cubic_spline(x_points, y_points):
    # set the size equal to the lens of the x-values
    size = len(x_points)
    # set the matrix equal to the size
    matrix_A: np.array = np.zeros((size, size))

    # start creating the matrix:
    # the starting value is 1
    matrix_A[0][0] = 1
    # subtract the x-values
    matrix_A[1][0] = x_points[1] - x_points[0]
    # double the additions of two x-value subtractions
    matrix_A[1][1] = 2 * ((x_points[1] - x_points[0]) + (x_points[2] - x_points[1]))
    # subtract x-values
    matrix_A[1][2] = x_points[2] - x_points[1]
    matrix_A[2][1] = x_points[2] - x_points[1]
    # double the additions of two x-value subtractions
    matrix_A[2][2] = 2 * ((x_points[3] - x_points[2]) + (x_points[2] - x_points[1]))
    # subtract x-values
    matrix_A[2][3] = x_points[3] - x_points[2]
    # starts and ends the same way
    matrix_A[3][3] = 1
    
    # print the matrix and add a newline
    print(matrix_A, "\n")

    # start preparing for the first vector
    
    # c0 and c3 both equal 0
    c0 = 0
    # c1 and c2 are the subtraction of 3 divided by the subtraction of x-values multiplied by 
    # the subtraction of y-values
    c1 = ((3 / (x_points[2] - x_points[1])) * (y_points[2] - y_points[1])) -\
          ((3 / (x_points[1] - x_points[0])) * (y_points[1] - y_points[0]))
    c2 = ((3 / (x_points[3] - x_points[2])) * (y_points[3] - y_points[2])) -\
          ((3 / (x_points[2] - x_points[1])) * (y_points[2] - y_points[1]))
    c3 = 0
    
    # set vector_b equal to an array of c0, c1, c2, and c3
    vector_b = np.array([c0, c1, c2, c3])
    
    # print vector_b and add a newline
    print(vector_b, "\n")

    # start the second vector
    # set function_one equal to matrix A
    function_one = [[matrix_A]]
    # set function_two equal to the values c0, c1, c2, c3
    function_two = [[c0], [c1], [c2], [c3]]

    # use solve from scipy to solve the system of equation, and set equal to system_of_equations
    system_of_equations = solve(function_one, function_two)

    # transpose the system_of_equations, but only the first line
    system_of_equations = system_of_equations.T[0]
    # print the system_of_equations and add a newline
    print(system_of_equations, "\n")

# main function:
if __name__ == "__main__":
    # requirement from assignment to cover decimal places
    np.set_printoptions(precision = 7, suppress = True, linewidth = 100)
    
    # 1 - Neville's Method (looking for last row and column value)
    # x-values
    x_points = [3.6, 3.8, 3.9]
    # y-values
    y_points = [1.675, 1.436, 1.318]
    # approximated value
    approximated_x = 3.7 
    nevilles_method(x_points, y_points, approximated_x)

    # 2 and 3 - Newton's Forward Method (looking for degree approximations and value approximations)
    # x-values
    x_values = [7.2, 7.4, 7.5, 7.6]
    # y-values
    f_x = [23.5492, 25.3913, 26.8224, 27.4589]
    newton_method_and_approx(x_values, f_x)

    # 4 - Hermite Interpolation (looking for full matrix)
    hermite_interpolation()

    # 5 - Cubic Spline (looking for a matrix A, vector B, and vector D)
    # x-values
    x = [2, 5, 8, 10]
    # y-values
    y = [3, 5, 7, 9]
    cubic_spline(x, y)