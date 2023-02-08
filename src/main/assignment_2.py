import numpy as np
import decimal

# 1. Using Neville's method, find the 2nd degree interpolating value for f(3.7) for the 
#    following set of data: x - 3.6, 3.8, 3.9; f(x) - 1.675, 1.436, 1.318
def nevilles_method():
    x0 = 3.6
    x1 = 3.8
    x2 = 3.9
    approx_x = 3.7
    f_x0 = 1.675
    f_x1 = 1.436
    f_x2 = 1.318
    Q_0_0 = f_x0
    Q_1_0 = f_x1
    Q_2_0 = f_x2
    Q_1_1: float = ((1) / (x1 - x0)) * (((approx_x - x0) * Q_1_0) - ((approx_x - x1) * Q_0_0))
    # print(Q_1_1) # just to check the value (comment out later)
    Q_2_1: float = ((1) / (x2 - x1)) * (((approx_x - x1) * Q_2_0) - ((approx_x - x2) * Q_1_0))
    # print(Q_2_1) # just to check the value (comment out later)
    Q_2_2: float = ((1) / (x2 - x0)) * (((approx_x - x0) * Q_2_1) - ((approx_x - x2) * Q_1_1))
    print(Q_2_2, "\n")


# 2. Using Newton's forward method, print out the polynomial approximations for degrees 1, 2,
#    and 3 using the following set of data: 
#       (a) Hint, create the table first
#       (b) x - 7.2, 7.4, 7.5, 7.6; f(x) - 23.5492, 25.3913, 26.8224, 27.4589

def fake_two():
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
    print(first_dd_1)
    second_dd_1 = (first_dd_2 - first_dd_1) / (x2 - x0)
    second_dd_2 = (first_dd_3 - first_dd_2) / (x3 - x1)
    print(second_dd_1)
    third_dd = (second_dd_2 - second_dd_1) / (x3 - x0)
    print(third_dd)

def divided_difference(x_points, y_points):
    # set up the matrix
    size = len(x_points)
    matrix: np.array = np.zeros((size, size))

    # fill the matrix
    for index, row in enumerate(matrix):
        row[0] = y_points[index]

    for i in range(1, size):
        for j in range(1, i + 1):
            numerator = y_points[i] - y_points[i - 1]
            denominator = x_points[i] - x_points[i - j]
            operation = (numerator) / (denominator)
            matrix[i][j] = decimal.Decimal(operation)


    print(matrix, "\n") # checking, comment out later
    #print(matrix[1][1])
    #print(matrix[2][2])
    #print(matrix[3][3])
    return matrix
    
# 3. Using the results from 3, approximate f(7.3)
def approximating_value(matrix, x_points, value):
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]

    for index in range(1, len(x_points)):
        polynomial_coefficient = matrix[index][index]

        reoccuring_x_span *= (value - x_points[index - 1])
        mult_operation = polynomial_coefficient * reoccuring_x_span
        reoccuring_px_result += mult_operation
    
    print(reoccuring_px_result)
    return reoccuring_px_result

 

# 4. Using the divided difference method, print out the Hermite polynomial approximation matrix
#       x = 3.6, 3.8, 3.9
#       f(x) = 1.675, 1.436, 1.318
#       f'(x) = -1.195, -1.188, -1.182
def get_x_span(i, j):
    if i == j:
        return i, j - i

    elif j > i:
        return j - i, i
        
    elif j < i:
        return j - i - 2, i

def apply_div_diff(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i + 2):
            # something get left and diagonal left
            left: float = matrix[i][j - 1]
            diagonal_left: float = matrix[i - 1][j - 1]
            numerator: float = left - diagonal_left
            
            # something get x_spand
            start, end = get_x_span(i, j)
            
            # something get operation
            list_of_xs_involved = []
            for index in range(start, end):
                list_of_xs_involved.append(matrix[index][0])

            unique_xs = set(list_of_xs_involved)
            list_of_uniques = list(unique_xs)

            denominator = list_of_uniques[-1] - list_of_uniques[0]

            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation
    print(matrix)

def hermite_interpolation():
    x_points = [1.3, 1.6, 1.9]
    y_points = [.6200, .4554, .2818]

    slopes = [-.522, -.569, -.581]
    
    num_of_points = len(x_points)
    matrix = np.zeros((num_of_points *2, num_of_points * 2))

    # populate x values
    index = 0
    for x in range(0, len(matrix), 2):
        matrix[x][0] = x_points[index]
        matrix[x + 1][0] = x_points[index]
        index += 1

    # prepopulate
    index = 0
    for y in range(0, len(matrix), 2):
        matrix[y][0] = y_points[index]
        matrix[y + 1][0] = y_points[index]
        index += 1

    # prepopulate derivatives ( every other row )
    index = 0
    for x in range(1, len(matrix), 2):
        matrix[x][2] = slopes[index]
        index += 1

    # apply the divided differences
    apply_div_diff(matrix)

# 5. Using cubic spline interpolation, solve for the following using this set of data:
#    x - 2, 5, 8, 10; f(x) - 3, 5, 7, 9
#       (a) Find matrix A
#       (b) Vector b
#       (c) Vector c       
            

if __name__ == "__main__":
    np.set_printoptions(precision = 7, suppress = True, linewidth = 100)
    # nevilles_method()

    #fake_two()

    # 2
    x_points = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    divided_table = divided_difference(x_points, y_points)

    approximated_value = 7.3
    # final_approximation = approximating_value(divided_table, x_points, approximated_value) 

    # 4
    # hermite_interpolation()