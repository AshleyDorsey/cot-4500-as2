# cot-4500-as2
 
#### By **Ashley Dorsey**

#### Assignment Covering Topics from Chapter 3

## Technologies Used

- Python

- GitHub

## Description

Question 1 uses Neville's Method to approximate the value associated with x = 3.7. By creating the matrix associated with the x- and y-values, it is easy to find this value as it is the last value in the matrix (last row and column).

Question 2 and 3 use Newton's Forward Method to find polynomial and value approximations. Question 2 focuses on finding the polynomial approximations for the first, second, and third degrees by using the x- and y- values to find the values associated with each degree (equations found in the notes). Question 3 uses the values found in Question 2 to approximate the value associated with x = 7.3, meaning we put the polynomial approximations into an equation (multiplying each by the range of x-values that affect each approximation) in order to finally produce a final approximation.

Question 4 creates the Hermite Interpolation matrix, when given the x-values, y-values, and slopes. It then uses divided difference to retrieve the rest of the matrix values, and has a size double the lens of the x-values.

Question 5 uses cubic spline to create one matrix and two vectors. The matrix is made up of values of 0, 1, ranges of x's, and double multiple ranges of x's. The first vector is found by using the equation (3 / (x[index] - x[index])) * (y[index] - y[index]) - (3 / x[index] - x[index]) * (y[index] - y[index]). The index changes based on the range you are looking for for each part. The second vector takes the matrix and vector we already have and solves for the system of equations that they produce.

## Setup/Installation Requirements

- Create a requirements.txt. This can be done by typing into the terminal "pip freeze > requirements.txt".

- Install numpy by typing into the terminal "pip install numpy". However, if this is already installed, you may not need to install again.

- Install scipy by typing into the terminal "python -m pip install numpy." Then use this to acquire solve by typing "from scipy.linalg import solve" at the top of the code where the other imports are.

- At the top of the main function set up the print requirements by typing "np.setprintoptions(precision = 7, suppress = True, linewidth = 100)."

- In the command line, enter "python assignment_2.py" and then hit ENTER to run.