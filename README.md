# EE-399

Introduction
This code was created for EE 399 Introduction to Machine Learning, HW 1 submission by Sabrina Hwang. 
The code implements a least-squares curve fitting technique to find the parameters of a given function 
that best fits a given dataset. Additionally, it generates a 2D error landscape by sweeping through 
different values of the function parameters and fixing two parameters at a time.

Code Description
The code is written in Python and uses the following libraries:
    `numpy` for numerical computing
    `matplotlib` for data visualization
    `math' for mathematical functions
    'scipy' for curve fitting
    
Finding Minimum Error and Optimizing Parameters:
The code reads a dataset of 31 points and defines a function to fit the data using least-squares curve 
fitting. The function func(x, A, B, C, D) is a combination of a cosine function and a linear function 
with four parameters A, B, C, D that are to be optimized. The curve_fit function from scipy library is 
used to find the optimized values of the parameters. Then, the minimum error between the function and 
the dataset is calculated, and the results are printed along with a plot of the function fit to the 
data.

Generating 2D Error Landscape:
The code also generates a 2D error landscape by sweeping through different values of the function 
parameters and fixing two parameters at a time. The error is calculated for each combination of 
parameter values, and the results are plotted using pcolor from matplotlib library.

The code first fixes A and B parameters and sweeps through C and D parameters, then fixes A and C 
parameters and sweeps through B and D parameters, and finally fixes A and D parameters and sweeps 
through B and C parameters. The min function is used to find the minimum error and the corresponding 
parameter values.

Usage:
To run the code, simply run the Python file hw1.py in any Python environment. The output will be 
printed to the console and displayed in a pop-up window. The matplotlib library is required to display 
the 2D error landscape plot.

Conclusion:
This code demonstrates how least-squares curve fitting can be used to find the parameters of a function 
that best fit a given dataset. Additionally, it shows how a 2D error landscape can be generated to 
visualize the relationship between the function parameters and the error. The code can be used as a 
starting point for more complex curve fitting problems and for exploring the relationship between 
different function parameters.
