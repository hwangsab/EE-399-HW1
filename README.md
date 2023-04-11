# EE-399: Introduction to Machine Learning
#### HW 1 Submission
#### Sabrina Hwang

## Abstract:
This code was created for EE 399 Introduction to Machine Learning, HW 1 submission by Sabrina Hwang. 
The code implements a least-squares curve fitting technique to find the parameters of a given function 
that best fits a given dataset. Additionally, it generates a 2D error landscape by sweeping through 
different values of the function parameters and fixing two parameters at a time.

The accompanying Python code performs optimization and machine learning on the models, of which the 
accuracy of them are then evaluated using the least squared error calculations. 

## Introduction and Overview:
Fitting data to models remains a fundamental theme throughout optimization and machine learning processes. 
This assignment in particular exercises tasks of fitting various kinds of models to a fixed 2D dataset. 
This dataset consists of 31 points, which are then fit to a function that is the combination of a cosine
function, a linear function, and a constant value. 

## Theoretical Background:
The theoretical foundation for this code is based on the concept of linear regression, which is a 
statistical method used to analyze the relationship between two variables. In simple linear regression, 
the goal is to find a line that best fits the data points, where one variable is considered the
dependent variable and the other is considered the independent variable. The line is determined by 
minimizing the sum of squared differences between the predicted values and the actual values.

This method can be extended to multiple linear regression, where there are more than one independent 
variables. In this case, the goal is to find a plane that best fits the data points. 

The models for this assignment are fit to the data with the least-squares error equation:
$$E=\sqrt{(1/n)\sum_{j=1}^{n}(f(x_j)-y_j)^2}$$

As mentioned before, the function structure represents a combination of a cosine function, a linear 
function, and a constant, of which are determined by the parameters $A$, $B$, $C$, and $D$. This structure
can be mathematically represented by the function $$f(x)=Acos(Bx)+Cx+D$$
These parameters are then optimized with Python code. 

## Algorithm Implementation and Development:
This homework assignment works around the following dataset:
```
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```

Completion of this project and development of the algorithm was accomplished through Python as our 
primary programming language. 

### Code Description
The code is written in Python and uses the following libraries:
    `numpy` for numerical computing
    `matplotlib` for data visualization
    `math` for mathematical functions
    `scipy` for curve fitting
    
### Finding Minimum Error and Optimizing Parameters
The code reads a dataset of 31 points and defines a function to fit the data using least-squares curve 
fitting. The function func(x, A, B, C, D) is a combination of a cosine function and a linear function 
with four parameters A, B, C, D that are to be optimized. The curve_fit function from scipy library is 
used to find the optimized values of the parameters. Then, the minimum error between the function and 
the dataset is calculated, and the results are printed along with a plot of the function fit to the 
data.

### Generating 2D Error Landscape:
The code also generates a 2D error landscape by sweeping through different values of the function 
parameters and fixing two parameters at a time. The error is calculated for each combination of 
parameter values, and the results are plotted using pcolor from matplotlib library.

The code first fixes A and B parameters and sweeps through C and D parameters, then fixes A and C 
parameters and sweeps through B and D parameters, and finally fixes A and D parameters and sweeps 
through B and C parameters. The min function is used to find the minimum error and the corresponding 
parameter values.

#### Problem 1:
#### Problem 2:
#### Problem 3:
#### Problem 4:

## Computational Results:

### Usage
To run the code, simply run the Python file hw1.py in any Python environment. The output will be 
printed to the console and displayed in a pop-up window. The matplotlib library is required to display 
the 2D error landscape plot.

#### Problem 1:
#### Problem 2:
#### Problem 3:
#### Problem 4:

## Summary and Conclusions:
This code demonstrates how least-squares curve fitting can be used to find the parameters of a function 
that best fit a given dataset. Additionally, it shows how a 2D error landscape can be generated to 
visualize the relationship between the function parameters and the error. The code can be used as a 
starting point for more complex curve fitting problems and for exploring the relationship between 
different function parameters.
