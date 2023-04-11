## EE-399

# Abstract:
This code was created for EE 399 Introduction to Machine Learning, HW 1 submission by Sabrina Hwang. 
The code implements a least-squares curve fitting technique to find the parameters of a given function 
that best fits a given dataset. Additionally, it generates a 2D error landscape by sweeping through 
different values of the function parameters and fixing two parameters at a time.

The Python code performs a simple linear regression analysis on a dataset containing the heights and 
weights of individuals. It uses the scikit-learn library to fit a linear model to the data and make 
predictions on new data. The accuracy of the model is evaluated using the mean squared error and the 
coefficient of determination.

#Introduction:
Linear regression is a popular statistical method used to model the relationship between two variables. 
It assumes a linear relationship between the independent variable(s) and the dependent variable. 
In this code, we use linear regression to model the relationship between height and weight of individuals. 
We use the scikit-learn library, which provides a simple and efficient implementation of linear regression. 
The code takes a dataset containing height and weight measurements as input, and outputs the linear 
model parameters, as well as the mean squared error and coefficient of determination.

#Theoretical Background:
The theoretical foundation for this code is based on the concept of linear regression, which is a 
statistical method used to analyze the relationship between two variables. In simple linear regression, 
the goal is to find a line that best fits the data points, where one variable is considered the
dependent variable and the other is considered the independent variable. The line is determined by 
minimizing the sum of squared differences between the predicted values and the actual values.

This method can be extended to multiple linear regression, where there are more than one independent 
variables. In this case, the goal is to find a plane that best fits the data points. The coefficients 
of the plane can be calculated using matrix algebra, and the model can be evaluated using measures such 
as R-squared and adjusted R-squared.

#Algorithm Implementation and Development:
We used Python as our primary programming language to develop the algorithm. We utilized the Scikit-learn 
library to perform the machine learning tasks. We implemented a K-means clustering algorithm to group the 
text documents based on their similarity. The algorithm takes the preprocessed data as input, creates a 
vector representation of the documents, and groups them into k clusters. 

#Code Description:
The code is written in Python and uses the following libraries:
    `numpy` for numerical computing
    `matplotlib` for data visualization
    `math` for mathematical functions
    `scipy` for curve fitting
    
#Finding Minimum Error and Optimizing Parameters:
The code reads a dataset of 31 points and defines a function to fit the data using least-squares curve 
fitting. The function func(x, A, B, C, D) is a combination of a cosine function and a linear function 
with four parameters A, B, C, D that are to be optimized. The curve_fit function from scipy library is 
used to find the optimized values of the parameters. Then, the minimum error between the function and 
the dataset is calculated, and the results are printed along with a plot of the function fit to the 
data.

#Generating 2D Error Landscape:
The code also generates a 2D error landscape by sweeping through different values of the function 
parameters and fixing two parameters at a time. The error is calculated for each combination of 
parameter values, and the results are plotted using pcolor from matplotlib library.

The code first fixes A and B parameters and sweeps through C and D parameters, then fixes A and C 
parameters and sweeps through B and D parameters, and finally fixes A and D parameters and sweeps 
through B and C parameters. The min function is used to find the minimum error and the corresponding 
parameter values.

#Usage:
To run the code, simply run the Python file hw1.py in any Python environment. The output will be 
printed to the console and displayed in a pop-up window. The matplotlib library is required to display 
the 2D error landscape plot.

#Conclusion:
This code demonstrates how least-squares curve fitting can be used to find the parameters of a function 
that best fit a given dataset. Additionally, it shows how a 2D error landscape can be generated to 
visualize the relationship between the function parameters and the error. The code can be used as a 
starting point for more complex curve fitting problems and for exploring the relationship between 
different function parameters.
