#!/usr/bin/env python
# coding: utf-8

# # EE 399 Introduction to Machine Learning: HW 1 Submission
# ### Sabrina Hwang

# In[92]:


import numpy as np
import matplotlib.pyplot as plt
import math as math

from scipy.optimize import curve_fit


# In[93]:


X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])


# ### II (i)
# Write a code to find the minimum error and determine the parameters A, B, C, D

# In[94]:


def func(x, A, B, C, D):
    return A*np.cos(B*x) + C*x + D

popt, pcov = curve_fit(func, X, Y)

A, B, C, D = popt

error = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

print("Minimum error:", error)
print("Optimized values:")
print("     A =", A)
print("     B =", B)
print("     C =", C)
print("     D =", D)

plt.scatter(X, Y)
plt.plot(X, func(X, *popt), 'r-', label='fit')
plt.legend()
plt.show()


# ### II (ii) 
# With the results of (i), fix two of the parameters and sweep through values of the other two parameters to generate a 2D loss (error) landscape. Do all combinations of two fixed parameters and two swept parameters. You can use something like pcolor to visualize the results in a grid. How many minima can you find as you sweep through parameters?

# In[95]:


fig, axs = plt.subplots(3, 2, figsize=(10, 12))
fig.suptitle('2D Error Landscape')
plt.subplots_adjust(hspace=0.5)

A = 1.0
B = 0.1

# Fix parameters A and B, sweep C and D
C_range = np.linspace(-5, 5, 100)
D_range = np.linspace(30, 60, 100)
C_grid, D_grid = np.meshgrid(C_range, D_range)
error_grid = np.zeros_like(C_grid)
for i in range(len(C_range)):
    for j in range(len(D_range)):
        C = C_range[i]
        D = D_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.subplot(321)
plt.pcolor(D_grid, C_grid, error_grid)
plt.xlabel('D')
plt.ylabel('C')
plt.title('Fix parameters A and B (A = {:.1f}, B = {:.1f})'.format(A, B))
plt.colorbar()

# Fix parameters A and C, sweep B and D
B_range = np.linspace(0.01, 1.0, 100)
D_range = np.linspace(30, 60, 100)
B_grid, D_grid = np.meshgrid(B_range, D_range)
error_grid = np.zeros_like(B_grid)
for i in range(len(B_range)):
    for j in range(len(D_range)):
        B = B_range[i]
        D = D_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.subplot(322)
plt.pcolor(D_grid, B_grid, error_grid)
plt.xlabel('D')
plt.ylabel('B')
plt.title('Fix parameters A and C (A = {:.1f}, C = {:.1f})'.format(A, C))
plt.colorbar()

# Fix parameters A and D, sweep B and C
B_range = np.linspace(0.01, 1.0, 100)
C_range = np.linspace(-5, 5, 100)
B_grid, C_grid = np.meshgrid(B_range, C_range)
error_grid = np.zeros_like(B_grid)
for i in range(len(B_range)):
    for j in range(len(C_range)):
        B = B_range[i]
        C = C_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.subplot(323)
plt.pcolor(C_grid, B_grid, error_grid)
plt.xlabel('C')
plt.ylabel('B')
plt.title('Fix parameters A and D (A = {:.1f}, D = {:.1f})'.format(A, D))
plt.colorbar()

# Fix parameters B and C, sweep A and D
A_range = np.linspace(0.1, 2.0, 100)
D_range = np.linspace(30, 60, 100)
A_grid, D_grid = np.meshgrid(A_range, D_range)
error_grid = np.zeros_like(A_grid)
for i in range(len(A_range)):
    for j in range(len(D_range)):
        A = A_range[i]
        D = D_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.subplot(324)
plt.pcolor(D_grid, A_grid, error_grid)
plt.xlabel('D')
plt.ylabel('A')
plt.title('Fix parameters B and C (B = {:.1f}, C = {:.1f})'.format(B, C))
plt.colorbar()

# Fix parameters B and D, sweep A and C
A_range = np.linspace(0.1, 2.0, 100)
C_range = np.linspace(-5, 5, 100)
A_grid, C_grid = np.meshgrid(A_range, C_range)
error_grid = np.zeros_like(A_grid)
for i in range(len(A_range)):
    for j in range(len(C_range)):
        A = A_range[i]
        C = C_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.subplot(325)
plt.pcolor(C_grid, A_grid, error_grid)
plt.xlabel('C')
plt.ylabel('A')
plt.title('Fix parameters B and D (B = {:.1f}, D = {:.1f})'.format(B, D))
plt.colorbar()

# Fix parameters C and D, sweep A and B
A_range = np.linspace(0.1, 2.0, 100)
B_range = np.linspace(0.01, 1.0, 100)
A_grid, B_grid = np.meshgrid(A_range, B_range)
error_grid = np.zeros_like(A_grid)
for i in range(len(A_range)):
    for j in range(len(B_range)):
        A = A_range[i]
        B = B_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.subplot(326)
plt.pcolor(B_grid, A_grid, error_grid)
plt.xlabel('B')
plt.ylabel('A')
plt.title('Fix parameters C and D (C = {:.1f}, D = {:.1f})'.format(C, D))
plt.colorbar()

plt.show()


# ### II (iii) 
# Using the first 20 data points as training data, fit a line, parabola and 19th degree
# polynomial to the data. Compute the least-square error for each of these over the training
# points. Then compute the least square error of these models on the test data which are
# the remaining 10 data points.

# In[96]:


import numpy as np
import matplotlib.pyplot as plt

# data
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

# split into train and test data
X_train = X[:20]
Y_train = Y[:20]
X_test = X[20:]
Y_test = Y[20:]

# fit line, parabola, and 19th degree polynomial
line_coeffs = np.polyfit(X_train, Y_train, 1)
parabola_coeffs = np.polyfit(X_train, Y_train, 2)
poly_coeffs = np.polyfit(X_train, Y_train, 19)

# compute predictions on train and test data
Y_line_train = np.polyval(line_coeffs, X_train)
Y_parabola_train = np.polyval(parabola_coeffs, X_train)
Y_poly_train = np.polyval(poly_coeffs, X_train)

Y_line_test = np.polyval(line_coeffs, X_test)
Y_parabola_test = np.polyval(parabola_coeffs, X_test)
Y_poly_test = np.polyval(poly_coeffs, X_test)

# compute least square error on train and test data
line_train_error = np.sum((Y_line_train - Y_train)**2)
parabola_train_error = np.sum((Y_parabola_train - Y_train)**2)
poly_train_error = np.sum((Y_poly_train - Y_train)**2)

line_test_error = np.sum((Y_line_test - Y_test)**2)
parabola_test_error = np.sum((Y_parabola_test - Y_test)**2)
poly_test_error = np.sum((Y_poly_test - Y_test)**2)

# plot data and fits
fig, ax = plt.subplots()
ax.plot(X_train, Y_train, 'bo', label='Training Data')
ax.plot(X_test, Y_test, 'ro', label='Test Data')
ax.plot(X_train, Y_line_train, 'g', label='Line Fit (Train)')
ax.plot(X_train, Y_parabola_train, 'c', label='Parabola Fit (Train)')
ax.plot(X_train, Y_poly_train, 'm', label='19th Degree Polynomial Fit (Train)')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()

# print least square errors
print("Line Train Error:", line_train_error)
print("Parabola Train Error:", parabola_train_error)
print("19th Degree Polynomial Train Error:", poly_train_error)
print("Line Test Data Error:", line_test_error)
print("Parabola Test Data Error:", parabola_test_error)
print("19th Degree Polynomial Test Data Error:", poly_test_error)


# ### II (iv) 
# Repeat (iii) but use the first 10 and last 10 data points as training data. Then fit the
# model to the test data (which are the 10 held out middle data points). Compare these
# results to (iii)

# In[98]:


import numpy as np
import matplotlib.pyplot as plt

# define the data
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

# split the data into training and test sets
X_train = np.concatenate((X[:10], X[-10:]))
Y_train = np.concatenate((Y[:10], Y[-10:]))
X_test = X[10:20]
Y_test = Y[10:20]

# fit a line to the training data
coefficients_line = np.polyfit(X_train, Y_train, 1)
line_fit = np.poly1d(coefficients_line)

# fit a parabola to the training data
coefficients_parabola = np.polyfit(X_train, Y_train, 2)
parabola_fit = np.poly1d(coefficients_parabola)

# fit a 19th degree polynomial to the training data
coefficients_poly = np.polyfit(X_train, Y_train, 19)
poly_fit = np.poly1d(coefficients_poly)

# evaluate the models on the test data
line_error = np.sum((Y_test - line_fit(X_test)) ** 2)
parabola_error = np.sum((Y_test - parabola_fit(X_test)) ** 2)
poly_error = np.sum((Y_test - poly_fit(X_test)) ** 2)

# plot the results
plt.plot(X_train, Y_train, 'bo', label='Training Data')
plt.plot(X_test, Y_test, 'ro', label='Test Data')
plt.plot(X_train, line_fit(X_train), 'r-', label=f'Line Fit (error={line_error:.2f})')
plt.plot(X_train, parabola_fit(X_train), 'g-', label=f'Parabola Fit (error={parabola_error:.2f})')
plt.plot(X_train, poly_fit(X_train), 'm-', label=f'19th Degree Poly Fit (error={poly_error:.2f})')
plt.legend(loc='upper left')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Model Fits')
plt.show()

