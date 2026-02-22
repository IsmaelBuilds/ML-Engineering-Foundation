import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    
    cost = cost / m
    
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_ / (2 * m)) * reg_cost
    
    return cost + reg_cost

print("Logistic Regression module with Regularization ready.")
