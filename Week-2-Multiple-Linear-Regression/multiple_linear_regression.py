import numpy as np

def compute_cost(X, y, w, b):
    """
    X: ndarray (m,n)
    y: ndarray (m,)
    w: ndarray (n,)
    b: scalar
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b              
        cost = cost + (f_wb_i - y[i])**2              
    cost = cost / (2 * m)                         
    return cost

# Exemple d'initialisation
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
w_init = np.array([0.39, 18.7, -53.3, -26])
b_init = 785.18

print(f"Cost with initial w, b: {compute_cost(X_train, y_train, w_init, b_init)}")
