import numpy as np

# Cost Function using a boucle for
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    Args:
      x (ndarray): Shape (m,) Training data (features)
      y (ndarray): Shape (m,) Target values
      w, b (scalar): Model parameters  
    """
    m = x.shape[0] 
    cost_sum = 0 
    for i in range(m):
        # Calculate the prediction for example i
        f_wb = w * x[i] + b
        # Calculate the squared error
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    
    # Final MSE cost formula
    total_cost = (1 / (2 * m)) * cost_sum  
    return total_cost

#  Gradient Calculation (Partial Derivatives)
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression.
    Calculates the partial derivatives of the cost function w.r.t w and b.
    """
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        # Prediction and error calculation
        f_wb = w * x[i] + b 
        error = f_wb - y[i]
        
        # Accumulate the derivatives
        dj_dw = dj_dw + error * x[i]
        dj_db = dj_db + error 
        
    # Average the gradients
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

# Gradient Descent Algorithm
def gradient_descent(x, y, w_in, b_in, alpha, num_iters): 
    """
    Performs gradient descent to fit w and b.
    """
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        # Calculate the slopes (gradients)
        dj_dw, dj_db = compute_gradient(x, y, w, b)     

        # Simultaneous update of parameters
        w = w - alpha * dj_dw                            
        b = b - alpha * dj_db                            
        
        # Print cost every 100 iterations to monitor convergence
        if i % 100 == 0:
            current_cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}: Cost {current_cost:.4f}")
        
    return w, b

# Testing with Course Data
if __name__ == "__main__":
    # x: House size | y: Price
    x_train = np.array([1.0, 2.0])   
    y_train = np.array([300.0, 500.0]) 

    # Run the model starting from zero
    w_final, b_final = gradient_descent(x_train, y_train, 0, 0, 0.1, 1000)
    
    print(f"\nFinal results:")
    print(f"w: {w_final:.2f}, b: {b_final:.2f}")
