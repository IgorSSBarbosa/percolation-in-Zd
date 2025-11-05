import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from richardson_extrapol import data_loader

def loglog_regression(x, y):
    """
    Calculate power law coefficients from log-log linear regression
    
    Parameters:
    x, y: numpy arrays of same size, x sorted with positive real numbers
    
    Returns:
    a_hat, b_hat, r_squared: coefficients for y = b * x^a
    """
    # Input validation
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        raise ValueError("Inputs must be numpy arrays")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("x and y must contain only positive values")
    
    # Take logarithms
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Perform linear regression on log-transformed data
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    # Convert back to power law coefficients
    a_hat = slope  # exponent
    b_hat = np.exp(intercept)  # coefficient
    r_squared = r_value**2
    
    return a_hat, b_hat, r_squared

def plot_loglog_regression(x, y, a_hat, b_hat, r_squared, parameter):
    """
    Plot both log-log plot and original data with linear regression
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Log-log plot with linear regression
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Generate points for regression line in log space
    x_log_fit = np.linspace(log_x.min(), log_x.max(), 100)
    y_log_fit = a_hat * x_log_fit + np.log(b_hat)
    
    ax1.scatter(log_x, log_y, alpha=0.7, label=f'{parameter}')
    ax1.plot(x_log_fit, y_log_fit, 'r-', linewidth=2, 
             label=f'Linear fit: y = {a_hat:.3f}x + {np.log(b_hat):.3f}')
    ax1.set_xlabel('log(x)')
    ax1.set_ylabel('log(y)')
    ax1.set_title(f'Log-Log Plot for {parameter}\n$R^2 = {r_squared:.4f}$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Original data with power law fit
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = b_hat * x_fit**a_hat
    
    ax2.scatter(x, y, alpha=0.7, label='Original data')
    ax2.plot(x_fit, y_fit, 'r-', linewidth=2, 
             label=f'Power law: $y = {b_hat:.3f}x^{{{a_hat:.3f}}}$')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Original Data with Power Law Fit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage and test
if __name__ == "__main__":

    # load data
    data_path='simulation_data/Z2parameters.json'
    parameter='cluster_vol'
    X_n, var = data_loader(data_path, parameter)
    x = range(1,len(X_n)+1)
    x,X_n = np.array(x), np.array(X_n)
    
    # Calculate coefficients
    a_hat, b_hat, r_squared = loglog_regression(x, X_n)
    
    print("Power Law Regression Results:")
    # print(f"True coefficients: a = {true_a}, b = {true_b}")
    print(f"Estimated coefficients: a_hat = {a_hat:.4f}, b_hat = {b_hat:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    
    # Plot results
    plot_loglog_regression(x, X_n, a_hat, b_hat, r_squared, parameter)