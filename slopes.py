import argparse
import numpy as np
from richardson_extrapol import data_loader
from ping_pong import initialize_ping_pong, run_ping_pong
from richardson_extrapol import richardson_extrapolation,parse_arguments,plot_data,plot_inverted_data,convergence_rate,plot_results,plot_zeta_estimates

def loglog_consecutive_slopes(x, y):
    """
    Calculate slopes between consecutive points in log-log plot
    
    Parameters:
    x, y: numpy arrays of same size, x sorted with positive real numbers
    
    Returns:
    slopes: array of slopes between consecutive points in log-log space
    """
    # Input validation
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        raise ValueError("Inputs must be numpy arrays")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if len(x) < 2:
        raise ValueError("Arrays must have at least 2 elements")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("x and y must contain only positive values")
    
    # Take logarithms
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Calculate slopes between consecutive points
    # slope = (log(y[i+1]) - log(y[i])) / (log(x[i+1]) - log(x[i]))
    delta_log_y = np.diff(log_y)
    delta_log_x = np.diff(log_x)
    
    # Avoid division by zero (though unlikely with positive x values)
    slopes = delta_log_y / delta_log_x
    
    return slopes

# Example usage and test
if __name__ == "__main__":
    # load data
    data_path='simulation_data/Z2parameters.json'
    parameter='cluster_vol'
    X_n, var = data_loader(data_path, parameter)
    var = var[1:]

    x = range(1,len(X_n)+1)
    x,X_n = np.array(x), np.array(X_n)

    slopes = loglog_consecutive_slopes(x,X_n)
    """
    slope ~ -1 -1/δ which inplies δ ~ 1/(1-slope)
    """
    delta = [1/(1-slope) for slope in slopes]
    print(f'δ = {delta}')

    # Applying Richardson extrapolation on the deltas
    parser = argparse.ArgumentParser(description = "Richardson Extrapolation arguments")
    args = parse_arguments(parser)
    plot_data(delta, var, args)
    plot_inverted_data(delta,args)
    zeta, zeta_estimates = convergence_rate(args,delta)
    R = richardson_extrapolation(delta,zeta)
    plot_results( delta, R, args)
    plot_zeta_estimates(args,zeta_estimates)

    # Apply ping-pong in the slopes
    args = initialize_ping_pong()
    run_ping_pong(args, delta)
