import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

def parse_arguments(parser):
    parser.add_argument(
        "--data_path",
        type=str,
        default='simulation_data/Z2parameters.json',
    )
    parser.add_argument(
        "--parameter",
        type=str,
        default='p_c2',
        help='parameter or function which you mmay want to load the data.'
    )
    parser.add_argument(
        "--basex",
        type=float,
        default=4,
    )
    parser.add_argument(
        "--basey",
        type=float,
        default=4,
    )
    parser.add_argument(
        "--ylimits",
        type=float,
        nargs=2,
        default=[0.4, 0.7],
        help="Y-axis limits for the plot (min, max)"
    )
    parser.add_argument(
        "--true_p_c",
        "-pc",
        type=float,
        default=None,
        help="True value of p_c to accelerate convergence of Zeta",
    )
    parser.add_argument(
        "--true_zeta",
        "-zeta",
        type=float,
        default=None,
        help="True value of zeta to accelerate convergence",
    )
    parser.add_argument(
        "--richardson_cancel",
        "-r",
        type = int,
        default=None,
        help="Number of Richardson Extrapolations cancalations will be printed"
    )
    parser.add_argument(
        "--type_expansion",
        "-type",
        choices=['exponential', 'power_law'],
        help='Chooses the type of expansion assumed to calculate the right bias correction estimator',
        default='exponential',
    )

    return parser.parse_args()

def richardson_extrapolation(X_n, zeta):
    """
    Perform Richardson extrapolation to improve the estimate of a limit.

    Parameters:
    X_n : array-like
        The function values at the current step.
    zeta : float
        The estimated convergence rate.

    Returns:
    float
        The improved estimate after Richardson extrapolation.
    """
    # Create a table to hold the extrapolated values
    R = np.zeros((len(X_n), len(X_n)))
    num_extrapolations = len(X_n)
    # Compute the first column of the table
    for i in range(num_extrapolations):
        R[i, 0] = X_n[i]

    # Fill in the rest of the table using Richardson's formula
    for j in range(1, num_extrapolations):
        for i in range(num_extrapolations - j):
            R[i, j] = (np.pow(zeta, j)*R[i + 1, j - 1] - R[i, j - 1]) / (np.pow(zeta, j) - 1)

    return R

def data_loader(data_path, parameter):
    with open(data_path,'r') as f:
        data_dict = json.load(f)
    data = data_dict[parameter]
    if parameter in ['p_c2']:
        data_np = np.array(data[1:])
    else:
        data_np = np.array(data)
    
    # if dimension is one no mean need to be taken
    dim = len(data_np.shape)
    if dim>1:
        X_n = np.mean(data, axis=1)
        var = np.var(data, axis=1, ddof=1)
    else:
        X_n = data_np
        var = [0]*data_np.shape[0]

    return X_n, var


def convergence_rate(args, X_n, p_c=None):
    """
    Estimate the convergence rate (zeta) from the sequence of estimates X_n.
    Suppose |X_n - p_c| ~ C * zeta^(-n) for some constant C.

    Parameters:
    X_n : array-like
        The function values at the current step.
    p_c : float, optional
        The true limit value, if known. If provided, it will be used to compute zeta.

    Returns:
    float
        The estimated convergence rate (zeta).
    """

    # Estimate zeta from the data
    if p_c is None:
        deltas = np.abs(np.diff(X_n))
        zeta_estimates = deltas[:-1] / deltas[1:]
    else:
        deltas = np.abs(X_n - p_c)
        zeta_estimates = deltas[:-1] / deltas[1:]

    zeta = np.mean(zeta_estimates)
    # If the true zeta is provided, use it directly
    if args.true_zeta is not None:
        zeta = args.true_zeta
    
    return zeta, zeta_estimates

def plot_inverted_data(X_n, args):
    '''
    Plot the inverted data and its variance.

    '''
    domain = [1/k for k in range(1, len(X_n)+1)]
    plt.figure(figsize=(16, 10))
    plt.plot(domain, np.array(X_n), 'o-')
    plt.xlabel('1/k')
    plt.xlim( 1/(len(X_n)+2) , 1.1) # set x limit to be between 0 and len(X_n)-1
    plt.ylabel(f'{args.parameter}in log scale')
    plt.xscale('log')
    plt.title('Original Data')
    if args.true_p_c is not None:
        plt.axhline(y=args.true_p_c, color='r', linestyle='--', label='True Limit: {:.4f}'.format(args.true_p_c))
        plt.legend()
    plt.grid()
    plt.show()

    

def plot_data(X_n, var, args):
    '''
    Plot the original data and its variance.
    '''

    plt.figure(figsize=(16, 10))
    plt.plot(range(len(X_n)), X_n, 'o-')
    plt.fill_between(range(len(X_n)), X_n - np.sqrt(var), X_n + np.sqrt(var), color='gray', alpha=0.3, label='1 Std Dev')   
    plt.xlabel('Index')
    plt.xscale('log')
    plt.xlim(0, len(X_n)) # set x limit to be between 0 and len(X_n)-1
    plt.ylabel(f'{args.parameter} in log scale')
    plt.title('Original Data')
    if args.true_p_c is not None:
        plt.axhline(y=args.true_p_c, color='r', linestyle='--', label='True Limit: {:.4f}'.format(args.true_p_c))
        plt.legend()
    plt.grid()
    plt.show()

def plot_results(X_n, R, args, p_c=None):
    plt.figure(figsize=(16, 10))
    plt.plot(range(len(X_n)), X_n, 'o-', label='Original Estimates')
    # Ploting only J cancelations if this number is provided or plotting all cancelations possible if not
    if args.richardson_cancel is not None:
        J = args.richardson_cancel
    else:
        J = len(X_n)

    for i in range(1, J):
        plt.plot(range(i,len(X_n)), R[:len(X_n)-i, i], 'o--', label=f'Richardson Extrapolated {i}')
    if p_c is not None:
        plt.axhline(y=p_c, color='r', linestyle='--', label='True Limit: {:.4f}'.format(p_c))
    plt.xlabel('Index')
    plt.xlim(0, len(X_n)) # set x limit to be between 0 and len(X_n)-1
    plt.ylabel(f'{args.parameter}')
    plt.title('Richardson Extrapolation')
    plt.legend()
    plt.grid()
    plt.show()

def plot_zeta_estimates(args, zeta_estimates):
    plt.figure(figsize=(16, 12))
    plt.plot(range(1, len(zeta_estimates) + 1), zeta_estimates, 'o-')
    plt.axhline(y=np.mean(zeta_estimates), color='r', linestyle='--', label='Mean Zeta: {:.4f}'.format(np.mean(zeta_estimates)))
    if args.true_zeta is not None:
        plt.axhline(y=args.true_zeta, color='g', linestyle='--', label='True Zeta: {:.4f}'.format(args.true_zeta))    
    plt.xlabel('Index')
    plt.ylabel('Zeta Estimates')
    plt.title('Convergence Rate (Zeta) Estimates')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description = "Richardson Extrapolation arguments")
    args = parse_arguments(parser)
    data_path = args.data_path

    # Load data
    X_n, var = data_loader(data_path, args.parameter)
    
    plot_data(X_n, var, args)
    plot_inverted_data(X_n, args)
    
    # Estimate convergence rate
    zeta, zeta_estimates = convergence_rate(args, X_n, p_c=args.true_p_c)

    R = richardson_extrapolation(X_n, zeta=zeta)

    plot_results(X_n, R, args, p_c=args.true_p_c)
    plot_zeta_estimates(args, zeta_estimates)


if __name__ == "__main__":
    
    main()

