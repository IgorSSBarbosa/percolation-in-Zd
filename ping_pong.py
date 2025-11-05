import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from richardson_extrapol import parse_arguments as richardson_parse_arguments
from richardson_extrapol import richardson_extrapolation, data_loader

def translate_str_to_list(s):
    """
    Translate a string representation of a list into an actual list of integers.
    Example: "1,2,3" -> [1, 2, 3]
    """
    return [int(item) for item in s.split(',')]

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

    # ignore infinite values
    zeta_estimates = zeta_estimates[np.isfinite(zeta_estimates)]

    zeta = np.mean(zeta_estimates)
    
    return zeta

def plot_ping_pong(zeta_estimate, p_c_estimate, args, label=None):
    if label is None:
        # color the points changing to a colormap from blue to red
        c = plt.cm.coolwarm(np.linspace(0, 1, 100))
    else:
        # Each label gets a different colormap
        J = args.cancellation_index if isinstance(args.cancellation_index, int) else max(args.cancellation_index) + 1
        c = plt.cm.viridis(np.linspace(0, 1, J))

    if args.sample_plot:
        # plot 1000 random points for better visualization, but keep the order of the points
        sample_idx = np.random.choice(range(len(p_c_estimate)), size=min(1000, len(p_c_estimate)), replace=False)
        sample_idx.sort()

        for i in tqdm(sample_idx, desc="Plotting Ping-Pong Estimates"):
            plt.scatter(zeta_estimate[i], p_c_estimate[i], color=c[i % 100], s=15)
    else:
        # plot all points
        pbar = range(len(p_c_estimate))
        for i in tqdm(pbar, desc=f"Plotting Ping-Pong Estimates for Cancellation Index J={label if label else 'Single J'}"):
            if label is not None:
                plt.scatter(zeta_estimate[i], p_c_estimate[i], color=c[label % J], s=15)
                if i == 0:
                    plt.scatter(zeta_estimate[i], p_c_estimate[i], color=c[label % J], s=15, label=f"Cancellation Index J={label}")
            else:
                plt.scatter(zeta_estimate[i], p_c_estimate[i], color=c[i % 100], s=15)

    plt.xlabel('zeta Estimate')
    plt.ylabel('p_c Estimate')
    plt.title('Ping-Pong Richardson Extrapolation Estimates')
    plt.legend()
    plt.grid()

def parse_arguments(parser):
    # include the number of ping pong steps as argument
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of ping pong steps to simulate",
    )
    parser.add_argument(
        "-J",
        "--cancellation_index",
        default=1,
        help="Richardson extrapolation index for ping-pong method",
    )
    parser.add_argument(
        "--sample_plot",
        action="store_true",
        help="Sample points for plotting",
    )
    return richardson_parse_arguments(parser)

def initialize_ping_pong():
    parser = argparse.ArgumentParser(description="Richardson Extrapolation Analysis")
    args = parse_arguments(parser)
    # set the real values of p_c and zeta if known for better visualization
    args.true_p_c = 0.5
    
    # translate cancellation_index if it's a string representation of a list
    if isinstance(args.cancellation_index, str):
        args.cancellation_index = translate_str_to_list(args.cancellation_index)

    return args

def ping_pong(J, args, step, X_n):

    # initialize ping-pong estimators
    ping_pong_steps = args.steps
    p_c_estimate = np.zeros(ping_pong_steps)
    zeta_estimate = np.zeros(ping_pong_steps)

    # Initial estimate
    p_c_estimate[0] = X_n[-1]

    for step in tqdm(range(1, ping_pong_steps + 1), leave=True, desc=f"Ping-Pong Steps for J={J}"):
            
        # Estimate convergence rate
        zeta_estimate[step-1] = convergence_rate(args, X_n=X_n ,p_c = p_c_estimate[step-1])
        if step < ping_pong_steps:
            # Update Richardson extrapolated estimate for next ping-pong step
            R = richardson_extrapolation(X_n, zeta_estimate[step-1])
            p_c_estimate[step] = R[J, -1 - J]

    return p_c_estimate, zeta_estimate

def run_ping_pong(args, X_n):

    J = args.cancellation_index

    if isinstance(J, int):
        p_c_estimate, zeta_estimate = ping_pong(J, args, args.steps, X_n)
    elif isinstance(J, list):
        p_c_estimate = {}
        zeta_estimate = {}
        for j in J:
            p_c_estimate[j], zeta_estimate[j] = ping_pong(j, args, args.steps, X_n)


    # Plot results
    plt.figure(figsize=(12, 6))
    if isinstance(J, int):
        plot_ping_pong(zeta_estimate, p_c_estimate, args)
    elif isinstance(J, list):
        for j in J:
            plot_ping_pong(zeta_estimate[j], p_c_estimate[j], args, label=j)
    # plot true p_c and true zeta as a red point
    if args.true_zeta and args.true_p_c:
        plt.plot(args.true_zeta, args.true_p_c, 'rx', markersize=15, label='True Value (p_c, zeta)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    args = initialize_ping_pong()
    # Load data
    X_n, var = data_loader(args.data_path, args.parameter)

    X_n, var = np.array(X_n), np.array(var)
    run_ping_pong(args, X_n)