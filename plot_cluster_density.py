import numpy as np
import matplotlib.pyplot as plt
import json


def plot_polynomials(poly_dict, p_range=(0, 1), num_points=1000, title='Cluster density'):
    """
    Plot polynomial functions from a dictionary.
    
    Parameters:
    poly_dict (dict): Dictionary with keys as labels and values as polynomial strings
    p_range (tuple): Range of p values to plot (default: (0, 1))
    num_points (int): Number of points to evaluate (default: 1000)
    """
    
    # Create array of p values
    p = np.linspace(p_range[0], p_range[1], num_points)
    
    # Set up the plot
    plt.figure(figsize=(16, 12))
    
    # Define a color cycle for distinct colors
    colors = plt.cm.autumn(np.linspace(0, 1, len(poly_dict)))
    
    # Process and plot each polynomial
    for (key, poly_str), color in zip(poly_dict.items(), colors):
        # Replace p** with p^ for numpy compatibility and handle the expressions
        # Convert the string to a lambda function that can be evaluated
        try:
            # Replace p** with p** and handle the expressions properly
            # We'll create a safe evaluation function
            def safe_eval(p_val):
                # Replace p with the actual value in the expression
                expr = poly_str.replace('p', f'({p_val})')
                return eval(expr)
            
            # Vectorize the function to work with arrays
            vec_eval = np.vectorize(safe_eval)
            y = vec_eval(p)
            
            # Plot with label and color
            plt.plot(p, y, label=f'k_{key}(p)', color=color, linewidth=2)
            
        except Exception as e:
            print(f"Error processing polynomial {key}: {e}")
            continue
    
    # Customize the plot
    plt.xlabel('p', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.85)
    # plt.xlim(p_range)
    

# Example usage with your input
if __name__ == "__main__":
    
    cluster_density_file = 'Z2clusterdensity.json'
    with open(cluster_density_file, 'r') as f:
        data = json.load(f)

    cluster_density_poly = data['cluster_density']
    derivative = data['derivative']
    
    plot_polynomials(cluster_density_poly)
    plot_polynomials(derivative,title='Derivative')
    # Show the plot
    plt.tight_layout()
    plt.show()