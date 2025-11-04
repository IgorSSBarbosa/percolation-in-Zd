import json
import matplotlib.pyplot as plt
import numpy as np

def cluster_density_eval(p, cluster_density_file):
    """ Evaluate the number of cluster by vertex n(s,p) in p"""
    # load the polynomials
    with open(cluster_density_file, 'r') as f:
        data = json.load(f)
    cluster_density_poly = data["cluster_density"]

    # evaluate at p
    y = dict()
    for (key, poly_str) in cluster_density_poly.items():
        try:
            # Replace p** with p** and handle the expressions properly
            # We'll create a safe evaluation function
            def safe_eval(p_val):
                # Replace p with the actual value in the expression
                expr = poly_str.replace('p', f'({p_val})')
                return eval(expr)
            
            # Vectorize the function to work with arrays
            vec_eval = np.vectorize(safe_eval)
            y[key] = vec_eval(p)

        except Exception as e:
            print(f'Error processing polynomial {key}: {e}')
            continue

    return y

def theta(y, p):
    v = [0]*len(y)
    for (key, value) in y.items():
        v[int(key)-1] = int(key)*value
    theta_compl = sum(v)
    theta = [1 - x for x in theta_compl]
    plt.figure(figsize=(16,12))
    plt.plot(p, theta, label='θ(p)', linewidth=2)
    plt.xlabel('p',fontsize=12)
    plt.ylabel('θ(p)',fontsize=12)
    plt.title(f'Percolation Probability, s={len(y)}',fontsize=14)
    plt.legend()
    return theta

def mean_cluster_size(y,p):
    v = [0]*len(y)
    for (key, value) in y.items():
        v[int(key)-1] = (int(key)**2)*value

    qui = sum(v)
    plt.figure(figsize=(16,12))
    plt.plot(p, qui, label='χ_f(p)', linewidth=2)
    plt.xlabel('p',fontsize=12)
    plt.ylabel('χ_f(p)',fontsize=12)
    plt.title(f'Mean cluster density, s={len(y)}',fontsize=14)
    plt.legend()
    return qui

def critical_parameter(y, numb_points):
    v = [0]*len(y)
    p_c = [0]*len(y)
    for (key, value) in y.items():
        v[int(key)-1] = int(key)*value
    p_c = np.argmax(v, axis=1)/numb_points

    plt.figure(figsize=(16,12))
    plt.plot(p_c, label='Critical point', linewidth=2)
    plt.xlabel('s',fontsize=12)
    plt.ylabel('p_c',fontsize=12)
    plt.title(f'Critical Parameter, s={len(y)}',fontsize=14)
    plt.legend()
    return p_c
    


if __name__ == '__main__':
    cluster_density_file = 'Z2clusterdensity.json'
    p_range = [0,1]
    num_points = 1000
    p = np.linspace(p_range[0], p_range[1], num_points)

    y = cluster_density_eval(p, cluster_density_file)
    theta = theta(y,p)
    qui = mean_cluster_size(y,p)

    p_c = critical_parameter(y,num_points)
    plt.show()