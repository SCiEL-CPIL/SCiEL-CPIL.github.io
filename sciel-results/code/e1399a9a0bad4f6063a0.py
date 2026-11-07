import numpy as np

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    tau = T - t
    
    # Heuristic based on Avellaneda-Stoikov style approximation
    # d_b = 1/k - Delta_-
    # d_a = 1/k - Delta_+
    # Delta_- approx -gamma * sigma^2 * tau * (q - 0.5)
    # Delta_+ approx -gamma * sigma^2 * tau * (q + 0.5)
    
    q = inventory
    coeff = gamma * sigma**2 * tau
    
    d_b = 1.0 / k - coeff * (q - 0.5)
    d_a = 1.0 / k - coeff * (q + 0.5)
    
    # Handle boundaries
    # If q == Q, cannot bid. Set d_b to inf.
    # If q == -Q, cannot ask. Set d_a to inf.
    
    d_b = np.where(q == Q, np.inf, d_b)
    d_a = np.where(q == -Q, np.inf, d_a)
    
    # Clip to valid range?
    # "Values below -3/k ... invalid"
    # So d >= -3/k.
    # Also, if d is very large, it's fine (inf is allowed).
    # Let's clip lower bound.
    d_b = np.maximum(d_b, -3.0 / k)
    d_a = np.maximum(d_a, -3.0 / k)
    
    return d_b, d_a
