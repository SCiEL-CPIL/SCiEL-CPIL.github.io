import numpy as np

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n_paths = len(inventory)
    
    # Base depth
    d0 = 1.0 / k
    
    # Coefficient for inventory skew
    # Heuristic: balance risk penalty vs fill probability
    # c scales the shift in depth
    # Using gamma * sigma^2 / (A * k) seems to capture the relative strength
    # We might need to tune this constant.
    c = (gamma * sigma**2) / (A * k)
    
    # Calculate depths
    # d_b = d0 - c * q
    # d_a = d0 + c * q
    
    q = inventory
    bid_depths = d0 - c * q
    ask_depths = d0 + c * q
    
    # Ensure depths are not too small (min depth constraint -3/k)
    min_depth = -3.0 / k
    # We want positive depths usually, but if c*q is large, it can go negative.
    # Negative depth means quoting inside the mid? Or invalid?
    # "Values below -3/k ... make the whole result invalid."
    # So we must clip to min_depth.
    # However, if depth is negative, it's very aggressive.
    # Let's clip to min_depth.
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Handle boundaries
    # If q == Q, no ask fill allowed. Set ask_depth to inf.
    # If q == -Q, no bid fill allowed. Set bid_depth to inf.
    bid_depths[q == Q] = np.inf
    ask_depths[q == -Q] = np.inf
    
    return bid_depths, ask_depths
