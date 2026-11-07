import numpy as np

def heuristics(distance_matrix):
    """
    Computes a power-law inverse distance heuristic matrix for the TSP.
    
    The heuristic value for an edge (i, j) is inversely proportional to its distance
    raised to a power beta. A beta > 1 increases the bias towards shorter edges.
    """
    d = np.asarray(distance_matrix, dtype=float)
    
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
        
    n = d.shape[0]
    d_work = d.copy()
    
    # Set diagonal to infinity so 1/d becomes 0
    np.fill_diagonal(d_work, np.inf)
    
    # Avoid division by zero for off-diagonal zeros (unlikely in random points, but safe)
    # Replace 0 with a very small number or inf?
    # If d is 0, 1/0 is inf. We want to avoid inf in heuristic if possible, 
    # but the task says non-finite values make result invalid? 
    # "An exception, non-finite values or a wrong shape make the whole result invalid."
    # So we MUST avoid inf or nan in the output.
    
    # If d_work has 0s off-diagonal, 1/0 is inf.
    # Let's replace 0s with a very small positive number, e.g., 1e-9? 
    # Or just clip d_work to a minimum value.
    min_d = 1e-9
    d_work = np.where(d_work > min_d, d_work, min_d)
    # But diagonal is inf, so it stays inf.
    
    # Exponent beta
    beta = 2.0
    
    # Calculate heuristic
    # h = 1 / d^beta
    h = 1.0 / np.power(d_work, beta)
    
    # Diagonal is 1/inf = 0.
    
    # Normalize to mean 1 over off-diagonal elements
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Calculate mean of valid entries
    valid_h = h[mask]
    if len(valid_h) > 0:
        mean_h = np.mean(valid_h)
        if mean_h > 0:
            h = h / mean_h
        else:
            h = np.ones_like(h)
    else:
        h = np.ones_like(h)
        
    # Ensure non-finite values are handled
    # If any inf or nan slipped in, replace with 1e-9 or 0?
    # The task says non-finite values make result invalid.
    # So we must ensure output is finite.
    
    # Replace any non-finite values with 0 (which will become 1e-9)
    h = np.where(np.isfinite(h), h, 0.0)
    
    # Entries that are not positive become 1e-9
    h = np.where(h > 0, h, 1e-9)
    
    return h
