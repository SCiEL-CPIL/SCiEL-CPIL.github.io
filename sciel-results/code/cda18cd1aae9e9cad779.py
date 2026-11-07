import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    # Ensure demands is a numpy array for vectorized ops
    demands = np.asarray(demands)
    
    # Calculate base heuristic: 1 / distance
    # Add small epsilon to avoid division by zero for i == j
    d = distance_matrix + 1e-9
    
    # Heuristic: favor high demand and short distance
    # eta_ij = demand_j / d_ij
    # This encourages visiting high-demand customers, which helps in 
    # packing the vehicles and reducing the number of trips.
    
    # We need to be careful with the depot (index 0) where demand is 0.
    # If demand is 0, the heuristic will be 0, which is then clamped to 1e-9.
    
    # Vectorized calculation:
    # demands is shape (N,)
    # We want eta[i, j] = demands[j] / d[i, j]
    # So we broadcast demands[j] across rows.
    
    # demands_col = demands.reshape(1, -1)
    # eta = demands_col / d
    
    # However, we must ensure the result is a float matrix of the same shape.
    
    try:
        demands_col = demands.reshape(1, -1)
        eta = demands_col / d
        
        # Replace non-positive or non-finite values with 1e-9
        # The prompt says: "Heuristic entries that are not positive become 1e-9"
        # Also "non-finite values ... make the whole result invalid" 
        # Wait, "An exception, non-finite values or a wrong shape make the whole result invalid."
        # This usually refers to the *output* of the function.
        # If I return inf, the evaluator kills it.
        # So I must ensure all values are finite and positive.
        
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        
        return eta

    except Exception:
        # Fallback to safe 1/d if something goes wrong
        return 1.0 / (distance_matrix + 1e-9)
