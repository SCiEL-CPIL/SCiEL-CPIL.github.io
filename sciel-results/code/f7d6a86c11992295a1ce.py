import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    try:
        demands = np.asarray(demands, dtype=float)
        distance_matrix = np.asarray(distance_matrix, dtype=float)
        
        N = len(demands)
        
        if distance_matrix.shape != (N, N):
            return np.full((N, N), 1e-9)

        eps = 1e-9
        d = distance_matrix + eps
        
        # Base heuristic: inverse distance
        inv_dist = 1.0 / d
        
        # Demand scaling for arrival nodes (j > 0)
        # Prioritize high demand nodes to pack them early.
        demand_scale_arrival = 1.0 + demands
        
        # Demand scaling for departure nodes (i) when returning to depot (j=0)
        # Prioritize return if the current node has high demand (proxy for heavy load).
        # Note: This is a static approximation.
        
        # Initialize eta
        eta = np.zeros((N, N))
        
        # For customer-to-customer and depot-to-customer (j > 0)
        # eta[i, j] = inv_dist[i, j] * (1 + demands[j])
        for j in range(1, N):
            eta[:, j] = inv_dist[:, j] * (1.0 + demands[j])
            
        # For return to depot (j = 0)
        # eta[i, 0] = inv_dist[i, 0] * (1 + demands[i])
        # This makes it more attractive to return from high-demand nodes.
        eta[:, 0] = inv_dist[:, 0] * (1.0 + demands)
        
        # Clean up
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        
        return eta
        
    except Exception:
        try:
            N = len(demands)
        except:
            N = 0
        return np.full((N, N), 1e-9)
