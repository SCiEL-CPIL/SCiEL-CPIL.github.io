import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Heuristic matrix for ant colony optimization of capacitated vehicle routing.
    Returns an edge-attractiveness matrix (larger is more attractive).
    """
    start_time = time.time()
    DEADLINE = 320.0  # Leave 10s margin before 330s kill
    
    n = len(demands)
    D = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    d = np.asarray(demands, dtype=np.float64)
    
    eps = 1e-9
    
    # Base heuristic: inverse distance (standard ACO heuristic)
    # Using 1/d rather than 1/d^2 can sometimes provide better exploration
    D_safe = np.where(D > eps, D, 1.0)
    base = 1.0 / D_safe
    
    # Savings-based heuristic: reward edges that save distance
    # S_ij = D_0i + D_0j - D_ij
    D0 = D[0, :]
    D0_i = D0[:, np.newaxis]
    D0_j = D0[np.newaxis, :]
    S = D0_i + D0_j - D
    
    # Normalize savings to [0, 1] range roughly
    Denom = D0_i + D0_j
    Denom_safe = np.where(Denom > eps, Denom, 1.0)
    S_norm = np.where(S > 0, S / Denom_safe, 0.0)
    
    # Demand-aware heuristic: prefer visiting customers with higher demand
    # when remaining capacity is limited (though this is static, so we use
    # a mild bias)
    d_j = d[np.newaxis, :]
    max_demand = max(d.max(), 1.0)
    demand_bias = 1.0 + 0.2 * (d_j / max_demand)
    
    # Combine: base distance heuristic * savings boost * mild demand bias
    eta = base * (1.0 + 1.5 * S_norm) * demand_bias
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    eta = np.maximum(eta, 1e-9)
    
    # For the depot row/col, we want to encourage returning to depot
    # when the tour is complete or capacity is exceeded.
    # The depot's heuristic values should be moderate.
    
    return eta
