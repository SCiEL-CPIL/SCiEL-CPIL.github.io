"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Build an edge-attractiveness matrix by simulating ACO pheromone dynamics
    with a cooling schedule, combined with structural cycle-cover information.
    
    Strategy:
    1. Simulate a short ACO run (few ants, few iterations) to estimate which
       edges receive the most pheromone reinforcement under the actual routing
       constraints (capacity, depot returns).
    2. Use the final pheromone distribution as the primary heuristic signal.
    3. Blend with a distance-based term and cycle-cover structural term.
    4. Apply a cooling schedule to the heuristic exponent to smooth the result.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    n = distance_matrix.shape[0]
    d = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    capacity = float(capacity)
    
    # Ensure finite distances
    d = np.where(np.isfinite(d), d, 1e6)
    d = np.maximum(d, 1e-9)
    
    # --- Step 1: Simulate ACO pheromone accumulation ---
    # Use a small number of ants and iterations to estimate pheromone trajectories
    n_ants_sim = 10
    n_iter_sim = 15
    rho = 0.9  # pheromone decay
    alpha = 1.0  # pheromone weight
    beta = 1.0  # heuristic weight (we'll use a simple distance heuristic during sim)
    
    # Initial heuristic for simulation: inverse distance
    sim_heuristic = 1.0 / (d + 1e-9)
    np.fill_diagonal(sim_heuristic, 0.0)
    
    # Pheromone initialization
    tau = np.ones((n, n), dtype=np.float64)
    
    # Accumulate pheromone deposits
    tau_accum = np.zeros((n, n), dtype=np.float64)
    
    for it in range(n_iter_sim):
        # Check deadline
        if time.time() > deadline - 5.0:
            break
        
        # Each ant constructs a tour
        for ant in range(n_ants_sim):
            current = 0  # depot
            remaining_cap = capacity
            visited = [0]
            tour_edges = []
            
            while True:
                # Find unvisited customers that fit
                candidates = []
                for j in range(1, n):
                    if j not in visited and demands[j] <= remaining_cap:
                        candidates.append(j)
                
                if not candidates:
                    # Return to depot
                    tour_edges.append((current, 0))
                    current = 0
                    remaining_cap = capacity
                    visited = [0]
                    
                    # Check if all customers visited
                    if len(visited) == n:
                        break
                    continue
                
                # Compute probabilities
                probs = np.zeros(len(candidates))
                for idx, j in enumerate(candidates):
                    probs[idx] = (tau[current, j] ** alpha) * (sim_heuristic[current, j] ** beta)
                
                total = np.sum(probs)
                if total < 1e-30:
                    probs = np.ones(len(candidates)) / len(candidates)
                    total = 1.0
                else:
                    probs /= total
                
                # Select next customer
                r = np.random.uniform()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(candidates):
                    idx = len(candidates) - 1
                
                next_node = candidates[idx]
                tour_edges.append((current, next_node))
                current = next_node
                remaining_cap -= demands[current]
                visited.append(current)
            
            # Compute tour cost
            tour_cost = 0.0
            for i, j in tour_edges:
                tour_cost += d[i, j]
            
            # Deposit pheromone
            deposit = 1.0 / (tour_cost + 1e-9)
            for i, j in tour_edges:
                tau_accum[i, j] += deposit
        
        # Update pheromone
        tau = tau * rho + tau_accum
        tau_accum = np.zeros((n, n), dtype=np.float64)
        
        # Normalize to prevent overflow/underflow
        tau_max = np.max(tau)
        if tau_max > 0:
            tau /= tau_max
    
    # Use the accumulated pheromone as a heuristic signal
    # Apply a cooling exponent to smooth
    cooling_exp = 0.7  # < 1 to smooth, > 0 to preserve structure
    pheromone_heuristic = tau ** cooling_exp
    
    # --- Step 2: Distance-based heuristic ---
    dist_heuristic = 1.0 / (d**2 + 1e-4)
    
    # --- Step 3: Cycle cover structural term ---
    # Find minimum-cost cycle cover
    cost_matrix = d.copy()
    np.fill_diagonal(cost_matrix, 1e6)  # Prevent self-loops
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    
    cycle_cover = np.zeros((n, n), dtype=np.float64)
    for i, j in zip(row_idx, col_idx):
        if i != j:
            cycle_cover[i, j] = 1.0
    
    # --- Step 4: Clarke-Wright savings ---
    d_i0 = d[:, 0].reshape(-1, 1)
    d_0j = d[0, :].reshape(1, -1)
    S = d_i0 + d_0j - d
    S = np.maximum(S, 0.0)
    S_max = np.max(S)
    if S_max > 0:
        S_norm = S / S_max
    else:
        S_norm = np.zeros_like(S)
    
    # --- Step 5: Demand-aware adjustment ---
    gamma = 0.5
    demand_penalty = 1.0 / (1.0 + gamma * (demands / capacity))
    demand_penalty[0] = 1.0
    demand_mult = demand_penalty[np.newaxis, :]
    
    # --- Step 6: Combine all components ---
    # Normalize pheromone heuristic
    ph_max = np.max(pheromone_heuristic)
    if ph_max > 0:
        ph_norm = pheromone_heuristic / ph_max
    else:
        ph_norm = np.zeros_like(pheromone_heuristic)
    
    # Weights: pheromone simulation is the primary signal
    w_ph = 2.5
    w_dist = 1.0
    w_cc = 1.5
    w_s = 0.5
    
    ph_mult = 1.0 + w_ph * ph_norm
    cc_mult = 1.0 + w_cc * cycle_cover
    s_mult = 1.0 + w_s * S_norm
    
    heuristic = dist_heuristic * ph_mult * cc_mult * s_mult * demand_mult
    
    # Handle diagonal and invalid values
    np.fill_diagonal(heuristic, 0.0)
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
