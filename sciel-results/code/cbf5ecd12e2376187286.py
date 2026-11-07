"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Build an edge-attractiveness matrix that anticipates ACO pheromone convergence.
    
    Strategy: simulate a short ACO run with a cooling schedule on the heuristic
    exponent alpha. Early iterations favor distance (high alpha), later iterations
    incorporate demand-aware routing (lower alpha, demand-weighted distance).
    The final heuristic is a blend that encodes the expected reinforcement trajectory.
    
    We run a lightweight inner ACO (few ants, few iterations) to let pheromone
    concentrate on good edges, then use the resulting pheromone pattern to shape
    the static heuristic. This makes the heuristic 'look ahead' at where the
    colony will converge, biasing early ants toward edges that will be reinforced.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    capacity = float(capacity)
    
    # Ensure finite, positive
    dm = np.where(np.isfinite(distance_matrix), distance_matrix, 1e6)
    dm = np.maximum(dm, 1e-9)
    
    # --- Lightweight inner ACO to discover promising edges ---
    # Use a small number of ants and iterations to stay fast
    n_ants_inner = 10
    n_iters_inner = 15
    rho = 0.5  # fast evaporation for inner run
    tau_min = 1e-4
    tau_max = 1e2
    
    # Pheromone initialized uniformly
    tau = np.ones((n, n))
    
    # Track best tour found in inner run
    best_cost = np.inf
    best_tour = None
    
    # Cooling schedule for alpha: start high (distance-driven), decay to lower
    # (allow demand-aware routing to influence)
    alpha_start = 3.0
    alpha_end = 1.5
    
    for it in range(n_iters_inner):
        # Linear cooling
        alpha = alpha_start + (alpha_end - alpha_start) * (it / max(1, n_iters_inner - 1))
        
        # Build heuristic for this inner iteration
        # Demand-aware: penalize edges that lead to customers with high remaining demand
        # relative to what's left in the route. We approximate by weighting distance
        # by a factor that considers demand of the destination.
        # For a static heuristic matrix, we use: 1 / (dist * (1 + demand_j/capacity * 0.5))
        demand_factor = 1.0 + (demands / capacity) * 0.5
        demand_factor[0] = 1.0  # depot has no demand penalty
        
        # Heuristic: inverse of demand-weighted distance
        # eta[i,j] = 1 / (dm[i,j] * demand_factor[j])
        eta = 1.0 / (dm * demand_factor[np.newaxis, :])
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        # Run inner ants
        for ant in range(n_ants_inner):
            # Build tour
            tour = [0]
            current = 0
            remaining = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0
            
            while True:
                # Find feasible next customers
                feasible = []
                for j in range(1, n):
                    if not visited[j] and demands[j] <= remaining:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot
                    total_dist += dm[current, 0]
                    break
                
                # Compute probabilities
                probs = np.zeros(n)
                for j in feasible:
                    probs[j] = tau[current, j] * (eta[current, j] ** alpha)
                
                prob_sum = probs.sum()
                if prob_sum <= 0:
                    # Fallback: pick closest feasible
                    dists = np.array([dm[current, j] for j in feasible])
                    j = feasible[int(np.argmin(dists))]
                else:
                    probs = probs / prob_sum
                    # Sample
                    r = np.random.random()
                    cumsum = np.cumsum(probs)
                    j = int(np.searchsorted(cumsum, r))
                    if j >= n:
                        j = feasible[0]
                    else:
                        j = feasible[j] if j < len(feasible) else feasible[0]
                        # More careful: use inverse CDF on feasible subset
                        feasible_probs = np.array([probs[j2] for j2 in feasible])
                        fp_sum = feasible_probs.sum()
                        if fp_sum > 0:
                            feasible_probs = feasible_probs / fp_sum
                            r2 = np.random.random()
                            cum2 = np.cumsum(feasible_probs)
                            j = feasible[int(np.searchsorted(cum2, r2))]
                            if j >= len(feasible):
                                j = feasible[0]
                        else:
                            j = feasible[0]
                
                total_dist += dm[current, j]
                remaining -= demands[j]
                visited[j] = True
                tour.append(j)
                current = j
            
            # Update best
            if total_dist < best_cost:
                best_cost = total_dist
                best_tour = list(tour)
        
        # Pheromone update
        tau *= (1.0 - rho)
        if best_tour is not None:
            # Deposit pheromone on best tour edges
            deposit = 1.0 / max(best_cost, 1e-9)
            for i in range(len(best_tour) - 1):
                a, b = best_tour[i], best_tour[i + 1]
                tau[a, b] += deposit
                tau[b, a] += deposit
            # Also depot return
            if len(best_tour) > 1:
                tau[best_tour[-1], 0] += deposit
                tau[0, best_tour[-1]] += deposit
        
        # Clamp pheromone
        tau = np.clip(tau, tau_min, tau_max)
    
    # --- Build final heuristic from inner ACO results ---
    # The pheromone matrix tau now encodes which edges are promising.
    # Combine with base distance heuristic using a blend.
    
    # Base heuristic: inverse distance with demand awareness
    demand_factor = 1.0 + (demands / capacity) * 0.3
    demand_factor[0] = 1.0
    base_eta = 1.0 / (dm * demand_factor[np.newaxis, :])
    base_eta = np.where(np.isfinite(base_eta) & (base_eta > 0), base_eta, 1e-9)
    
    # Pheromone-guided component: use the inner tau as an attractiveness boost
    # Normalize tau to [1e-9, 1] range for blending
    tau_norm = tau.copy()
    tau_norm = np.where(tau_norm > 0, tau_norm, 1e-9)
    tau_max_val = tau_norm.max()
    if tau_max_val > 0:
        tau_norm = tau_norm / tau_max_val
    tau_norm = np.clip(tau_norm, 1e-9, 1.0)
    
    # Blend: geometric mean of base heuristic and pheromone-guided component
    # This gives edges that are both close AND frequently reinforced a boost
    final_heuristic = base_eta * (tau_norm ** 0.5)
    
    # Ensure positivity and finiteness
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), 
                               final_heuristic, 1e-9)
    
    # Make diagonal zero (no self-loops)
    np.fill_diagonal(final_heuristic, 0.0)
    
    return final_heuristic
# EVOLVE-BLOCK-END
