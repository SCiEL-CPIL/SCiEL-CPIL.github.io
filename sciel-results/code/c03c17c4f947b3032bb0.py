import numpy as np
import time
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix.
    Strategy: Multi-criteria heuristic search combining inverse distance and demand penalty.
    We search over parameters for:
    H[i,j] = (1/d_ij)^gamma * (1 / (1 + beta * demand_j))
    
    This biases towards shorter distances and penalizes high-demand nodes,
    encouraging the ACO to pack heavier customers earlier or in tighter clusters,
    potentially reducing the number of trips and empty-return distances.
    """
    start_time = time.time()
    deadline = start_time + 310.0  # 20s safety margin
    
    D = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    n = D.shape[0]
    
    # Precompute base features
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = np.where(D > 1e-10, 1.0 / D, 0.0)
    
    def simulate_aco(heuristic_matrix):
        """
        Runs the EXACT ACO protocol specified in the task contract.
        Returns the best cost found.
        """
        pheromone = np.ones_like(heuristic_matrix)
        best_total_cost = float('inf')
        
        num_ants = 20
        num_iterations = 30
        
        for _ in range(num_iterations):
            iteration_best_cost = float('inf')
            
            for ant in range(num_ants):
                current = 0
                current_load = 0.0
                total_dist = 0.0
                unvisited = set(range(1, n))
                
                while unvisited:
                    # Find feasible next nodes
                    feasible = []
                    for j in unvisited:
                        if demands[j] <= (capacity - current_load):
                            feasible.append(j)
                    
                    if not feasible:
                        # Return to depot
                        total_dist += D[current, 0]
                        break
                    
                    # Calculate probabilities
                    # p[j] = pheromone[current, j] * heuristic[current, j]
                    weights = np.array([pheromone[current, j] * heuristic_matrix[current, j] for j in feasible])
                    
                    total_w = np.sum(weights)
                    if total_w < 1e-10:
                        # Fallback to random
                        next_node = random.choice(feasible)
                    else:
                        probs = weights / total_w
                        # Use random.choices for weighted selection
                        next_node = random.choices(feasible, weights=probs, k=1)[0]
                    
                    # Move
                    total_dist += D[current, next_node]
                    current_load += demands[next_node]
                    unvisited.remove(next_node)
                    current = next_node
                    
                    if not unvisited:
                        total_dist += D[current, 0]
                        break
                
            if total_dist < iteration_best_cost:
                iteration_best_cost = total_dist
            
            if iteration_best_cost < best_total_cost:
                best_total_cost = iteration_best_cost
            
            # Update Pheromone
            pheromone *= 0.9
            if iteration_best_cost > 0:
                pheromone += 1.0 / iteration_best_cost
        
        return best_total_cost

    best_cost = float('inf')
    best_heuristic = None
    
    # Candidate parameters
    # gamma controls distance sensitivity
    # beta controls demand penalty (higher beta penalizes high demand more)
    gammas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    betas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    
    candidates = []
    for g in gammas:
        for b in betas:
            candidates.append((g, b))
    
    # Shuffle candidates to ensure we get a baseline early and explore broadly
    random.shuffle(candidates)
    
    for g, b in candidates:
        if time.time() > deadline - 10.0:
            break
        
        # Construct heuristic
        # H = (1/d)^g * (1 / (1 + b * demand))
        demand_factor = 1.0 / (1.0 + b * demands)
        
        if g == 1.0:
            dist_part = inv_dist
        else:
            dist_part = np.power(inv_dist, g)
            
        heuristic_matrix = dist_part * demand_factor[np.newaxis, :]
        
        # Sanitize
        heuristic_matrix = np.where(np.isfinite(heuristic_matrix) & (heuristic_matrix > 0), heuristic_matrix, 1e-9)
        
        # Run Simulation
        cost = simulate_aco(heuristic_matrix)
        
        if cost < best_cost:
            best_cost = cost
            best_heuristic = heuristic_matrix

    # If no heuristic was found (shouldn't happen), fallback to standard 1/d
    if best_heuristic is None:
        with np.errstate(divide='ignore', invalid='ignore'):
            best_heuristic = np.where(D > 0, D ** (-1.0), 1e-9)
        best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)

    return best_heuristic
