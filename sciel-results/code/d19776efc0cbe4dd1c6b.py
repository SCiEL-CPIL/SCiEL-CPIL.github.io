import numpy as np
import time
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix for 3 fixed instances.
    Uses a grid search over the exponent gamma in (1/d)^gamma,
    evaluated by a fast, vectorized proxy ACO.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    demands = np.array(demands)
    
    # Ensure distance_matrix is a numpy array
    D = np.asarray(distance_matrix)
    
    # Candidate gamma values
    # Gamma > 1 focuses on local edges. Gamma < 1 allows longer jumps.
    # Standard is 1.0. We search around this.
    candidate_gammas = np.arange(0.5, 3.5, 0.1)
    
    def run_proxy_aco(gamma, num_tours=50):
        """
        Fast proxy ACO using vectorized weight selection.
        """
        # Construct heuristic matrix for this gamma
        # D has 0 on diagonal. We want 1e-9 there.
        # For off-diagonal, we want D^(-gamma).
        # If D is 0 (shouldn't happen for i!=j in valid instances, but safe), handle it.
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = np.where(D > 0, D ** (-gamma), 1e-9)
        
        # Ensure diagonal is 1e-9 (or 0, but 1e-9 is safer for "positive" requirement if it's ever accessed, 
        # though ants don't pick self). The task says "Heuristic entries that are not positive become 1e-9".
        # D[i,i]=0 -> 0**(-gamma) is inf. np.where handles the D>0 part.
        # For D==0 and i!=j, it becomes 1e-9.
        # For i==j, D==0, so it becomes 1e-9.
        
        # Sanitize: any non-finite or non-positive becomes 1e-9
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        best_cost = float('inf')
        
        for _ in range(num_tours):
            current = 0
            current_load = 0.0
            total_distance = 0.0
            unvisited = np.array([True] * n)
            unvisited[0] = False # Depot is not a customer to visit
            
            # List of unvisited customer indices
            # We maintain a list for efficiency in selection
            unvisited_indices = list(range(1, n))
            
            while unvisited_indices:
                # Find feasible neighbors
                # Feasible if demand[j] <= capacity - current_load
                feasible_indices = []
                for j in unvisited_indices:
                    if demands[j] <= (capacity - current_load):
                        feasible_indices.append(j)
                
                if not feasible_indices:
                    # Return to depot
                    total_distance += D[current, 0]
                    break
                
                # Get weights for feasible nodes from current
                # eta[current, feasible_indices]
                weights = eta[current, feasible_indices]
                
                # Normalize
                total_w = np.sum(weights)
                if total_w == 0:
                    # Fallback to random
                    next_node = random.choice(feasible_indices)
                else:
                    probs = weights / total_w
                    # Use np.random.choice for vectorized selection
                    next_node = feasible_indices[np.random.choice(len(feasible_indices), p=probs)]
                
                # Move
                total_distance += D[current, next_node]
                current_load += demands[next_node]
                
                # Remove next_node from unvisited
                unvisited_indices.remove(next_node)
                current = next_node
                
                if not unvisited_indices:
                    # All visited, return to depot
                    total_distance += D[current, 0]
                    break
            
            if total_distance < best_cost:
                best_cost = total_cost if False else total_distance # Fixed typo in thought, code below is correct
        
        return best_cost

    best_gamma = 1.0
    best_proxy_cost = float('inf')
    
    # Time budget for search: 280s (leaving 20s margin)
    # We have ~30 gamma values.
    # Let's estimate time per gamma.
    # If we run 50 tours, each tour is ~50 steps. 2500 steps per gamma.
    # 30 gammas * 2500 steps = 75,000 steps.
    # Each step involves a small numpy operation and list manipulation.
    # This should be very fast (< 1s).
    
    for gamma in candidate_gammas:
        # Check time
        if time.time() - start_time > 280.0:
            break
            
        cost = run_proxy_aco(gamma, num_tours=50)
        if cost < best_proxy_cost:
            best_proxy_cost = cost
            best_gamma = gamma
            
    # Construct final heuristic matrix with best_gamma
    with np.errstate(divide='ignore', invalid='ignore'):
        final_heuristic = np.where(D > 0, D ** (-best_gamma), 1e-9)
    
    # Sanitize
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
    
    return final_heuristic
