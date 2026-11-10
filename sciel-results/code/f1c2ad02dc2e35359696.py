import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using Simulated Annealing to optimize
    a parameterized heuristic model.
    
    The model is: H[i,j] = exp(-alpha * dist[i,j]) * beta^(cluster_match)
    where cluster_match is 1 if i and j are in the same 'soft' cluster 
    (defined by a threshold on distance) and 0 otherwise.
    Instead of hard clustering, we optimize alpha and the cluster threshold 
    to minimize the cost of greedy routes constructed with this heuristic.
    
    Actually, a more direct approach: Optimize the heuristic matrix directly 
    via SA, using a greedy constructor to evaluate the cost.
    """
    deadline = time.time() + 320.0
    n = distance_matrix.shape[0]
    
    if distance_matrix.shape != (n, n):
        return 1.0 / (distance_matrix + 1e-9)

    # Ensure demands are integers for capacity check
    demands_int = np.array(demands, dtype=int)
    
    # Base heuristic: inverse distance
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    base_heuristic[np.diag_indices(n)] = 0.0
    
    def greedy_construct(heuristic_matrix):
        """
        Construct routes using a greedy strategy guided by the heuristic.
        Returns total distance.
        """
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        total_dist = 0.0
        current = 0
        remaining_cap = capacity
        
        while not np.all(visited):
            # Find next customer
            best_score = -1
            best_next = -1
            
            # Vectorized scoring for efficiency
            # Score = pheromone (1.0) * heuristic
            # Mask visited and infeasible nodes
            scores = heuristic_matrix[current].copy()
            scores[visited] = -1e10
            # Check capacity feasibility
            # demands_int is 1-indexed effectively (0 is depot)
            feasible_mask = (demands_int[1:] <= remaining_cap)
            # Create full mask
            full_feasible = np.ones(n, dtype=bool)
            full_feasible[0] = False # Depot is visited anyway, but just in case
            full_feasible[1:] = feasible_mask
            
            scores[~full_feasible] = -1e10
            
            # If no feasible node, return to depot
            if np.max(scores) <= -1e9:
                total_dist += distance_matrix[current, 0]
                current = 0
                visited[0] = True # Depot is always 'visited' in the sense that we return to it
                # Reset remaining capacity
                remaining_cap = capacity
                # Mark depot as visited again? No, we just continue.
                # The loop condition is 'not all visited'. 
                # We need to ensure we don't get stuck if only depot is unvisited.
                # But depot is index 0. If all customers are visited, loop ends.
                # So this is fine.
                continue
            
            best_next = np.argmax(scores)
            
            if best_next == -1 or scores[best_next] <= -1e9:
                # Should not happen if there are unvisited customers, but safety
                break
                
            total_dist += distance_matrix[current, best_next]
            visited[best_next] = True
            remaining_cap -= demands_int[best_next]
            current = best_next
            
        # Return to depot from last customer
        total_dist += distance_matrix[current, 0]
        
        return total_dist

    # Initial heuristic: inverse distance
    current_heuristic = base_heuristic.copy()
    # Normalize to avoid extreme values affecting SA noise
    current_heuristic /= np.max(current_heuristic)
    
    current_cost = greedy_construct(current_heuristic)
    best_heuristic = current_heuristic.copy()
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    cooling_rate = 0.995
    min_T = 1e-4
    max_iterations = 5000 # Upper bound, will stop by time
    
    # We will perturb the heuristic matrix by adding noise to a subset of entries
    # and scaling.
    
    iteration = 0
    while T > min_T and time.time() < deadline - 10:
        # Generate a neighbor
        # Strategy: Pick a random subset of edges and scale them, or add Gaussian noise
        # Let's try scaling random entries to explore the space of relative attractiveness
        
        neighbor_heuristic = current_heuristic.copy()
        
        # Perturb a small fraction of the matrix
        num_perturbations = max(1, int(0.05 * n * n))
        rows, cols = np.random.choice(n, num_perturbations), np.random.choice(n, num_perturbations)
        
        # Avoid diagonal
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]
        
        if len(rows) == 0:
            T *= cooling_rate
            continue
            
        # Apply random scaling factor
        # Multiplicative noise is better for attractiveness
        noise = np.random.lognormal(mean=0, sigma=0.5, size=len(rows))
        neighbor_heuristic[rows, cols] *= noise
        
        # Ensure positivity
        neighbor_heuristic = np.maximum(neighbor_heuristic, 1e-9)
        
        # Evaluate
        neighbor_cost = greedy_construct(neighbor_heuristic)
        
        # Acceptance criterion
        delta = neighbor_cost - current_cost
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_heuristic = neighbor_heuristic
            current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_heuristic = current_heuristic.copy()
        else:
            # Revert (current_heuristic is unchanged)
            pass
            
        T *= cooling_rate
        iteration += 1
        
    # Final processing
    # The heuristic needs to be positive.
    # Scale it back to a reasonable range? 
    # The ACO formula is pheromone * heuristic. 
    # Since pheromone starts at 1 and decays, the absolute scale of heuristic matters 
    # relative to pheromone. 
    # In the seed program, heuristic was roughly 1/distance. 
    # Our SA optimized a normalized version. 
    # Let's rescale best_heuristic to have a mean similar to base_heuristic 
    # to ensure it's in the "right" ballpark for the ACO dynamics, 
    # or just return it as is if it's positive.
    # However, if the heuristic values become very small, the probability 
    # proportional to pi * eta might be dominated by pi. 
    # The task says "Heuristic entries that are not positive become 1e-9".
    # It doesn't specify normalization.
    
    # To be safe, let's normalize the best heuristic to have a median of 1.0
    # or similar, or just ensure it's positive and finite.
    # Actually, looking at the seed, it used 1/dist. 
    # If our SA found a structure that works well with greedy, 
    # returning that structure directly is the goal.
    
    best_heuristic = np.maximum(best_heuristic, 1e-9)
    best_heuristic[np.diag_indices(n)] = 1e-9 # Diagonal should be non-positive/zero effectively
    
    return best_heuristic
