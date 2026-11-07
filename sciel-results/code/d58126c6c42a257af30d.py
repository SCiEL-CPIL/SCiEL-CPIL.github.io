import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Constructs a heuristic matrix by simulating ACO dynamics with a cooling schedule.
    The returned matrix encodes the expected pheromone reinforcement trajectory.
    """
    n = len(demands)
    
    # Base heuristic: inverse distance
    # Ensure no division by zero
    base_eta = 1.0 / (distance_matrix + 1e-9)
    
    # Simulation parameters
    n_ants = 20
    n_iters = 30
    evaporation = 0.9
    
    # Initialize pheromone
    tau = np.ones_like(distance_matrix)
    
    # Pre-compute demand array for faster access
    demands_arr = np.asarray(demands, dtype=np.float64)
    
    # Store the best tour found across all iterations
    global_best_tour = None
    global_best_cost = float('inf')
    
    for _ in range(n_iters):
        best_cost_iter = float('inf')
        best_tour_iter = None
        
        for _ in range(n_ants):
            current_node = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            remaining_capacity = capacity
            tour = [0]
            current_cost = 0.0
            
            while not np.all(visited):
                # Find feasible neighbors
                # Neighbors j such that not visited[j] and demands[j] <= remaining_capacity
                # Vectorized check
                unvisited = np.where(~visited)[0]
                unvisited = unvisited[unvisited > 0]
                
                if len(unvisited) == 0:
                    break # Should not happen as depot is visited but we break when all customers visited
                    
                feasible = unvisited[demands_arr[unvisited] <= remaining_capacity]
                
                if len(feasible) == 0:
                    # Return to depot
                    current_cost += distance_matrix[current_node, 0]
                    current_node = 0
                    remaining_capacity = capacity
                    continue
                
                # Calculate probabilities
                # prob proportional to tau[i, j] * base_eta[i, j]
                vals = tau[current_node, feasible] * base_eta[current_node, feasible]
                
                # Ensure positive and finite
                vals[~np.isfinite(vals)] = 1e-9
                vals[vals <= 0] = 1e-9
                
                total_prob = np.sum(vals)
                
                if total_prob == 0:
                    # Fallback to nearest
                    dists = distance_matrix[current_node, feasible]
                    next_node = feasible[np.argmin(dists)]
                else:
                    probs = vals / total_prob
                    r = np.random.random()
                    cum_prob = np.cumsum(probs)
                    next_idx = np.searchsorted(cum_prob, r)
                    if next_idx >= len(feasible):
                        next_idx = len(feasible) - 1
                    next_node = feasible[next_idx]
                
                # Move
                current_cost += distance_matrix[current_node, next_node]
                remaining_capacity -= demands_arr[next_node]
                visited[next_node] = True
                tour.append(next_node)
                current_node = next_node
            
            # Return to depot
            current_cost += distance_matrix[current_node, 0]
            tour.append(0)
            
            if current_cost < best_cost_iter:
                best_cost_iter = current_cost
                best_tour_iter = tour[:]
                
            if current_cost < global_best_cost:
                global_best_cost = current_cost
                global_best_tour = tour[:]
        
        # Update pheromone
        if best_cost_iter < float('inf') and best_tour_iter is not None:
            tau *= evaporation
            deposit = 1.0 / best_cost_iter
            # Standard ACO: reinforce best tour
            for i in range(len(best_tour_iter) - 1):
                u = best_tour_iter[i]
                v = best_tour_iter[i+1]
                tau[u, v] += deposit
                tau[v, u] += deposit
                
            # Ensure positivity
            tau[tau <= 0] = 1e-9

    # Final cleanup
    heur_matrix = tau.copy()
    
    # Boost the edges of the global best tour to strongly guide the real ACO
    # This compensates for any potential flattening of pheromone in the real ACO
    # and ensures the heuristic points to a known good solution.
    if global_best_tour is not None:
        for i in range(len(global_best_tour) - 1):
            u = global_best_tour[i]
            v = global_best_tour[i+1]
            # Multiply by a factor to make these edges highly attractive
            # We want these edges to be significantly more attractive than others
            heur_matrix[u, v] *= 10.0
            heur_matrix[v, u] *= 10.0

    heur_matrix[~np.isfinite(heur_matrix)] = 1e-9
    heur_matrix[heur_matrix <= 0] = 1e-9
    
    return heur_matrix
