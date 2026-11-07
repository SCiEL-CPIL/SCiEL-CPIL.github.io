import numpy as np
import time
import sys

def _aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, seed=42):
    """
    Fast internal ACO simulation matching the contract's parameters.
    Used to evaluate heuristic quality during the heuristic construction phase.
    """
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = np.zeros(n_ants)
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[np.random.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = np.random.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths[ant] = tour_len
            
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
        
        tau *= 0.9
        
        current_best = np.min(tour_lengths)
        if current_best < best_tour_len:
            best_tour_len = current_best
            
    return best_tour_len

def _two_opt_improve(tour, dist, max_passes=100):
    """
    Apply 2-opt improvement to a tour.
    """
    n = len(tour)
    if n < 4:
        return tour, sum(dist[tour[i], tour[(i+1)%n]] for i in range(n))
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                u1 = tour[(i - 1) % n]
                v1 = tour[i]
                u2 = tour[j]
                v2 = tour[(j + 1) % n]
                
                delta = dist[u1, u2] + dist[v1, v2] - dist[u1, v1] - dist[u2, v2]
                if delta < -1e-10:
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
                    
    total_len = 0.0
    for i in range(n):
        total_len += dist[tour[i], tour[(i + 1) % n]]
    return tour, total_len

def _nearest_neighbor_tour(dist, start=0, seed=None):
    """
    Generate a tour using Nearest Neighbor heuristic.
    """
    if seed is not None:
        np.random.seed(seed)
        
    n = len(dist)
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    tour = [start]
    
    for _ in range(n - 1):
        dists = dist[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    return tour

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
        
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    # Strategy: Multi-Source Structural Fusion
    # 1. Generate a diverse set of high-quality tours using different initializations and local search.
    # 2. Aggregate edge statistics (frequency and quality) across these tours.
    # 3. Construct a heuristic matrix that rewards edges that are frequently part of good tours.
    # 4. This guides the ACO to explore a "basin of attraction" around good solutions rather than a single path.
    
    # Step 1: Generate diverse tours
    tours = []
    tour_lengths = []
    
    # Base seeds for diversity
    base_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for seed in base_seeds:
        if time.time() > deadline - 60:
            break
            
        # Generate initial tour with NN
        # We can vary the start node or use random perturbations, but NN from 0 is standard.
        # To get diversity, we can perturb the distance matrix slightly or use different random states for tie-breaking?
        # NN is deterministic. To get diversity, let's use Random Insertion or perturbed NN.
        # Let's try Random Insertion for some seeds.
        
        if seed < 5:
            # Standard NN
            tour = _nearest_neighbor_tour(safe_dist, start=0)
        else:
            # Random Insertion: Start with 2 nodes, insert next node at best position
            np.random.seed(seed)
            tour = [0, np.random.randint(1, n)]
            unvisited = list(range(1, n))
            unvisited.remove(tour[1])
            np.random.shuffle(unvisited)
            
            for node in unvisited:
                best_pos = 0
                best_cost = float('inf')
                for i in range(len(tour)):
                    u = tour[i]
                    v = tour[(i + 1) % len(tour)]
                    cost = safe_dist[u, node] + safe_dist[node, v] - safe_dist[u, v]
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = i + 1
                tour.insert(best_pos, node)
                
        # Improve with 2-opt
        tour, length = _two_opt_improve(tour, safe_dist)
        tours.append(tour)
        tour_lengths.append(length)
        
    if not tours:
        # Fallback if no time
        return 1.0 / safe_dist

    # Step 2: Aggregate edge statistics
    # We want to weight edges by:
    # 1. Frequency: How many tours contain this edge?
    # 2. Quality: The length of the tour containing the edge (shorter is better).
    # 
    # Heuristic value H[i,j] ~ Sum( 1 / (Length_t * Frequency_t) ) ?
    # Or simpler: H[i,j] = Sum( w_t * I_edge_in_t ) where w_t = 1/Length_t
    
    edge_weight = np.zeros((n, n))
    
    min_len = min(tour_lengths)
    
    for tour, length in zip(tours, tour_lengths):
        # Weight this tour's edges by its quality
        w = 1.0 / (length - min_len + 1.0) # Normalize so best tour has highest weight
        
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            # Directed or undirected? 
            # The ACO uses directed pheromone, but TSP is symmetric.
            # The heuristic is usually symmetric. 
            # Let's accumulate symmetrically to encourage both directions, 
            # or just follow the direction of the tour?
            # The contract says "travel direction only" for pheromone deposit.
            # Heuristic is static. 
            # If we make it symmetric, it helps the ant go either way.
            # If we make it directed, it forces the specific order.
            # Given the "backbone" insight, directed might be stronger for following the tour.
            # But "consensus" implies undirected structural importance.
            # Let's try symmetric accumulation first, as it's more robust to direction reversal in 2-opt.
            edge_weight[u, v] += w
            edge_weight[v, u] += w

    # Step 3: Construct Heuristic
    # Combine structural weight with inverse distance to avoid zero values and maintain locality.
    # H = (1 + edge_weight) * (1 / dist)
    
    base_heur = 1.0 / safe_dist
    
    # Normalize edge_weight to be comparable to base_heur scale
    # Max edge_weight is sum of weights of all tours containing that edge.
    # Let's scale it so that the max edge_weight is around 10x the median base_heur?
    # Or just use it as a multiplier.
    
    max_w = np.max(edge_weight)
    if max_w > 0:
        # Scale weights so they have a significant impact
        # Let's aim for the best edges to be ~10x more attractive than base
        scaled_w = edge_weight / max_w * 10.0
    else:
        scaled_w = np.zeros((n, n))
        
    final_heur = (1.0 + scaled_w) * base_heur
    
    # Ensure validity
    final_heur[~np.isfinite(final_heur)] = 1e-9
    final_heur[final_heur <= 0] = 1e-9
    
    # Step 4: Refine with ACO simulation if time permits
    # We can try a few variations of the scaling factor or blending with pure inverse distance.
    
    best_heur = final_heur
    best_score = _aco_simulate(safe_dist, final_heur, seed=123)
    
    if time.time() < deadline - 30:
        # Try different scaling factors for the structural weight
        for scale_factor in [5.0, 20.0, 50.0]:
            scaled_w_test = edge_weight / max_w * scale_factor if max_w > 0 else np.zeros((n, n))
            h_test = (1.0 + scaled_w_test) * base_heur
            h_test[~np.isfinite(h_test)] = 1e-9
            h_test[h_test <= 0] = 1e-9
            
            score = _aco_simulate(safe_dist, h_test, seed=123)
            if score < best_score:
                best_score = score
                best_heur = h_test
                
        # Try blending with pure inverse distance (less structural bias)
        blend_alpha = 0.5
        h_blend = (1.0 - blend_alpha) * base_heur + blend_alpha * best_heur
        h_blend[~np.isfinite(h_blend)] = 1e-9
        h_blend[h_blend <= 0] = 1e-9
        score = _aco_simulate(safe_dist, h_blend, seed=123)
        if score < best_score:
            best_score = score
            best_heur = h_blend
            
    return best_heur
