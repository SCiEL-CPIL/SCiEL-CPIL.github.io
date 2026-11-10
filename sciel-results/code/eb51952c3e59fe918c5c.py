import numpy as np
import time
import random

def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic for ant colony orienteering.
    
    Strategy:
    - Use an internal GRASP-like search to find high-quality tours.
    - Construct the heuristic matrix based on the edges used in these high-quality tours.
    - This makes the heuristic instance-specific and aware of the global constraint (max_len).
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Safety check for finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(distance_matrix)):
        return np.ones((n, n), dtype=float) * 1e-9

    # Initialize heuristic with a base greedy value
    # Base heuristic: prize[j] / (d[i,j] + d[j,0])
    cost = distance_matrix + distance_matrix[:, 0][None, :]
    eps = 1e-9
    cost_safe = np.maximum(cost, eps)
    prize_safe = np.maximum(prizes[None, :], eps)
    
    H_base = prize_safe / cost_safe
    # Normalize H_base to [0, 1] for blending later
    max_base = np.max(H_base)
    if max_base > 0:
        H_base = H_base / max_base
    else:
        H_base = np.zeros_like(H_base)

    # Internal Search: GRASP-like
    # We will generate several candidate tours and keep the best ones.
    # A tour is a list of node indices, starting and ending at 0.
    
    best_tours = []
    best_prizes = []
    
    num_candidates = 20  # Number of random greedy constructions
    # Adjust number based on time
    if time.time() > deadline - 10.0:
        num_candidates = 5

    def get_tour_prize_length(tour):
        """
        tour: list of nodes [0, n1, n2, ..., 0]
        Returns (total_prize, total_length)
        """
        if len(tour) < 2:
            return 0.0, 0.0
        total_prize = 0.0
        total_len = 0.0
        for i in range(len(tour) - 1):
            u = tour[i]
            v = tour[i+1]
            total_len += distance_matrix[u, v]
            total_prize += prizes[v]
        return total_prize, total_len

    def greedy_construct(random_seed):
        """
        Construct a tour using a randomized greedy strategy.
        """
        rng = np.random.RandomState(random_seed)
        visited = {0}
        current = 0
        current_len = 0.0
        tour = [0]
        
        while True:
            # Find all feasible next nodes
            candidates = []
            for j in range(n):
                if j in visited:
                    continue
                # Check if we can go to j and return to 0
                step_cost = distance_matrix[current, j]
                return_cost = distance_matrix[j, 0]
                if current_len + step_cost + return_cost <= max_len:
                    # Calculate a score for this candidate
                    # We want high prize, low cost.
                    # Add some randomness to explore
                    score = prizes[j] / (step_cost + return_cost + eps)
                    # Randomize the score slightly
                    score *= (1.0 + 0.1 * rng.rand())
                    candidates.append((score, j))
            
            if not candidates:
                break
            
            # Sort by score descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Pick the best candidate (or a random one from the top few for diversity)
            # Here we just pick the best to be fast, but the seed varies the scores
            best_j = candidates[0][1]
            
            tour.append(best_j)
            visited.add(best_j)
            current_len += distance_matrix[current, best_j]
            current = best_j
            
        tour.append(0)
        return tour

    def local_search(tour):
        """
        Simple local search: try to insert any unvisited node if it fits.
        Try to remove low-value nodes to make room for high-value ones.
        """
        visited = set(tour)
        current_tour = list(tour)
        
        # Pass 1: Insert any feasible unvisited node that adds prize
        improved = True
        while improved:
            improved = False
            prize, length = get_tour_prize_length(current_tour)
            
            # Find best insertion point for each unvisited node
            best_gain = 0
            best_node = -1
            best_pos = -1
            
            for j in range(n):
                if j in visited:
                    continue
                
                # Try inserting j at every position
                for k in range(1, len(current_tour)): # Positions between 0 and end
                    u = current_tour[k-1]
                    v = current_tour[k]
                    
                    # New length if inserted between u and v
                    old_len_segment = distance_matrix[u, v]
                    new_len_segment = distance_matrix[u, j] + distance_matrix[j, v]
                    delta_len = new_len_segment - old_len_segment
                    
                    if length + delta_len <= max_len:
                        # Gain is prize[j]
                        gain = prizes[j]
                        if gain > best_gain:
                            best_gain = gain
                            best_node = j
                            best_pos = k
            
            if best_node != -1:
                current_tour.insert(best_pos, best_node)
                visited.add(best_node)
                improved = True
        
        # Pass 2: Try to swap or remove/replace
        # Simple removal of low prize nodes if it allows adding a much better one?
        # This is complex. Let's stick to insertion for now as it's safe.
        
        return current_tour

    # Run GRASP
    for i in range(num_candidates):
        if time.time() > deadline - 5.0:
            break
        seed = i * 12345 + 42
        tour = greedy_construct(seed)
        tour = local_search(tour)
        prize, length = get_tour_prize_length(tour)
        
        if prize > 0:
            best_tours.append(tour)
            best_prizes.append(prize)

    # If no tours found, fall back to base heuristic
    if not best_tours:
        H = H_base
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H

    # Construct final heuristic from best tours
    # We want to boost edges that appear in the best tours.
    # H_final = H_base + sum( alpha_k * I(edge in tour_k) )
    # Let's weight by the prize of the tour.
    
    H_final = H_base.copy()
    
    # Normalize best_prizes to weights
    max_prize = max(best_prizes) if best_prizes else 1.0
    if max_prize > 0:
        weights = [p / max_prize for p in best_prizes]
    else:
        weights = [1.0 / len(best_prizes) for _ in best_prizes]

    # Only use the top 5 tours to save time and avoid noise
    # Sort tours by prize descending
    sorted_indices = np.argsort(best_prizes)[::-1]
    top_k = min(5, len(sorted_indices))
    
    for idx in range(top_k):
        tour = best_tours[sorted_indices[idx]]
        w = weights[sorted_indices[idx]]
        
        for i in range(len(tour) - 1):
            u = tour[i]
            v = tour[i+1]
            # Boost the edge (u, v)
            # The boost should be significant enough to guide the ACO
            H_final[u, v] += 1.0 * w

    # Ensure values are finite and positive
    H_final = np.where(np.isfinite(H_final) & (H_final > 0), H_final, 1e-9)
    
    # Optional: Normalize H_final to a reasonable range?
    # The ACO uses pheromone * heuristic. If heuristic is too large, it dominates.
    # But since we are adding to a normalized base, it should be okay.
    # Let's cap it to avoid extreme values.
    max_val = np.max(H_final)
    if max_val > 10.0:
        H_final = H_final / max_val * 10.0

    if H_final.shape != (n, n):
        return np.ones((n, n), dtype=float) * 1e-9

    return H_final
