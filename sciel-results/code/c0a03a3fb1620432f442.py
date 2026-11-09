import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin

    n = distance_matrix.shape[0]
    customers = np.arange(1, n)
    eps = 1e-9

    # 1. Base Heuristic: Inverse Distance
    base_heur = 1.0 / (distance_matrix + eps)
    np.fill_diagonal(base_heur, 0.0)

    # 2. Savings Heuristic Signal
    d_to_depot = distance_matrix[:, 0]
    savings = d_to_depot[:, None] + d_to_depot[None, :] - distance_matrix

    cust_savings = savings[np.ix_(customers, customers)]
    cust_savings = np.clip(cust_savings, 0, None)

    max_savings = np.max(cust_savings)
    if max_savings > 0:
        norm_savings = cust_savings / max_savings
    else:
        norm_savings = np.zeros_like(cust_savings)

    # 3. Lagrangian Relaxation-based Edge Weight Heuristic
    # Solve a relaxed CVRP by constructing a "penalized" tour structure.
    # The idea is to find edges that are frequently used in near-optimal
    # relaxed solutions (TSP-like but with capacity penalties).
    
    # We approximate the dual optimization by running a simplified
    # constructive heuristic that penalizes capacity violations,
    # and count edge frequencies.
    
    edge_count = np.zeros((n, n), dtype=float)
    
    # Run multiple randomized constructive passes with Lagrangian-style penalties
    # to estimate which edges are "robustly" good under capacity constraints.
    n_passes = 50
    rng = np.random.RandomState(42)
    
    for pass_idx in range(n_passes):
        if time.time() > deadline - 50:  # Leave time for final processing
            break
            
        # Randomize the starting point and neighbor selection
        unvisited = set(customers.tolist())
        current = 0
        remaining_cap = capacity
        tour_edges = []
        
        # First, pick a random start customer
        if unvisited:
            start_node = rng.choice(list(unvisited))
            tour_edges.append((0, start_node))
            current = start_node
            remaining_cap -= demands[start_node]
            unvisited.remove(start_node)
        
        while unvisited:
            # Find eligible customers
            eligible = [c for c in unvisited if demands[c] <= remaining_cap]
            
            if not eligible:
                # Return to depot and reload
                tour_edges.append((current, 0))
                current = 0
                remaining_cap = capacity
                continue
            
            # Score each eligible customer with a Lagrangian-style penalty
            # The penalty encourages packing routes (higher remaining capacity usage is better)
            # and favors shorter distances.
            
            best_score = -float('inf')
            best_node = None
            
            # Add some randomness to avoid getting stuck in local patterns
            noise_scale = 0.1 * (1.0 - pass_idx / n_passes)
            
            for c in eligible:
                dist = distance_matrix[current, c]
                # Penalty term: encourage using capacity efficiently
                # If we pick c, remaining capacity becomes remaining_cap - demands[c]
                # We want to maximize the "fill" of the route, so higher remaining_cap - demands[c] is worse
                # Actually, we want to minimize distance, but also want to pack routes.
                # A simple proxy: minimize dist + lambda * (1 - (remaining_cap - demands[c]) / capacity)
                # This encourages picking customers that use up capacity.
                
                fill_factor = 1.0 - (remaining_cap - demands[c]) / capacity
                score = -dist - 0.5 * fill_factor + noise_scale * rng.randn()
                
                if score > best_score:
                    best_score = score
                    best_node = c
            
            if best_node is not None:
                tour_edges.append((current, best_node))
                remaining_cap -= demands[best_node]
                unvisited.remove(best_node)
                current = best_node
        
        # Close the tour
        if current != 0:
            tour_edges.append((current, 0))
        
        # Count edges
        for u, v in tour_edges:
            edge_count[u, v] += 1.0
            edge_count[v, u] += 1.0  # Symmetric

    # Normalize edge counts
    max_count = np.max(edge_count)
    if max_count > 0:
        norm_edge_freq = edge_count / max_count
    else:
        norm_edge_freq = np.zeros_like(edge_count)

    # 4. Combine Heuristics
    # H = Base * (1 + alpha * Savings + beta * EdgeFreq)
    alpha = 1.0
    beta = 0.8
    
    final_heur = base_heur.copy()
    
    # Apply boosts to all edges (including depot-customer)
    full_boost = 1.0 + alpha * norm_savings + beta * norm_edge_freq
    
    # norm_savings is only defined for customer-customer, so we need to handle depot edges
    # For depot-customer edges, use a simpler boost
    depot_boost = 1.0 + beta * norm_edge_freq[0, 1:]  # Depot to customers
    final_heur[0, 1:] *= depot_boost
    final_heur[1:, 0] *= depot_boost
    
    # For customer-customer edges, use the full boost
    cust_mask = np.zeros((n, n), dtype=bool)
    cust_mask[np.ix_(customers, customers)] = True
    
    cust_boost = 1.0 + alpha * norm_savings + beta * norm_edge_freq[np.ix_(customers, customers)]
    final_heur[cust_mask] *= cust_boost[np.ix_(customers, customers)]
    
    # Ensure positivity and finiteness
    final_heur = np.where(final_heur > 0, final_heur, 1e-9)
    final_heur = np.where(np.isfinite(final_heur), final_heur, 1e-9)
    
    if final_heur.shape != distance_matrix.shape:
        final_heur = 1.0 / (distance_matrix + eps)
    
    return final_heur
