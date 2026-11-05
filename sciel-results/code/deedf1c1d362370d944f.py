import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Deterministic Greedy Forward-Backward Pass (Coordinate Descent)
    1. Initialize scores using a strong static heuristic (Inverse Degree + Hub Penalty).
    2. Simulate the greedy decoding to get a baseline Independent Set size.
    3. Perform deterministic sweeps:
       - Forward pass: For each vertex in current order, check if moving it earlier (swapping with predecessor) improves the set size.
       - Backward pass: For each vertex in current order, check if moving it later (swapping with successor) improves the set size.
       This avoids the randomness and potential stagnation of random pairwise swaps by systematically exploring the neighborhood of the current ordering.
    4. Return the final scores.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    # 1. Initial Heuristic: Low Degree + Hub Penalty
    # We want high scores for vertices that are "safe" to pick early.
    # Low degree is good. Being adjacent to high-degree hubs is bad (they block many others).
    deg = adj.sum(axis=1).astype(float)
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    hub_penalty = neighbor_deg_sum / safe_deg
    
    # Base score: prefer low degree, penalize hub adjacency
    scores = -deg - 0.5 * hub_penalty
    
    # Ensure unique scores to avoid tie-breaking ambiguity in argsort
    # Add a tiny unique offset
    scores += np.linspace(0, 1e-6, n)

    def simulate_fast(s):
        """Fast greedy simulation using numpy boolean array."""
        order = np.argsort(-s)
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return count

    # 2. Baseline
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # 3. Deterministic Local Search
    # We will perform sweeps. In each sweep, we iterate through the vertices in their current order.
    # We attempt to swap adjacent pairs in the ordering if it improves the objective.
    # This is equivalent to a bubble-sort-like optimization on the ordering.
    
    max_time = 115.0
    no_improve_sweeps = 0
    max_no_improve_sweeps = 5
    
    # Current order
    current_order = np.argsort(-best_scores)
    
    while time.time() - start_time < max_time and no_improve_sweeps < max_no_improve_sweeps:
        improved = False
        
        # Forward Pass: Try to move each vertex earlier in the order
        # We iterate from the end to the beginning to allow a vertex to "bubble up"
        # but for a simple adjacent swap check, we just check each pair (i, i+1) in the order array.
        
        # To make it more effective, we can try swapping any vertex with its immediate predecessor.
        # Let's iterate through the order indices.
        for idx in range(1, n):
            if time.time() - start_time > max_time:
                break
            
            v = current_order[idx]
            u = current_order[idx - 1]
            
            # If u and v are neighbors, swapping them might change which one gets picked first.
            # If they are not neighbors, swapping them doesn't affect the relative "blocking" of each other,
            # but it might affect their relationship with other vertices? 
            # Actually, in a greedy decoder, the order only matters for who gets picked first.
            # If u and v are not neighbors, picking u first doesn't block v, and picking v first doesn't block u.
            # However, picking u first might block some w that v would have blocked, or vice versa.
            # So swapping non-neighbors can also matter.
            
            # Swap in scores
            best_scores[u], best_scores[v] = best_scores[v], best_scores[u]
            new_val = simulate_fast(best_scores)
            
            if new_val > best_val:
                best_val = new_val
                improved = True
                # Update order to reflect the swap
                current_order[idx - 1], current_order[idx] = v, u
                # We don't break, we continue the sweep with the new order
            else:
                # Revert
                best_scores[u], best_scores[v] = best_scores[v], best_scores[u]
        
        # Backward Pass: Try to move each vertex later in the order
        # This helps if a vertex is blocking another vertex that would be better off earlier?
        # Actually, if a swap in the forward pass didn't improve, a backward pass (swapping i with i+1)
        # is just the reverse of the forward pass check on the pair (i, i+1).
        # So a single forward pass that allows "bubbling" might be sufficient if we repeat it.
        # However, the simple adjacent swap in a single forward pass only allows a vertex to move left.
        # To allow movement right, we need a backward pass or a bidirectional check.
        
        # Let's do a backward pass to allow vertices to move right (lower priority)
        for idx in range(n - 2, -1, -1):
            if time.time() - start_time > max_time:
                break
            
            v = current_order[idx]
            u = current_order[idx + 1]
            
            # Swap in scores
            best_scores[u], best_scores[v] = best_scores[v], best_scores[u]
            new_val = simulate_fast(best_scores)
            
            if new_val > best_val:
                best_val = new_val
                improved = True
                # Update order
                current_order[idx], current_order[idx + 1] = u, v
            else:
                # Revert
                best_scores[u], best_scores[v] = best_scores[v], best_scores[u]
        
        if not improved:
            no_improve_sweeps += 1
        else:
            no_improve_sweeps = 0
            
    return best_scores
