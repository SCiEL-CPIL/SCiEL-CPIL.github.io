import numpy as np
import time
import random

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Permutation-based Local Search (Hill Climbing with Restarts)
    1. Initialize a permutation of vertices using a strong static heuristic (Low Degree + Hub Penalty).
    2. Evaluate the permutation by simulating the greedy decoder.
    3. Perform local search on the permutation:
       - Randomly swap two vertices or insert a vertex at a new position.
       - Accept if the new permutation yields a larger or equal independent set (with slight bias to larger).
       - If stuck (no improvement for K steps), perform a random "kick" (perturb the permutation).
    4. If time remains, restart from a different heuristic or the best found so far with a new kick.
    5. Return scores corresponding to the best permutation found (rank-based scores).
    """
    start_time = time.time()
    max_time = 140.0  # Safety margin for 150s limit
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Pre-compute neighbor lists for fast access
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    def simulate(order):
        """
        Simulates the greedy decoder for a given permutation of vertices.
        Returns the size of the independent set.
        """
        active_set = set(range(n))
        count = 0
        for v in order:
            if v in active_set:
                count += 1
                active_set.remove(v)
                for u in neighbors_list[v]:
                    if u in active_set:
                        active_set.remove(u)
        return count

    # 1. Initial Heuristic
    deg = adj.sum(axis=1)
    # Calculate a "hub penalty": vertices connected to high-degree nodes are less desirable
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    hub_penalty = neighbor_deg_sum / safe_deg
    
    # Score for sorting: We want low degree and low hub penalty to be "better" (higher priority)
    # So we sort by ascending (degree + hub_penalty)
    init_scores = deg.astype(float) + 0.1 * hub_penalty
    # Add small random noise to break ties and provide diversity for restarts
    rng = np.random.default_rng(42)
    
    best_order = None
    best_size = -1
    
    # We will run multiple "sessions" of local search
    # Each session starts from a heuristic or a perturbed best solution
    sessions = 0
    max_sessions = 50 # Cap on sessions, but time is the real limit
    
    while time.time() - start_time < max_time:
        sessions += 1
        
        # Initialize current order
        if sessions == 1:
            # First session: Pure heuristic
            sort_idx = np.argsort(init_scores)
            current_order = list(sort_idx)
        else:
            # Subsequent sessions: Start from best found, then perturb (kick)
            if best_order is None:
                sort_idx = np.argsort(init_scores + rng.random(n) * 0.1)
                current_order = list(sort_idx)
            else:
                current_order = list(best_order)
                # Perform a random kick: swap 5-10 random pairs
                num_kicks = rng.integers(5, 15)
                for _ in range(num_kicks):
                    i, j = rng.integers(0, n, size=2)
                    current_order[i], current_order[j] = current_order[j], current_order[i]
        
        current_size = simulate(current_order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = list(current_order)
        
        # Local Search Loop
        no_improve_count = 0
        max_no_improve = 200 # If stuck for 200 moves, break to restart
        
        while time.time() - start_time < max_time and no_improve_count < max_no_improve:
            # Generate a random move
            # 50% chance: Swap two random positions
            # 50% chance: Insert a vertex at a new random position
            move_type = rng.integers(0, 2)
            
            if move_type == 0:
                i, j = rng.integers(0, n, size=2)
                if i == j:
                    continue
                # Swap
                current_order[i], current_order[j] = current_order[j], current_order[i]
                new_size = simulate(current_order)
                
                if new_size >= current_size:
                    if new_size > current_size:
                        no_improve_count = 0
                        if new_size > best_size:
                            best_size = new_size
                            best_order = list(current_order)
                    else:
                        # Equal size, accept with some probability to explore? 
                        # Or just reject to stay focused? 
                        # Let's reject equal size to force progress, unless we are stuck.
                        # Actually, accepting equal size can help escape plateaus.
                        # But for speed, let's only accept strictly better or equal if stuck.
                        if no_improve_count > 50:
                            current_size = new_size
                            no_improve_count = 0
                        else:
                            # Revert
                            current_order[i], current_order[j] = current_order[j], current_order[i]
                            no_improve_count += 1
                else:
                    # Revert
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                    no_improve_count += 1
            else:
                # Insertion move
                i = rng.integers(0, n)
                v = current_order[i]
                # Remove v
                current_order.pop(i)
                # Insert at random position j
                j = rng.integers(0, n)
                current_order.insert(j, v)
                
                new_size = simulate(current_order)
                
                if new_size >= current_size:
                    if new_size > current_size:
                        no_improve_count = 0
                        if new_size > best_size:
                            best_size = new_size
                            best_order = list(current_order)
                    else:
                        if no_improve_count > 50:
                            current_size = new_size
                            no_improve_count = 0
                        else:
                            # Revert: remove from j, insert back at i
                            current_order.pop(j)
                            current_order.insert(i, v)
                            no_improve_count += 1
                else:
                    # Revert
                    current_order.pop(j)
                    current_order.insert(i, v)
                    no_improve_count += 1

    # Construct final scores from best_order
    # The decoder picks in descending score order.
    # So the vertex at index 0 in best_order should have the highest score.
    # Score = n - 1 - index
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - 1 - idx
        
    return scores
