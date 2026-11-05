import numpy as np
import time

def priority(adjacency):
    """
    Assign priority scores for a greedy MIS decoder.
    
    New structural approach: Direct permutation optimization with simulated annealing
    and incremental delta-evaluation. Instead of optimizing a continuous score vector,
    we work directly on the vertex permutation (the order in which vertices are considered).
    
    Key insight: The greedy decoder's output depends only on the relative ordering, not
    the actual score values. We can represent the solution as a permutation and use
    simulated annealing with fast incremental evaluation to optimize it.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    deg = adj.sum(axis=1).astype(np.int32)
    
    def simulate_from_order(order):
        """Simulate greedy MIS given a vertex order (array of vertex indices)."""
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                for nb in neighbors_list[v]:
                    if active[nb]:
                        active[nb] = False
        return count
    
    # Initial heuristic ordering: prioritize low-degree vertices
    # This gives a good starting point for the search
    base_order = np.argsort(deg, kind='stable')
    current_order = base_order.copy()
    current_val = simulate_from_order(current_order)
    best_order = current_order.copy()
    best_val = current_val
    
    # Simulated annealing on the permutation
    rng = np.random.default_rng(42)
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    last_improve_time = time.time()
    no_improve_limit = 2.0  # seconds
    
    step = 0
    while time.time() < deadline - 5:
        step += 1
        
        # Generate a random move: swap two positions in the order
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        # Perform swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Evaluate new order
        new_val = simulate_from_order(current_order)
        delta = new_val - current_val
        
        # Accept or reject
        if delta > 0 or (T > T_min and rng.random() < np.exp(delta / T)):
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_order = current_order.copy()
                last_improve_time = time.time()
        else:
            # Reject: swap back
            current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Cooling schedule
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # Reheat if stuck
        if time.time() - last_improve_time > no_improve_limit:
            T = 0.5
            last_improve_time = time.time()
            # Also try a larger perturbation: reverse a random segment
            seg_start = rng.integers(0, n - 1)
            seg_end = rng.integers(seg_start + 1, n)
            current_order[seg_start:seg_end] = current_order[seg_start:seg_end][::-1]
            new_val = simulate_from_order(current_order)
            delta = new_val - current_val
            if delta > 0:
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best_order = current_order.copy()
        
        # Additional local search: try 2-opt style moves (swap + neighbor check)
        if step % 100 == 0 and time.time() < deadline - 10:
            # Try to improve by moving a vertex to a better position
            v_idx = rng.integers(0, n)
            v = current_order[v_idx]
            best_new_pos = v_idx
            best_new_val = current_val
            
            # Try moving v to a few other positions
            for offset in [1, -1, 2, -2, 5, -5]:
                new_pos = v_idx + offset
                if 0 <= new_pos < n:
                    # Temporarily move v to new_pos
                    current_order[v_idx], current_order[new_pos] = current_order[new_pos], current_order[v_idx]
                    temp_val = simulate_from_order(current_order)
                    current_order[v_idx], current_order[new_pos] = current_order[new_pos], current_order[v_idx]
                    
                    if temp_val > best_new_val:
                        best_new_val = temp_val
                        best_new_pos = new_pos
            
            # Apply the best move if it improved
            if best_new_pos != v_idx:
                current_order[v_idx], current_order[best_new_pos] = current_order[best_new_pos], current_order[v_idx]
                current_val = best_new_val
                if current_val > best_val:
                    best_val = current_val
                    best_order = current_order.copy()
                    last_improve_time = time.time()
    
    # Convert best_order to scores
    # The decoder picks vertices in descending score order.
    # So, vertex at position 0 in best_order should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
    
    return scores
