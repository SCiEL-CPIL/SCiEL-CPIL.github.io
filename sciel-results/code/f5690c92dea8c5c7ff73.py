import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Precompute neighbor lists as numpy arrays for fast indexing
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = adjacency.sum(axis=1).astype(np.float64)
    
    # Fast greedy MIS evaluation
    def greedy_mis_size(order):
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                if len(neighbors[v]) > 0:
                    selected[neighbors[v]] = True
        return count
    
    # Simulated Annealing based search for better orderings
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_order = None
    
    def update_best(order, size):
        nonlocal best_size, best_order
        if size > best_size:
            best_size = size
            best_order = order[:]
    
    # Generate initial orderings
    initial_orders = []
    
    # 1. Minimum degree (ascending degree) - often good for MIS
    initial_orders.append(np.argsort(degrees))
    
    # 2. Maximum degree (descending degree)
    initial_orders.append(np.argsort(-degrees))
    
    # 3. Random
    initial_orders.append(rng.permutation(n))
    
    # 4. Noisy degree
    noisy_deg = degrees + rng.normal(0, 1, n)
    initial_orders.append(np.argsort(noisy_deg))
    
    # 5. Another random
    initial_orders.append(rng.permutation(n))
    
    # 6. Degree squared (emphasize high degree)
    initial_orders.append(np.argsort(-degrees**2))
    
    # 7. Inverse degree (emphasize low degree)
    inv_deg = 1.0 / (degrees + 1)
    initial_orders.append(np.argsort(-inv_deg))
    
    # Evaluate initial orders
    for order in initial_orders:
        size = greedy_mis_size(order)
        update_best(order, size)
    
    # Simulated Annealing
    # We'll run SA on each initial order to find better solutions
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.999
    
    # For each initial order, run SA
    for r in range(len(initial_orders)):
        if time.time() > deadline - 15:
            break
        
        order = initial_orders[r].copy()
        current_size = greedy_mis_size(order)
        update_best(order, current_size)
        
        T = T_start
        sa_steps = 0
        max_sa_steps = 10000
        
        while T > T_end and sa_steps < max_sa_steps:
            if time.time() > deadline - 5:
                break
            
            # Pick two random indices to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                sa_steps += 1
                T *= cooling_rate
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            delta = new_size - current_size
            
            if delta >= 0:
                # Accept improvement
                current_size = new_size
                update_best(order, current_size)
            else:
                # Accept worse solution with probability exp(delta/T)
                if rng.random() < np.exp(delta / T):
                    current_size = new_size
                else:
                    # Revert
                    order[i], order[j] = order[j], order[i]
            
            sa_steps += 1
            T *= cooling_rate
    
    # Additional local search (hill climbing) on the best order found
    if best_order is not None and time.time() < deadline - 10:
        order = best_order.copy()
        current_size = greedy_mis_size(order)
        
        # Hill climbing with random restarts within the best neighborhood
        for step in range(5000):
            if time.time() > deadline - 3:
                break
                
            # Pick two random indices to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            
            if new_size >= current_size:
                current_size = new_size
                update_best(order, current_size)
            else:
                # Revert
                order[i], order[j] = order[j], order[i]
    
    # Convert best_order to scores
    # The decoder sorts by descending score.
    # So the first element in best_order should have the highest score.
    if best_order is None:
        # Fallback: min degree
        scores = np.zeros(n)
        order = np.argsort(degrees)
        for rank, v in enumerate(order):
            scores[v] = n - rank
        return scores
    
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
