import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    # Convert to list of sets for efficient neighbor manipulation
    neighbors = [set(np.where(adjacency[i])[0]) for i in range(n)]
    initial_degrees = [len(neighbors[i]) for i in range(n)]
    
    best_score = -1
    best_scores_array = np.zeros(n)
    
    def evaluate_order(order):
        # Construct scores
        scores = np.zeros(n)
        for i, v in enumerate(order):
            scores[v] = n - i
        # Run decoder
        # We can just return len(order) if we trust the simulation, 
        # but running the actual decoder is safer and fast.
        order_dec = np.argsort(-scores)
        selected = 0
        blocked = np.zeros(n, dtype=bool)
        for v in order_dec:
            if not blocked[v]:
                selected += 1
                blocked[adjacency[v]] = True
                blocked[v] = True
        return selected, scores

    def simulate(strategy, random_seed=None):
        alive = np.ones(n, dtype=bool)
        neighbors_copy = [set(neighbors[i]) for i in range(n)]
        current_degrees = [len(neighbors_copy[i]) for i in range(n)]
        order = []
        
        rng = np.random.default_rng(random_seed) if random_seed is not None else None
        
        while np.any(alive):
            candidates = [i for i in range(n) if alive[i]]
            if not candidates:
                break
            
            if strategy == 'min_degree':
                if random_seed is not None:
                    # Tie break randomly among min degree
                    min_deg = min(current_degrees[i] for i in candidates)
                    min_cands = [i for i in candidates if current_degrees[i] == min_deg]
                    best_v = rng.choice(min_cands)
                else:
                    best_v = min(candidates, key=lambda i: (current_degrees[i], i))
            elif strategy == 'max_sum':
                def cost(i):
                    s = 0
                    for u in neighbors_copy[i]:
                        if alive[u]:
                            s += current_degrees[u]
                    return -s # Maximize sum -> Minimize -sum
                best_v = min(candidates, key=lambda i: (cost(i), i))
            elif strategy == 'min_sum':
                def cost(i):
                    s = 0
                    for u in neighbors_copy[i]:
                        if alive[u]:
                            s += current_degrees[u]
                    return s
                best_v = min(candidates, key=lambda i: (cost(i), i))
            else:
                best_v = min(candidates, key=lambda i: (current_degrees[i], i))
                
            order.append(best_v)
            
            to_remove = set(neighbors_copy[best_v])
            to_remove.add(best_v)
            
            for u in to_remove:
                if alive[u]:
                    alive[u] = False
                    for w in neighbors_copy[u]:
                        if alive[w]:
                            neighbors_copy[w].remove(u)
                            current_degrees[w] -= 1
                            
        return order

    # Strategy 1: Min Degree (Deterministic)
    order1 = simulate('min_degree')
    size1, scores1 = evaluate_order(order1)
    if size1 > best_score:
        best_score = size1
        best_scores_array = scores1
        
    if time.time() > deadline:
        return best_scores_array

    # Strategy 2: Max Sum (Deterministic)
    order2 = simulate('max_sum')
    size2, scores2 = evaluate_order(order2)
    if size2 > best_score:
        best_score = size2
        best_scores_array = scores2
        
    if time.time() > deadline:
        return best_scores_array

    # Strategy 3: Min Sum (Deterministic)
    order3 = simulate('min_sum')
    size3, scores3 = evaluate_order(order3)
    if size3 > best_score:
        best_score = size3
        best_scores_array = scores3
        
    if time.time() > deadline:
        return best_scores_array

    # Strategy 4-9: Min Degree (Random Tie-Break)
    for i in range(6):
        if time.time() > deadline:
            break
        order_r = simulate('min_degree', random_seed=i)
        size_r, scores_r = evaluate_order(order_r)
        if size_r > best_score:
            best_score = size_r
            best_scores_array = scores_r

    return best_scores_array
