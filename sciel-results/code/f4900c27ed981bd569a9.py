import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety (evaluator kills at 90s)
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    # Build conflict graph
    node_set = set(valid_nodes)
    conflicts = {v: set() for v in valid_nodes}
    for i_idx, v in enumerate(valid_nodes):
        for u in valid_nodes[i_idx + 1:]:
            if v + u in squares:
                conflicts[v].add(u)
                conflicts[u].add(v)
    
    # Precompute degrees for ordering
    degrees = {v: len(conflicts[v]) for v in valid_nodes}
    
    def is_independent(v, chosen):
        return not (conflicts[v] & chosen)
        
    def greedy_mis(order):
        chosen = set()
        for v in order:
            if is_independent(v, chosen):
                chosen.add(v)
        return chosen

    def local_search(chosen, max_iters=500):
        chosen = set(chosen)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # 1. Try adding nodes
            for v in valid_nodes:
                if v in chosen:
                    continue
                if is_independent(v, chosen):
                    chosen.add(v)
                    improved = True
            
            # 2. Try 1-Swaps
            for v in valid_nodes:
                if v in chosen:
                    continue
                conf_in_chosen = conflicts[v] & chosen
                if len(conf_in_chosen) == 1:
                    u = next(iter(conf_in_chosen))
                    # Since conf_in_chosen has size 1, removing u makes v independent
                    chosen.discard(u)
                    chosen.add(v)
                    improved = True
                elif len(conf_in_chosen) > 1:
                    # Try 2-Swap: remove two conflicts, add v
                    # Only beneficial if we can add back 2 other nodes, but for now
                    # let's just do 1-swaps for speed. 2-swaps are complex.
                    pass
            
        return chosen

    def perturbation(chosen, k):
        # Remove k random nodes
        chosen = set(chosen)
        to_remove = random.sample(list(chosen), min(k, len(chosen)))
        for u in to_remove:
            chosen.discard(u)
        # Re-add greedily from the removed + some random others?
        # Just run local search again on the perturbed set
        return local_search(chosen, max_iters=50)

    best = set()
    random.seed(42)
    
    # Initial strategies
    orders = [
        valid_nodes,
        sorted(valid_nodes, key=lambda x: len(conflicts[x])), # Degree ascending
        sorted(valid_nodes, key=lambda x: -len(conflicts[x])), # Degree descending
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        chosen = greedy_mis(order)
        chosen = local_search(chosen, max_iters=200)
        if len(chosen) > len(best):
            best = chosen

    # Iterated Local Search
    # Start from best or random
    current = set(best)
    if not current:
        current = greedy_mis(valid_nodes)
        
    # Structural change: Use a more aggressive restart strategy and mixed perturbation
    # Also, try to find a better initial solution using simulated annealing-like approach
    # or just more aggressive ILS with varying perturbation sizes
    
    for i in range(200): # Increased restarts
        if time.time() > deadline:
            break
        
        # Varying perturbation size
        k = random.randint(2, 15)
        current = perturbation(current, k)
        
        if len(current) > len(best):
            best = current
            current = set(best)
            
        # Occasionally restart from scratch with random order
        if i % 5 == 0:
            random_order = valid_nodes[:]
            random.shuffle(random_order)
            cand = greedy_mis(random_order)
            cand = local_search(cand, max_iters=100)
            if len(cand) > len(best):
                best = cand
                current = set(best)
            elif len(cand) > len(current):
                current = cand

    # Final polish: try to improve the best solution with more iterations
    if time.time() < deadline - 5:
        polished = local_search(best, max_iters=1000)
        if len(polished) > len(best):
            best = polished

    if not best:
        best = {valid_nodes[0]}
        
    return list(best)
