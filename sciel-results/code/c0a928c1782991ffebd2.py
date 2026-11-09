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
            
            # 1. Try adding nodes (sorted by degree ascending for better choices)
            candidates = [v for v in valid_nodes if v not in chosen and is_independent(v, chosen)]
            candidates.sort(key=lambda x: len(conflicts[x]))
            for v in candidates:
                if is_independent(v, chosen):
                    chosen.add(v)
                    improved = True
            
            # 2. Try 1-Swaps: remove one conflict, add v
            for v in valid_nodes:
                if v in chosen:
                    continue
                conf_in_chosen = conflicts[v] & chosen
                if len(conf_in_chosen) == 1:
                    u = next(iter(conf_in_chosen))
                    # Check if removing u allows adding v and potentially more
                    chosen.discard(u)
                    chosen.add(v)
                    improved = True
                elif len(conf_in_chosen) == 2:
                    # Try 2-Swap: remove two conflicts, add v
                    # Only beneficial if we can add back at least 2 other nodes
                    conf_list = list(conf_in_chosen)
                    u1, u2 = conf_list[0], conf_list[1]
                    
                    # Temporarily remove u1, u2 and add v
                    chosen.discard(u1)
                    chosen.discard(u2)
                    chosen.add(v)
                    
                    # Try to add back any nodes that were blocked by u1 or u2
                    added_back = 0
                    for w in valid_nodes:
                        if w in chosen:
                            continue
                        if w == v:
                            continue
                        if is_independent(w, chosen):
                            chosen.add(w)
                            added_back += 1
                        if time.time() > deadline:
                            break
                    
                    # If we added back at least as many as we removed, keep the change
                    if added_back >= 1:
                        improved = True
                    else:
                        # Revert
                        chosen.discard(v)
                        chosen.add(u1)
                        chosen.add(u2)
            
        return chosen

    def simulated_annealing(chosen, T_start=1.0, T_end=0.1, cooling=0.99):
        chosen = set(chosen)
        T = T_start
        while T > T_end and time.time() < deadline - 2:
            T *= cooling
            # Pick a random move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                # Find a random node that can be added
                candidates = [v for v in valid_nodes if v not in chosen and is_independent(v, chosen)]
                if candidates:
                    v = random.choice(candidates)
                    new_chosen = chosen | {v}
                    # Always accept improvement
                    chosen = new_chosen
            elif move_type == 'remove':
                if chosen:
                    u = random.choice(list(chosen))
                    new_chosen = chosen - {u}
                    # Accept with probability exp(-delta/T), delta = 1 (worse)
                    if random.random() < math.exp(-1.0 / T):
                        chosen = new_chosen
            elif move_type == 'swap':
                if chosen:
                    u = random.choice(list(chosen))
                    # Find v not in chosen such that v is independent of chosen \ {u}
                    # v can conflict with u, but not with others
                    potential_v = [v for v in valid_nodes if v not in chosen and is_independent(v, chosen - {u})]
                    if potential_v:
                        v = random.choice(potential_v)
                        new_chosen = (chosen - {u}) | {v}
                        # Size is same, so delta = 0. Always accept to explore.
                        chosen = new_chosen
                        
            # Periodic local search to fix up
            if random.random() < 0.05:
                chosen = local_search(chosen, max_iters=10)
                
        return chosen

    def perturbation(chosen, k):
        # Remove k random nodes, preferring high-degree nodes for more impactful changes
        chosen = set(chosen)
        if not chosen:
            return chosen
        
        # Sort by degree descending to target high-conflict nodes
        sorted_chosen = sorted(chosen, key=lambda x: -len(conflicts[x]))
        to_remove = random.sample(sorted_chosen, min(k, len(chosen)))
        for u in to_remove:
            chosen.discard(u)
        # Re-add greedily from the removed + some random others?
        # Just run local search again on the perturbed set
        return local_search(chosen, max_iters=50)

    best = set()
    random.seed(42)
    
    # Initial strategies with more diverse orderings
    orders = [
        valid_nodes,
        sorted(valid_nodes, key=lambda x: len(conflicts[x])), # Degree ascending
        sorted(valid_nodes, key=lambda x: -len(conflicts[x])), # Degree descending
        sorted(valid_nodes, key=lambda x: (len(conflicts[x]), x)), # Degree then value
        sorted(valid_nodes, key=lambda x: (-len(conflicts[x]), x)), # Negative degree then value
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        chosen = greedy_mis(order)
        chosen = local_search(chosen, max_iters=300)
        if len(chosen) > len(best):
            best = chosen

    # Iterated Local Search with improved perturbation
    current = set(best)
    if not current:
        current = greedy_mis(valid_nodes)
        
    # More aggressive ILS with varying perturbation sizes and occasional full restarts
    for i in range(300):
        if time.time() > deadline:
            break
        
        # Varying perturbation size - smaller for fine-tuning, larger for exploration
        if i < 50:
            k = random.randint(1, 5)
        elif i < 150:
            k = random.randint(3, 10)
        else:
            k = random.randint(5, 20)
            
        # Occasionally use Simulated Annealing instead of simple perturbation
        if random.random() < 0.2:
            current = simulated_annealing(current, T_start=1.0, T_end=0.5, cooling=0.95)
        else:
            current = perturbation(current, k)
        
        if len(current) > len(best):
            best = current
            current = set(best)
            
        # Occasionally restart from scratch with random order
        if i % 7 == 0:
            random_order = valid_nodes[:]
            random.shuffle(random_order)
            cand = greedy_mis(random_order)
            cand = local_search(cand, max_iters=150)
            if len(cand) > len(best):
                best = cand
                current = set(best)
            elif len(cand) > len(current):
                current = cand

    # Final polish: try to improve the best solution with more iterations
    if time.time() < deadline - 5:
        polished = local_search(best, max_iters=2000)
        if len(polished) > len(best):
            best = polished
    
    # One more final attempt with a different strategy
    if time.time() < deadline - 3 and best:
        # Try removing a few random nodes from best and re-optimizing
        for attempt in range(10):
            if time.time() > deadline - 2:
                break
            trial = set(best)
            to_remove = random.sample(list(trial), min(3, len(trial)))
            for u in to_remove:
                trial.discard(u)
            trial = local_search(trial, max_iters=100)
            if len(trial) > len(best):
                best = trial
        
        # Final Simulated Annealing polish
        if time.time() < deadline - 1:
            trial = simulated_annealing(best, T_start=0.5, T_end=0.1, cooling=0.98)
            if len(trial) > len(best):
                best = trial

    if not best:
        best = {valid_nodes[0]}
        
    return list(best)
