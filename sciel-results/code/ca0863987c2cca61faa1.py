<<<<<<< SEARCH
def _fast_anneal(values, n, m, edge_sp, time_budget=5.0):
    """
    Fast simulated annealing on edge selection.
    Avoids O(N) candidate filtering by using rejection sampling or direct random pick from all edges.
    """
    start_time = time.time()
    all_edges = list(combinations(range(n), 2))
    total_edges = len(all_edges)
    
    # Initialize with greedy: pick edges with highest collision potential
    # For speed, we just pick random initially or a simple greedy if time permits
    # Let's do a quick greedy init: sort by (sum_count + prod_count) is expensive to compute globally.
    # Instead, just start with a random set of m edges.
    current_edges = set(random.sample(all_edges, m))
    
    current_sums_set, current_prods_set, current_score = _score_detailed(values, list(current_edges))
    best_score = current_score
    best_edges = set(current_edges)
    
    sum_counts = Counter()
    prod_counts = Counter()
    for e in current_edges:
        s, p = edge_sp[e]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    max_iter = 50000 # Fixed high number, time checked inside
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > _DEADLINE - 2:
            break
        if current_score <= 3:
            break
        
        # Pick random edge to remove
        remove_edge = random.choice(list(current_edges))
        
        # Pick random edge to add (from all edges)
        # If it's the same or already in set, we just skip or pick again.
        # To avoid infinite loops, we just pick a random index.
        add_edge = all_edges[random.randint(0, total_edges - 1)]
        
        if add_edge == remove_edge:
            continue
            
        if add_edge in current_edges:
            # If we pick an edge already in the set, it's not a valid swap (no change in set size)
            # We can either skip or treat it as a null move. Skipping is safer.
            continue

        rs, rp = edge_sp[remove_edge]
        as_, ap = edge_sp[add_edge]
        
        # Compute delta
        delta_sums = 0
        if sum_counts[rs] == 1:
            delta_sums -= 1
        if sum_counts[as_] == 0:
            delta_sums += 1
        
        delta_prods = 0
        if prod_counts[rp] == 1:
            delta_prods -= 1
        if prod_counts[ap] == 0:
            delta_prods += 1
        
        new_num_sums = len(current_sums_set) + delta_sums
        new_num_prods = len(current_prods_set) + delta_prods
        new_score = max(new_num_sums, new_num_prods)
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or (T > T_min and random.random() < math.exp(-delta_score / T)):
            # Apply move
            current_edges.discard(remove_edge)
            current_edges.add(add_edge)
            
            # Update counts
            sum_counts[rs] -= 1
            if sum_counts[rs] == 0:
                del sum_counts[rs]
            prod_counts[rp] -= 1
            if prod_counts[rp] == 0:
                del prod_counts[rp]
            
            sum_counts[as_] += 1
            prod_counts[ap] += 1
            
            # Update sets
            if sum_counts[rs] == 0:
                current_sums_set.discard(rs)
            if sum_counts[as_] == 1:
                current_sums_set.add(as_)
            
            if prod_counts[rp] == 0:
                current_prods_set.discard(rp)
            if prod_counts[ap] == 1:
                current_prods_set.add(ap)
            
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
        
        T = max(T_min, T * cooling)
    
    return best_score, best_edges
=======
def _fast_anneal(values, n, m, edge_sp, time_budget=5.0):
    """
    Fast simulated annealing on edge selection with greedy init and polish.
    """
    start_time = time.time()
    all_edges = list(combinations(range(n), 2))
    total_edges = len(all_edges)
    
    # Initialize with greedy: pick edges with highest collision potential
    current_edges = _greedy_init_edges(n, m, edge_sp)
    
    current_sums_set, current_prods_set, current_score = _score_detailed(values, list(current_edges))
    best_score = current_score
    best_edges = set(current_edges)
    
    sum_counts = Counter()
    prod_counts = Counter()
    for e in current_edges:
        s, p = edge_sp[e]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    max_iter = 50000 # Fixed high number, time checked inside
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > _DEADLINE - 2:
            break
        if current_score <= 3:
            break
        
        # Pick random edge to remove
        remove_edge = random.choice(list(current_edges))
        
        # Pick random edge to add (from all edges)
        add_edge = all_edges[random.randint(0, total_edges - 1)]
        
        if add_edge == remove_edge:
            continue
            
        if add_edge in current_edges:
            continue

        rs, rp = edge_sp[remove_edge]
        as_, ap = edge_sp[add_edge]
        
        # Compute delta
        delta_sums = 0
        if sum_counts[rs] == 1:
            delta_sums -= 1
        if sum_counts[as_] == 0:
            delta_sums += 1
        
        delta_prods = 0
        if prod_counts[rp] == 1:
            delta_prods -= 1
        if prod_counts[ap] == 0:
            delta_prods += 1
        
        new_num_sums = len(current_sums_set) + delta_sums
        new_num_prods = len(current_prods_set) + delta_prods
        new_score = max(new_num_sums, new_num_prods)
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or (T > T_min and random.random() < math.exp(-delta_score / T)):
            # Apply move
            current_edges.discard(remove_edge)
            current_edges.add(add_edge)
            
            # Update counts
            sum_counts[rs] -= 1
            if sum_counts[rs] == 0:
                del sum_counts[rs]
            prod_counts[rp] -= 1
            if prod_counts[rp] == 0:
                del prod_counts[rp]
            
            sum_counts[as_] += 1
            prod_counts[ap] += 1
            
            # Update sets
            if sum_counts[rs] == 0:
                current_sums_set.discard(rs)
            if sum_counts[as_] == 1:
                current_sums_set.add(as_)
            
            if prod_counts[rp] == 0:
                current_prods_set.discard(rp)
            if prod_counts[ap] == 1:
                current_prods_set.add(ap)
            
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
        
        T = max(T_min, T * cooling)
    
    # Polish pass: try all single-edge swaps from best state
    if time.time() < _DEADLINE - 5:
        polish_start = time.time()
        polish_budget = min(5.0, _DEADLINE - 5 - polish_start)
        
        # Rebuild state from best_edges
        polish_edges = set(best_edges)
        polish_sums_set, polish_prods_set, polish_score = _score_detailed(values, list(polish_edges))
        polish_sum_counts = Counter()
        polish_prod_counts = Counter()
        for e in polish_edges:
            s, p = edge_sp[e]
            polish_sum_counts[s] += 1
            polish_prod_counts[p] += 1
        
        improved = True
        while improved and time.time() - polish_start < polish_budget:
            improved = False
            for rem in list(polish_edges):
                if time.time() - polish_start > polish_budget:
                    break
                rs, rp = edge_sp[rem]
                for add in all_edges:
                    if add in polish_edges:
                        continue
                    as_, ap = edge_sp[add]
                    
                    delta_sums = 0
                    if polish_sum_counts[rs] == 1:
                        delta_sums -= 1
                    if polish_sum_counts.get(as_, 0) == 0:
                        delta_sums += 1
                    
                    delta_prods = 0
                    if polish_prod_counts[rp] == 1:
                        delta_prods -= 1
                    if polish_prod_counts.get(ap, 0) == 0:
                        delta_prods += 1
                    
                    new_num_sums = len(polish_sums_set) + delta_sums
                    new_num_prods = len(polish_prods_set) + delta_prods
                    new_score = max(new_num_sums, new_num_prods)
                    
                    if new_score < polish_score:
                        # Apply swap
                        polish_edges.discard(rem)
                        polish_edges.add(add)
                        
                        polish_sum_counts[rs] -= 1
                        if polish_sum_counts[rs] == 0:
                            del polish_sum_counts[rs]
                        polish_prod_counts[rp] -= 1
                        if polish_prod_counts[rp] == 0:
                            del polish_prod_counts[rp]
                        
                        polish_sum_counts[as_] += 1
                        polish_prod_counts[ap] += 1
                        
                        if polish_sum_counts[rs] == 0:
                            polish_sums_set.discard(rs)
                        if polish_sum_counts[as_] == 1:
                            polish_sums_set.add(as_)
                        
                        if polish_prod_counts[rp] == 0:
                            polish_prods_set.discard(rp)
                        if polish_prod_counts[ap] == 1:
                            polish_prods_set.add(ap)
                        
                        polish_score = new_score
                        if polish_score < best_score:
                            best_score = polish_score
                            best_edges = set(polish_edges)
                        improved = True
                        break
    
    return best_score, best_edges
>>>>>>> REPLACE

