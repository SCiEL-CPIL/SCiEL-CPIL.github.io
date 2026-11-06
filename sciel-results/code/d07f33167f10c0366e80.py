import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Generate all triples
    all_triples = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                all_triples.append((i, j, k))
    
    triple_set = set(all_triples)
    n_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Pre-compute conflict groups: for each triple idx, list of groups (each group is a tuple of 7 indices)
    # If all 7 indices in a group are in H, then adding idx creates a K(2,2,2)
    conflict_groups = [[] for _ in range(n_triples)]
    
    for idx, T in enumerate(all_triples):
        a, b, c = T
        T_set = frozenset((a, b, c))
        for a_prime in range(16):
            if a_prime in T_set:
                continue
            for b_prime in range(16):
                if b_prime in T_set or b_prime == a_prime:
                    continue
                if a_prime == b or b_prime == a:
                    continue
                for c_prime in range(16):
                    if c_prime in T_set or c_prime in (a_prime, b_prime):
                        continue
                    if c_prime == a or c_prime == b:
                        continue
                    others = [
                        (a, b, c_prime),
                        (a, b_prime, c),
                        (a, b_prime, c_prime),
                        (a_prime, b, c),
                        (a_prime, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b_prime, c_prime)
                    ]
                    valid = True
                    idxs = []
                    for o in others:
                        if o not in triple_set:
                            valid = False
                            break
                        idxs.append(triple_to_idx[o])
                    if valid:
                        conflict_groups[idx].append(tuple(idxs))
    
    def greedy_solve(seed, time_limit, start_with_star=True):
        random.seed(seed)
        H_set = set()
        
        if start_with_star:
            for idx, t in enumerate(all_triples):
                if 0 in t:
                    H_set.add(idx)
        
        candidates = [idx for idx in range(n_triples) if idx not in H_set]
        random.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > time_limit:
                break
            safe = True
            for group in conflict_groups[idx]:
                all_in = True
                for gidx in group:
                    if gidx not in H_set:
                        all_in = False
                        break
                if all_in:
                    safe = False
                    break
            if safe:
                H_set.add(idx)
        
        return H_set
    
    best_H_set = set()
    best_count = 0
    
    # Multiple restarts with different seeds
    seeds = [42, 1, 2, 3, 7, 13, 42, 100, 200, 300]
    for i, seed in enumerate(seeds):
        remaining = deadline - time.time()
        if remaining < 8:
            break
        t_limit = min(deadline - 3, time.time() + remaining * 0.7)
        H_set = greedy_solve(seed, t_limit, start_with_star=True)
        count = len(H_set)
        if count > best_count:
            best_count = count
            best_H_set = H_set
    
    # Try without star initialization
    for seed in [50, 51, 52, 53, 54]:
        remaining = deadline - time.time()
        if remaining < 8:
            break
        t_limit = min(deadline - 3, time.time() + remaining * 0.7)
        H_set = greedy_solve(seed, t_limit, start_with_star=False)
        count = len(H_set)
        if count > best_count:
            best_count = count
            best_H_set = H_set
    
    # Local search: try to add more triples by removing some and re-adding
    remaining = deadline - time.time()
    if remaining > 20:
        # Try swapping: remove one triple, try to add multiple others
        H_set = set(best_H_set)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Find triples that are currently blocked
            blocked = []
            for idx in range(n_triples):
                if idx in H_set:
                    continue
                is_blocked = False
                for group in conflict_groups[idx]:
                    all_in = True
                    for gidx in group:
                        if gidx not in H_set:
                            all_in = False
                            break
                    if all_in:
                        is_blocked = True
                        break
                if is_blocked:
                    blocked.append(idx)
            
            # Try removing one triple from H and see if we can add more
            for idx in range(n_triples):
                if time.time() > deadline - 10:
                    break
                if idx not in H_set:
                    continue
                # Try removing idx
                H_set.remove(idx)
                added = 0
                for bidx in blocked:
                    if bidx == idx:
                        continue
                    safe = True
                    for group in conflict_groups[bidx]:
                        all_in = True
                        for gidx in group:
                            if gidx not in H_set:
                                all_in = False
                                break
                        if all_in:
                            safe = False
                            break
                    if safe:
                        H_set.add(bidx)
                        added += 1
                if added > 1:  # Net gain
                    improved = True
                else:
                    H_set.add(idx)  # Put it back
                if time.time() > deadline - 10:
                    break
            
            if len(H_set) > best_count:
                best_count = len(H_set)
                best_H_set = set(H_set)
    
    return [list(all_triples[idx]) for idx in best_H_set]
