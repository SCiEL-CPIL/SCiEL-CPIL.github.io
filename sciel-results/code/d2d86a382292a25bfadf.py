import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = m
    
    # Precompute compatibility
    compat = [set() for _ in range(n_triples)]
    for i in range(n_triples):
        mi = triple_masks[i]
        for j in range(i + 1, n_triples):
            mj = triple_masks[j]
            if (mi & mj).bit_count() <= 1:
                compat[i].add(j)
                compat[j].add(i)
    
    def get_alpha(T_indices):
        """Compute independence number of the triple system using DFS."""
        if not T_indices:
            return 16
        
        # Build adjacency of 'forbidden' pairs for each point
        # neighbors[p] contains pairs (a, b) such that {p, a, b} is a triple in T
        neighbors = [[] for _ in range(16)]
        for idx in T_indices:
            a, b, c = all_triples[idx]
            # For point a, the other two are b, c
            neighbors[a].append((b, c))
            neighbors[b].append((a, c))
            neighbors[c].append((a, b))
            
        best_alpha = 0
        
        def dfs(mask, count, start):
            nonlocal best_alpha
            if count > best_alpha:
                best_alpha = count
            # Prune: even if we pick all remaining points, we can't beat best_alpha
            if count + (16 - start) <= best_alpha:
                return
            
            for p in range(start, 16):
                # Check if p can be added to mask
                # p is safe if no pair (a,b) in neighbors[p] is fully in mask
                safe = True
                for a, b in neighbors[p]:
                    if (mask >> a) & 1 and (mask >> b) & 1:
                        safe = False
                        break
                if safe:
                    dfs(mask | (1 << p), count + 1, p + 1)
        
        dfs(0, 0, 0)
        return best_alpha
    
    def greedy_packing(order):
        """Build a maximal packing following the given order."""
        T = []
        T_set = set()
        for idx in order:
            compatible = all(idx in compat[t] for t in T_set)
            if compatible:
                T.append(idx)
                T_set.add(idx)
        return T
    
    def local_search(T, time_limit):
        """Improve T via swaps and additions."""
        current_T = T[:]
        current_T_set = set(current_T)
        current_alpha = get_alpha(current_T)
        best_T = current_T[:]
        best_alpha = current_alpha
        
        while time.time() < time_limit:
            improved = False
            
            # Try adding compatible triples
            for _ in range(8):
                idx = random.randint(0, n_triples - 1)
                if idx in current_T_set:
                    continue
                ok = all(idx in compat[t] for t in current_T_set)
                if ok:
                    new_T = current_T + [idx]
                    new_alpha = get_alpha(new_T)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_T = new_T
                        current_T = new_T
                        current_T_set.add(idx)
                        improved = True
                        break
                    elif new_alpha == best_alpha and len(new_T) < 35:
                        current_T = new_T
                        current_T_set.add(idx)
                        improved = True
                        break
            
            # Try single swaps
            if len(current_T) > 0:
                for _ in range(8):
                    remove_idx = random.choice(current_T)
                    add_idx = random.randint(0, n_triples - 1)
                    if add_idx == remove_idx or add_idx in current_T_set:
                        continue
                    
                    ok = all(add_idx in compat[t] for t in current_T_set if t != remove_idx)
                    if ok:
                        new_T = [t for t in current_T if t != remove_idx]
                        new_T.append(add_idx)
                        new_alpha = get_alpha(new_T)
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            best_T = new_T
                            current_T = new_T
                            current_T_set = set(new_T)
                            improved = True
                            break
            
            # Try 2-for-2 swaps
            if len(current_T) >= 2:
                for _ in range(4):
                    r1, r2 = random.sample(current_T, 2)
                    a1 = random.randint(0, n_triples - 1)
                    a2 = random.randint(0, n_triples - 1)
                    if a1 == a2 or a1 in current_T_set or a2 in current_T_set:
                        continue
                    if a1 == r1 or a1 == r2 or a2 == r1 or a2 == r2:
                        continue
                    
                    remaining = [t for t in current_T if t != r1 and t != r2]
                    ok1 = all(a1 in compat[t] for t in remaining)
                    ok2 = all(a2 in compat[t] for t in remaining)
                    if ok1 and ok2 and (a1 in compat[a2]):
                        new_T = remaining + [a1, a2]
                        new_alpha = get_alpha(new_T)
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            best_T = new_T
                            current_T = new_T
                            current_T_set = set(new_T)
                            improved = True
                            break
            
            if not improved and random.random() < 0.15:
                order = list(range(n_triples))
                random.shuffle(order)
                fresh = greedy_packing(order)
                fresh_alpha = get_alpha(fresh)
                if fresh_alpha < best_alpha:
                    best_alpha = fresh_alpha
                    best_T = fresh
                    current_T = fresh
                    current_T_set = set(fresh)
        
        return best_T, best_alpha
    
    best_alpha = 16
    best_system = []
    
    # Multiple restarts with local search
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        order = list(range(n_triples))
        random.shuffle(order)
        
        T = greedy_packing(order)
        alpha = get_alpha(T)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = T[:]
        
        # If we've found a good solution, do local search
        if alpha <= 13 and time.time() < deadline - 20:
            T, alpha = local_search(T, deadline - 15)
            if alpha < best_alpha:
                best_alpha = alpha
                best_system = T[:]
        
        # Early termination if we reach likely optimum
        if best_alpha <= 7:
            break
    
    if not best_system:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    return [list(all_triples[i]) for i in best_system]
