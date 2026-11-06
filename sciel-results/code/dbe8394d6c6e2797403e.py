import time
import random
import math

def construct():
    start_time = time.time()
    # Safety margin: 80s out of 90s limit. 
    deadline = start_time + 80.0
    
    n = 256
    
    # Perfect squares up to 512 (max sum 256+256)
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
        
    # Diagonal constraint: 2a is a perfect square
    # a = 2 * m^2
    diagonal_bad = set()
    m = 1
    while 2 * m * m <= n:
        diagonal_bad.add(2 * m * m)
        m += 1
        
    # Available vertices: those not violating diagonal constraint
    available_vertices = [i for i in range(1, n + 1) if i not in diagonal_bad]
    available_set = set(available_vertices)
    avail_list = sorted(available_vertices)
    
    # Build conflict graph
    conflicts = {v: set() for v in available_vertices}
    
    for a in available_vertices:
        for b in available_vertices:
            if a < b and (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    # Precompute degree for ordering
    degrees = {v: len(conflicts[v]) for v in available_vertices}
    
    def greedy_min_degree(pool):
        current_pool = set(pool)
        selected = set()
        while current_pool:
            min_deg = float('inf')
            best_v = None
            # Use precomputed static degree as a heuristic if dynamic is too slow, 
            # but dynamic is better. For N=256, dynamic is fine.
            for v in current_pool:
                deg = len(conflicts[v] & current_pool)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if min_deg == 0:
                        break
            if best_v is None:
                break
            selected.add(best_v)
            to_remove = {best_v} | (conflicts[best_v] & current_pool)
            current_pool -= to_remove
        return selected

    def greedy_weighted(pool, temperature=1.5):
        """
        Stochastic greedy: pick v with prob proportional to exp(-deg * temperature).
        """
        current_pool = set(pool)
        selected = set()
        while current_pool:
            # Calculate degrees in current_pool
            degs = {}
            min_deg = float('inf')
            for v in current_pool:
                d = len(conflicts[v] & current_pool)
                degs[v] = d
                if d < min_deg:
                    min_deg = d
            
            # If min_deg is 0, we can pick any isolated node. 
            # To ensure we don't miss optimal structures, if there are isolated nodes, 
            # picking one is always safe. 
            if min_deg == 0:
                isolated = [v for v in current_pool if degs[v] == 0]
                # Random choice among isolated for diversity
                v = random.choice(isolated)
            else:
                # Soft-min: weight = exp(-(d - min_deg) / temperature)
                # Higher temperature -> flatter distribution -> more randomness
                weights = {}
                total_w = 0.0
                for v in current_pool:
                    w = math.exp(-(degs[v] - min_deg) / temperature)
                    weights[v] = w
                    total_w += w
                
                r = random.random() * total_w
                cumulative = 0.0
                v = None
                # Shuffle to break ties in deterministic iteration order
                # But we need to map back to node.
                nodes = list(current_pool)
                random.shuffle(nodes)
                for node in nodes:
                    cumulative += weights[node]
                    if r <= cumulative:
                        v = node
                        break
                if v is None:
                    v = nodes[-1]
            
            selected.add(v)
            to_remove = {v} | (conflicts[v] & current_pool)
            current_pool -= to_remove
        return selected

    def local_search(s, time_limit):
        """
        Local Search with:
        1. Add any compatible vertex.
        2. 1-for-1 Swap: Replace u with v if v has fewer conflicts with (S - {u}) than u has with S? 
           No, standard swap: Remove u, add v. Valid if v has no conflicts with S\{u} and v != u.
           But we want to maximize size, so swaps are only useful if they allow *adding* later.
           Actually, a pure 1-for-1 swap doesn't change size. 
           The current program does 1-for-2 swaps. Let's stick to that or similar.
           
        Improved Local Search:
        1. Try to add any vertex.
        2. Try 1-for-2 swaps (as before).
        3. Try 2-for-3 swaps (harder, maybe skip).
        4. Perturbation: Remove k vertices, refill.
        """
        current = set(s)
        in_set = set(current)
        best_in_ls = set(current)
        best_size = len(current)
        no_improve_count = 0
        max_no_improve = 10
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try adding any vertex
            for v in avail_list:
                if v in in_set:
                    continue
                if not (conflicts[v] & in_set):
                    current.add(v)
                    in_set.add(v)
                    improved = True
                    no_improve_count = 0
                    if len(current) > best_size:
                        best_size = len(current)
                        best_in_ls = set(current)
                    break
            
            if improved:
                continue
                
            # 2. Try 1-for-2 swap
            # We look for u in current and v1, v2 not in current such that:
            # - v1, v2 are only blocked by u (i.e., conflicts[v] & in_set == {u})
            # - v1, v2 are not adjacent to each other
            # If found, remove u, add v1, v2. Size increases by 1.
            
            # Optimization: Precompute for each u in current, the list of v not in current 
            # such that conflicts[v] & in_set == {u}
            
            # To make this faster, we can iterate v not in in_set and check blockers.
            # But that's O(N^2) per step.
            # Let's try a targeted approach:
            # For each v not in in_set:
            #   blockers = conflicts[v] & in_set
            #   if len(blockers) == 1:
            #       u = blockers.pop()
            #       v is a candidate to replace u
            
            # We can maintain a list of "swap candidates" for each u.
            # But dynamic maintenance is complex. For N=256, O(N^2) is ~65k ops. 
            # Python can do ~10M ops/sec. So 65k is ~6ms. 
            # If we do this every step, and we have ~100 steps, that's 0.6s. Acceptable.
            
            # Let's implement the O(N^2) check but optimize the inner loop.
            
            # Build map: u -> list of v such that v is blocked only by u
            swap_candidates = {u: [] for u in current}
            
            for v in avail_list:
                if v in in_set:
                    continue
                blockers = conflicts[v] & in_set
                if len(blockers) == 1:
                    u = blockers.pop()
                    if u in swap_candidates:
                        swap_candidates[u].append(v)
                        
            # Now try to find a valid swap
            found_swap = False
            # Sort by number of candidates to find a solution faster? 
            # Or just iterate u
            for u in list(current):
                candidates = swap_candidates[u]
                if len(candidates) < 2:
                    continue
                
                # Find pair (v1, v2) in candidates that are not adjacent
                # candidates list is small usually
                L = len(candidates)
                for i in range(L):
                    v1 = candidates[i]
                    if found_swap: break
                    for j in range(i + 1, L):
                        v2 = candidates[j]
                        if v1 not in conflicts[v2]:
                            # Valid swap
                            current.remove(u)
                            in_set.remove(u)
                            current.add(v1)
                            in_set.add(v1)
                            current.add(v2)
                            in_set.add(v2)
                            improved = True
                            found_swap = True
                            no_improve_count = 0
                            if len(current) > best_size:
                                best_size = len(current)
                                best_in_ls = set(current)
                            break
                            
            if improved:
                continue
            
            # 3. Try 2-for-1 swap? (Reduce size to allow larger additions later? No, we want max size)
            # What about 2-for-3? 
            # Remove u1, u2. Add v1, v2, v3.
            # This is harder.
            
            # 4. Perturbation
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb: remove 2-4 random vertices
                k_remove = random.randint(2, 4)
                to_remove = random.sample(list(current), min(k_remove, len(current)))
                for v in to_remove:
                    current.remove(v)
                    in_set.remove(v)
                
                # Refill greedily
                refill_improved = True
                while refill_improved:
                    refill_improved = False
                    for v in avail_list:
                        if v in in_set:
                            continue
                        if not (conflicts[v] & in_set):
                            current.add(v)
                            in_set.add(v)
                            refill_improved = True
                            break
                
                no_improve_count = 0
                # Don't update best_in_ls here unless size is larger, but we just removed, so likely smaller
                # If we get stuck in a local min, perturbation helps escape.
        
        return best_in_ls

    best_set = set()
    best_size = 0
    
    # Initial greedy
    s0 = greedy_min_degree(available_vertices)
    best_set = set(s0)
    best_size = len(s0)
    
    # Local search on initial
    s_ls = local_search(s0, deadline - 15)
    if len(s_ls) > best_size:
        best_size = len(s_ls)
        best_set = set(s_ls)
        
    # Multiple restarts with Weighted Greedy
    random.seed(42)
    
    # Tune temperature?
    # Start with lower temp (more greedy) and increase?
    # Or just use a range.
    
    restart_count = 0
    while time.time() < deadline - 2:
        restart_count += 1
        # Vary temperature for diversity
        # temp = 0.5 + random.random() * 1.5
        temp = 1.0
        s_init = greedy_weighted(available_vertices, temperature=temp)
        
        # Local search
        s_res = local_search(s_init, deadline - 2)
        
        if len(s_res) > best_size:
            best_size = len(s_res)
            best_set = set(s_res)
            
    return list(best_set)
