import time
import random
import math

def construct():
    start_time = time.time()
    # Safety margin: 55s out of 60s limit. 
    # If we exceed this, we return the best found so far.
    deadline = start_time + 55.0
    
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
    
    # Build conflict graph
    # conflicts[v] = set of u such that u+v is square and u != v
    # We only care about conflicts between available vertices
    conflicts = {v: set() for v in available_vertices}
    
    for a in available_vertices:
        for b in available_vertices:
            if a < b and (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    # Precompute sorted available vertices for deterministic iteration if needed
    avail_list = sorted(available_vertices)
    
    # Precompute neighbor lists for faster access
    neighbor_lists = {v: list(conflicts[v]) for v in available_vertices}

    def greedy_init_weighted(pool, seed=None, base_temp=1.5):
        """
        Weighted Greedy Init with adaptive temperature.
        At each step, select a vertex with probability proportional to exp(-degree * temperature).
        Temperature decreases over time to become more greedy.
        """
        if seed is not None:
            random.seed(seed)
        
        current_pool = set(pool)
        selected = set()
        
        # Adaptive temperature: start higher, cool down
        temp = base_temp
        cooling_rate = 0.95
        
        while current_pool:
            # Calculate degrees in current_pool
            degrees = {}
            min_deg = float('inf')
            for v in current_pool:
                # Degree is number of neighbors in current_pool
                deg = len(conflicts[v] & current_pool)
                degrees[v] = deg
                if deg < min_deg:
                    min_deg = deg
            
            # If we have isolated nodes (deg 0), pick randomly for diversity
            if min_deg == 0:
                isolated = [v for v, d in degrees.items() if d == 0]
                v = random.choice(isolated)
            else:
                # Calculate weights with current temperature
                weights = {}
                total_w = 0.0
                for v, deg in degrees.items():
                    w = math.exp(-(deg - min_deg) * temp)
                    weights[v] = w
                    total_w += w
                
                # Random selection based on weights
                r = random.random() * total_w
                cumulative = 0.0
                v = None
                for node in current_pool:
                    cumulative += weights[node]
                    if r <= cumulative:
                        v = node
                        break
                if v is None:
                    v = random.choice(list(current_pool))
                
                # Cool down temperature
                temp *= cooling_rate
            
            selected.add(v)
            # Remove v and its neighbors from pool
            to_remove = {v} | (conflicts[v] & current_pool)
            current_pool -= to_remove
            
        return selected

    def local_search(s, time_limit, allow_worse=False):
        """
        Enhanced Local Search with:
        1. Add any compatible vertex.
        2. 1-for-2 Swap.
        3. 2-for-3 Swap: Remove u1, u2, add v1, v2, v3 if feasible.
        4. Perturbation with simulated annealing acceptance.
        """
        current = set(s)
        in_set = set(current)
        
        improved = True
        best_in_ls = set(current)
        best_size = len(current)
        
        # Simulated annealing parameters
        sa_temp = 5.0
        sa_cooling = 0.9
        sa_min_temp = 0.1
        
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try adding any vertex that is not in set and has no conflicts with set
            for v in avail_list:
                if v in in_set:
                    continue
                if not (conflicts[v] & in_set):
                    current.add(v)
                    in_set.add(v)
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_in_ls = set(current)
                    break
            
            if not improved:
                # 2. Try 1-for-2 swap
                swap_done = False
                for u in list(current):
                    candidates = []
                    for v in avail_list:
                        if v in in_set:
                            continue
                        if u not in conflicts[v]:
                            continue
                        common = conflicts[v] & in_set
                        if len(common) == 1:
                            candidates.append(v)
                    
                    if len(candidates) >= 2:
                        for i in range(len(candidates)):
                            if swap_done:
                                break
                            for j in range(i + 1, len(candidates)):
                                if candidates[i] not in conflicts[candidates[j]]:
                                    current.remove(u)
                                    in_set.remove(u)
                                    current.add(candidates[i])
                                    in_set.add(candidates[i])
                                    current.add(candidates[j])
                                    in_set.add(candidates[j])
                                    improved = True
                                    swap_done = True
                                    if len(current) > best_size:
                                        best_size = len(current)
                                        best_in_ls = set(current)
                                    break
                        if swap_done:
                            break
                
                if not swap_done:
                    # 3. Try 2-for-3 swap (more powerful but expensive)
                    two_swap_done = False
                    current_list = list(current)
                    for i in range(len(current_list)):
                        if two_swap_done:
                            break
                        u1 = current_list[i]
                        for j in range(i + 1, len(current_list)):
                            if two_swap_done:
                                break
                            u2 = current_list[j]
                            if u1 in conflicts[u2]:
                                continue
                            
                            # Find candidates that are only blocked by {u1, u2}
                            candidates = []
                            for v in avail_list:
                                if v in in_set:
                                    continue
                                common = conflicts[v] & in_set
                                if common == {u1, u2} or common == {u1} or common == {u2}:
                                    # Check that v doesn't conflict with other elements in {u1,u2}
                                    candidates.append(v)
                            
                            if len(candidates) >= 3:
                                # Find three non-adjacent candidates
                                for a_idx in range(len(candidates)):
                                    if two_swap_done:
                                        break
                                    for b_idx in range(a_idx + 1, len(candidates)):
                                        if two_swap_done:
                                            break
                                        if candidates[a_idx] in conflicts[candidates[b_idx]]:
                                            continue
                                        for c_idx in range(b_idx + 1, len(candidates)):
                                            if two_swap_done:
                                                break
                                            if candidates[a_idx] in conflicts[candidates[c_idx]]:
                                                continue
                                            if candidates[b_idx] in conflicts[candidates[c_idx]]:
                                                continue
                                            
                                            # Perform 2-for-3 swap
                                            current.remove(u1)
                                            in_set.remove(u1)
                                            current.remove(u2)
                                            in_set.remove(u2)
                                            current.add(candidates[a_idx])
                                            in_set.add(candidates[a_idx])
                                            current.add(candidates[b_idx])
                                            in_set.add(candidates[b_idx])
                                            current.add(candidates[c_idx])
                                            in_set.add(candidates[c_idx])
                                            improved = True
                                            two_swap_done = True
                                            if len(current) > best_size:
                                                best_size = len(current)
                                                best_in_ls = set(current)
                                            break
                        if two_swap_done:
                            break
                    if two_swap_done:
                        improved = True
                        continue
                
                if not improved:
                    # 4. Perturbation with simulated annealing
                    if time.time() > time_limit - 2:
                        break
                    
                    k_remove = random.randint(2, 5)
                    to_remove = random.sample(list(current), min(k_remove, len(current)))
                    old_size = len(current)
                    
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
                    
                    new_size = len(current)
                    
                    # Simulated annealing acceptance
                    if new_size >= old_size:
                        improved = True
                        if new_size > best_size:
                            best_size = new_size
                            best_in_ls = set(current)
                    else:
                        # Accept worse with probability exp(-delta/temp)
                        delta = old_size - new_size
                        if sa_temp > sa_min_temp:
                            accept_prob = math.exp(-delta / sa_temp)
                            if random.random() < accept_prob:
                                improved = True
                        sa_temp *= sa_cooling

        if len(best_in_ls) > len(current):
            return best_in_ls
        return current

    def greedy_min_degree(pool):
        current_pool = set(pool)
        selected = set()
        while current_pool:
            min_deg = float('inf')
            best_v = None
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

    best_set = set()
    best_size = 0
    
    # Initial greedy
    s0 = greedy_min_degree(available_vertices)
    best_set = set(s0)
    best_size = len(s0)
    
    # Local search on initial
    s_ls = local_search(s0, deadline - 8)
    if len(s_ls) > best_size:
        best_size = len(s_ls)
        best_set = set(s_ls)
        
    # Multiple restarts with Weighted Greedy
    random.seed(42)
    
    restart_count = 0
    while time.time() < deadline - 2:
        restart_count += 1
        # Vary base temperature for diversity
        base_temp = 1.0 + 0.5 * math.sin(restart_count * 0.7)
        s_init = greedy_init_weighted(available_vertices, seed=restart_count, base_temp=base_temp)
        
        s_res = local_search(s_init, deadline - 2)
        
        if len(s_res) > best_size:
            best_size = len(s_res)
            best_set = set(s_res)
            
    return list(best_set)
