import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    def time_ok():
        return time.time() < deadline
    
    # Build Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    full_mask = (1 << n) - 1
    comp_bits = [0] * n
    for i in range(n):
        comp_bits[i] = (full_mask ^ adj_bits[i]) & ~(1 << i)
    
    def compute_k4s(n, adj_bits):
        k4s = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    common = adj_bits[i] & adj_bits[j]
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & adj_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    k4s.append((i, j, k, l))
        return k4s
    
    def compute_ind4s(n, adj_bits):
        full_mask = (1 << n) - 1
        comp_bits = [0] * n
        for i in range(n):
            comp_bits[i] = (full_mask ^ adj_bits[i]) & ~(1 << i)
        ind4s = []
        for i in range(n):
            for j in range(i + 1, n):
                if (comp_bits[i] >> j) & 1:
                    common = comp_bits[i] & comp_bits[j]
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & comp_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    ind4s.append((i, j, k, l))
        return ind4s
    
    # Precompute for seed
    current_k4s = compute_k4s(n, adj_bits)
    current_ind4s = compute_ind4s(n, adj_bits)
    
    def has_k5_bitset(n, adj_bits):
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    common = adj_bits[i] & adj_bits[j]
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & adj_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    common3 = common2 & adj_bits[l]
                                    if common3 & ~((1 << (l + 1)) - 1):
                                        return True
        return False
    
    def has_ind5_bitset(n, adj_bits):
        comp_bits = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            comp_bits[i] = (full_mask ^ adj_bits[i]) & ~(1 << i)
        
        for i in range(n):
            for j in range(i + 1, n):
                if (comp_bits[i] >> j) & 1:
                    common = comp_bits[i] & comp_bits[j]
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & comp_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    common3 = common2 & comp_bits[l]
                                    if common3 & ~((1 << (l + 1)) - 1):
                                        return True
        return False
    
    def is_valid(n, adj_bits):
        return not has_k5_bitset(n, adj_bits) and not has_ind5_bitset(n, adj_bits)
    
    # Verify seed
    if not is_valid(n, adj_bits):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    edges.append((i, j))
        return n, edges
    
    best_n = n
    best_adj_bits = list(adj_bits)
    
    def find_valid_neighbor_set(n, adj_bits, k4s, ind4s):
        """
        Find a subset S of {0..n-1} such that:
        1. S contains no K4 from the graph.
        2. V \ S contains no Ind4 from the graph.
        Uses Simulated Annealing / Hill Climbing on the conflict count.
        """
        if not time_ok():
            return None
            
        all_mask = (1 << n) - 1
        # Precompute masks for K4s and Ind4s for fast subset check
        k4_masks = []
        for k4 in k4s:
            mask = 0
            for v in k4:
                mask |= (1 << v)
            k4_masks.append(mask)
            
        ind4_masks = []
        for ind4 in ind4s:
            mask = 0
            for v in ind4:
                mask |= (1 << v)
            ind4_masks.append(mask)
        
        # Initial state: Random
        S = 0
        for i in range(n):
            if random.random() < 0.5:
                S |= (1 << i)
        
        def calc_cost(S):
            cost = 0
            # Count K4s fully in S
            for mask in k4_masks:
                if (S & mask) == mask:
                    cost += 1
            # Count Ind4s fully in ~S (i.e., S & mask == 0)
            for mask in ind4_masks:
                if (S & mask) == 0:
                    cost += 1
            return cost
        
        current_cost = calc_cost(S)
        if current_cost == 0:
            return S
            
        best_S = S
        best_cost = current_cost
        
        # Simulated Annealing parameters
        T = 10.0
        T_min = 0.1
        cooling = 0.995
        max_iters = 20000
        
        for it in range(max_iters):
            if not time_ok():
                break
            if T < T_min:
                break
                
            # Pick a random vertex to flip
            v = random.randint(0, n - 1)
            bit = 1 << v
            new_S = S ^ bit
            
            # Calculate delta cost efficiently?
            # For now, just recalculate if it's fast enough.
            # n ~ 40, k4s ~ few hundreds. This is fast.
            new_cost = calc_cost(new_S)
            
            delta = new_cost - current_cost
            
            if delta < 0:
                S = new_S
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_S = S
                    if best_cost == 0:
                        return S
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < pow(2.718281828, -delta / T):
                    S = new_S
                    current_cost = new_cost
            
            T *= cooling
            
        if best_cost == 0:
            return best_S
        return None
    
    # Attempt to extend the graph
    current_n = n
    current_adj_bits = list(adj_bits)
    current_k4s = current_k4s
    current_ind4s = current_ind4s
    
    for target_n in range(n + 1, 43):
        if not time_ok():
            break
        
        found = False
        # Try multiple restarts for the local search
        max_restarts = 50
        for r in range(max_restarts):
            if not time_ok():
                break
            
            S = find_valid_neighbor_set(current_n, current_adj_bits, current_k4s, current_ind4s)
            if S is not None:
                # Add the vertex
                new_adj_bits = list(current_adj_bits)
                new_adj_bits.append(S)
                for i in range(current_n):
                    if (S >> i) & 1:
                        new_adj_bits[i] |= (1 << current_n)
                
                # Verify
                if is_valid(current_n + 1, new_adj_bits):
                    current_adj_bits = new_adj_bits
                    current_n += 1
                    best_n = current_n
                    best_adj_bits = list(current_adj_bits)
                    # Recompute K4s and Ind4s
                    current_k4s = compute_k4s(current_n, current_adj_bits)
                    current_ind4s = compute_ind4s(current_n, current_adj_bits)
                    found = True
                    break
        
        if not found:
            # If we can't add a vertex, maybe we can improve the current graph
            # to allow addition? Or just stop.
            # Given time constraints, stopping is safer.
            break
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj_bits[i] >> j) & 1:
                edges.append((i, j))
    
    return best_n, edges
