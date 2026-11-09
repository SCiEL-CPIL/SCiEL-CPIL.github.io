import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = 55.0
    
    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set
    
    # Precompute LCMs
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # 1. Find all bad triples and build the conflict graph G
    # G[u] is a bitmask of v such that there exists w where {u,v,w} is a bad triple.
    conflict_graph = [0] * (N + 1)
    bad_triple_masks = []
    
    # Precompute lcm table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            lcm_table[a][b] = lcm(a, b)
            
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lab == lcm_table[a][c] == lcm_table[b][c]:
                    mask = (1 << a) | (1 << b) | (1 << c)
                    bad_triple_masks.append(mask)
                    # Update conflict graph
                    conflict_graph[a] |= (1 << b)
                    conflict_graph[b] |= (1 << a)
                    conflict_graph[a] |= (1 << c)
                    conflict_graph[c] |= (1 << a)
                    conflict_graph[b] |= (1 << c)
                    conflict_graph[c] |= (1 << b)
                    
    if not bad_triple_masks:
        return [i for i in range(1, N + 1) if full_mask & (1 << i)]
    
    # Precompute for each element, the list of bad triple indices it participates in
    element_triples = [[] for _ in range(N + 1)]
    for idx, t in enumerate(bad_triple_masks):
        x = t
        while x:
            lsb = x & -x
            idx_elem = lsb.bit_length() - 1
            element_triples[idx_elem].append(idx)
            x ^= lsb
    
    def is_valid_fast(mask):
        """Check validity using precomputed element_triples for speed"""
        for i in range(1, N + 1):
            if mask & (1 << i):
                for t_idx in element_triples[i]:
                    t = bad_triple_masks[t_idx]
                    if (mask & t) == t:
                        return False
        return True
        
    def greedy_mis_on_graph():
        # Greedy Independent Set on the conflict graph G.
        remaining = full_mask
        selected = 0
        
        while remaining:
            min_deg = N + 1
            best_u = 0
            temp = remaining
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                deg = (conflict_graph[u] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_u = u
                temp ^= lsb
                
            if best_u == 0:
                break
                
            selected |= (1 << best_u)
            remaining &= ~(1 << best_u)
            remaining &= ~conflict_graph[best_u]
            
        return selected

    def add_back(mask):
        improved = True
        while improved:
            improved = False
            for i in range(1, N + 1):
                if not (mask & (1 << i)):
                    new_mask = mask | (1 << i)
                    if is_valid_fast(new_mask):
                        mask = new_mask
                        improved = True
        return mask
        
    def swap_improve(mask):
        # 1 out, 2 in
        improved = True
        while improved:
            improved = False
            in_mask = [i for i in range(1, N + 1) if mask & (1 << i)]
            not_in_mask = [i for i in range(1, N + 1) if not (mask & (1 << i))]
            
            if not not_in_mask:
                break
                
            for i in range(len(not_in_mask)):
                v = not_in_mask[i]
                for j in range(i + 1, len(not_in_mask)):
                    w = not_in_mask[j]
                    
                    M_prime = mask | (1 << v) | (1 << w)
                    blocking_triples = [t for t in bad_triple_masks if (M_prime & t) == t]
                    
                    if not blocking_triples:
                        if is_valid_fast(M_prime):
                            mask = M_prime
                            improved = True
                            break
                    
                    common_intersection = None
                    for t in blocking_triples:
                        if common_intersection is None:
                            common_intersection = t
                        else:
                            common_intersection &= t
                    
                    if common_intersection is None:
                        continue
                        
                    candidate_u = None
                    x = common_intersection
                    while x:
                        lsb = x & -x
                        idx = lsb.bit_length() - 1
                        if mask & (1 << idx):
                            candidate_u = idx
                            break
                        x ^= lsb
                    
                    if candidate_u is not None:
                        mask = (mask & ~(1 << candidate_u)) | (1 << v) | (1 << w)
                        improved = True
                        break
            
            if improved:
                continue
                
        return mask
    
    def swap2_improve(mask):
        """2 out, 2 in improvement"""
        improved = True
        while improved:
            improved = False
            in_mask = [i for i in range(1, N + 1) if mask & (1 << i)]
            not_in_mask = [i for i in range(1, N + 1) if not (mask & (1 << i))]
            
            if len(not_in_mask) < 2 or len(in_mask) < 2:
                break
            
            # Limit search to avoid O(n^4) explosion
            max_pairs = min(len(not_in_mask), 30)
            max_remove_pairs = min(len(in_mask), 30)
            
            for i in range(max_pairs):
                v = not_in_mask[i]
                for j in range(i + 1, max_pairs):
                    w = not_in_mask[j]
                    
                    M_base = mask | (1 << v) | (1 << w)
                    blocking_triples = [t for t in bad_triple_masks if (M_base & t) == t]
                    
                    if not blocking_triples:
                        if is_valid_fast(M_base):
                            mask = M_base
                            improved = True
                            break
                    
                    # Find pairs (u1, u2) in in_mask such that removing both fixes all blocking triples
                    common_intersection = None
                    for t in blocking_triples:
                        if common_intersection is None:
                            common_intersection = t
                        else:
                            common_intersection &= t
                    
                    if common_intersection is None:
                        continue
                    
                    # Need to find u1, u2 in in_mask & common_intersection
                    candidates_in_common = []
                    x = common_intersection
                    while x:
                        lsb = x & -x
                        idx = lsb.bit_length() - 1
                        if mask & (1 << idx):
                            candidates_in_common.append(idx)
                        x ^= lsb
                    
                    if len(candidates_in_common) < 2:
                        continue
                    
                    # Try removing pairs from candidates_in_common
                    for a_idx in range(min(len(candidates_in_common), max_remove_pairs)):
                        u1 = candidates_in_common[a_idx]
                        for b_idx in range(a_idx + 1, min(len(candidates_in_common), max_remove_pairs)):
                            u2 = candidates_in_common[b_idx]
                            
                            new_mask = M_base & ~(1 << u1) & ~(1 << u2)
                            if is_valid_fast(new_mask) and new_mask.bit_count() > mask.bit_count():
                                mask = new_mask
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if improved:
                continue
                
        return mask
    
    def polish(mask):
        """High-precision polish: alternate add_back, swap_improve, swap2_improve"""
        for _ in range(3):
            mask = add_back(mask)
            mask = swap_improve(mask)
            mask = swap2_improve(mask)
            mask = add_back(mask)
        return mask
    
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy MIS on the conflict graph
    m = greedy_mis_on_graph()
    m = polish(m)
    size = m.bit_count()
    if size > best_size:
        best_size = size
        best_mask = m
        
    # Strategy 2: Randomized Greedy MIS on the conflict graph - more iterations
    random.seed(42)
    for _ in range(500):
        if time.time() - start_time > deadline - 5:
            break
        
        remaining = full_mask
        selected = 0
        
        while remaining:
            min_deg = N + 1
            candidates = []
            temp = remaining
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                deg = (conflict_graph[u] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
                temp ^= lsb
                
            if not candidates:
                break
                
            u = random.choice(candidates)
            selected |= (1 << u)
            remaining &= ~(1 << u)
            remaining &= ~conflict_graph[u]
            
        m = selected
        m = polish(m)
        size = m.bit_count()
        if size > best_size:
            best_size = size
            best_mask = m
    
    # Strategy 3: Greedy Removal
    def greedy_remove(full_mask, randomize=False):
        mask = full_mask
        while True:
            active_triples = [t for t in bad_triple_masks if (mask & t) == t]
            if not active_triples:
                break
            
            counts = [0] * (N + 1)
            for t in active_triples:
                x = t
                while x:
                    lsb = x & -x
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    x ^= lsb
            
            max_count = 0
            candidates = []
            for i in range(1, N + 1):
                if mask & (1 << i):
                    if counts[i] > max_count:
                        max_count = counts[i]
                        candidates = [i]
                    elif counts[i] == max_count:
                        candidates.append(i)
            
            if not candidates:
                break
                
            if randomize:
                u = random.choice(candidates)
            else:
                u = candidates[0]
            
            mask &= ~(1 << u)
            
            if time.time() - start_time > deadline - 5:
                break
        
        return mask

    for _ in range(100):
        if time.time() - start_time > deadline - 5:
            break
        m = greedy_remove(full_mask, randomize=True)
        m = polish(m)
        size = m.bit_count()
        if size > best_size:
            best_size = size
            best_mask = m

    # Final polish on best
    best_mask = polish(best_mask)
    
    # Convert mask to list
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    return result
