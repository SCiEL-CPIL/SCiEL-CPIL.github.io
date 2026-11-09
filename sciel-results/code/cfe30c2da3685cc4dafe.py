import time
import random

def _popcount(x):
    return bin(x).count('1')

def _has_K4_bitset(adj_bits, n):
    for i in range(n):
        for j in range(i+1, n):
            if not (adj_bits[i] & (1 << j)):
                continue
            common_ij = adj_bits[i] & adj_bits[j]
            for k in range(j+1, n):
                if not (common_ij & (1 << k)):
                    continue
                common_ijk = common_ij & adj_bits[k]
                mask_l = ~((1 << (k+1)) - 1)
                if common_ijk & mask_l:
                    return True
    return False

def _has_indep_6_bitset(adj_bits, n):
    mask_all = (1 << n) - 1
    comp = [0]*n
    for i in range(n):
        comp[i] = (~adj_bits[i]) & mask_all
        comp[i] &= ~(1 << i)
        
    def find_clique(candidates, size):
        if size == 0:
            return True
        if _popcount(candidates) < size:
            return False
        v_bit = candidates & -candidates
        v = v_bit.bit_length() - 1
        new_cands = candidates & comp[v]
        if find_clique(new_cands, size - 1):
            return True
        candidates &= ~v_bit
        if find_clique(candidates, size):
            return True
        return False

    return find_clique(mask_all, 6)

def _is_valid_bitset(adj_bits, n):
    return not _has_K4_bitset(adj_bits, n) and not _has_indep_6_bitset(adj_bits, n)

def _edges_from_bitset(adj_bits, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                edges.append([i, j])
    return edges

def _local_search_bitset(adj_bits, n, deadline):
    curr = adj_bits[:]
    no_improve = 0
    
    while time.time() < deadline:
        if no_improve > 1000:
            break
            
        # Check K4
        if _has_K4_bitset(curr, n):
            k4 = _find_K4(curr, n)
            if k4:
                u, v, w, x = k4
                pairs = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                ru, rv = random.choice(pairs)
                curr[ru] &= ~(1 << rv)
                curr[rv] &= ~(1 << ru)
            continue
            
        # Check Indep 6
        if _has_indep_6_bitset(curr, n):
            indep = _find_indep_6(curr, n)
            if indep:
                u, v = random.sample(indep, 2)
                curr[u] |= (1 << v)
                curr[v] |= (1 << u)
            continue
            
        # Valid
        return curr
    return None

def _find_K4(adj_bits, n):
    for i in range(n):
        for j in range(i+1, n):
            if not (adj_bits[i] & (1 << j)): continue
            common_ij = adj_bits[i] & adj_bits[j]
            for k in range(j+1, n):
                if not (common_ij & (1 << k)): continue
                common_ijk = common_ij & adj_bits[k]
                mask_l = ~((1 << (k+1)) - 1)
                cands = common_ijk & mask_l
                if cands:
                    l_bit = cands & -cands
                    l = l_bit.bit_length() - 1
                    return (i, j, k, l)
    return None

def _find_indep_6(adj_bits, n):
    mask_all = (1 << n) - 1
    comp = [0]*n
    for i in range(n):
        comp[i] = (~adj_bits[i]) & mask_all
        comp[i] &= ~(1 << i)
        
    path = []
    def backtrack(candidates, size):
        if size == 0:
            return True
        if _popcount(candidates) < size:
            return False
        v_bit = candidates & -candidates
        v = v_bit.bit_length() - 1
        path.append(v)
        new_cands = candidates & comp[v]
        if backtrack(new_cands, size - 1):
            return True
        path.pop()
        candidates &= ~v_bit
        if backtrack(candidates, size):
            return True
        return False
        
    if backtrack(mask_all, 6):
        return path
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try n=35
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        n = 35
        p = 0.5
        adj_bits = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
                    
        res = _local_search_bitset(adj_bits, n, deadline)
        if res is not None:
            if _is_valid_bitset(res, n):
                return {'n': n, 'edges': _edges_from_bitset(res, n)}
                
    # Try n=34
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        n = 34
        adj_bits = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        res = _local_search_bitset(adj_bits, n, deadline)
        if res is not None:
            if _is_valid_bitset(res, n):
                return {'n': n, 'edges': _edges_from_bitset(res, n)}
                
    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
