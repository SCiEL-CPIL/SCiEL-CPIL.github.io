import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    
    # 1. Precompute conflict masks
    conflict_mask = [[0] * (N + 1) for _ in range(N + 1)]
    
    # Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Build conflict masks
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g_ij = gcd_table[i][j]
            mask = 0
            for k in range(1, N + 1):
                if k == i or k == j:
                    continue
                g_ik = gcd_table[i][k]
                g_jk = gcd_table[j][k]
                if g_ij == g_ik == g_jk:
                    mask |= (1 << k)
            conflict_mask[i][j] = mask
            conflict_mask[j][i] = mask

    def is_valid_addition(s_mask, x):
        """Check if adding x to s_mask keeps it valid. Fast: only check pairs involving x."""
        if (s_mask >> x) & 1:
            return True  # already in set
        # For each y in s_mask, check if there's a z in s_mask such that {x,y,z} is bad
        # i.e., conflict_mask[x][y] & s_mask should be 0 for all y in s_mask
        temp = s_mask
        while temp:
            lsb_y = temp & -temp
            y = lsb_y.bit_length() - 1
            temp ^= lsb_y
            if (conflict_mask[x][y] & s_mask) != 0:
                return False
        return True

    def find_bad_triple(s_mask):
        """Find a bad triple in s_mask. Returns (i, j, k) or None."""
        temp = s_mask
        while temp:
            lsb_i = temp & -temp
            i = lsb_i.bit_length() - 1
            temp ^= lsb_i
            temp_j = s_mask & ~((1 << (i + 1)) - 1)
            while temp_j:
                lsb_j = temp_j & -temp_j
                j = lsb_j.bit_length() - 1
                temp_j ^= lsb_j
                candidates = conflict_mask[i][j] & s_mask
                if candidates:
                    lsb_k = candidates & -candidates
                    k = lsb_k.bit_length() - 1
                    return (i, j, k)
        return None

    def greedy_build(order):
        """Build a valid set by adding elements in the given order if possible."""
        s_mask = 0
        for x in order:
            if is_valid_addition(s_mask, x):
                s_mask |= (1 << x)
        return s_mask

    def expand(s_mask):
        """Try to add elements to s_mask in random order."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            elements = [i for i in range(1, N + 1) if not ((s_mask >> i) & 1)]
            random.shuffle(elements)
            for x in elements:
                if is_valid_addition(s_mask, x):
                    s_mask |= (1 << x)
                    changed = True
        return s_mask

    def local_search(s_mask):
        """Local search with 1-for-1, 2-for-3, and 3-for-4 swaps."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            elements_in = [i for i in range(1, N + 1) if (s_mask >> i) & 1]
            elements_out = [i for i in range(1, N + 1) if not ((s_mask >> i) & 1)]
            
            # Try to add any element
            random.shuffle(elements_out)
            for x in elements_out:
                if is_valid_addition(s_mask, x):
                    s_mask |= (1 << x)
                    improved = True
                    break
            
            if improved:
                continue
                
            # Stuck: try swaps
            random.shuffle(elements_in)
            random.shuffle(elements_out)
            
            # 1-for-1 swap: remove u, add v (same size, may enable further additions)
            for u in elements_in[:20]:
                for v in elements_out[:20]:
                    if time.time() > deadline:
                        break
                    new_mask = (s_mask & ~(1 << u)) | (1 << v)
                    # Check validity: need to verify no bad triple in new_mask
                    # Quick check: only triples involving v (since we removed u)
                    # Actually need full check for correctness
                    if _is_valid_full(new_mask):
                        s_mask = new_mask
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            
            # 2-for-3 swap: remove {u1,u2}, add {v1,v2,v3} (net +1)
            for idx in range(min(10, len(elements_in) - 1)):
                for idx2 in range(idx + 1, min(10, len(elements_in))):
                    if time.time() > deadline:
                        break
                    u1, u2 = elements_in[idx], elements_in[idx2]
                    base = s_mask & ~(1 << u1) & ~(1 << u2)
                    for v1 in elements_out[:15]:
                        for v2 in elements_out[:15]:
                            if v1 == v2:
                                continue
                            for v3 in elements_out[:15]:
                                if v3 == v1 or v3 == v2:
                                    continue
                                new_mask = base | (1 << v1) | (1 << v2) | (1 << v3)
                                if _is_valid_full(new_mask):
                                    s_mask = new_mask
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            
            # 3-for-4 swap: remove {u1,u2,u3}, add {v1,v2,v3,v4} (net +1)
            for idx in range(min(5, len(elements_in) - 2)):
                for idx2 in range(idx + 1, min(5, len(elements_in) - 1)):
                    for idx3 in range(idx2 + 1, min(5, len(elements_in))):
                        if time.time() > deadline:
                            break
                        u1, u2, u3 = elements_in[idx], elements_in[idx2], elements_in[idx3]
                        base = s_mask & ~(1 << u1) & ~(1 << u2) & ~(1 << u3)
                        for v1 in elements_out[:10]:
                            for v2 in elements_out[:10]:
                                if v1 == v2:
                                    continue
                                for v3 in elements_out[:10]:
                                    if v3 == v1 or v3 == v2:
                                        continue
                                    for v4 in elements_out[:10]:
                                        if v4 == v1 or v4 == v2 or v4 == v3:
                                            continue
                                        new_mask = base | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                        if _is_valid_full(new_mask):
                                            s_mask = new_mask
                                            improved = True
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
        return s_mask

    def _is_valid_full(s_mask):
        """Full validity check."""
        temp = s_mask
        while temp:
            lsb_i = temp & -temp
            i = lsb_i.bit_length() - 1
            temp ^= lsb_i
            temp_j = s_mask & ~((1 << (i + 1)) - 1)
            while temp_j:
                lsb_j = temp_j & -temp_j
                j = lsb_j.bit_length() - 1
                temp_j ^= lsb_j
                if (conflict_mask[i][j] & s_mask) != 0:
                    return False
        return True

    best_size = 0
    best_set = 0
    
    random.seed(42)
    
    # Strategy: multiple restarts with different construction orders
    for run in range(80):
        if time.time() > deadline:
            break
        
        # Different construction strategies
        if run % 4 == 0:
            # Natural order
            order = list(range(1, N + 1))
        elif run % 4 == 1:
            # Reverse order
            order = list(range(N, 0, -1))
        elif run % 4 == 2:
            # Random order
            order = list(range(1, N + 1))
            random.shuffle(order)
        else:
            # Sort by number of conflicts (fewest first)
            conflict_count = [0] * (N + 1)
            for i in range(1, N + 1):
                for j in range(i + 1, N + 1):
                    cnt = bin(conflict_mask[i][j]).count('1')
                    conflict_count[i] += cnt
                    conflict_count[j] += cnt
            order = sorted(range(1, N + 1), key=lambda x: conflict_count[x])
        
        s = greedy_build(order)
        s = expand(s)
        
        size = bin(s).count("1")
        if size > best_size:
            best_size = size
            best_set = s
        
        # Local search with swaps
        s = local_search(s)
        
        size = bin(s).count("1")
        if size > best_size:
            best_size = size
            best_set = s

    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    if not result:
        return [1]
    return result
