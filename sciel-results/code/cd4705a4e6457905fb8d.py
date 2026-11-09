import time
import random

# EVOLVE-BLOCK-START

def _check_validity_fast(adj, n):
    """
    Fast check for validity.
    Returns True if valid, False if invalid.
    """
    mask = (1 << n) - 1
    
    # Check K4
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if not (ai >> j) & 1:
                continue
            aj = adj[j]
            common_ij = ai & aj
            common_ij &= mask ^ ((1 << (j+1)) - 1)
            
            while common_ij:
                k_bit = common_ij & -common_ij
                k = k_bit.bit_length() - 1
                common_ij ^= k_bit
                
                ak = adj[k]
                candidates_l = (common_ij & ak) & (mask ^ ((1 << (k+1)) - 1))
                if candidates_l:
                    return False
            
    # Check I6
    comp = [0] * n
    for i in range(n):
        comp[i] = mask ^ adj[i] ^ (1 << i)
        
    for i in range(n):
        ci = comp[i]
        for j in range(i + 1, n):
            if not (ci >> j) & 1:
                continue
            cj = comp[j]
            common_ij = ci & cj
            common_ij &= mask ^ ((1 << (j+1)) - 1)
            
            while common_ij:
                k_bit = common_ij & -common_ij
                k = k_bit.bit_length() - 1
                common_ij ^= k_bit
                
                ck = comp[k]
                candidates_l = (common_ij & ck) & (mask ^ ((1 << (k+1)) - 1))
                
                while candidates_l:
                    l_bit = candidates_l & -candidates_l
                    l = l_bit.bit_length() - 1
                    candidates_l ^= l_bit
                    
                    cl = comp[l]
                    candidates_m = (candidates_l & cl) & (mask ^ ((1 << (l+1)) - 1))
                    
                    while candidates_m:
                        m_bit = candidates_m & -candidates_m
                        m = m_bit.bit_length() - 1
                        candidates_m ^= m_bit
                        
                        cm = comp[m]
                        candidates_o = (candidates_m & cm) & (mask ^ ((1 << (m+1)) - 1))
                        if candidates_o:
                            return False
                            
    return True

def _find_violation(adj, n):
    """
    Returns (type, vertices) or None.
    """
    mask = (1 << n) - 1
    
    # Check K4
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if not (ai >> j) & 1:
                continue
            aj = adj[j]
            common_ij = ai & aj
            common_ij &= mask ^ ((1 << (j+1)) - 1)
            
            while common_ij:
                k_bit = common_ij & -common_ij
                k = k_bit.bit_length() - 1
                common_ij ^= k_bit
                
                ak = adj[k]
                candidates_l = (common_ij & ak) & (mask ^ ((1 << (k+1)) - 1))
                if candidates_l:
                    l_bit = candidates_l & -candidates_l
                    l = l_bit.bit_length() - 1
                    return ('K4', [i, j, k, l])
            
    # Check I6
    comp = [0] * n
    for i in range(n):
        comp[i] = mask ^ adj[i] ^ (1 << i)
        
    for i in range(n):
        ci = comp[i]
        for j in range(i + 1, n):
            if not (ci >> j) & 1:
                continue
            cj = comp[j]
            common_ij = ci & cj
            common_ij &= mask ^ ((1 << (j+1)) - 1)
            
            while common_ij:
                k_bit = common_ij & -common_ij
                k = k_bit.bit_length() - 1
                common_ij ^= k_bit
                
                ck = comp[k]
                candidates_l = (common_ij & ck) & (mask ^ ((1 << (k+1)) - 1))
                
                while candidates_l:
                    l_bit = candidates_l & -candidates_l
                    l = l_bit.bit_length() - 1
                    candidates_l ^= l_bit
                    
                    cl = comp[l]
                    candidates_m = (candidates_l & cl) & (mask ^ ((1 << (l+1)) - 1))
                    
                    while candidates_m:
                        m_bit = candidates_m & -candidates_m
                        m = m_bit.bit_length() - 1
                        candidates_m ^= m_bit
                        
                        cm = comp[m]
                        candidates_o = (candidates_m & cm) & (mask ^ ((1 << (m+1)) - 1))
                        if candidates_o:
                            o_bit = candidates_o & -candidates_o
                            o = o_bit.bit_length() - 1
                            return ('I6', [i, j, k, l, m, o])
                            
    return None

def _flip_edge(adj, u, v):
    if (adj[u] >> v) & 1:
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
    else:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

def _search_repair(n_target, time_limit):
    """
    Try to find a valid graph on n_target vertices using repair.
    """
    start_time = time.time()
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
            
        adj = [0] * n_target
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        for step in range(5000):
            if time.time() - start_time > time_limit:
                return None, 0
                
            v = _find_violation(adj, n_target)
            if v is None:
                return adj, n_target
                
            v_type, verts = v
            if v_type == 'K4':
                pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
                u_idx, v_idx = random.choice(pairs)
                u, v = verts[u_idx], verts[v_idx]
            else:
                pairs = [(i, j) for i in range(6) for j in range(i+1, 6)]
                u_idx, v_idx = random.choice(pairs)
                u, v = verts[u_idx], verts[v_idx]
                
            _flip_edge(adj, u, v)
            
    return None, 0

def construct():
    start = time.time()
    deadline = 145.0
    
    best_adj = None
    best_n = 0
    
    # Try high n first
    # 35 is the target.
    time_budgets = {
        35: 40.0,
        34: 30.0,
        33: 20.0,
        32: 15.0,
        31: 10.0,
        30: 5.0,
        29: 5.0,
        28: 5.0,
        27: 5.0,
        26: 5.0,
        25: 5.0,
        24: 5.0,
        23: 5.0,
        22: 5.0,
        21: 5.0,
        20: 5.0
    }
    
    for n in range(35, 19, -1):
        if time.time() - start > deadline - 5:
            break
            
        t_start = time.time()
        adj, found_n = _search_repair(n, time_budgets.get(n, 5.0))
        
        if found_n > best_n:
            best_n = found_n
            best_adj = adj
            
        if found_n == n:
            break
            
    if best_adj is None:
        # Fallback
        best_n = 1
        best_adj = [0]
        
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
