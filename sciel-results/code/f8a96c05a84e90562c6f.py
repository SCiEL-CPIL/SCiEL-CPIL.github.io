import random
import time
from itertools import combinations

def is_k4_free(adj, n):
    """Fast K4-free check using bitmasks."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common_ab = adj[a] & adj[b]
            common_ab &= ~((1 << (b + 1)) - 1)
            while common_ab:
                lsb = common_ab & -common_ab
                c = lsb.bit_length() - 1
                common_abc = adj[a] & adj[b] & adj[c]
                common_abc &= ~((1 << (c + 1)) - 1)
                if common_abc:
                    return False
                common_ab ^= lsb
    return True

def compute_largest_tf(adj, n, deadline):
    """
    Find the largest triangle-free induced subgraph using branch and bound.
    """
    best_found = 0
    
    def backtrack(start_idx, chosen_count, chosen_mask):
        nonlocal best_found
        if time.time() > deadline:
            return
        if chosen_count > best_found:
            best_found = chosen_count
        if start_idx >= n:
            return
        if chosen_count + (n - start_idx) <= best_found:
            return
        
        v = start_idx
        
        # Try including v
        nbrs_v_in_chosen = adj[v] & chosen_mask
        creates_tri = False
        m = nbrs_v_in_chosen
        while m:
            lsb_u = m & -m
            u = lsb_u.bit_length() - 1
            m ^= lsb_u
            other_nbrs = nbrs_v_in_chosen & ~lsb_u
            if adj[u] & other_nbrs:
                creates_tri = True
                break
        if not creates_tri:
            backtrack(start_idx + 1, chosen_count + 1, chosen_mask | (1 << v))
        
        # Try excluding v
        backtrack(start_idx + 1, chosen_count, chosen_mask)
    
    backtrack(0, 0, 0)
    return best_found

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 18
    
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    best_l = n + 1
    best_adj = None
    
    def update_best(adj, l):
        nonlocal best_l, best_adj
        if l < best_l:
            best_l = l
            best_adj = adj[:]
    
    random.seed(42)
    
    # Turan T(18,3) as baseline
    adj_turan = [0] * n
    parts3 = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts3[i]:
                for v in parts3[j]:
                    adj_turan[u] |= (1 << v)
                    adj_turan[v] |= (1 << u)
    
    l = compute_largest_tf(adj_turan, n, deadline)
    update_best(adj_turan, l)
    
    if best_adj is not None and time.time() < deadline - 5:
        current_adj = best_adj[:]
        current_l = best_l
        T = 1.0
        
        for step in range(50000):
            if time.time() > deadline - 2:
                break
            
            i, j = random.sample(range(n), 2)
            current_adj[i] ^= (1 << j)
            current_adj[j] ^= (1 << i)
            
            if not is_k4_free(current_adj, n):
                current_adj[i] ^= (1 << j)
                current_adj[j] ^= (1 << i)
                continue
            
            new_l = compute_largest_tf(current_adj, n, deadline)
            
            if new_l < current_l:
                current_l = new_l
                update_best(current_adj, new_l)
            elif new_l == current_l:
                pass
            else:
                delta = new_l - current_l
                if random.random() < pow(2.71828, -delta / T):
                    pass
                else:
                    current_adj[i] ^= (1 << j)
                    current_adj[j] ^= (1 << i)
            
            T *= 0.999
    
    if best_adj is None:
        return adj_to_edges(adj_turan)
    
    return adj_to_edges(best_adj)
