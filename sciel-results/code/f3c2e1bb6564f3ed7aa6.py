import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    def count_k5_and_indep5(adj_masks, n):
        """Count all K5 and independent 5-sets in the graph."""
        num_k5 = 0
        num_indep5 = 0
        for i in range(n - 4):
            mi = adj_masks[i]
            for j in range(i + 1, n - 3):
                if (mi >> j) & 1:
                    # i-j is an edge; need k,l,m all adjacent to i,j and to each other
                    common = mi & adj_masks[j]
                    # Candidates must be > j
                    common &= ~((1 << (j + 1)) - 1)
                    # Get candidate list
                    cands = []
                    tmp = common
                    while tmp:
                        bit = tmp & (-tmp)
                        cands.append(bit.bit_length() - 1)
                        tmp ^= bit
                    for k_idx in range(len(cands) - 2):
                        k = cands[k_idx]
                        mk = adj_masks[k]
                        for l_idx in range(k_idx + 1, len(cands) - 1):
                            l = cands[l_idx]
                            if not ((mk >> l) & 1):
                                continue
                            ml = adj_masks[l]
                            for m_idx in range(l_idx + 1, len(cands)):
                                m = cands[m_idx]
                                if (mk >> m) & 1 and (ml >> m) & 1:
                                    num_k5 += 1
                else:
                    # i-j is NOT an edge; need k,l,m all non-adjacent to i,j and to each other
                    # Non-neighbors of i (excluding self): ~(mi | (1<<i))
                    non_i = ~(mi | (1 << i))
                    non_j = ~(adj_masks[j] | (1 << j))
                    common = non_i & non_j
                    common &= ~((1 << (j + 1)) - 1)  # only vertices > j
                    common &= (1 << n) - 1  # mask to n bits
                    cands = []
                    tmp = common
                    while tmp:
                        bit = tmp & (-tmp)
                        cands.append(bit.bit_length() - 1)
                        tmp ^= bit
                    for k_idx in range(len(cands) - 2):
                        k = cands[k_idx]
                        mk = adj_masks[k]
                        for l_idx in range(k_idx + 1, len(cands) - 1):
                            l = cands[l_idx]
                            if (mk >> l) & 1:
                                continue
                            ml = adj_masks[l]
                            for m_idx in range(l_idx + 1, len(cands)):
                                m = cands[m_idx]
                                if not ((mk >> m) & 1) and not ((ml >> m) & 1):
                                    num_indep5 += 1
        return num_k5, num_indep5

    def is_valid(adj_masks, n):
        k5, i5 = count_k5_and_indep5(adj_masks, n)
        return k5 == 0 and i5 == 0

    def get_edges(adj_masks, n):
        edges = []
        for i in range(n):
            mi = adj_masks[i]
            for j in range(i + 1, n):
                if (mi >> j) & 1:
                    edges.append((i, j))
        return edges

    def count_affected(adj_masks, n, u, v):
        """Count K5 and Indep5 containing the pair (u,v) using bitmask acceleration."""
        num_k5 = 0
        num_indep5 = 0
        is_edge = (adj_masks[u] >> v) & 1
        mu = adj_masks[u]
        mv = adj_masks[v]
        
        # Build candidate set: vertices w not in {u,v}
        if is_edge:
            # For K5: w must be adjacent to both u and v
            common = mu & mv
            common &= ~(1 << u)
            common &= ~(1 << v)
            # For Indep5 with edge (u,v): impossible since u-v is an edge
        else:
            # For K5 with non-edge (u,v): impossible
            # For Indep5: w must be non-adjacent to both u and v
            non_u = ~(mu | (1 << u))
            non_v = ~(mv | (1 << v))
            common = non_u & non_v
            common &= ~(1 << u)
            common &= ~(1 << v)
            common &= (1 << n) - 1
        
        # Extract candidate vertices
        cands = []
        tmp = common
        while tmp:
            bit = tmp & (-tmp)
            cands.append(bit.bit_length() - 1)
            tmp ^= bit
        
        nc = len(cands)
        for a_idx in range(nc - 2):
            a = cands[a_idx]
            ma = adj_masks[a]
            for b_idx in range(a_idx + 1, nc - 1):
                b = cands[b_idx]
                ab_edge = (ma >> b) & 1
                mb = adj_masks[b]
                for c_idx in range(b_idx + 1, nc):
                    c = cands[c_idx]
                    ac_edge = (ma >> c) & 1
                    bc_edge = (mb >> c) & 1
                    
                    if is_edge:
                        # K5: a,b,c must form triangle (all edges)
                        if ab_edge and ac_edge and bc_edge:
                            num_k5 += 1
                    else:
                        # Indep5: a,b,c must have no edges
                        if not ab_edge and not ac_edge and not bc_edge:
                            num_indep5 += 1
        return num_k5, num_indep5

    def init_counts(adj_masks, n):
        return count_k5_and_indep5(adj_masks, n)

    # Try Paley graph on 41 vertices first
    n = 41
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    
    adj_masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
    
    if is_valid(adj_masks, n):
        return n, get_edges(adj_masks, n)
    
    # Simulated annealing on n=41
    current_k5, current_i5 = init_counts(adj_masks, n)
    best_cost = current_k5 + current_i5
    best_adj = adj_masks[:]
    
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        old_k5_aff, old_i5_aff = count_affected(adj_masks, n, u, v)
        
        mask_u = adj_masks[u]
        mask_v = adj_masks[v]
        adj_masks[u] = mask_u ^ (1 << v)
        adj_masks[v] = mask_v ^ (1 << u)
        
        new_k5_aff, new_i5_aff = count_affected(adj_masks, n, u, v)
        
        current_k5 = current_k5 - old_k5_aff + new_k5_aff
        current_i5 = current_i5 - old_i5_aff + new_i5_aff
        new_cost = current_k5 + current_i5
        
        delta = new_cost - best_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept move
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = adj_masks[:]
                if best_cost == 0:
                    return n, get_edges(adj_masks, n)
        else:
            # Revert
            adj_masks[u] = mask_u
            adj_masks[v] = mask_v
            current_k5 = current_k5 - new_k5_aff + old_k5_aff
            current_i5 = current_i5 - new_i5_aff + old_i5_aff
        
        T = max(T_min, T * cooling)
    
    if is_valid(best_adj, n):
        return n, get_edges(best_adj, n)
    
    # Try n=40 with multiple random restarts
    if time.time() < deadline - 30:
        n2 = 40
        for restart in range(20):
            adj2 = [0] * n2
            for i in range(n2):
                for j in range(i + 1, n2):
                    if random.random() < 0.5:
                        adj2[i] |= (1 << j)
                        adj2[j] |= (1 << i)
            
            k5_2, i5_2 = init_counts(adj2, n2)
            best_cost_2 = k5_2 + i5_2
            best_adj_2 = adj2[:]
            T2 = 3.0
            T2_min = 0.05
            cooling2 = 0.999
            
            while time.time() < deadline - 5:
                u = random.randint(0, n2 - 1)
                v = random.randint(0, n2 - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                old_k5_aff, old_i5_aff = count_affected(adj2, n2, u, v)
                
                mask_u = adj2[u]
                mask_v = adj2[v]
                adj2[u] = mask_u ^ (1 << v)
                adj2[v] = mask_v ^ (1 << u)
                
                new_k5_aff, new_i5_aff = count_affected(adj2, n2, u, v)
                
                k5_2 = k5_2 - old_k5_aff + new_k5_aff
                i5_2 = i5_2 - old_i5_aff + new_i5_aff
                new_cost_2 = k5_2 + i5_2
                
                delta2 = new_cost_2 - best_cost_2
                
                if delta2 < 0 or random.random() < np.exp(-delta2 / T2):
                    if new_cost_2 < best_cost_2:
                        best_cost_2 = new_cost_2
                        best_adj_2 = adj2[:]
                        if best_cost_2 == 0:
                            return n2, get_edges(adj2, n2)
                else:
                    adj2[u] = mask_u
                    adj2[v] = mask_v
                    k5_2 = k5_2 - new_k5_aff + old_k5_aff
                    i5_2 = i5_2 - new_i5_aff + old_i5_aff
                
                T2 = max(T2_min, T2 * cooling2)
            
            if is_valid(best_adj_2, n2):
                return n2, get_edges(best_adj_2, n2)
    
    # Final fallback: Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj37 = [0] * 37
    for i in range(37):
        for j in range(i + 1, 37):
            if (j - i) % q in residues:
                adj37[i] |= (1 << j)
                adj37[j] |= (1 << i)
    
    return 37, get_edges(adj37, 37)
