import time
from math import gcd

def construct():
    """
    Returns the maximum cardinality subset of [1, 255] containing 255,
    such that gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    # 1. Identify candidate nodes: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    candidates = []
    for i in range(1, 256):
        if gcd(i, 255) > 1:
            candidates.append(i)
            
    n = len(candidates)
    
    # Map value to index in candidates list for quick lookup
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # 2. Build adjacency using bitsets for fast operations
    adj_bits = [0] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
                
    # Find index of 255
    idx_255 = val_to_idx[255]
    
    # Since 255 is connected to all candidates (by construction),
    # the max clique containing 255 = 1 + max clique of G \ {255}
    
    # Build subgraph excluding 255
    old_to_new = {}
    new_to_old = []
    new_idx = 0
    for i in range(n):
        if i != idx_255:
            old_to_new[i] = new_idx
            new_to_old.append(i)
            new_idx += 1
            
    m = new_idx  # number of nodes in subgraph
    
    # Build adjacency bitsets for subgraph
    sub_adj_bits = [0] * m
    for i in range(m):
        old_i = new_to_old[i]
        for j in range(m):
            if i != j:
                old_j = new_to_old[j]
                if adj_bits[old_i] & (1 << old_j):
                    sub_adj_bits[i] |= (1 << j)
    
    # Degrees in subgraph
    degrees = [bin(sub_adj_bits[i]).count('1') for i in range(m)]
    
    # Order vertices by degree descending
    order = sorted(range(m), key=lambda x: degrees[x], reverse=True)
    
    # Remap to ordered indices
    old_to_ordered = [0] * m
    ordered_to_old = [0] * m
    for new_pos, old_pos in enumerate(order):
        old_to_ordered[old_pos] = new_pos
        ordered_to_old[new_pos] = old_pos
    
    # Build ordered adjacency
    ord_adj_bits = [0] * m
    for i in range(m):
        old_i = ordered_to_old[i]
        for j in range(m):
            if i != j:
                old_j = ordered_to_old[j]
                if sub_adj_bits[old_i] & (1 << old_j):
                    ord_adj_bits[i] |= (1 << j)
    
    # Tomita-style maximum clique with coloring bound
    best_size = 0
    best_clique = []
    
    def expand_clique(R, P):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if len(R) + len(P) <= best_size:
            return
            
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
            
        # Greedy coloring to get upper bound
        p_list = list(P)
        
        # Compute degrees in P
        p_deg = {}
        for u in p_list:
            cnt = 0
            for v in p_list:
                if u != v and (ord_adj_bits[u] & (1 << v)):
                    cnt += 1
            p_deg[u] = cnt
            
        p_list.sort(key=lambda x: p_deg[x], reverse=True)
        
        # Greedy coloring: assign smallest available color
        colors = {}
        for u in p_list:
            used = set()
            for v in p_list:
                if v in colors and (ord_adj_bits[u] & (1 << v)):
                    used.add(colors[v])
            c = 0
            while c in used:
                c += 1
            colors[u] = c
            
        max_color = max(colors.values()) if colors else 0
        
        if len(R) + max_color <= best_size:
            return
            
        # Branch: iterate through vertices in order of increasing color
        p_sorted_by_color = sorted(p_list, key=lambda x: colors[x])
        
        for u in p_sorted_by_color:
            if time.time() > deadline:
                return
                
            if len(R) + colors[u] <= best_size:
                continue
                
            new_R = R + [u]
            new_P = [v for v in P if v != u and (ord_adj_bits[u] & (1 << v))]
            
            expand_clique(new_R, new_P)
            
            P.remove(u)
            
            if not P:
                break
    
    # Initial call
    initial_P = set(range(m))
    expand_clique([], initial_P)
    
    # Reconstruct the clique
    final_values = [255]
    for node_idx in best_clique:
        old_idx = ordered_to_old[node_idx]
        orig_idx = new_to_old[old_idx]
        final_values.append(candidates[orig_idx])
        
    if 255 not in final_values:
        final_values.append(255)
        
    return final_values

# Run and verify
if __name__ == "__main__":
    res = construct()
    print(len(res))
    # Verify
    valid = True
    for i in range(len(res)):
        if 255 not in res:
            valid = False
        for j in range(i+1, len(res)):
            if gcd(res[i], res[j]) == 1:
                valid = False
                break
    print("Valid:", valid)
