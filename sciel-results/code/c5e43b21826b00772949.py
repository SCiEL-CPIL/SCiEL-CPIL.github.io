import time
import random
import numpy as np

def _paley_colors(n):
    """Generate Paley graph coloring for prime n."""
    squares = set()
    for x in range(1, n // 2 + 1):
        squares.add((x * x) % n)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _count_mono_k22(adj):
    """
    Count monochromatic K2,2 subgraphs.
    adj is a list of lists or numpy array. 
    Returns count of mono K2,2.
    A mono K2,2 of color c exists for pairs (u,v) and (x,y) if edges (u,x),(u,y),(v,x),(v,y) are all c.
    This is equivalent to u and v having at least 2 common neighbors in color c.
    So sum over all pairs (u,v) of C(common_neighbors(u,v), 2).
    """
    n = adj.shape[0]
    total = 0
    for c in (0, 1):
        # Build bitmasks for color c
        masks = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i, j] == c:
                    masks[i] |= (1 << j)
        
        for u in range(n):
            for v in range(u + 1, n):
                # Common neighbors in color c
                common = masks[u] & masks[v]
                # Remove u and v themselves if they are connected? 
                # In simple graph, no self loops. 
                # adj[u,u] is 0. So masks[u] does not have bit u.
                # So common does not have u or v.
                cnt = bin(common).count('1')
                if cnt >= 2:
                    total += cnt * (cnt - 1) // 2
    return total

def _has_mono_k222(adj):
    """
    Check if there is a monochromatic K2,2,2.
    Returns True if found.
    """
    n = adj.shape[0]
    for c in (0, 1):
        masks = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i, j] == c:
                    masks[i] |= (1 << j)
        
        for u in range(n):
            for v in range(u + 1, n):
                S = masks[u] & masks[v]
                S &= ~(1 << u)
                S &= ~(1 << v)
                
                # Iterate over pairs b1, b2 in S
                # We need to find b1 < b2 in S such that
                # there exist c1, c2 in S \ {b1, b2} such that
                # c1, c2 are in masks[b1] & masks[b2]
                
                # Extract indices from S
                # S is an int
                # Get list of set bits
                s_indices = []
                temp = S
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    s_indices.append(idx)
                    temp ^= lsb
                    
                m = len(s_indices)
                if m < 4:
                    continue
                
                # Check all pairs in s_indices
                for i in range(m):
                    b1 = s_indices[i]
                    for j in range(i + 1, m):
                        b2 = s_indices[j]
                        
                        # Common neighbors of b1 and b2 in color c
                        common_b = masks[b1] & masks[b2]
                        # Must be in S, and not b1, b2
                        common_b &= S
                        common_b &= ~(1 << b1)
                        common_b &= ~(1 << b2)
                        
                        cnt = bin(common_b).count('1')
                        if cnt >= 2:
                            return True
    return False

def _sa_search(adj_init, deadline):
    """
    Run Simulated Annealing to minimize count of mono K2,2.
    """
    n = adj_init.shape[0]
    adj = adj_init.copy()
    current_energy = _count_mono_k22(adj)
    
    if current_energy == 0:
        if not _has_mono_k222(adj):
            return adj, 0
            
    T_start = 5.0
    T_end = 0.01
    steps = 5000
    
    alpha = (T_end / T_start) ** (1.0 / steps)
    T = T_start
    best_adj = adj.copy()
    best_energy = current_energy
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Pick random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        while i == j:
            j = random.randint(0, n - 1)
            
        if i > j:
            i, j = j, i
            
        old_color = adj[i, j]
        new_color = 1 - old_color
        
        # Apply flip
        adj[i, j] = new_color
        adj[j, i] = new_color
        
        new_energy = _count_mono_k22(adj)
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_adj = adj.copy()
                if best_energy == 0:
                    if not _has_mono_k222(best_adj):
                        return best_adj, 0
        else:
            # Revert
            adj[i, j] = old_color
            adj[j, i] = old_color
            
        T *= alpha
        
    return best_adj, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Hardcoded valid n=29
    colors_29 = _paley_colors(29)
    
    # Try to find n=30
    n_target = 30
    # Initialize with random or Paley-like
    # Let's try a few random restarts
    
    best_result = {'n': 29, 'colors': colors_29}
    
    # Attempt 1: Random init
    random.seed(42)
    adj_init = np.random.randint(0, 2, (n_target, n_target))
    adj_init = np.triu(adj_init, 1)
    adj_init = adj_init + adj_init.T
    np.fill_diagonal(adj_init, 0)
    
    adj_res, energy_res = _sa_search(adj_init, deadline)
    if energy_res == 0 and not _has_mono_k222(adj_res):
        colors = []
        for i in range(n_target):
            for j in range(i+1, n_target):
                colors.append(adj_res[i, j])
        return {'n': n_target, 'colors': colors}
        
    # Attempt 2: Paley 29 + random vertex
    if time.time() < deadline - 10:
        adj_paley_29 = np.zeros((29, 29), dtype=np.int8)
        idx = 0
        for i in range(29):
            for j in range(i+1, 29):
                c = colors_29[idx]
                adj_paley_29[i, j] = c
                adj_paley_29[j, i] = c
                idx += 1
                
        adj_init_2 = np.zeros((n_target, n_target), dtype=np.int8)
        adj_init_2[:29, :29] = adj_paley_29
        for i in range(29):
            c = random.randint(0, 1)
            adj_init_2[i, 29] = c
            adj_init_2[29, i] = c
            
        adj_res2, energy_res2 = _sa_search(adj_init_2, deadline)
        if energy_res2 == 0 and not _has_mono_k222(adj_res2):
            colors = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    colors.append(adj_res2[i, j])
            return {'n': n_target, 'colors': colors}

    return best_result
