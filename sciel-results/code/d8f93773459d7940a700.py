import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Seed solution n=9
    seed_n = 9
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    best = {'n': seed_n, 'colors': seed_colors}
    
    # Try to find higher n
    # n=13 is 78 edges, n=12 is 66, n=11 is 55.
    # We will try 13, then 12, then 11.
    
    for n in [13, 12, 11]:
        if time.time() > deadline - 2.0:
            break
            
        # Allocate time: more time for harder problems?
        # Or equal time?
        # Let's give 30s to 13, 20s to 12, 10s to 11.
        budget = 30.0 if n == 13 else (20.0 if n == 12 else 10.0)
        end_time = time.time() + budget
        
        res = _find_coloring(n, end_time)
        if res:
            return res
            
    return best

def _find_coloring(n, deadline):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Precompute neighbors for fast access? No, adj is dynamic.
    
    # Random seed for reproducibility in local tests, but random in eval?
    # Use random.shuffle
    
    for attempt in range(10000):
        if time.time() > deadline:
            break
            
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        
        def check_c6(u, v, c):
            nu = adj[c][u]
            nv = adj[c][v]
            
            # Iterate x1 in nu
            x1_mask = nu
            while x1_mask:
                lsb = x1_mask & -x1_mask
                x1 = lsb.bit_length() - 1
                x1_mask ^= lsb
                if x1 == v: continue
                
                # Iterate x4 in nv
                x4_mask = nv
                while x4_mask:
                    lsb4 = x4_mask & -x4_mask
                    x4 = lsb4.bit_length() - 1
                    x4_mask ^= lsb4
                    if x4 == u or x4 == x1: continue
                    
                    # N(x1) and N(x4) excluding forbidden {u, v, x1, x4}
                    forbidden = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
                    
                    nx1 = adj[c][x1] & ~forbidden
                    nx4 = adj[c][x4] & ~forbidden
                    
                    # Check if union of N(x2) for x2 in nx1 intersects nx4
                    # Union = 0
                    # for x2 in nx1: Union |= adj[c][x2]
                    # if Union & nx4: return True
                    
                    # Optimization:
                    # If nx1 is empty, skip.
                    if not nx1: continue
                    if not nx4: continue
                    
                    union = 0
                    vx2 = nx1
                    while vx2:
                        lsb2 = vx2 & -vx2
                        x2 = lsb2.bit_length() - 1
                        vx2 ^= lsb2
                        union |= adj[c][x2]
                        
                    if union & nx4:
                        return True
            return False

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == num_edges:
                return True
            
            u, v = edges[idx]
            
            # Symmetry breaking
            if idx == 0:
                c_vals = [0]
            else:
                c_vals = [0, 1, 2, 3]
                random.shuffle(c_vals)
                
            for c in c_vals:
                if check_c6(u, v, c):
                    continue
                    
                colors[idx] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if backtrack(idx + 1):
                    return True
                    
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors[idx] = -1
                
            return False
            
        if backtrack(0):
            return {'n': n, 'colors': colors}
            
    return None
