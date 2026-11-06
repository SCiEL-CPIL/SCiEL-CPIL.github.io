import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # We want to maximize n.
    # Try 17 first. If it fails, try 16, etc.
    # But 17 is the target.
    
    # Fallback solution for n=9 (seed)
    # This is a valid coloring, just small.
    def get_seed_solution(n):
        # Simple construction: color(i,j) = (i+j) % 4? No, might have C6.
        # The seed in the prompt was n=9.
        # Let's just return a hardcoded valid n=13 or similar if we fail?
        # Actually, the prompt says "Seed is n=9".
        # We should try to beat 9.
        # My backtracker should easily find n=15, 16.
        pass

    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline - 10: # Keep 10s buffer
            break
            
        # Allocate time based on n
        # n=17 is hard, give it most time.
        time_limit = 100.0 if n == 17 else 20.0
        target_end = time.time() + time_limit
        
        result = _search(n, target_end)
        if result:
            return result
            
    # If all fail, return a small valid solution
    # We need a valid solution for n=10 at least.
    # Let's just run a very short search for n=10 to be sure.
    if time.time() < deadline - 1:
        res = _search(10, time.time() + 1.0)
        if res:
            return res
            
    # Absolute fallback: n=1
    return {'n': 1, 'colors': []}

def _search(n, deadline):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Map edge to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # State
    colors = [-1] * num_edges
    adj = [[0] * n for _ in range(4)]
    
    # Precompute bit masks for vertices?
    # We use Python ints as bitsets.
    
    def is_valid_add(u, v, c):
        """
        Check if adding edge (u, v) with color c creates a C6.
        This is true if there is a path of length 5 from u to v in G_c.
        Path: u - x1 - x2 - x3 - x4 - v
        """
        nu = adj[c][u]
        nv = adj[c][v]
        
        # Iterate x1 in N(u)
        x1_mask = nu
        while x1_mask:
            lsb = x1_mask & -x1_mask
            x1 = lsb.bit_length() - 1
            x1_mask ^= lsb
            
            # x1 must be distinct from v (handled by mask? No, v might be in nu? 
            # In a simple graph, u and v are not adjacent in G_c yet. 
            # But wait, if we are adding (u,v), they are NOT in G_c. 
            # So v is NOT in nu. u is NOT in nv.
            # So x1 != v is guaranteed if graph is simple and (u,v) not yet added.
            # However, we must ensure x1 != u (guaranteed) and x1 != v (guaranteed).
            
            # Iterate x4 in N(v)
            x4_mask = nv
            while x4_mask:
                lsb4 = x4_mask & -x4_mask
                x4 = lsb4.bit_length() - 1
                x4_mask ^= lsb4
                
                # x4 must be distinct from u and x1
                if x4 == u or x4 == x1:
                    continue
                
                # Forbidden vertices for x2, x3: u, v, x1, x4
                forbidden = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
                
                # x2 in N(x1) \ forbidden
                nx1 = adj[c][x1] & ~forbidden
                
                # We need to check if there exists x2 in nx1 and x3 in N(x4) \ forbidden
                # such that (x2, x3) is an edge in G_c.
                
                # Optimization:
                # Let S2 = N(x1) \ forbidden
                # Let S3 = N(x4) \ forbidden
                # We need an edge between S2 and S3.
                
                # Compute Union of Neighbors of S2
                # U = Union( N(x2) for x2 in S2 )
                # If U intersects S3, then there is an edge.
                
                union_neighbors = 0
                temp = nx1
                while temp:
                    lsb2 = temp & -temp
                    x2 = lsb2.bit_length() - 1
                    temp ^= lsb2
                    union_neighbors |= adj[c][x2]
                    
                # Check intersection with S3
                nx4 = adj[c][x4] & ~forbidden
                if union_neighbors & nx4:
                    return False
                    
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
            
        if idx == num_edges:
            return True
            
        u, v = edges[idx]
        
        # Symmetry Breaking
        # Color 0 is fixed for edge (0,1)
        # Color 1 is fixed for edge (0,2) if (0,2) != 0?
        # Let's use a standard symmetry breaking for 4 colors:
        # c(0,1) = 0
        # c(0,2) in {0, 1}
        # c(0,3) in {0, 1, 2}
        
        if idx == 0:
            c_vals = [0]
        elif idx == 1: # Edge (0,2)
            c_vals = [0, 1]
            random.shuffle(c_vals)
        elif idx == 2: # Edge (0,3)
            c_vals = [0, 1, 2]
            random.shuffle(c_vals)
        else:
            c_vals = [0, 1, 2, 3]
            random.shuffle(c_vals)
            
        for c in c_vals:
            if not is_valid_add(u, v, c):
                continue
                
            # Assign
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
            if backtrack(idx + 1):
                return True
                
            # Unassign
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[idx] = -1
            
        return False

    # Multiple restarts
    # Since the search is deterministic given the random shuffles, 
    # we just run it once with random shuffles.
    # If it fails, we can restart.
    
    max_restarts = 100
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Reset state
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        
        if backtrack(0):
            return {'n': n, 'colors': colors}
            
    return None
