import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin

    # Try n from 17 down to 9
    # We want to maximize n, so start high.
    
    # Seed solution (n=9) to ensure we don't return worse than seed
    seed_n = 9
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    best = {'n': seed_n, 'colors': seed_colors}
    
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
            
        # Allocate time based on n.
        # Harder problems (higher n) get more time initially, but we must be fast.
        # n=17: 25s, n=16: 15s, n=15: 10s, n=14: 8s, n=13: 5s, n=12: 5s, n=11: 5s, n=10: 5s
        time_limit = 25.0 if n == 17 else (15.0 if n == 16 else (10.0 if n == 15 else 8.0))
        target_end = time.time() + time_limit
        
        result = _search(n, target_end)
        if result:
            return result
            
    return best

def _search(n, deadline):
    # Backtracking search with strong pruning
    # Edges in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Precompute edge index for O(1) lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Colors for each edge, -1 for unassigned
    colors = [-1] * num_edges
    
    # Adjacency bitsets for each color
    # adj[c][u] is a bitmask of vertices connected to u with color c
    adj = [[0] * n for _ in range(4)]
    
    def check_c6(u, v, c):
        # Check if adding edge (u, v) with color c creates a C6
        # We need a path of length 5 from u to v in G_c (current adj)
        # Path: u - x1 - x2 - x3 - x4 - v
        # x1 in N(u), x4 in N(v)
        # x2 - x3 is an edge in G_c
        # x2 in N(x1), x3 in N(x4)
        # x2 != x3, and all distinct from u, v, x1, x4
        
        nu = adj[c][u]
        nv = adj[c][v]
        
        # Iterate x1 in nu
        x1_mask = nu
        while x1_mask:
            lsb = x1_mask & -x1_mask
            x1 = lsb.bit_length() - 1
            x1_mask ^= lsb
            
            # x1 must not be v
            if x1 == v: continue
            
            # Iterate x4 in nv
            x4_mask = nv
            while x4_mask:
                lsb4 = x4_mask & -x4_mask
                x4 = lsb4.bit_length() - 1
                x4_mask ^= lsb4
                
                # x4 must not be u or x1
                if x4 == u or x4 == x1: continue
                
                # Check if there is a path x1 - x2 - x3 - x4
                # x2 in N(x1), x3 in N(x4), edge (x2, x3) in G_c
                # x2 != x3, x2 not in {u, v, x1, x4}, x3 not in {u, v, x1, x4}
                
                # Forbidden mask for x2, x3: u, v, x1, x4
                forbidden = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
                
                # N(x1) and N(x4) with forbidden removed
                nx1 = adj[c][x1] & ~forbidden
                nx4 = adj[c][x4] & ~forbidden
                
                # We need to check if there is any edge (x2, x3) in G_c
                # where x2 in nx1 and x3 in nx4.
                # This is equivalent to checking if for any x2 in nx1,
                # (adj[c][x2] & nx4) is non-zero.
                
                # Optimization: Precompute the union of neighbors of x1's neighbors?
                # No, just iterate. nx1 is usually small.
                
                vx2 = nx1
                while vx2:
                    lsb2 = vx2 & -vx2
                    x2 = lsb2.bit_length() - 1
                    vx2 ^= lsb2
                    
                    if adj[c][x2] & nx4:
                        return True # C6 found
                        
        return False

    # Heuristic: Random restarts with random color ordering
    # and symmetry breaking.
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
            
        if idx == num_edges:
            return True
            
        u, v = edges[idx]
        
        # Symmetry breaking:
        # 1. First edge (0,1) must be color 0.
        if idx == 0:
            c_vals = [0]
        else:
            # 2. Second edge (0,2) can be 0 or 1.
            # If it's 1, it fixes the second color.
            # If it's 0, we haven't used color 1 yet.
            # Let's enforce: the first edge of a new color must be the smallest possible.
            # This is complex. A simpler symmetry breaking:
            # Count used colors. If we are trying to assign a new color,
            # it must be the smallest unused color.
            
            used_colors = set()
            for i in range(idx):
                used_colors.add(colors[i])
                
            available = [0, 1, 2, 3]
            if idx == 1:
                # Edge (0,2). (0,1) is 0.
                # We can use 0 or 1.
                c_vals = [0, 1]
            else:
                # General case
                c_vals = [0, 1, 2, 3]
                random.shuffle(c_vals)
                
            # Prune: If we try to use a color that hasn't been used yet,
            # it must be the smallest such color.
            # However, shuffling makes this hard to enforce simply.
            # Let's stick to simple symmetry: (0,1)=0.
            # And maybe (0,2) in {0,1}.
            
        for c in c_vals:
            # Additional symmetry check:
            # If c is not in used_colors, and there is a smaller color c' not in used_colors,
            # then c is symmetric to c'. We should only try c if it's the smallest unused.
            if idx > 0:
                used_colors = set(colors[:idx])
                if c not in used_colors:
                    smaller_unused = [x for x in range(4) if x < c and x not in used_colors]
                    if smaller_unused:
                        continue
                        
            if check_c6(u, v, c):
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

    # Try several random restarts
    for _ in range(2000):
        if time.time() > deadline:
            break
            
        # Reset state
        for i in range(num_edges):
            colors[i] = -1
        for c in range(4):
            for i in range(n):
                adj[c][i] = 0
                
        if backtrack(0):
            return {'n': n, 'colors': colors}
            
    return None
