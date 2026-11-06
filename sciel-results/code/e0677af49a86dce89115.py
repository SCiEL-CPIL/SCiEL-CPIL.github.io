import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback: Greedy for small n
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to find larger n
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
            
        # 1. Try Greedy
        greedy_res = greedy_solve(n, deadline)
        if greedy_res is not None:
            return greedy_res
            
        # 2. Try Backtracking
        bt_res = solve_n_backtrack(n, deadline)
        if bt_res is not None:
            return bt_res
            
        # If both fail, continue to smaller n
        # We can update best_n with greedy if it succeeded for smaller n?
        # No, we are iterating down. If n=17 fails, we try n=16.
        # If n=16 greedy succeeds, we return it.
        
    return {'n': best_n, 'colors': best_colors}

def greedy_solve(n, deadline):
    # colors[i][j] for i < j
    colors = [[-1] * n for _ in range(n)]
    # adj[c][u] bitmask
    adj = [[0] * n for _ in range(4)]
    
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for u, v in edges:
        if time.time() > deadline:
            return None
        placed = False
        for c in range(4):
            if not creates_c6(c, u, v, adj):
                colors[u][v] = c
                colors[v][u] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                placed = True
                break
        if not placed:
            return None
            
    # Flatten colors
    flat_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            flat_colors.append(colors[i][j])
            
    return {'n': n, 'colors': flat_colors}

def creates_c6(c, u, v, adj):
    # Check if adding edge (u,v) of color c creates a C6
    # i.e., is there a path of length 5 from u to v in adj[c]?
    # BFS from u up to depth 5
    visited = 1 << u
    frontier = 1 << u
    
    for _ in range(5):
        next_frontier = 0
        # Iterate through nodes in frontier
        # Since n <= 20, we can just loop 0..n-1
        # Or use bit operations
        f = frontier
        while f:
            # Get lowest set bit
            lsb = f & -f
            idx = lsb.bit_length() - 1
            f ^= lsb
            next_frontier |= adj[c][idx]
        next_frontier &= ~visited
        if next_frontier & (1 << v):
            return True
        visited |= next_frontier
        frontier = next_frontier
        if frontier == 0:
            break
            
    return False

def solve_n_backtrack(n, deadline):
    # Try different symmetry breaking prefixes
    # c(0,1)=0
    # c(0,2) in 0..3
    # c(0,3) in 0..3
    # c(0,4) in 0..3 ?
    # Let's just fix c(0,1)=0, c(0,2)=1, c(0,3)=2
    # And also try c(0,3)=0, c(0,3)=1, c(0,3)=3
    # Actually, let's iterate c(0,2) and c(0,3)
    
    colors = [[-1] * n for _ in range(n)]
    adj = [[0] * n for _ in range(4)]
    
    def backtrack(edge_idx):
        if time.time() > deadline:
            return False
        if edge_idx == n * (n - 1) // 2:
            return True
            
        u = edge_idx // (n - 1) # This mapping is tricky
        # Let's precompute edge list
        pass
        
    # Precompute edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)
    
    def bt(k):
        if time.time() > deadline:
            return False
        if k == m:
            return True
            
        u, v = edges[k]
        for c in range(4):
            if not creates_c6(c, u, v, adj):
                colors[u][v] = c
                colors[v][u] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if bt(k + 1):
                    return True
                    
                # Unassign
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors[u][v] = -1
                colors[v][u] = -1
                
        return False

    # Symmetry breaking: Fix first few edges
    # We will try a few specific assignments for the first 3 edges
    # e0: (0,1), e1: (0,2), e2: (0,3)
    
    for c1 in range(4):
        for c2 in range(4):
            # Reset state
            for i in range(n):
                for j in range(n):
                    colors[i][j] = -1
                for c in range(4):
                    adj[c][i] = 0
                    
            # Assign e0
            colors[0][1] = 0
            colors[1][0] = 0
            adj[0][0] |= (1 << 1)
            adj[0][1] |= (1 << 0)
            
            # Assign e1
            colors[0][2] = c1
            colors[2][0] = c1
            adj[c1][0] |= (1 << 2)
            adj[c1][2] |= (1 << 0)
            
            # Assign e2
            colors[0][3] = c2
            colors[3][0] = c2
            adj[c2][0] |= (1 << 3)
            adj[c2][3] |= (1 << 0)
            
            # Start backtracking from edge index 3
            if bt(3):
                flat_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        flat_colors.append(colors[i][j])
                return {'n': n, 'colors': flat_colors}
                
            if time.time() > deadline:
                return None
                
    return None
