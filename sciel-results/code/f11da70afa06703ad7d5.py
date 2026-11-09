import sys
import time

def construct():
    n = 30
    # Edge list in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    colors = [None] * len(edges)
    
    # Bitsets for neighbors
    # adj[0][u] is bitset of neighbors connected to u by color 0
    # adj[1][u] is bitset of neighbors connected to u by color 1
    adj = [[0] * n for _ in range(2)]
    
    # Map edge (i,j) to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    def get_color(u, v):
        return colors[edge_idx[(u, v)]]
        
    def set_color(u, v, c):
        idx = edge_idx[(u, v)]
        colors[idx] = c
        # Update bitsets
        adj[c][u] |= (1 << v)
        adj[c][v] |= (1 << u)
        
    def unset_color(u, v, c):
        idx = edge_idx[(u, v)]
        colors[idx] = None
        adj[c][u] &= ~(1 << v)
        adj[c][v] &= ~(1 << u)

    def has_k222(u, v, c):
        # Check if adding edge (u,v) of color c creates a mono K2,2,2
        # u, v are in different parts.
        # We need x paired with u, y paired with v, z,w paired with each other.
        # x in N_c(v), y in N_c(u), edge(x,y)=c.
        # z,w in N_c(u) cap N_c(v) cap N_c(x) cap N_c(y).
        
        # Get bitsets
        # N_c(v) is adj[c][v]. But u is not in N_c(v) yet? 
        # We are checking BEFORE adding (u,v). 
        # So adj[c][v] does NOT contain u.
        # We need x such that edge(v,x) is c. So x in adj[c][v].
        # We need y such that edge(u,y) is c. So y in adj[c][u].
        
        A = adj[c][v]
        B = adj[c][u]
        
        # Iterate x in A
        x = 0
        while A:
            if A & 1:
                # x is 0? No, we need to extract index.
                pass
            A >>= 1
            x += 1
        # This loop is inefficient. Use bit manipulation.
        
        # Better: iterate over bits in A
        a_bits = []
        temp_A = adj[c][v]
        idx_x = 0
        while temp_A:
            if temp_A & 1:
                a_bits.append(idx_x)
            temp_A >>= 1
            idx_x += 1
            
        b_bits = []
        temp_B = adj[c][u]
        idx_y = 0
        while temp_B:
            if temp_B & 1:
                b_bits.append(idx_y)
            temp_B >>= 1
            idx_y += 1
            
        for x in a_bits:
            if x == u: continue # x cannot be u
            # Check if edge(x, v) is c? Yes, x in N_c(v).
            # We need y in B such that edge(x, y) is c.
            # And then check S.
            for y in b_bits:
                if y == v: continue # y cannot be v
                if x == y: continue # x, y distinct
                # Check if edge(x, y) is c
                if get_color(x, y) == c:
                    # Check if there are at least 2 common neighbors of u, v, x, y in color c
                    # S = N_c(u) & N_c(v) & N_c(x) & N_c(y)
                    S = adj[c][u] & adj[c][v] & adj[c][x] & adj[c][y]
                    # Count bits in S
                    if S & (S - 1): # at least 2 bits set
                        return True
        return False

    def backtrack(idx):
        if idx == len(edges):
            return True
            
        u, v = edges[idx]
        
        # Try color 0
        if not has_k222(u, v, 0):
            set_color(u, v, 0)
            if backtrack(idx + 1):
                return True
            unset_color(u, v, 0)
            
        # Try color 1
        if not has_k222(u, v, 1):
            set_color(u, v, 1)
            if backtrack(idx + 1):
                return True
            unset_color(u, v, 1)
            
        return False

    # Symmetry breaking: Fix first few edges
    # We can force (0,1) to be 0.
    # And maybe (0,2) to be 0?
    # Let's just force (0,1)=0 to break the global color swap symmetry.
    set_color(0, 1, 0)
    
    # Start backtracking from edge index 1
    # We need to be careful: the backtracking function expects to assign edges[idx].
    # We already assigned edges[0].
    # So we call backtrack(1).
    
    # Add a timeout guard
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # We need to modify backtrack to check time.
    # Since Python recursion is slow, we might need to be careful.
    # But n=30 is small enough if pruning is strong.
    
    # Let's implement a manual stack or just rely on recursion with time check.
    # Recursion depth is 435. Python default limit is 1000. OK.
    
    # To handle timeout, we can't easily break out of deep recursion in Python without exceptions.
    # We will use an exception for timeout.
    
    class TimeoutError(Exception):
        pass
        
    def backtrack_timed(idx):
        if time.time() > deadline:
            raise TimeoutError()
            
        if idx == len(edges):
            return True
            
        u, v = edges[idx]
        
        # Try color 0
        if not has_k222(u, v, 0):
            set_color(u, v, 0)
            if backtrack_timed(idx + 1):
                return True
            unset_color(u, v, 0)
            
        # Try color 1
        if not has_k222(u, v, 1):
            set_color(u, v, 1)
            if backtrack_timed(idx + 1):
                return True
            unset_color(u, v, 1)
            
        return False

    try:
        if backtrack_timed(1):
            return {'n': n, 'colors': colors}
    except TimeoutError:
        pass
        
    # If 30 fails, try 29
    n = 29
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    colors = [None] * len(edges)
    adj = [[0] * n for _ in range(2)]
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    def get_color_n(u, v):
        return colors[edge_idx[(u, v)]]
        
    def set_color_n(u, v, c):
        idx = edge_idx[(u, v)]
        colors[idx] = c
        adj[c][u] |= (1 << v)
        adj[c][v] |= (1 << u)
        
    def unset_color_n(u, v, c):
        idx = edge_idx[(u, v)]
        colors[idx] = None
        adj[c][u] &= ~(1 << v)
        adj[c][v] &= ~(1 << u)

    def has_k222_n(u, v, c):
        A = adj[c][v]
        B = adj[c][u]
        
        a_bits = []
        temp_A = A
        idx_x = 0
        while temp_A:
            if temp_A & 1:
                a_bits.append(idx_x)
            temp_A >>= 1
            idx_x += 1
            
        b_bits = []
        temp_B = B
        idx_y = 0
        while temp_B:
            if temp_B & 1:
                b_bits.append(idx_y)
            temp_B >>= 1
            idx_y += 1
            
        for x in a_bits:
            if x == u: continue
            for y in b_bits:
                if y == v: continue
                if x == y: continue
                if get_color_n(x, y) == c:
                    S = adj[c][u] & adj[c][v] & adj[c][x] & adj[c][y]
                    if S & (S - 1):
                        return True
        return False

    def backtrack_n(idx):
        if time.time() > deadline:
            raise TimeoutError()
        if idx == len(edges):
            return True
        u, v = edges[idx]
        if not has_k222_n(u, v, 0):
            set_color_n(u, v, 0)
            if backtrack_n(idx + 1):
                return True
            unset_color_n(u, v, 0)
        if not has_k222_n(u, v, 1):
            set_color_n(u, v, 1)
            if backtrack_n(idx + 1):
                return True
            unset_color_n(u, v, 1)
        return False

    set_color_n(0, 1, 0)
    try:
        if backtrack_n(1):
            return {'n': n, 'colors': colors}
    except TimeoutError:
        pass

    # Fallback to seed solution
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}

# EVOLVE-BLOCK-END
