import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Try n from 17 down to 9
    # We want to maximize n, so start high.
    # If we fail on 17, try 16, etc.
    # But 17 might be hard. Maybe try 13 first to ensure a good score, then use remaining time for 17?
    # Better: Try 17 with a time limit. If fail, try 16, etc.
    
    best_result = {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
    
    # Known good lower bound to ensure we don't return worse than seed
    # Seed is n=9.
    
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
            
        # Allocate time based on n. Harder problems get more time?
        # Or just a fixed slice per n?
        # Let's give 20s to 17, 10s to 16, 5s to 15, etc.
        time_limit = 20.0 if n == 17 else (10.0 if n == 16 else (5.0 if n == 15 else 2.0))
        target_end = time.time() + time_limit
        
        result = _search(n, target_end)
        if result:
            return result
            
    return best_result

def _search(n, deadline):
    # Backtracking search
    # Edges in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Colors for each edge, -1 for unassigned
    colors = [-1] * num_edges
    
    # Adjacency bitsets for each color
    # adj[c][u] is a bitmask of vertices connected to u with color c
    adj = [[0] * n for _ in range(4)]
    
    # Precompute edge index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return edge_idx[(i, j)]
        
    def is_valid_add(u, v, c):
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
        # To be efficient, we can iterate over set bits
        # But Python bit iteration is slow.
        # Alternative: Check if there exists x1, x4 such that (N(x1) & N(x4) & ~(forbidden)) != 0
        # Forbidden: u, v, x1, x4.
        
        # Let's iterate x1
        x1_mask = nu
        while x1_mask:
            # Extract lowest set bit
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
                
                # N(x1) and N(x4)
                nx1 = adj[c][x1]
                nx4 = adj[c][x4]
                
                # Common neighbors? No, we need an edge between a neighbor of x1 and a neighbor of x4.
                # Let's iterate x2 in N(x1)
                # Optimization: If (N(x1) & N(x4)) has a bit, it's a common neighbor (path length 2 via same node? No, C6 requires distinct nodes).
                # We need x2 != x3.
                
                # Let's just check if there is any pair (x2, x3) such that x2 in N(x1), x3 in N(x4), and (x2, x3) is an edge in G_c.
                # This is equivalent to: Does G_c contain an edge (x2, x3) where x2 in N(x1) \ Forbidden and x3 in N(x4) \ Forbidden?
                
                # Forbidden mask for x2, x3: u, v, x1, x4
                forbidden = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
                
                valid_x2s = nx1 & ~forbidden
                valid_x3s = nx4 & ~forbidden
                
                # We need to check if any edge (x2, x3) in G_c exists with x2 in valid_x2s and x3 in valid_x3s.
                # We can iterate x2 in valid_x2s
                vx2 = valid_x2s
                while vx2:
                    lsb2 = vx2 & -vx2
                    x2 = lsb2.bit_length() - 1
                    vx2 ^= lsb2
                    
                    # Check if x2 is connected to any x3 in valid_x3s
                    # adj[c][x2] & valid_x3s
                    if adj[c][x2] & valid_x3s:
                        return False # C6 found
                
        return True

    # Heuristic: Order vertices or edges?
    # Standard backtracking on lexicographic edges is slow.
    # Use "fail-first": pick edge with fewest available colors?
    # Or just random restarts with greedy initialization?
    
    # Let's try a simple backtracking with forward checking?
    # Given time, let's try randomized restarts with a simple backtracker.
    
    for attempt in range(10000):
        if time.time() > deadline:
            break
            
        # Randomize edge order? Or just restart search?
        # For simplicity, we'll do a depth-first search with random value ordering.
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if idx == num_edges:
                return colors[:]
                
            u, v = edges[idx]
            
            # Try colors 0-3 in random order
            col_order = [0, 1, 2, 3]
            random.shuffle(col_order)
            
            for c in col_order:
                # Check if color c is possible
                # Symmetry breaking:
                # First edge must be 0
                if idx == 0 and c != 0:
                    continue
                # Second edge (0,2) must be 0 or 1?
                # If (0,1)=0, then (0,2) can be 0 or 1. If 1, we fix color 1 to be the second new color.
                # This is tricky. Let's just use (0,1)=0.
                
                if not is_valid_add(u, v, c):
                    continue
                
                # Assign
                colors[idx] = c
                # Update adj
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                res = backtrack(idx + 1)
                if res:
                    return res
                
                # Unassign
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors[idx] = -1
                
            return None
            
        # Reset state for each restart?
        # No, backtrack modifies local state.
        # But `adj` and `colors` are closed over. We need to reset them.
        # It's better to have a class or pass state.
        # For speed, let's just recreate the state.
        
        # Actually, the `is_valid_add` uses `adj`.
        # We need to reset `adj` and `colors` before each attempt.
        
        # This is inefficient in Python.
        # Let's try a different approach: Greedy construction with repair?
        
    return None

# The above is too slow/complex for Python within 120s for n=17.
# Let's stick to finding n=10..13 which is easier.
# Or use a known construction for n=17 if I can guess it?
# No, let's just optimize the search for smaller n.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Seed solution
    seed_n = 9
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    
    best = {'n': seed_n, 'colors': seed_colors}
    
    # Try to improve
    for n in range(17, 9, -1):
        if time.time() > deadline - 5: # Stop if close to end
            break
            
        # Time budget for this n
        budget = 15.0 if n >= 16 else 5.0
        end_time = time.time() + budget
        
        res = _find_coloring(n, end_time)
        if res:
            return res
            
    return best

def _find_coloring(n, deadline):
    # Use a simpler, faster check for small n
    # For n <= 13, the number of edges is small (78 for n=13).
    # Backtracking might work.
    
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Use numpy for bitsets? No, Python ints are fine.
    
    # We will use a randomized backtracking search.
    # To speed up, we'll use a "watched" approach or just fast bit operations.
    
    import random
    
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        
        # Random restart: shuffle edge order?
        # No, keep lexicographic but randomize color order.
        
        def check_c6(u, v, c):
            # Returns True if adding (u,v) creates C6
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
                    
                    # Check path x1-x2-x3-x4
                    # x2 in N(x1), x3 in N(x4), edge (x2,x3) in Gc
                    # x2, x3 distinct from u, v, x1, x4
                    
                    forbidden = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
                    
                    nx1 = adj[c][x1] & ~forbidden
                    nx4 = adj[c][x4] & ~forbidden
                    
                    # Check if any x2 in nx1 is connected to any x3 in nx4
                    # We can check: for each x2 in nx1, is (adj[c][x2] & nx4) non-zero?
                    vx2 = nx1
                    while vx2:
                        lsb2 = vx2 & -vx2
                        x2 = lsb2.bit_length() - 1
                        vx2 ^= lsb2
                        if adj[c][x2] & nx4:
                            return True
            return False

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == num_edges:
                return True
            
            u, v = edges[idx]
            
            # Symmetry: first edge 0
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
