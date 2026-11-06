import time
import random
from itertools import combinations

def _has_k222_fast(adj, n):
    """
    Check if graph with adjacency bitmask list adj on n vertices contains K2,2,2.
    K2,2,2 is complete tripartite with parts of size 2.
    Parts: {a,b}, {c,d}, {e,f}. All 12 edges between different parts must exist.
    No edges required within parts.
    """
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            # a,b are edge (part 1)
            for c in range(b+1, n):
                if not (adj[a] >> c) & 1 or not (adj[b] >> c) & 1:
                    continue
                # a-c and b-c edges exist
                for d in range(c+1, n):
                    if not (adj[a] >> d) & 1 or not (adj[b] >> d) & 1:
                        continue
                    # a-d and b-d edges exist. Now {a,b} and {c,d} form a K4 minus possibly (c,d).
                    # We need two more vertices e,f that connect to all of a,b,c,d.
                    cn_mask = adj[a] & adj[b] & adj[c] & adj[d]
                    cn_mask &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                    if cn_mask.bit_count() >= 2:
                        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Step 1: Generate base n=29 solution (Paley 29)
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    n_base = q
    adj_base = [0] * n_base
    for i in range(n_base):
        for j in range(i+1, n_base):
            diff = (j - i) % q
            if diff in squares:
                adj_base[i] |= (1 << j)
                adj_base[j] |= (1 << i)
    
    # Step 2: Try to extend to n=30
    n_target = 30
    
    # Precompute base adj as bitmask list for speed
    # adj_base[i] already has bits set for neighbors in color 1
    
    # For extension, x[i] = color of edge (29, i)
    # Color 1 edges: adj_base plus edges where x[i]=1
    # Color 0 edges: complement of color 1
    
    def count_violations_for_x(x):
        """Count K2,2,2 violations in both colors for the extension defined by x."""
        full = (1 << n_target) - 1
        adj1 = [0] * n_target
        # Copy base
        for i in range(n_base):
            adj1[i] = adj_base[i]
        # Add new vertex edges
        adj1[29] = 0
        for i in range(n_base):
            if x[i] == 1:
                adj1[29] |= (1 << i)
                adj1[i] |= (1 << 29)
        
        # Build color 0 adjacency
        adj0 = [0] * n_target
        for i in range(n_target):
            adj0[i] = full ^ adj1[i] ^ (1 << i)
        
        # Count violations
        count = 0
        if _has_k222_fast(adj1, n_target):
            count += 1
        if _has_k222_fast(adj0, n_target):
            count += 1
        return count
    
    def is_valid_x(x):
        full = (1 << n_target) - 1
        adj1 = [0] * n_target
        for i in range(n_base):
            adj1[i] = adj_base[i]
        adj1[29] = 0
        for i in range(n_base):
            if x[i] == 1:
                adj1[29] |= (1 << i)
                adj1[i] |= (1 << 29)
        
        adj0 = [0] * n_target
        for i in range(n_target):
            adj0[i] = full ^ adj1[i] ^ (1 << i)
        
        if _has_k222_fast(adj1, n_target):
            return False
        if _has_k222_fast(adj0, n_target):
            return False
        return True
    
    # Local search with simulated annealing-like approach
    random.seed(42)
    best_x = None
    
    # Try multiple random restarts with hill climbing
    for restart in range(5000):
        if time.time() > deadline:
            break
        
        x = [random.randint(0, 1) for _ in range(n_base)]
        
        # Check if already valid
        if is_valid_x(x):
            best_x = x[:]
            break
        
        # Hill climbing: try single flips that reduce violations
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_viol = count_violations_for_x(x)
            
            # Try each flip
            for i in range(n_base):
                if time.time() > deadline:
                    break
                x[i] = 1 - x[i]
                new_viol = count_violations_for_x(x)
                if new_viol < current_viol:
                    current_viol = new_viol
                    improved = True
                    if new_viol == 0:
                        best_x = x[:]
                        break
                else:
                    x[i] = 1 - x[i]  # revert
                
                if best_x is not None:
                    break
        
        if best_x is not None:
            break
    
    if best_x is not None:
        # Construct final answer
        n = n_target
        colors = []
        adj = [0] * n
        for i in range(n_base):
            adj[i] = adj_base[i]
        adj[29] = 0
        for i in range(n_base):
            if best_x[i] == 1:
                adj[29] |= (1 << i)
                adj[i] |= (1 << 29)
        
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    colors.append(1)
                else:
                    colors.append(0)
        
        return {'n': n, 'colors': colors}

    # Fallback: return n=29
    colors = []
    for i in range(n_base):
        for j in range(i+1, n_base):
            if (adj_base[i] >> j) & 1:
                colors.append(1)
            else:
                colors.append(0)
    return {'n': n_base, 'colors': colors}
