import time
import random
import numpy as np

def _get_edges(n):
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def _color_index(n, i, j):
    # Index in the flat list for edge (i, j) with i < j
    # Number of edges for k < i is k*(k-1)/2
    # Edges for i are (i, i+1) ... (i, j)
    # Index = sum_{k=0}^{i-1} k + (j - i - 1)
    # sum_{k=0}^{i-1} k = i*(i-1)/2
    return i*(i-1)//2 + (j - i - 1)

def _build_paley(p):
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _count_violations(colors, n, v):
    """
    Count monochromatic K2,2,2 subgraphs involving vertex v.
    """
    violations = 0
    
    # Build adjacency lists for color 0 and color 1 for vertex v
    # And for all other vertices, we need to check cross edges.
    
    # For efficiency, let's precompute the color matrix or just use the list.
    # n <= 31, so n^2 is small.
    
    # Get neighbors of v for color 0 and 1
    neigh0 = []
    neigh1 = []
    for u in range(n):
        if u == v:
            continue
        if u < v:
            idx = _color_index(n, u, v)
        else:
            idx = _color_index(n, v, u)
        c = colors[idx]
        if c == 0:
            neigh0.append(u)
        else:
            neigh1.append(u)
            
    def check_color(neigh):
        count = 0
        L = len(neigh)
        if L < 4:
            return 0
        # Iterate over all subsets of 4 neighbors
        # There are C(L, 4) subsets.
        # For each subset, check if it can be partitioned into two pairs
        # such that all cross edges are of the correct color.
        
        # To speed up, we can iterate over pairs of pairs.
        # Or just iterate over all 4-combinations.
        # L is at most n-1 = 30. C(30,4) = 27405.
        # For each, we check 3 partitions.
        # 27405 * 3 * 4 edge checks = ~300k ops.
        # This is fast enough for local search steps if we don't do too many.
        
        for i in range(L):
            u1 = neigh[i]
            for j in range(i+1, L):
                u2 = neigh[j]
                for k in range(j+1, L):
                    u3 = neigh[k]
                    for l in range(k+1, L):
                        u4 = neigh[l]
                        
                        # Partition 1: {u1,u2} and {u3,u4}
                        # Check edges (u1,u3), (u1,u4), (u2,u3), (u2,u4)
                        # All must be color c (which is determined by the list 'neigh', so c is 0 if neigh is neigh0, 1 if neigh1)
                        c_target = 0 if neigh == neigh0 else 1
                        
                        def edge_color(a, b):
                            if a < b:
                                idx = _color_index(n, a, b)
                            else:
                                idx = _color_index(n, b, a)
                            return colors[idx]
                            
                        # Check partition 1
                        if (edge_color(u1, u3) == c_target and
                            edge_color(u1, u4) == c_target and
                            edge_color(u2, u3) == c_target and
                            edge_color(u2, u4) == c_target):
                            count += 1
                            
                        # Check partition 2: {u1,u3} and {u2,u4}
                        if (edge_color(u1, u2) == c_target and
                            edge_color(u1, u4) == c_target and
                            edge_color(u3, u2) == c_target and
                            edge_color(u3, u4) == c_target):
                            count += 1
                            
                        # Check partition 3: {u1,u4} and {u2,u3}
                        if (edge_color(u1, u2) == c_target and
                            edge_color(u1, u3) == c_target and
                            edge_color(u4, u2) == c_target and
                            edge_color(u4, u3) == c_target):
                            count += 1
                            
        return count
        
    violations += check_color(neigh0)
    violations += check_color(neigh1)
    
    return violations

def _is_valid(colors, n):
    """
    Check if the coloring is valid (no mono K2,2,2).
    """
    # We only need to check vertices that might have been added or changed?
    # For safety, check all.
    # But _count_violations for a vertex v counts K222s containing v.
    # Each K222 has 6 vertices, so it will be counted 6 times (once for each vertex).
    # So if sum of counts is 0, it's valid.
    
    total = 0
    for v in range(n):
        total += _count_violations(colors, n, v)
        
    return total == 0

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Start with Paley 29
    p = 29
    n = p
    colors = _build_paley(p)
    
    # Try to add vertices
    for new_n in range(n + 1, 32):
        if time.time() > deadline:
            break
            
        # Initialize new vertex edges randomly
        # Or use a heuristic: copy from a similar vertex?
        # Let's try random init and local search.
        
        best_colors = None
        best_violations = float('inf')
        
        # Multiple restarts
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Copy old colors
            new_colors = colors[:]
            
            # Add edges for new vertex (new_n - 1)
            v_new = new_n - 1
            for i in range(v_new):
                new_colors.append(random.randint(0, 1))
                
            # Local search
            cur_colors = new_colors[:]
            cur_violations = _count_violations(cur_colors, new_n, v_new)
            
            # Note: _count_violations only checks K222s containing v_new.
            # The old part is valid. So total violations = violations containing v_new.
            
            if cur_violations == 0:
                best_colors = cur_colors
                best_violations = 0
                break
                
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try flipping each edge connected to v_new
                # There are v_new edges.
                # We can also try flipping edges in the old graph?
                # Flipping old edges might break the old validity.
                # Let's stick to flipping new edges first.
                
                # Randomly pick an edge to flip to avoid getting stuck?
                # Or iterate through all.
                for i in range(v_new):
                    if time.time() > deadline:
                        break
                    # Flip edge (i, v_new)
                    # Index:
                    if i < v_new:
                        idx = _color_index(new_n, i, v_new)
                    else:
                        idx = _color_index(new_n, v_new, i)
                        
                    old_val = cur_colors[idx]
                    cur_colors[idx] = 1 - old_val
                    
                    new_violations = _count_violations(cur_colors, new_n, v_new)
                    
                    if new_violations < cur_violations:
                        cur_violations = new_violations
                        improved = True
                        if cur_violations == 0:
                            break
                    else:
                        cur_colors[idx] = old_val
                        
                if cur_violations == 0:
                    break
                    
            if cur_violations < best_violations:
                best_violations = cur_violations
                best_colors = cur_colors[:]
                
            if best_violations == 0:
                break
                
        if best_violations == 0:
            colors = best_colors
            n = new_n
        else:
            # If we couldn't fix it, stop trying to add more?
            # Or continue to next size?
            # Usually if n fails, n+1 is harder.
            # But maybe a different random init would work?
            # Given time constraints, we might just stop.
            # However, we can try to improve the current n if we haven't reached 30/31.
            # But the task is to maximize n.
            # If we can't add 30, we return 29.
            break
            
    return {'n': n, 'colors': colors}
