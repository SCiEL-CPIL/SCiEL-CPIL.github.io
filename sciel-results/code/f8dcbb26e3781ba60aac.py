import time
import random
import numpy as np

def _get_edges(n):
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def _color_index(n, i, j):
    # Index in the flat list for edge (i, j) with i < j
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

def _count_violations_fast(colors, n, v):
    """
    Count monochromatic K2,2,2 subgraphs involving vertex v using vectorized operations.
    """
    violations = 0
    
    # Build adjacency matrix for color 0 and color 1
    # adj[c][i][j] = 1 if edge (i,j) has color c
    adj = [np.zeros((n, n), dtype=np.int8) for _ in range(2)]
    
    for i in range(n):
        for j in range(i+1, n):
            idx = _color_index(n, i, j)
            c = colors[idx]
            adj[c][i][j] = 1
            adj[c][j][i] = 1
    
    for c in range(2):
        # Get neighbors of v in color c
        neigh = np.where(adj[c][v] == 1)[0]
        L = len(neigh)
        if L < 4:
            continue
        
        # For each pair of disjoint pairs from neigh, check if all cross edges are color c
        # We need to find K2,2,2 which is a complete tripartite graph K2,2,2
        # This means we need 6 vertices partitioned into 3 groups of 2, with all edges between groups present
        
        # Alternative: iterate over all 6-subsets and check if they form K2,2,2
        # But that's expensive. Let's use a different approach.
        
        # Actually, K2,2,2 as a subgraph means we need 6 vertices that can be partitioned 
        # into 3 pairs such that all 12 edges between different pairs are present.
        
        # Let's iterate over all 6-combinations of neighbors
        from itertools import combinations
        
        for combo in combinations(neigh, 6):
            a, b, c1, d, e, f = combo
            # Try all 15 ways to partition into 3 pairs
            # Pairings: (a,b)(c1,d)(e,f), (a,b)(c1,e)(d,f), (a,b)(c1,f)(d,e),
            #           (a,c1)(b,d)(e,f), (a,c1)(b,e)(d,f), (a,c1)(b,f)(d,e),
            #           (a,d)(b,c1)(e,f), (a,d)(b,e)(c1,f), (a,d)(b,f)(c1,e),
            #           (a,e)(b,c1)(d,f), (a,e)(b,d)(c1,f), (a,e)(b,f)(c,d),
            #           (a,f)(b,c1)(d,e), (a,f)(b,d)(c1,e), (a,f)(b,e)(c1,d)
            
            pairings = [
                ((a,b),(c1,d),(e,f)),
                ((a,b),(c1,e),(d,f)),
                ((a,b),(c1,f),(d,e)),
                ((a,c1),(b,d),(e,f)),
                ((a,c1),(b,e),(d,f)),
                ((a,c1),(b,f),(d,e)),
                ((a,d),(b,c1),(e,f)),
                ((a,d),(b,e),(c1,f)),
                ((a,d),(b,f),(c1,e)),
                ((a,e),(b,c1),(d,f)),
                ((a,e),(b,d),(c1,f)),
                ((a,e),(b,f),(c,d)),
                ((a,f),(b,c1),(d,e)),
                ((a,f),(b,d),(c1,e)),
                ((a,f),(b,e),(c1,d))
            ]
            
            for p1, p2, p3 in pairings:
                # Check all 12 edges between different pairs
                valid = True
                for u in p1:
                    for w in p2:
                        if adj[c][u][w] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                    for w in p3:
                        if adj[c][u][w] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    for u in p2:
                        for w in p3:
                            if adj[c][u][w] == 0:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    violations += 1
    
    return violations

def _is_valid_full(colors, n):
    """
    Check if the coloring is valid (no mono K2,2,2) by checking all vertices.
    """
    for v in range(n):
        if _count_violations_fast(colors, n, v) > 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Start with Paley 29
    p = 29
    n = p
    colors = _build_paley(p)
    
    # Verify the starting coloring is valid
    if not _is_valid_full(colors, n):
        # Should not happen, but just in case
        return {'n': n, 'colors': colors}
    
    # Try to add vertices
    for new_n in range(n + 1, 32):
        if time.time() > deadline:
            break
            
        best_colors = None
        best_violations = float('inf')
        
        # Multiple restarts
        for restart in range(30):
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
            cur_violations = _count_violations_fast(cur_colors, new_n, v_new)
            
            if cur_violations == 0:
                best_colors = cur_colors
                best_violations = 0
                break
                
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try flipping each edge connected to v_new
                for i in range(v_new):
                    if time.time() > deadline:
                        break
                    idx = _color_index(new_n, i, v_new)
                    
                    old_val = cur_colors[idx]
                    cur_colors[idx] = 1 - old_val
                    
                    new_violations = _count_violations_fast(cur_colors, new_n, v_new)
                    
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
            # Verify the new coloring is fully valid
            if _is_valid_full(best_colors, new_n):
                colors = best_colors
                n = new_n
            else:
                # Should not happen if local search was correct, but safety check
                break
        else:
            break
            
    # Final validation
    if not _is_valid_full(colors, n):
        # Fallback: return the valid Paley 29
        colors = _build_paley(29)
        n = 29
        
    return {'n': n, 'colors': colors}
