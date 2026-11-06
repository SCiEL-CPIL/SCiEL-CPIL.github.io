import random
import time

def construct():
    # Try to find a valid coloring for n=26, 27, 28
    # Local Search approach
    for n in [26, 27, 28]:
        if time.time() > 110:
            break
        # Multiple restarts
        for restart in range(200):
            if time.time() > 115:
                break
            result = local_search(n)
            if result is not None:
                return {'n': n, 'colors': result}
    
    # Fallback to n=26 if we somehow didn't find it (should be rare for LS)
    # Or a smaller n if time is tight
    n = 26
    # If LS failed, we might return a random one which is invalid, but we must return valid.
    # Actually, LS is very robust for n=26.
    # If it fails, try n=25?
    for n in [25, 24]:
        result = local_search(n)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Absolute fallback: n=11 valid construction (from seed)
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def local_search(n):
    # colors[i][j] is the color of edge (i,j), i<j
    # Use a 2D list for easy access, or 1D array
    # 1D array mapping (i,j) to index
    # index = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
    # Simpler: use a dictionary or 2D list of size n x n
    # 2D list is fine for n<=29
    
    colors = [[-1] * n for _ in range(n)]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Random initialization
    for i, j in edges:
        colors[i][j] = random.randint(0, 4)
        colors[j][i] = colors[i][j]
        
    # Helper to get neighbors in color c
    # We can compute on the fly or maintain sets.
    # For n=26, computing on the fly is fast enough if we are not doing millions of steps.
    # But LS might do many steps.
    # Let's maintain adjacency lists for each color.
    
    neighbors = [ [set() for _ in range(n)] for _ in range(5) ]
    
    def add_edge(i, j, c):
        if colors[i][j] != -1:
            old_c = colors[i][j]
            neighbors[old_c][i].remove(j)
            neighbors[old_c][j].remove(i)
        colors[i][j] = c
        colors[j][i] = c
        neighbors[c][i].add(j)
        neighbors[c][j].add(i)
        
    # Initialize neighbors
    for i, j in edges:
        c = colors[i][j]
        neighbors[c][i].add(j)
        neighbors[c][j].add(i)
        
    # Function to find all violations
    # Violation: pair (u, v) with >= 2 common neighbors in color c
    def find_violations():
        violations = []
        for c in range(5):
            for u in range(n):
                # Only check v > u to avoid duplicates?
                # The condition is symmetric.
                for v in range(u + 1, n):
                    # Common neighbors of u and v in color c
                    # Intersection of neighbors[c][u] and neighbors[c][v]
                    # Optimization: if len < 2, skip
                    if len(neighbors[c][u]) < 2 or len(neighbors[c][v]) < 2:
                        continue
                    inter = neighbors[c][u].intersection(neighbors[c][v])
                    if len(inter) >= 2:
                        # Store the violation
                        # We need to know which edges to change.
                        # Any edge in the C4 formed by u, v and any two from inter.
                        # Let w1, w2 be two from inter.
                        # The C4 is u-w1-v-w2-u.
                        # Edges: (u,w1), (w1,v), (v,w2), (w2,u).
                        # All in color c.
                        # We can pick one of these edges to change.
                        # To be specific, store (u, v, c, w1, w2)
                        violations.append((u, v, c, list(inter)[:2]))
        return violations
        
    # Main loop
    max_steps = 5000
    for step in range(max_steps):
        if time.time() > 115:
            return None
            
        violations = find_violations()
        if not violations:
            # Valid coloring found
            out_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    out_colors.append(colors[i][j])
            return out_colors
            
        # Pick a random violation
        u, v, c, ws = random.choice(violations)
        w1, w2 = ws
        
        # The C4 is u - w1 - v - w2 - u
        # Edges involved: (u, w1), (w1, v), (v, w2), (w2, u)
        # Pick one edge to recolor
        # Let's pick the edge (u, w1) as an example, or random from the 4
        edge_candidates = [
            (min(u, w1), max(u, w1)),
            (min(w1, v), max(w1, v)),
            (min(v, w2), max(v, w2)),
            (min(w2, u), max(w2, u))
        ]
        
        ei, ej = random.choice(edge_candidates)
        
        # Change color of (ei, ej) to a random other color
        old_c = colors[ei][ej]
        new_c = random.randint(0, 4)
        while new_c == old_c:
            new_c = random.randint(0, 4)
            
        add_edge(ei, ej, new_c)
        
    return None
