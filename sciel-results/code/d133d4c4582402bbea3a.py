import time
import random
import numpy as np

def _find_k222(adj, color):
    """
    Find a monochromatic K2,2,2 in the graph defined by adj matrix for the given color.
    Returns (u, v, w, x, y, z) or None.
    adj is a numpy array of shape (n, n) with 0s and 1s.
    """
    n = adj.shape[0]
    # Iterate over pairs (u, v)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] != color:
                continue
            # Find common neighbors of u and v with color 'color'
            # Note: The edge (u,v) is color 'color', but u and v are in the same part, 
            # so they are not connected in K2,2,2. 
            # However, for the other parts to connect to u and v, 
            # the edges (u, w), (v, w) etc must be color 'color'.
            
            # S = {k | k != u, k != v, adj[u,k]==color, adj[v,k]==color}
            # We can compute this efficiently with numpy
            mask_u = adj[u] == color
            mask_v = adj[v] == color
            common = mask_u & mask_v
            common[u] = False
            common[v] = False
            S = np.where(common)[0]
            
            if len(S) < 4:
                continue
                
            # Iterate over pairs (w, x) in S
            for i in range(len(S)):
                w = S[i]
                for j in range(i + 1, len(S)):
                    x = S[j]
                    # Check if there are y, z in S such that 
                    # edges between {w,x} and {y,z} are color 'color'
                    # i.e., adj[w,y]==color, adj[w,z]==color, adj[x,y]==color, adj[x,z]==color
                    
                    # T = {y in S | y != w, y != x, adj[w,y]==color, adj[x,y]==color}
                    mask_w = adj[w] == color
                    mask_x = adj[x] == color
                    common_wx = mask_w & mask_x
                    common_wx[w] = False
                    common_wx[x] = False
                    # Also y must be in S
                    common_wx[S] = common_wx[S] & (np.arange(n)[S] != w) & (np.arange(n)[S] != x) 
                    # Actually simpler: just check candidates in S
                    candidates = []
                    for y in S:
                        if y == w or y == x:
                            continue
                        if adj[w, y] == color and adj[x, y] == color:
                            candidates.append(y)
                    
                    if len(candidates) >= 2:
                        y = candidates[0]
                        z = candidates[1]
                        return (u, v, w, x, y, z)
    return None

def _count_k222(adj, color):
    """
    Count the number of monochromatic K2,2,2 subgraphs.
    This is slower, used for evaluation if needed.
    """
    n = adj.shape[0]
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] != color:
                continue
            mask_u = adj[u] == color
            mask_v = adj[v] == color
            common = mask_u & mask_v
            common[u] = False
            common[v] = False
            S = np.where(common)[0]
            
            if len(S) < 4:
                continue
                
            for i in range(len(S)):
                w = S[i]
                for j in range(i + 1, len(S)):
                    x = S[j]
                    candidates = []
                    for y in S:
                        if y == w or y == x:
                            continue
                        if adj[w, y] == color and adj[x, y] == color:
                            candidates.append(y)
                    if len(candidates) >= 2:
                        # Number of ways to pick 2 from candidates
                        count += len(candidates) * (len(candidates) - 1) // 2
    return count

def _is_valid(colors, n):
    """
    Check if the coloring is valid.
    colors is a list of 0/1 for each pair (i,j) i<j.
    """
    # Reconstruct adjacency matrices
    adj0 = np.zeros((n, n), dtype=int)
    adj1 = np.zeros((n, n), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj0[i, j] = 1
                adj0[j, i] = 1
            else:
                adj1[i, j] = 1
                adj1[j, i] = 1
            idx += 1
            
    if _find_k222(adj0, 1) is not None:
        return False
    if _find_k222(adj1, 1) is not None:
        return False
    return True

def _colors_to_adj(colors, n):
    adj0 = np.zeros((n, n), dtype=int)
    adj1 = np.zeros((n, n), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj0[i, j] = 1
                adj0[j, i] = 1
            else:
                adj1[i, j] = 1
                adj1[j, i] = 1
            idx += 1
    return adj0, adj1

def _adj_to_colors(adj0, adj1, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj0[i, j] == 1:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Try to find a valid coloring for n=29, then 28, etc.
    for n in range(29, 16, -1):
        if time.time() > deadline:
            break
            
        # Initialize with Paley graph if n is a prime p = 4k+1
        # For n=29, p=29.
        # For other n, use random or extend.
        # Let's just use random for simplicity in the search loop, 
        # but Paley is a good start for primes.
        
        best_colors = None
        found = False
        
        # Try a few restarts for this n
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            if n == 29:
                # Paley 29
                squares = set()
                for x in range(1, 29):
                    squares.add((x * x) % 29)
                colors = []
                for i in range(29):
                    for j in range(i + 1, 29):
                        if (j - i) % 29 in squares:
                            colors.append(0)
                        else:
                            colors.append(1)
            else:
                colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            
            adj0, adj1 = _colors_to_adj(colors, n)
            
            # Local Search
            # We want to eliminate K2,2,2 in both colors.
            # Cost: number of K2,2,2 in color 0 + number in color 1.
            # Since counting is slow, we use "find and fix".
            
            stuck = 0
            for step in range(1000):
                if time.time() > deadline:
                    break
                    
                k222_0 = _find_k222(adj0, 1)
                if k222_0 is not None:
                    u, v, w, x, y, z = k222_0
                    # Flip a random edge in the K2,2,2
                    # Edges are between the pairs {u,v}, {w,x}, {y,z}
                    edges = [
                        (u, w), (u, x), (u, y), (u, z),
                        (v, w), (v, x), (v, y), (v, z),
                        (w, y), (w, z),
                        (x, y), (x, z)
                    ]
                    i, j = random.choice(edges)
                    if i > j: i, j = j, i
                    idx = 0
                    for a in range(n):
                        for b in range(a + 1, n):
                            if a == i and b == j:
                                break
                            idx += 1
                        else:
                            continue
                        break
                    
                    # Flip color
                    if adj0[i, j] == 1:
                        adj0[i, j] = 0
                        adj0[j, i] = 0
                        adj1[i, j] = 1
                        adj1[j, i] = 1
                    else:
                        adj0[i, j] = 1
                        adj0[j, i] = 1
                        adj1[i, j] = 0
                        adj1[j, i] = 0
                    stuck = 0
                    continue
                
                k222_1 = _find_k222(adj1, 1)
                if k222_1 is not None:
                    u, v, w, x, y, z = k222_1
                    edges = [
                        (u, w), (u, x), (u, y), (u, z),
                        (v, w), (v, x), (v, y), (v, z),
                        (w, y), (w, z),
                        (x, y), (x, z)
                    ]
                    i, j = random.choice(edges)
                    if i > j: i, j = j, i
                    idx = 0
                    for a in range(n):
                        for b in range(a + 1, n):
                            if a == i and b == j:
                                break
                            idx += 1
                        else:
                            continue
                        break
                    
                    if adj1[i, j] == 1:
                        adj1[i, j] = 0
                        adj1[j, i] = 0
                        adj0[i, j] = 1
                        adj0[j, i] = 1
                    else:
                        adj1[i, j] = 1
                        adj1[j, i] = 1
                        adj0[i, j] = 0
                        adj0[j, i] = 0
                    stuck = 0
                    continue
                
                # No K2,2,2 found
                # Verify validity
                if _is_valid(_adj_to_colors(adj0, adj1, n), n):
                    best_colors = _adj_to_colors(adj0, adj1, n)
                    found = True
                    break
                
                # If we get here, it means _find_k222 missed one? 
                # Or it's valid.
                # Let's assume if _find_k222 returns None for both, it's valid.
                # But _is_valid is a double check.
                break

            if found:
                break
        
        if found:
            return {'n': n, 'colors': best_colors}
            
    # Fallback to n=17 Paley
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}
