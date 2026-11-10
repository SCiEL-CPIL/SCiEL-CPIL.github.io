import time
import random

def count_c4s(colors, n):
    # colors is list of length n*(n-1)//2
    # Reconstruct adjacency matrices for each color
    # Or just count common neighbors
    # For each pair (u, v), count common neighbors w such that (u,w) and (v,w) have same color.
    # A C4 is u-v-w-x-u.
    # Equivalent: for each color c, the graph G_c must be C4-free.
    # G_c is C4-free iff for any u, v, |N_c(u) \cap N_c(v)| <= 1.
    
    # Build adjacency sets for each color
    adj = [set() for _ in range(n)] # adj[i] is set of neighbors? No, need color.
    # Let's use a list of lists of (neighbor, color)
    # Or 5 sets of neighbors for each vertex.
    neigh = [ [set() for _ in range(5)] for _ in range(n) ]
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            neigh[i][c].add(j)
            neigh[j][c].add(i)
            idx += 1
            
    c4_count = 0
    for c in range(5):
        for u in range(n):
            for v in range(u+1, n):
                # Count common neighbors in color c
                # If > 1, we have a C4 (actually multiple C4s)
                # We just need to know if it's > 1 to invalidate, 
                # but for optimization, counting is good.
                # Intersection size
                # To be fast, we can iterate over smaller set
                if len(neigh[u][c]) < len(neigh[v][c]):
                    common = 0
                    for w in neigh[u][c]:
                        if w in neigh[v][c]:
                            common += 1
                            if common > 1:
                                c4_count += (common - 1) # Approximate penalty
                                break
                else:
                    common = 0
                    for w in neigh[v][c]:
                        if w in neigh[u][c]:
                            common += 1
                            if common > 1:
                                c4_count += (common - 1)
                                break
    return c4_count

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Fallback: Seed solution n=11
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_n = n_seed
    best_colors = colors_seed
    
    # Try to find valid coloring for n=12
    for target_n in [12, 13, 14]:
        if time.time() > deadline - 5:
            break
            
        # Local search
        # Init random
        num_edges = target_n * (target_n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Optimize
        for step in range(5000):
            if time.time() > deadline - 2:
                break
            cur_c4s = count_c4s(colors, target_n)
            if cur_c4s == 0:
                # Found valid
                best_n = target_n
                best_colors = colors
                break
                
            # Pick a random edge involved in a C4? 
            # Or just random edge and try changing color.
            # To be efficient, pick random edge.
            idx = random.randint(0, num_edges - 1)
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            colors[idx] = new_c
            new_c4s = count_c4s(colors, target_n)
            if new_c4s < cur_c4s:
                pass # Keep
            else:
                colors[idx] = old_c # Revert
                
    return {'n': best_n, 'colors': best_colors}
