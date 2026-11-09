import time
import random
import math

def count_c4s_fast(n, colors):
    """
    Count C4s using bitmasks for adjacency.
    colors is a list of length n*(n-1)//2.
    """
    # adj[c][u] is a bitmask of neighbors of u in color c
    adj = [[0] * n for _ in range(5)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    total_c4s = 0
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[c][u] & adj[c][v]
                k = common.bit_count()
                if k >= 2:
                    total_c4s += k * (k - 1) // 2
    return total_c4s

def count_c4s(n, colors):
    """
    Count the number of C4s in the 5-coloring.
    colors is a list of length n*(n-1)//2.
    """
    # Build adjacency lists for each color
    # adj[c][u] = set of v such that edge (u,v) has color c
    adj = [ [set() for _ in range(n)] for _ in range(5) ]
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
            
    total_c4s = 0
    for c in range(5):
        for u in range(n):
            for v in range(u+1, n):
                # Count common neighbors in color c
                # Only if u and v are NOT connected by color c? 
                # A C4 is u-x-v-y-u. The edges are (u,x), (x,v), (v,y), (y,u).
                # So u and v are opposite. They do not need to be connected.
                # However, if u and v ARE connected by color c, that edge is not part of this specific C4 definition unless we consider the cycle u-x-v-y-u where (u,v) is a diagonal.
                # The definition "no two vertices with two common neighbours in one colour" implies that if u and v have 2 common neighbors x,y in color c, then u-x-v-y-u is a C4 (if x!=y and u!=v).
                # Note: The edge (u,v) itself is irrelevant to the existence of the C4 formed by the 4 vertices {u,v,x,y}.
                # So we just count pairs (u,v) with |N_c(u) \cap N_c(v)| >= 2.
                common = adj[c][u] & adj[c][v]
                k = len(common)
                if k >= 2:
                    total_c4s += k * (k - 1) // 2
    return total_c4s

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_n = 0
    best_colors = None
    
    # Try to find a solution for n=26 down to 20
    for n in range(26, 19, -1):
        if time.time() > deadline:
            break
            
        num_edges = n * (n - 1) // 2
        
        # Multiple restarts for this n
        found = False
        restarts = 20
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            # Initialize with random colors
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            current_c4s = count_c4s_fast(n, colors)
            if current_c4s == 0:
                best_n = n
                best_colors = colors
                found = True
                break
                
            best_c4s = current_c4s
            best_colors_sa = colors[:]
            
            T = 1.0
            # Adaptive max iterations based on remaining time
            max_iters = 50000
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                    
                idx = random.randint(0, num_edges - 1)
                old_c = colors[idx]
                new_c = random.randint(0, 4)
                if old_c == new_c:
                    continue
                    
                colors[idx] = new_c
                new_c4s = count_c4s_fast(n, colors)
                
                if new_c4s < current_c4s:
                    current_c4s = new_c4s
                    if current_c4s < best_c4s:
                        best_c4s = current_c4s
                        best_colors_sa = colors[:]
                    if current_c4s == 0:
                        best_n = n
                        best_colors = colors
                        found = True
                        break
                else:
                    # Accept with probability exp(-delta/T)
                    delta = new_c4s - current_c4s
                    if delta > 0:
                        prob = math.exp(-delta / T) if T > 0 else 0
                        if random.random() < prob:
                            current_c4s = new_c4s
                        else:
                            colors[idx] = old_c
                    else:
                        current_c4s = new_c4s
                        if current_c4s < best_c4s:
                            best_c4s = current_c4s
                            best_colors_sa = colors[:]
                            
                if it % 500 == 0:
                    T *= 0.98
                    
            if found:
                break
                
        if found:
            break
            
    if best_colors is None:
        # Fallback: Return a known small valid coloring
        # n=11 from seed is valid.
        n = 11
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        best_n = n

    return {'n': best_n, 'colors': best_colors}
