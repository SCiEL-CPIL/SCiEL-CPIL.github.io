"""Construct a dense edge-coloured graph on 20 vertices with rainbow 4-cycles."""
import time
import itertools
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    n = 20
    num_colors = 20

    # Precompute all 4-cycles as quadruples (a, b, c, d) where a < b < c < d
    # and the edges are (a,b), (b,c), (c,d), (d,a)
    # Actually, a 4-cycle is a-b-c-d-a where all 4 vertices are distinct and
    # the edges (a,b), (b,c), (c,d), (d,a) all exist in the graph.
    # We enumerate all ordered 4-tuples of distinct vertices up to rotation/reflection.
    # For efficiency, we precompute for each edge (u,v) the list of 4-cycles it belongs to.
    # A 4-cycle containing edge (u,v) has the form u-v-x-y-u where x,y are distinct
    # from u,v and from each other, and edges (v,x), (x,y), (y,u) exist.
    
    # Precompute: for each pair (u,v) with u<v, list of (x,y) such that
    # u-v-x-y-u is a potential 4-cycle (x != u,v, y != u,v,x, and y != x)
    # We need edges (v,x), (x,y), (y,u) to also be present.
    
    # For fast checking, we'll use a 2D array color[u][v] = color or -1 if no edge
    # and a 2D boolean array adj[u][v] = True if edge exists
    
    def try_greedy_construction(seed):
        """Greedy construction with vectorized 4-cycle checks."""
        rng = random.Random(seed)
        
        # color[u][v] for u < v, -1 means no edge
        color = np.full((n, n), -1, dtype=np.int8)
        
        # All possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(all_edges)
        
        # For each edge (u,v), precompute potential 4-cycle partners
        # A 4-cycle u-v-x-y-u requires:
        #   - x != u, x != v
        #   - y != u, y != v, y != x
        #   - edges (v,x), (x,y), (y,u) exist
        # We'll check these dynamically as edges are added.
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            # Try colors in random order
            colors = list(range(num_colors))
            rng.shuffle(colors)
            for c in colors:
                # Tentatively set color
                color[u, v] = c
                color[v, u] = c
                
                # Check all 4-cycles containing edge (u,v)
                # Form: u-v-x-y-u
                valid = True
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if color[v, x] == -1:
                        continue
                    for y in range(n):
                        if y == u or y == v or y == x:
                            continue
                        if color[x, y] == -1:
                            continue
                        if color[y, u] == -1:
                            continue
                        # Check 4-cycle u-v-x-y-u
                        c1 = color[u, v]
                        c2 = color[v, x]
                        c3 = color[x, y]
                        c4 = color[y, u]
                        if len(set([c1, c2, c3, c4])) != 4:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                else:
                    color[u, v] = -1
                    color[v, u] = -1
            # If no color worked, edge (u,v) is skipped
            
        # Count edges
        num_edges = 0
        for i in range(n):
            for j in range(i+1, n):
                if color[i, j] != -1:
                    num_edges += 1
        
        return num_edges, color

    # Try multiple seeds
    best_count = 0
    best_color = None
    random.seed(42)
    
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        count, col = try_greedy_construction(seed)
        if count > best_count:
            best_count = count
            best_color = col.copy()
            if best_count > 150:
                break

    # Convert to list format
    result = []
    if best_color is not None:
        for i in range(n):
            for j in range(i+1, n):
                if best_color[i, j] != -1:
                    result.append([i, j, int(best_color[i, j])])
    
    return result
# EVOLVE-BLOCK-END
