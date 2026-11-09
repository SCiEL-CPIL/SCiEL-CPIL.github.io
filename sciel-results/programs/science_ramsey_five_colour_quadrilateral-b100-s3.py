"""Program for R_5(C4) maximization via deterministic construction and local search."""
import random
import itertools
import numpy as np
from collections import defaultdict

def construct():
    # Try to find a valid coloring for n=26 using local search, then fallbacks
    # We use a time-bounded approach to stay within 120s
    
    # First, try a fast deterministic construction for smaller n that we know works
    # Then attempt to scale up via local search
    
    # Known valid: n=11 via cyclic construction
    def cyclic_11():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = (j - i) % n
                d = min(d, n - d)
                colors.append(d - 1)  # 0,1,2,3,4
        return n, colors
    
    def validate(n, colors):
        """Check no monochromatic C4: no two vertices share >=2 common neighbors in any color."""
        # For each color, build adjacency sets
        color_adj = [set() for _ in range(5)]
        for idx, (i, j) in enumerate(itertools.combinations(range(n), 2)):
            c = colors[idx]
            color_adj[c].add((i, j))
        
        # For each color, check for C4: equivalent to no two vertices having 2+ common neighbors
        # Build neighbor lists per color
        color_neighbors = [defaultdict(set) for _ in range(5)]
        for c in range(5):
            for (i, j) in color_adj[c]:
                color_neighbors[c][i].add(j)
                color_neighbors[c][j].add(i)
        
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    if u in color_neighbors[c] and v in color_neighbors[c]:
                        common = color_neighbors[c][u] & color_neighbors[c][v]
                        if len(common) >= 2:
                            return False
        return True
    
    # Local search: start from random, minimize number of monochromatic C4s
    def count_mono_c4(n, colors):
        """Count number of (pair of vertices, color) where they share >=2 common neighbors in that color."""
        # More precisely: for each color, count pairs with >=2 common neighbors
        color_neighbors = [defaultdict(set) for _ in range(5)]
        for idx, (i, j) in enumerate(itertools.combinations(range(n), 2)):
            c = colors[idx]
            color_neighbors[c][i].add(j)
            color_neighbors[c][j].add(i)
        
        count = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    if u in color_neighbors[c] and v in color_neighbors[c]:
                        common = color_neighbors[c][u] & color_neighbors[c][v]
                        if len(common) >= 2:
                            count += len(common) - 1  # count excess common neighbors
        return count
    
    def local_search(n, max_iters=50000):
        """Try to find valid coloring via local search."""
        random.seed(42)
        # Start with random coloring
        colors = [random.randint(0, 4) for _ in range(n*(n-1)//2)]
        
        def get_edge_idx(i, j):
            return i*n + j - (i*(i+1))//2 - 1  # lexicographic index
        
        # Actually let's use a simpler indexing
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_map[(i,j)] = idx
                idx += 1
        
        current_count = count_mono_c4(n, colors)
        
        for it in range(max_iters):
            if current_count == 0:
                return n, colors[:], True
            
            # Pick a random edge and try changing its color
            edge_idx = random.randint(0, len(colors)-1)
            # Reverse map: which (i,j) is edge_idx?
            # Find i,j from idx
            # idx = i*n - i*(i+1)/2 - 1 + (j-i-1) ... let's just iterate
            # Better: precompute reverse
            # Actually let's just rebuild adjacency each time (expensive but n<=26)
            
            old_color = colors[edge_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            colors[edge_idx] = new_color
            new_count = count_mono_c4(n, colors)
            
            if new_count <= current_count:
                current_count = new_count
            else:
                colors[edge_idx] = old_color
        
        return n, colors[:], current_count == 0
    
    # Try increasing n from 26 down to 11
    for n in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if n == 11:
            n_val, colors_val = cyclic_11()
            if validate(n_val, colors_val):
                return {'n': n_val, 'colors': colors_val}
            else:
                # Should never happen
                return {'n': 11, 'colors': [0]*55}
        
        # Try local search for this n
        n_out, colors_out, is_valid = local_search(n, max_iters=30000)
        if is_valid:
            return {'n': n_out, 'colors': colors_out}
    
    # Final fallback: return n=11
    n_val, colors_val = cyclic_11()
    return {'n': n_val, 'colors': colors_val}
