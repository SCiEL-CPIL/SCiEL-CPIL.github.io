import itertools
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute all K4s and their edge indices
    quads = list(itertools.combinations(range(n), 4))
    quad_edges = []
    for a, b, c, d in quads:
        # Edge indices in the flattened list (lexicographic order of pairs)
        # Pair (i,j) index: if i<j, index = i*n - i*(i+1)//2 + (j-i-1) is wrong.
        # Let's use a standard mapping.
        # Total edges: 120.
        # idx(i, j) for i < j:
        # Sum of edges for vertices 0 to i-1: i*(n-1) - i*(i-1)//2 ? No.
        # Edges involving 0: 15. 1: 14. ... i: 15-i.
        # Offset for vertex i is sum_{k=0}^{i-1} (15-k) = 15*i - i*(i-1)/2.
        # Then add (j - i - 1).
        def edge_idx(i, j):
            if i > j: i, j = j, i
            return i * (2 * n - 1 - i) // 2 + (j - i - 1)
        
        quad_edges.append([
            edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
            edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
        ])

    def count_colors(coloring):
        # Returns the number of distinct colors used
        # Assuming colors are integers 0..K-1, max+1 is a good proxy if contiguous,
        # but distinct count is safer.
        return len(set(coloring))

    def is_valid_fast(coloring):
        """Check if every K4 has at least 5 distinct colors."""
        for edges in quad_edges:
            colors = [coloring[e] for e in edges]
            if len(set(colors)) < 5:
                return False
        return True

    def make_coloring_formula(formula_fn, n=16):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(formula_fn(i, j))
        return colors

    best_coloring = None
    best_num_colors = float('inf')

    # 1. Try structured formulas
    # We want to find a coloring with <= 15 colors.
    # Try moduli 13, 14, 15, 16.
    
    formula_candidates = []
    
    # Linear forms
    for mod in range(13, 17):
        for a in range(1, 6):
            for b in range(1, 6):
                if a == b: continue
                formula_candidates.append(lambda i, j, a=a, b=b, mod=mod: (a*i + b*j) % mod)
    
    # Difference based
    for mod in range(13, 17):
        for a in range(1, 6):
            formula_candidates.append(lambda i, j, a=a, mod=mod: (a * (i - j)) % mod)
            
    # Quadratic
    for mod in range(13, 17):
        for a in range(1, 4):
            for b in range(1, 4):
                formula_candidates.append(lambda i, j, a=a, b=b, mod=mod: (a*i*i + b*j*j) % mod)

    for fn in formula_candidates:
        if time.time() > deadline: break
        cand = make_coloring_formula(fn)
        nc = count_colors(cand)
        if nc >= best_num_colors: continue
        if is_valid_fast(cand):
            best_num_colors = nc
            best_coloring = cand
            if nc <= 13: # Found optimal or near-optimal
                return cand

    # 2. Local Search / Hill Climbing
    # Start from a known valid coloring (e.g., (i+j)%17)
    # and try to reduce the number of colors.
    
    def local_search(start_coloring, max_iter=50000):
        nonlocal best_coloring, best_num_colors
        coloring = list(start_coloring)
        current_nc = count_colors(coloring)
        if current_nc >= best_num_colors:
            return
        
        # We want to minimize the number of distinct colors.
        # Strategy: Try to change an edge's color to another existing color
        # to potentially reduce the total distinct count, or just to explore space.
        # Better: Try to map high-index colors to low-index colors.
        
        # Sort edges by color frequency? Or just random.
        # Let's try to reduce the max color index first, then distinct count.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to change each edge to a "better" color
            # A "better" color is one that is already used by other edges,
            # allowing us to potentially drop the old color if it becomes unused.
            
            # Get current distinct colors
            distinct_colors = sorted(set(coloring))
            
            for e_idx in range(num_edges):
                if time.time() > deadline: break
                old_c = coloring[e_idx]
                # Try changing to other colors
                # Prioritize colors that are "common" or lower index
                for new_c in distinct_colors:
                    if new_c == old_c: continue
                    coloring[e_idx] = new_c
                    if is_valid_fast(coloring):
                        # Check if this reduced the number of colors
                        new_nc = count_colors(coloring)
                        if new_nc < current_nc:
                            current_nc = new_nc
                            improved = True
                            # Update best
                            if current_nc < best_num_colors:
                                best_num_colors = current_nc
                                best_coloring = coloring[:]
                                if best_num_colors <= 13:
                                    return
                        else:
                            # Even if not reduced, we might want to keep it if it helps future moves?
                            # For simplicity, we only accept if it improves or maintains distinct count
                            # but allows us to drop a color.
                            # Actually, just accepting valid moves with same distinct count
                            # can lead to dead ends. Let's only accept if it reduces distinct count.
                            pass
                    coloring[e_idx] = old_c
        
        # If we didn't reduce distinct count, we are stuck in a local minimum.
        # We can try random perturbations.
        
    # Start local search from several initial points
    initial_colorings = []
    
    # 1. Seed (i+j)%17
    initial_colorings.append(make_coloring_formula(lambda i, j: (i + j) % 17))
    
    # 2. Random valid colorings? Hard to generate directly.
    # Instead, use permutations of the seed.
    for _ in range(5):
        perm = list(range(17))
        random.shuffle(perm)
        base = make_coloring_formula(lambda i, j: (i + j) % 17)
        permuted = [perm[c] for c in base]
        # Renumber to 0..16
        unique = sorted(set(permuted))
        mapping = {v: i for i, v in enumerate(unique)}
        initial_colorings.append([mapping[c] for c in permuted])
        
    # 3. Try some other simple valid ones if found in step 1
    if best_coloring:
        initial_colorings.append(best_coloring)

    for start in initial_colorings:
        if time.time() > deadline: break
        local_search(start)
        
    if best_coloring is None:
        # Fallback to seed
        return make_coloring_formula(lambda i, j: (i + j) % 17)
        
    return best_coloring

# EVOLVE-BLOCK-END
