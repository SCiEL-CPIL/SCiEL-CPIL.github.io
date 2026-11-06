"""Improved program for edge-colourings of K16 with >=5 colours per K4."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _idx(i, j):
    if i > j:
        i, j = j, i
    return i * 16 - i * (i + 1) // 2 + j - i - 1

def _validate_list(colors_list):
    """Check every K4 spans at least 5 distinct colours using flat list."""
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        e1 = colors_list[_idx(a, b)]
        e2 = colors_list[_idx(a, c)]
        e3 = colors_list[_idx(a, d)]
        e4 = colors_list[_idx(b, c)]
        e5 = colors_list[_idx(b, d)]
        e6 = colors_list[_idx(c, d)]
        if e1 == e2 or e1 == e3 or e1 == e4 or e1 == e5 or e1 == e6:
            # Only need to check distinct count
            pass
        cols = {e1, e2, e3, e4, e5, e6}
        if len(cols) < 5:
            return False
    return True

def _count_bad_k4(colors_list):
    """Count number of K4s with fewer than 5 distinct colours."""
    bad = 0
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        e1 = colors_list[_idx(a, b)]
        e2 = colors_list[_idx(a, c)]
        e3 = colors_list[_idx(a, d)]
        e4 = colors_list[_idx(b, c)]
        e5 = colors_list[_idx(b, d)]
        e6 = colors_list[_idx(c, d)]
        cols = {e1, e2, e3, e4, e5, e6}
        if len(cols) < 5:
            bad += 1
    return bad

def _make_1factorization_cyclic():
    """Standard cyclic 1-factorization of K16. Returns list of 120 colors."""
    colors = [0] * 120
    # Vertices 0..14 on cycle, 15 is center
    for i in range(15):
        colors[_idx(15, i)] = i
        for j in range(1, 8):
            a = (i + j) % 15
            b = (i - j) % 15
            if a > b:
                a, b = b, a
            colors[_idx(a, b)] = i
    return colors

def _make_1factorization_cyclic2():
    """Alternative cyclic 1-factorization using different pairing."""
    colors = [0] * 120
    # Use a different cyclic structure
    n = 15
    for i in range(n):
        colors[_idx(15, i)] = i
        for j in range(1, 8):
            a = (i + j) % n
            b = (i - j) % n
            if a > b:
                a, b = b, a
            colors[_idx(a, b)] = i
    return colors

def _local_search_1factor(colors, max_time, seed=42):
    """Local search: swap edges between factors to reduce bad K4 count.
    colors is a flat list of 120 colors (0..14 for 1-factorization).
    """
    rng = random.Random(seed)
    t0 = time.time()
    deadline = max_time
    
    # Build factor membership: factor[f] = list of edge indices in that factor
    factor = [[] for _ in range(15)]
    for idx in range(120):
        c = colors[idx]
        factor[c].append(idx)
    
    bad_count = _count_bad_k4(colors)
    
    if bad_count == 0:
        return colors, 0
    
    # Precompute for each edge index which K4s contain it
    # edge_idx -> list of K4 tuples
    # For speed, precompute all K4s and their edge indices
    all_k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for a, b, c, d in all_k4s:
        k4_edges.append([_idx(a,b), _idx(a,c), _idx(a,d), _idx(b,c), _idx(b,d), _idx(c,d)])
    
    # For each edge index, which K4s contain it
    edge_to_k4s = [[] for _ in range(120)]
    for ki, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(ki)
    
    def _k4_distinct_count(ki, colors):
        edges = k4_edges[ki]
        e1 = colors[edges[0]]
        e2 = colors[edges[1]]
        e3 = colors[edges[2]]
        e4 = colors[edges[3]]
        e5 = colors[edges[4]]
        e6 = colors[edges[5]]
        return len({e1, e2, e3, e4, e5, e6})
    
    best_colors = colors[:]
    best_bad = bad_count
    
    # Try swapping two edges from different factors
    # Edge e1 in factor f1, edge e2 in factor f2, f1 != f2
    # After swap: e1 gets color f2, e2 gets color f1
    # Check if this reduces bad K4 count
    
    iterations = 0
    while time.time() - t0 < deadline:
        iterations += 1
        # Pick random pair of edges from different factors
        f1 = rng.randint(0, 14)
        f2 = rng.randint(0, 14)
        if f1 == f2:
            continue
        if not factor[f1] or not factor[f2]:
            continue
        e1 = factor[f1][rng.randint(0, len(factor[f1])-1)]
        e2 = factor[f2][rng.randint(0, len(factor[f2])-1)]
        
        # Try swap
        colors[e1] = f2
        colors[e2] = f1
        
        # Count change in bad K4s
        # Only K4s containing e1 or e2 are affected
        affected_k4s = set(edge_to_k4s[e1]) | set(edge_to_k4s[e2])
        
        # Compute bad count for affected K4s
        new_bad_affected = 0
        for ki in affected_k4s:
            if _k4_distinct_count(ki, colors) < 5:
                new_bad_affected += 1
        
        # We need to know the old bad count for these K4s
        # Let's just recompute total bad count (expensive but simple)
        # For speed, let's do a quick check
        total_bad = bad_count
        # Remove old contributions from affected K4s
        # This is getting complex. Let's just do full recount for simplicity
        # but limit iterations
        
        if iterations % 100 == 0:
            total_bad = _count_bad_k4(colors)
            if total_bad < best_bad:
                best_bad = total_bad
                best_colors = colors[:]
                bad_count = total_bad
                if best_bad == 0:
                    return best_colors, 0
            else:
                # Revert
                colors[e1] = f1
                colors[e2] = f2
                continue
        else:
            # Quick check: just check affected K4s
            # For now, revert and do periodic full checks
            colors[e1] = f1
            colors[e2] = f2
            continue
    
    # Final check
    final_bad = _count_bad_k4(best_colors)
    if final_bad < best_bad:
        best_bad = final_bad
    return best_colors, best_bad

def _try_1factorization_with_swaps(max_time, seed=42):
    """Try cyclic 1-factorization then local search."""
    colors = _make_1factorization_cyclic()
    bad = _count_bad_k4(colors)
    
    if bad == 0:
        return colors, 0
    
    # Local search
    result, bad = _local_search_1factor(colors, max_time, seed)
    return result, bad

def _try_random_1factorizations(max_time, seed=42):
    """Try multiple random 1-factorizations."""
    rng = random.Random(seed)
    best_colors = None
    best_bad = float('inf')
    
    t0 = time.time()
    attempts = 0
    
    while time.time() - t0 < max_time and attempts < 50:
        attempts += 1
        # Generate a random 1-factorization
        # Use random permutation of vertices and cyclic construction
        perm = list(range(16))
        rng.shuffle(perm)
        
        colors = [0] * 120
        # Map original vertex v to perm[v]
        # Build factorization on permuted vertices
        for i in range(15):
            # Edge (15, i) in original -> (perm[15], perm[i])
            a, b = perm[15], perm[i]
            if a > b:
                a, b = b, a
            colors[_idx(a, b)] = i
            for j in range(1, 8):
                a_orig = (i + j) % 15
                b_orig = (i - j) % 15
                a, b = perm[a_orig], perm[b_orig]
                if a > b:
                    a, b = b, a
                colors[_idx(a, b)] = i
        
        bad = _count_bad_k4(colors)
        if bad < best_bad:
            best_bad = bad
            best_colors = colors[:]
            if best_bad == 0:
                return best_colors, 0
        
        # Try local search on this one if time permits
        if time.time() - t0 < max_time * 0.7:
            result, bad2 = _local_search_1factor(colors, max_time * 0.2, seed + attempts)
            if bad2 < best_bad:
                best_bad = bad2
                best_colors = result[:]
                if best_bad == 0:
                    return best_colors, 0
    
    return best_colors, best_bad

def _make_17coloring():
    """The (i+j) % 17 construction - guaranteed valid, uses 17 colors."""
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def construct():
    t0 = time.time()
    deadline = 115  # safety margin before 120s kill
    
    # First, try to find a valid 15-coloring
    # Strategy 1: Cyclic 1-factorization + local search
    colors15, bad = _try_1factorization_with_swaps(30, seed=42)
    if bad == 0:
        return colors15
    
    # Strategy 2: Try random 1-factorizations
    remaining = deadline - (time.time() - t0)
    if remaining > 5:
        colors15b, bad_b = _try_random_1factorizations(remaining * 0.5, seed=123)
        if bad_b == 0:
            return colors15b
    
    # If we found a 15-coloring with few bad K4s, try more local search
    remaining = deadline - (time.time() - t0)
    if remaining > 5 and colors15 is not None and bad > 0:
        colors15c, bad_c = _local_search_1factor(colors15, remaining * 0.5, seed=999)
        if bad_c == 0:
            return colors15c
    
    # Fallback: 17-coloring (guaranteed valid)
    return _make_17coloring()

# EVOLVE-BLOCK-END
