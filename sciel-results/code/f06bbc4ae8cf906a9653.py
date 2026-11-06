import time
import itertools
import random
import numpy as np

def _check_k4_fast(colors):
    """Check every K4 has at least 5 distinct colours. Returns True if valid."""
    n = 16
    edge = np.zeros((n, n), dtype=np.int32)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge[i, j] = colors[idx]
            edge[j, i] = colors[idx]
            idx += 1
    
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    cs = np.array([edge[a,b], edge[a,c], edge[a,d],
                                   edge[b,c], edge[b,d], edge[c,d]], dtype=np.int32)
                    if len(np.unique(cs)) < 5:
                        return False
    return True

def _num_colours(colors):
    return len(set(colors))

def _try_merge(current_colors, c1, c2):
    """Try merging color c2 into c1. Returns new colors if valid, else None."""
    new_colors = [c1 if c == c2 else c for c in current_colors]
    if _check_k4_fast(new_colors):
        return new_colors
    return None

def _greedy_merge(colors, deadline, max_restarts=30):
    """Apply greedy color merging with multiple restarts."""
    best = colors[:]
    best_count = _num_colours(best)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        current = best[:]
        current_count = best_count
        unique = list(set(current))
        
        # Randomize merge order
        rng = random.Random(restart * 7919 + 13)
        pairs = list(itertools.combinations(unique, 2))
        rng.shuffle(pairs)
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            current_unique = list(set(current))
            pairs = list(itertools.combinations(current_unique, 2))
            rng.shuffle(pairs)
            
            for c1, c2 in pairs:
                if c1 not in set(current) or c2 not in set(current):
                    continue
                new_colors = _try_merge(current, c1, c2)
                if new_colors is not None:
                    current = new_colors
                    current_count = _num_colours(current)
                    if current_count < best_count:
                        best_count = current_count
                        best = current[:]
                    improved = True
                    break
    
    return best, best_count

def _local_search(colors, deadline):
    """Try individual edge recoloring to reduce color count."""
    best = colors[:]
    best_count = _num_colours(best)
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Try changing individual edges to existing colors
    for _ in range(50):
        if time.time() > deadline - 3:
            break
        
        current = best[:]
        current_count = best_count
        used_colors = set(current)
        
        # Try each edge, try changing to each used color
        for e in range(num_edges):
            if time.time() > deadline - 3:
                break
            old_c = current[e]
            for new_c in used_colors:
                if new_c == old_c:
                    continue
                candidate = current[:]
                candidate[e] = new_c
                if _check_k4_fast(candidate):
                    nc = _num_colours(candidate)
                    if nc < current_count:
                        current = candidate
                        current_count = nc
                        used_colors = set(current)
                        if current_count < best_count:
                            best_count = current_count
                            best = current[:]
    
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    best_colors = None
    best_count = 999
    
    # Phase 1: Systematic search over algebraic forms
    # Try (a*i + b*j) % p for various parameters
    primes = [13, 17, 19, 23, 29, 31]
    
    for p in primes:
        if time.time() > deadline - 60:
            break
        for a in range(1, p):
            for b in range(a, p):
                colors = [(a * i + b * j) % p for i in range(n) for j in range(i + 1, n)]
                cnt = _num_colours(colors)
                if cnt < best_count:
                    if _check_k4_fast(colors):
                        best_colors = colors[:]
                        best_count = cnt
                        if best_count <= 13:
                            return best_colors
    
    # Also try (i + j) % p specifically (the base case)
    for p in [13, 17, 19]:
        if time.time() > deadline - 60:
            break
        colors = [(i + j) % p for i in range(n) for j in range(i + 1, n)]
        cnt = _num_colours(colors)
        if cnt < best_count:
            if _check_k4_fast(colors):
                best_colors = colors[:]
                best_count = cnt
    
    # Szekeres-Wilson as another base
    colors_sw = [0] * num_edges
    edge_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if i == 15 or j == 15:
                k = j if i == 15 else i
                colors_sw[edge_idx] = k
            else:
                s = (i + j) % 15
                k = (s * 8) % 15
                colors_sw[edge_idx] = k
            edge_idx += 1
    
    if _check_k4_fast(colors_sw):
        cnt = _num_colours(colors_sw)
        if cnt < best_count:
            best_colors = colors_sw[:]
            best_count = cnt
    
    # Fallback if nothing found
    if best_colors is None:
        colors_fb = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        best_colors = colors_fb
        best_count = _num_colours(best_colors)
    
    # Phase 2: Greedy merging with multiple restarts
    if time.time() < deadline - 20:
        best_colors, best_count = _greedy_merge(best_colors, deadline - 10, max_restarts=30)
    
    # Phase 3: Local search (edge recoloring)
    if time.time() < deadline - 5:
        best_colors, best_count = _local_search(best_colors, deadline - 2)
    
    # Final verification
    if not _check_k4_fast(best_colors):
        # Should not happen, but fallback
        best_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    return best_colors
