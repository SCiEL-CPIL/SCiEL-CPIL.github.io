"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def _edge_idx(i, j):
    """Return the index in the lexicographic edge list for edge (i, j) with i < j."""
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def _build_cycles():
    """Build all 3 four-cycles for each 4-vertex subset."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    cycles.append((
                        _edge_idx(a, b),
                        _edge_idx(b, c),
                        _edge_idx(c, d),
                        _edge_idx(d, a)
                    ))
                    cycles.append((
                        _edge_idx(a, b),
                        _edge_idx(b, d),
                        _edge_idx(d, c),
                        _edge_idx(c, a)
                    ))
                    cycles.append((
                        _edge_idx(a, c),
                        _edge_idx(c, b),
                        _edge_idx(b, d),
                        _edge_idx(d, a)
                    ))
    return cycles

def _is_valid(colors, cycles):
    """Check no four-cycle has four distinct edge colors."""
    for e1, e2, e3, e4 in cycles:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if len({c1, c2, c3, c4}) == 4:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _algebraic_init(a, b, c, M):
    """Generate algebraic coloring (a*i + b*j + c) % M."""
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = _edge_idx(i, j)
            colors[idx] = (a * i + b * j + c) % M
    return colors

def _local_search(colors, cycles, edge_cycle_info, deadline, seed):
    """Fast local search: track used colors, prefer new colors, check only affected cycles."""
    rng = random.Random(seed)
    best = list(colors)
    best_count = _count_colors(best)
    used_colors = set(best)
    
    edges = list(range(NUM_EDGES))
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(edges)
        
        for e in edges:
            if time.time() >= deadline:
                break
            
            old_c = best[e]
            
            # Build candidate list: prefer unused colors, then used
            unused = [c for c in range(MAX_COLORS) if c not in used_colors]
            rng.shuffle(unused)
            used_list = list(used_colors)
            rng.shuffle(used_list)
            candidates = unused[:12] + used_list[:6]
            rng.shuffle(candidates)
            
            best_candidate = None
            best_new_count = best_count
            
            for new_c in candidates:
                if new_c == old_c:
                    continue
                
                # Check validity against all cycles containing edge e
                valid = True
                for ci in edge_cycle_info[e]:
                    e1, e2, e3, e4 = cycles[ci]
                    c1 = best[e1] if e1 != e else new_c
                    c2 = best[e2] if e2 != e else new_c
                    c3 = best[e3] if e3 != e else new_c
                    c4 = best[e4] if e4 != e else new_c
                    if len({c1, c2, c3, c4}) == 4:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Estimate new color count
                if new_c not in used_colors:
                    if best.count(old_c) <= 1:
                        new_count = best_count
                    else:
                        new_count = best_count + 1
                else:
                    new_count = best_count
                
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_candidate = new_c
            
            if best_candidate is not None:
                if best.count(old_c) <= 1:
                    used_colors.discard(old_c)
                best[e] = best_candidate
                used_colors.add(best_candidate)
                best_count = best_new_count
                improved = True
    
    return best, best_count

def _aggressive_refine(colors, cycles, edge_cycle_info, deadline):
    """Aggressive refinement: try all 120 colors per edge to maximize diversity."""
    best = list(colors)
    best_count = _count_colors(best)
    used_colors = set(best)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e in range(NUM_EDGES):
            if time.time() >= deadline:
                break
            
            old_c = best[e]
            
            # Try all unused colors first
            for new_c in range(MAX_COLORS):
                if new_c in used_colors:
                    continue
                
                # Check validity
                valid = True
                for ci in edge_cycle_info[e]:
                    e1, e2, e3, e4 = cycles[ci]
                    c1 = best[e1] if e1 != e else new_c
                    c2 = best[e2] if e2 != e else new_c
                    c3 = best[e3] if e3 != e else new_c
                    c4 = best[e4] if e4 != e else new_c
                    if len({c1, c2, c3, c4}) == 4:
                        valid = False
                        break
                
                if valid:
                    if best.count(old_c) <= 1:
                        used_colors.discard(old_c)
                    best[e] = new_c
                    used_colors.add(new_c)
                    best_count += 1
                    improved = True
                    break
    
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    cycles = _build_cycles()
    
    # Precompute edge-to-cycle mapping
    edge_cycle_info = [[] for _ in range(NUM_EDGES)]
    for ci, (e1, e2, e3, e4) in enumerate(cycles):
        edge_cycle_info[e1].append(ci)
        edge_cycle_info[e2].append(ci)
        edge_cycle_info[e3].append(ci)
        edge_cycle_info[e4].append(ci)
    
    best_colors = None
    best_count = 0
    
    rng = random.Random(42)
    
    # Build diverse parameter list - focus on small moduli that work well
    params = []
    
    # Systematic small-modulus search - these are the most promising
    for M in [4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 80, 100, 120]:
        max_ab = min(M, 15)
        for a in range(1, max_ab + 1):
            for b in range(1, max_ab + 1):
                if a == b and M <= 16:
                    continue
                for c in [0, 1, 2]:
                    params.append((a, b, c, M))
            if len(params) > 300:
                break
        if len(params) > 300:
            break
    
    # Random inits with varied moduli
    for _ in range(80):
        M = rng.choice([4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 80, 100, 120])
        a = rng.randint(1, max(1, M - 1))
        b = rng.randint(1, max(1, M - 1))
        c = rng.randint(0, max(0, M - 1))
        params.append((a, b, c, M))
    
    # Shuffle to get diversity early
    rng.shuffle(params)
    
    for pi, (a, b, c, M) in enumerate(params):
        if time.time() >= deadline - 5:
            break
        
        colors = _algebraic_init(a, b, c, M)
        
        # Check validity
        if not _is_valid(colors, cycles):
            continue
        
        count = _count_colors(colors)
        
        # Local search to improve
        if time.time() < deadline - 3:
            ls_colors, ls_count = _local_search(colors, cycles, edge_cycle_info, deadline - 1, seed=pi)
            if ls_count > count:
                count = ls_count
                colors = ls_colors
        
        if count > best_count:
            best_count = count
            best_colors = list(colors)
    
    # If we have a good result, do aggressive refinement
    if best_colors is not None and best_count > 0 and time.time() < deadline - 2:
        refined, refined_count = _aggressive_refine(best_colors, cycles, edge_cycle_info, deadline - 1)
        if refined_count > best_count:
            best_count = refined_count
            best_colors = refined
    
    # If no valid coloring found, return all zeros (valid but score 1)
    if best_colors is None:
        return [0] * NUM_EDGES
    
    return best_colors
# EVOLVE-BLOCK-END
