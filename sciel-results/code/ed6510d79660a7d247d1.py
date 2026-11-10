"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def _build_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    cycles.append((_edge_idx(a, b), _edge_idx(b, c), _edge_idx(c, d), _edge_idx(d, a)))
                    cycles.append((_edge_idx(a, b), _edge_idx(b, d), _edge_idx(d, c), _edge_idx(c, a)))
                    cycles.append((_edge_idx(a, c), _edge_idx(c, b), _edge_idx(b, d), _edge_idx(d, a)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

# Precompute edge-to-cycles mapping
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(CYCLES):
    for e in cyc:
        EDGE_TO_CYCLES[e].append(ci)

def _is_valid(colors):
    for e1, e2, e3, e4 in CYCLES:
        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _algebraic_init(a, b, c, M):
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = _edge_idx(i, j)
            colors[idx] = (a * i + b * j + c) % M
    return colors

def _check_edge_change(colors, edge, new_color):
    """Check if changing edge to new_color maintains validity."""
    for ci in EDGE_TO_CYCLES[edge]:
        e1, e2, e3, e4 = CYCLES[ci]
        c1 = colors[e1] if e1 != edge else new_color
        c2 = colors[e2] if e2 != edge else new_color
        c3 = colors[e3] if e3 != edge else new_color
        c4 = colors[e4] if e4 != edge else new_color
        if len({c1, c2, c3, c4}) == 4:
            return False
    return True

def _local_search(colors, deadline, rng, max_unused_priority=True):
    """Local search with unused-color preference and perturbation escapes."""
    best = list(colors)
    best_count = _count_colors(best)
    
    # Track color usage for quick unused-color selection
    color_usage = {}
    for c in best:
        color_usage[c] = color_usage.get(c, 0) + 1
    
    improved = True
    no_improve_streak = 0
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_streak += 1
        
        # Perturbation escape: if stuck for a while, make random valid changes
        if no_improve_streak > 5 and rng.random() < 0.3:
            for _ in range(rng.randint(3, 10)):
                e = rng.randint(0, NUM_EDGES - 1)
                old_c = best[e]
                # Try random colors
                for _ in range(20):
                    new_c = rng.randint(0, MAX_COLORS - 1)
                    if new_c == old_c:
                        continue
                    if _check_edge_change(best, e, new_c):
                        best[e] = new_c
                        # Update usage
                        color_usage[old_c] -= 1
                        if color_usage[old_c] == 0:
                            del color_usage[old_c]
                        color_usage[new_c] = color_usage.get(new_c, 0) + 1
                        break
                else:
                    best[e] = old_c
        
        edges = list(range(NUM_EDGES))
        rng.shuffle(edges)
        
        for e in edges:
            if time.time() >= deadline:
                break
            
            old_c = best[e]
            
            # Prefer unused colors to increase diversity
            if max_unused_priority:
                unused = [c for c in range(MAX_COLORS) if c not in color_usage]
                if unused:
                    candidates = unused + [rng.randint(0, MAX_COLORS - 1) for _ in range(10)]
                    rng.shuffle(candidates)
                else:
                    candidates = [rng.randint(0, MAX_COLORS - 1) for _ in range(20)]
            else:
                candidates = [rng.randint(0, MAX_COLORS - 1) for _ in range(20)]
            
            best_new_c = None
            for new_c in candidates:
                if new_c == old_c:
                    continue
                if _check_edge_change(best, e, new_c):
                    # Check if this increases or maintains color count
                    old_usage = color_usage.get(old_c, 0)
                    new_usage = color_usage.get(new_c, 0)
                    
                    if new_usage == 0:
                        # Using a new color - always beneficial
                        best_new_c = new_c
                        break
                    elif old_usage == 1 and new_usage > 0:
                        # Last use of old_c, new_c already used - neutral
                        best_new_c = new_c
                        break
                    elif old_usage > 1 and new_usage == 0:
                        # Introducing new color while old_c still used - beneficial
                        best_new_c = new_c
                        break
                    else:
                        # Check if it maintains count
                        best_new_c = new_c
                        break
            
            if best_new_c is not None:
                # Apply change
                best[e] = best_new_c
                old_count = color_usage.get(old_c, 0)
                new_count = color_usage.get(best_new_c, 0)
                
                color_usage[old_c] = old_count - 1
                if color_usage[old_c] == 0:
                    del color_usage[old_c]
                color_usage[best_new_c] = new_count + 1
                
                current_count = len(color_usage)
                if current_count > best_count:
                    best_count = current_count
                    improved = True
                    no_improve_streak = 0
                else:
                    # Rollback if no improvement
                    best[e] = old_c
                    color_usage[old_c] = old_count
                    color_usage[best_new_c] = new_count
                    if color_usage[best_new_c] == 0:
                        del color_usage[best_new_c]
    
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_colors = None
    best_count = 0
    
    rng = random.Random(42)
    
    # Generate diverse algebraic parameter combinations
    params = []
    
    # Small moduli with various coefficients
    for M in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        for a in range(1, min(M, 8) + 1):
            for b in range(1, min(M, 8) + 1):
                if a == b:
                    continue
                params.append((a, b, 0, M))
    
    # Larger moduli
    for M in [120, 119, 118, 117, 116, 115, 114, 112, 110, 108, 105, 100, 96, 90, 80, 72, 60, 48, 40, 36, 32, 24, 20, 16, 15, 12, 10]:
        for a in range(1, min(M, 6) + 1):
            for b in range(1, min(M, 6) + 1):
                if a == b:
                    continue
                params.append((a, b, 0, M))
    
    # Random parameters
    for _ in range(50):
        M = rng.choice([7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 120])
        a = rng.randint(1, min(M, 10) - 1)
        b = rng.randint(1, min(M, 10) - 1)
        c = rng.randint(0, M - 1)
        if a != b:
            params.append((a, b, c, M))
    
    rng.shuffle(params)
    
    # Try each parameter set
    for pi, (a, b, c, M) in enumerate(params):
        if time.time() >= deadline - 5.0:
            break
        
        colors = _algebraic_init(a, b, c, M)
        
        # Check validity
        if not _is_valid(colors):
            # Try to fix by randomizing violating edges
            fixed = False
            for _ in range(1000):
                violated = False
                for e1, e2, e3, e4 in CYCLES:
                    if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
                        violated = True
                        # Pick a random edge from the cycle and reassign
                        edge = rng.choice([e1, e2, e3, e4])
                        colors[edge] = rng.randint(0, MAX_COLORS - 1)
                        break
                if not violated:
                    fixed = True
                    break
            
            if not fixed:
                continue
        
        count = _count_colors(colors)
        if count > best_count:
            best_count = count
            best_colors = list(colors)
        
        # Local search to improve
        if time.time() < deadline - 3.0:
            ls_colors, ls_count = _local_search(colors, deadline - 1.0, rng)
            if ls_count > best_count:
                best_count = ls_count
                best_colors = ls_colors
    
    # Final aggressive improvement on the best solution
    if best_colors is not None and time.time() < deadline - 2.0:
        final_colors, final_count = _local_search(best_colors, deadline - 0.5, rng)
        if final_count > best_count:
            best_count = final_count
            best_colors = final_colors
    
    if best_colors is None:
        return [0] * NUM_EDGES
    
    return best_colors
