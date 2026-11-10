"""Rank-2 tensor decomposition: c(i,j) = p_i + q_j guarantees no 4-cycle has 4 distinct colors."""
import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
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

# Precompute which vertices are endpoints of each edge
EDGE_ENDPOINTS = []
for i in range(N):
    for j in range(i + 1, N):
        EDGE_ENDPOINTS.append((i, j))

def _count_colors(colors):
    return len(set(colors))

def _is_valid(colors):
    for e1, e2, e3, e4 in CYCLES:
        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
            return False
    return True

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

def _rank2_coloring(p, q, offset=0):
    """Generate coloring using rank-2 additive structure: color(i,j) = (p[i] + q[j] + offset) % MAX_COLORS.
    This structurally ensures any 4-cycle has at most 3 distinct edge colors.
    Proof: For cycle a-b-c-d-a, edges are p_a+q_b, p_b+q_c, p_c+q_d, p_d+q_a.
    Note (p_a+q_b) + (p_c+q_d) = (p_b+q_c) + (p_d+q_a) = p_a+p_b+p_c+p_d + q_a+q_b+q_c+q_d.
    So the sum of opposite edges are equal, meaning at most 3 distinct values in any 4-cycle.
    """
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = _edge_idx(i, j)
            colors[idx] = (p[i] + q[j] + offset) % MAX_COLORS
    return colors

def _rank2_coloring_symmetric(p, q, offset=0):
    """Symmetric variant: color(i,j) = (p[i] + p[j] + q[i] + q[j] + offset) % MAX_COLORS.
    Also structurally valid for the same additive closure reason.
    """
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = _edge_idx(i, j)
            colors[idx] = (p[i] + p[j] + q[i] + q[j] + offset) % MAX_COLORS
    return colors

def _generate_rank2_init(rng, variant='basic'):
    """Generate a random rank-2 initialization."""
    if variant == 'basic':
        # p and q are random values in 0..MAX_COLORS-1
        p = [rng.randint(0, MAX_COLORS - 1) for _ in range(N)]
        q = [rng.randint(0, MAX_COLORS - 1) for _ in range(N)]
        offset = rng.randint(0, MAX_COLORS - 1)
        return _rank2_coloring(p, q, offset)
    elif variant == 'symmetric':
        p = [rng.randint(0, MAX_COLORS - 1) for _ in range(N)]
        q = [rng.randint(0, MAX_COLORS - 1) for _ in range(N)]
        offset = rng.randint(0, MAX_COLORS - 1)
        return _rank2_coloring_symmetric(p, q, offset)
    elif variant == 'structured':
        # Use arithmetic progressions for p and q
        step1 = rng.randint(1, 20)
        step2 = rng.randint(1, 20)
        base1 = rng.randint(0, MAX_COLORS - 1)
        base2 = rng.randint(0, MAX_COLORS - 1)
        p = [(base1 + step1 * i) % MAX_COLORS for i in range(N)]
        q = [(base2 + step2 * j) % MAX_COLORS for j in range(N)]
        offset = rng.randint(0, MAX_COLORS - 1)
        return _rank2_coloring(p, q, offset)
    elif variant == 'binary':
        # Binary-like patterns
        p = [rng.randint(0, 3) * rng.randint(1, 30) for _ in range(N)]
        q = [rng.randint(0, 3) * rng.randint(1, 30) for _ in range(N)]
        offset = rng.randint(0, MAX_COLORS - 1)
        return _rank2_coloring(p, q, offset)
    else:
        p = [rng.randint(0, MAX_COLORS - 1) for _ in range(N)]
        q = [rng.randint(0, MAX_COLORS - 1) for _ in range(N)]
        return _rank2_coloring(p, q)

def _local_search(colors, deadline, rng, max_unused_priority=True):
    """Fast local search with unused-color preference."""
    best = list(colors)
    best_count = _count_colors(best)
    
    color_usage = {}
    for c in best:
        color_usage[c] = color_usage.get(c, 0) + 1
    
    improved = True
    no_improve_streak = 0
    iter_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_streak += 1
        iter_count += 1
        
        # Perturbation escape
        if no_improve_streak > 3 and rng.random() < 0.4:
            for _ in range(rng.randint(5, 15)):
                e = rng.randint(0, NUM_EDGES - 1)
                old_c = best[e]
                for _ in range(30):
                    new_c = rng.randint(0, MAX_COLORS - 1)
                    if new_c == old_c:
                        continue
                    if _check_edge_change(best, e, new_c):
                        best[e] = new_c
                        color_usage[old_c] -= 1
                        if color_usage[old_c] == 0:
                            del color_usage[old_c]
                        color_usage[new_c] = color_usage.get(new_c, 0) + 1
                        break
        
        edges = list(range(NUM_EDGES))
        rng.shuffle(edges)
        
        for e in edges:
            if time.time() >= deadline:
                break
            
            old_c = best[e]
            
            # Build candidate list: prefer unused colors
            if max_unused_priority:
                unused = [c for c in range(MAX_COLORS) if c not in color_usage]
                if unused:
                    candidates = unused[:20] + [rng.randint(0, MAX_COLORS - 1) for _ in range(10)]
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
                    best_new_c = new_c
                    break
            
            if best_new_c is not None:
                old_count = color_usage.get(old_c, 0)
                new_count = color_usage.get(best_new_c, 0)
                
                best[e] = best_new_c
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
                    # Rollback if no improvement in color count
                    best[e] = old_c
                    color_usage[old_c] = old_count
                    color_usage[best_new_c] = new_count
                    if color_usage[best_new_c] == 0:
                        del color_usage[best_new_c]
    
    return best, best_count

def _fix_violations(colors, rng, max_iters=2000):
    """Fix violations by reassigning random edges."""
    colors = list(colors)
    for _ in range(max_iters):
        violated = False
        for e1, e2, e3, e4 in CYCLES:
            if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
                violated = True
                edge = rng.choice([e1, e2, e3, e4])
                colors[edge] = rng.randint(0, MAX_COLORS - 1)
                break
        if not violated:
            break
    return colors, _is_valid(colors)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_colors = None
    best_count = 0
    
    rng = random.Random(42)
    
    # Strategy: Rank-2 tensor decomposition c(i,j) = p_i + q_j
    # This structurally guarantees validity (no 4-cycle with 4 distinct colors)
    # We generate many diverse initializations and run local search on each
    
    variants = ['basic', 'symmetric', 'structured', 'binary']
    num_inits = 30  # Number of different initializations to try
    
    init_configs = []
    for v in variants:
        for _ in range(num_inits // len(variants)):
            init_configs.append(v)
    
    rng.shuffle(init_configs)
    
    for variant in init_configs:
        if time.time() >= deadline - 5.0:
            break
        
        colors = _generate_rank2_init(rng, variant)
        
        # Rank-2 colorings are structurally valid, but let's verify
        if not _is_valid(colors):
            colors, valid = _fix_violations(colors, rng)
            if not valid:
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
