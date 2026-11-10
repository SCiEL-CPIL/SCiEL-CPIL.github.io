"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def _build_k4_edges(n=16):
    """Precompute for each K4, the list of edge indices it uses."""
    idx = [[-1]*n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    k4_edge_lists = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        edges = [idx[a][b], idx[a][c], idx[a][d], idx[b][c], idx[b][d], idx[c][d]]
        k4_edge_lists.append(edges)
    return idx, k4_edge_lists

def _count_violations_fast(colours, k4_edge_lists, min_colours=5):
    """Count K4s with fewer than min_colours distinct colours using precomputed edge lists."""
    violations = 0
    for edges in k4_edge_lists:
        cs = set(colours[e] for e in edges)
        if len(cs) < min_colours:
            violations += 1
    return violations

def _eval_colours_fast(colours, k4_edge_lists, n=16, min_colours=5):
    """Return (num_colours_used, num_violations)."""
    used = len(set(colours))
    viol = _count_violations_fast(colours, k4_edge_lists, min_colours)
    return used, viol

def _try_sum_mod(k, n=16, offset=0):
    """Colour (i,j) = (i+j+offset) mod k."""
    colours = [((i + j + offset) % k) for i in range(n) for j in range(i+1, n)]
    return colours

def _try_diff_mod(k, n=16, offset=0):
    """Colour (i,j) = (j-i+offset) mod k."""
    colours = [((j - i + offset) % k) for i in range(n) for j in range(i+1, n)]
    return colours

def _try_product_mod(k, n=16, offset=0):
    """Colour (i,j) = (i*j+offset) mod k."""
    colours = [((i * j + offset) % k) for i in range(n) for j in range(i+1, n)]
    return colours

def _try_sum_product_mod(k1, k2, n=16, o1=0, o2=0):
    """Colour (i,j) = ((i+j+o1) % k1) * k2 + ((i*j+o2) % k2)."""
    k = k1 * k2
    colours = [(((i + j + o1) % k1) * k2 + ((i * j + o2) % k2)) % k for i in range(n) for j in range(i+1, n)]
    return colours

def _local_search_fast(colours, n=16, min_colours=5, max_iter=5000, deadline=None, seed=42):
    """Fast hill-climb using precomputed K4 edge lists."""
    rng = random.Random(seed)
    idx, k4_edge_lists = _build_k4_edges(n)
    
    best_colours = colours[:]
    best_viol = _count_violations_fast(best_colours, k4_edge_lists, min_colours)
    current_colours = colours[:]
    current_viol = best_viol
    
    # Precompute which K4s each edge belongs to
    edge_to_k4s = defaultdict(list)
    for k4_idx, edges in enumerate(k4_edge_lists):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if current_viol == 0:
            break
        
        # Find a violating K4
        quad_idx = None
        for k4_idx in range(len(k4_edge_lists)):
            edges = k4_edge_lists[k4_idx]
            cs = set(current_colours[e] for e in edges)
            if len(cs) < min_colours:
                quad_idx = k4_idx
                break
        if quad_idx is None:
            break
        
        edges_in_quad = k4_edge_lists[quad_idx]
        
        # Try to improve by changing one edge in this K4 to a new colour
        improved = False
        max_k = max(current_colours) + 1
        for e in edges_in_quad:
            old_c = current_colours[e]
            for new_c in range(max_k + 1):
                if new_c == old_c:
                    continue
                current_colours[e] = new_c
                v = _count_violations_fast(current_colours, k4_edge_lists, min_colours)
                if v < current_viol:
                    current_viol = v
                    improved = True
                    break
                current_colours[e] = old_c
            if improved:
                break
        
        if not improved:
            # Try random perturbation on a random edge
            ei = rng.randrange(len(current_colours))
            old_c = current_colours[ei]
            current_colours[ei] = rng.randint(0, max_k)
            v = _count_violations_fast(current_colours, k4_edge_lists, min_colours)
            if v <= current_viol:
                current_viol = v
            else:
                current_colours[ei] = old_c
        
        if current_viol < best_viol:
            best_viol = current_viol
            best_colours = current_colours[:]
    
    return best_colours, best_viol

def _greedy_construct(n=16, max_colours=15, seed=42):
    """Greedy construction: assign colours one edge at a time."""
    rng = random.Random(seed)
    idx, k4_edge_lists = _build_k4_edges(n)
    
    # For each K4, track which edges are assigned and their colours
    colours = [None] * (n*(n-1)//2)
    
    # Process edges in lexicographic order
    for i in range(n):
        for j in range(i+1, n):
            ei = idx[i][j]
            # Try colours from 0 to max_colours-1
            assigned = False
            for c in range(max_colours):
                colours[ei] = c
                # Check if this assignment causes any K4 violation
                # Only need to check K4s that include this edge
                ok = True
                for k4_idx in range(len(k4_edge_lists)):
                    edges = k4_edge_lists[k4_idx]
                    if ei not in edges:
                        continue
                    # Check if all edges in this K4 are assigned
                    if any(colours[e] is None for e in edges):
                        continue
                    cs = set(colours[e] for e in edges)
                    if len(cs) < 5:
                        ok = False
                        break
                if ok:
                    assigned = True
                    break
            if not assigned:
                # Use a new colour
                colours[ei] = max_colours
                max_colours += 1
    
    return colours

# EVOLVE-BLOCK-END

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 16
    best_colours = None
    best_used = float('inf')
    best_viol = float('inf')
    
    idx, k4_edge_lists = _build_k4_edges(n)
    
    # Phase 1: Try structured constructions with fast evaluation
    # Try sum mod k
    for k in range(13, 20):
        for offset in range(k):
            if time.time() > deadline:
                break
            colours = _try_sum_mod(k, n, offset)
            used, viol = _eval_colours_fast(colours, k4_edge_lists, n)
            if viol == 0 and used < best_used:
                best_used = used
                best_viol = viol
                best_colours = colours[:]
    
    # Try diff mod k
    for k in range(13, 20):
        for offset in range(k):
            if time.time() > deadline:
                break
            colours = _try_diff_mod(k, n, offset)
            used, viol = _eval_colours_fast(colours, k4_edge_lists, n)
            if viol == 0 and used < best_used:
                best_used = used
                best_viol = viol
                best_colours = colours[:]
    
    # Try product mod k
    for k in range(13, 20):
        for offset in range(k):
            if time.time() > deadline:
                break
            colours = _try_product_mod(k, n, offset)
            used, viol = _eval_colours_fast(colours, k4_edge_lists, n)
            if viol == 0 and used < best_used:
                best_used = used
                best_viol = viol
                best_colours = colours[:]
    
    # Try combined constructions
    for k1 in range(2, 8):
        for k2 in range(2, 8):
            if k1 * k2 > 25:
                continue
            for o1 in range(k1):
                for o2 in range(k2):
                    if time.time() > deadline:
                        break
                    colours = _try_sum_product_mod(k1, k2, n, o1, o2)
                    used, viol = _eval_colours_fast(colours, k4_edge_lists, n)
                    if viol == 0 and used < best_used:
                        best_used = used
                        best_viol = viol
                        best_colours = colours[:]
    
    # Phase 2: If no valid solution, try greedy construction
    if best_colours is None:
        for seed in range(20):
            if time.time() > deadline:
                break
            colours = _greedy_construct(n, max_colours=15, seed=seed)
            used, viol = _eval_colours_fast(colours, k4_edge_lists, n)
            if viol == 0 and used < best_used:
                best_used = used
                best_viol = viol
                best_colours = colours[:]
            elif viol < best_viol:
                best_viol = viol
                best_colours = colours[:]
    
    # Phase 3: Local search to fix violations or reduce colours
    if best_colours is not None:
        if best_viol > 0:
            # Fix violations
            fixed, viol = _local_search_fast(best_colours, n, 5, max_iter=3000, deadline=deadline, seed=42)
            if viol == 0:
                used = len(set(fixed))
                best_colours = fixed
                best_used = used
                best_viol = 0
        else:
            # Already valid, try to reduce colours
            # Remap to compact range
            colour_map = {}
            next_c = 0
            for c in best_colours:
                if c not in colour_map:
                    colour_map[c] = next_c
                    next_c += 1
            best_colours = [colour_map[c] for c in best_colours]
            best_used = len(colour_map)
    
    # Fallback: use sum mod 17 (known to be valid)
    if best_colours is None:
        best_colours = _try_sum_mod(17, n, 0)
        best_used = 17
        best_viol = 0
    
    # Final compression
    if best_colours is not None:
        colour_map = {}
        next_c = 0
        for c in best_colours:
            if c not in colour_map:
                colour_map[c] = next_c
                next_c += 1
        best_colours = [colour_map[c] for c in best_colours]
    
    return best_colours
# EVOLVE-BLOCK-END
