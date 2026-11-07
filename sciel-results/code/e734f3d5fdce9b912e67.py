"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    from autoresearch_suite.science_tcs_expansion import check_lift

    # Build edge map: idx -> (type, pos)
    edge_map = {}
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            if v == (u + 1) % m:
                pos = u
            else:
                pos = v
            edge_map[idx] = ('top', pos)
        elif u >= m and v >= m:
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                pos = u_local
            else:
                pos = v_local
            edge_map[idx] = ('bottom', pos)
        else:
            if u < m:
                pos = u
            else:
                pos = v - m
            edge_map[idx] = ('match', pos)

    def make_signs(top_signs, bottom_signs, match_signs):
        """Construct full sign array from per-position sign arrays."""
        signs = np.ones(n_edges, dtype=int)
        for idx in range(n_edges):
            etype, pos = edge_map[idx]
            if etype == 'top':
                signs[idx] = top_signs[pos]
            elif etype == 'bottom':
                signs[idx] = bottom_signs[pos]
            else:
                signs[idx] = match_signs[pos]
        return signs

    def evaluate_signs(signs):
        try:
            radius, _ = check_lift(graph, signs)
            return radius
        except:
            return float('inf')

    best_radius = float('inf')
    best_signs = None

    def try_candidate(signs):
        nonlocal best_radius, best_signs
        if time.time() > DEADLINE:
            return
        r = evaluate_signs(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    # Analytic approach: For prism graphs, the optimal configuration
    # minimizes spectral radius by making cycle products as -1 as possible
    # while keeping the graph unbalanced (for connected lift).
    
    # Strategy: Try structured patterns that are known to give low spectral radius
    # For even m: alternating signs on cycles gives spectral radius 2
    # For odd m: need at least one -1 per cycle, near-minimal is 2*cos(pi/m) + 2*cos(pi/m)
    
    # Pattern 1: All +1 except one -1 on top cycle (minimal unbalance)
    for pos in range(m):
        top = np.ones(m, dtype=int)
        top[pos] = -1
        bottom = np.ones(m, dtype=int)
        match = np.ones(m, dtype=int)
        try_candidate(make_signs(top, bottom, match))

    # Pattern 2: All +1 except one -1 on bottom cycle
    for pos in range(m):
        top = np.ones(m, dtype=int)
        bottom = np.ones(m, dtype=int)
        bottom[pos] = -1
        match = np.ones(m, dtype=int)
        try_candidate(make_signs(top, bottom, match))

    # Pattern 3: All +1 except one -1 on matching
    for pos in range(m):
        top = np.ones(m, dtype=int)
        bottom = np.ones(m, dtype=int)
        match = np.ones(m, dtype=int)
        match[pos] = -1
        try_candidate(make_signs(top, bottom, match))

    # Pattern 4: Alternating on top, +1 elsewhere (for even m)
    if m % 2 == 0:
        top = np.ones(m, dtype=int)
        top[1::2] = -1
        bottom = np.ones(m, dtype=int)
        match = np.ones(m, dtype=int)
        try_candidate(make_signs(top, bottom, match))
        
        top = np.ones(m, dtype=int)
        bottom = np.ones(m, dtype=int)
        bottom[1::2] = -1
        match = np.ones(m, dtype=int)
        try_candidate(make_signs(top, bottom, match))

    # Pattern 5: Alternating on matching
    match = np.ones(m, dtype=int)
    match[1::2] = -1
    top = np.ones(m, dtype=int)
    bottom = np.ones(m, dtype=int)
    try_candidate(make_signs(top, bottom, match))

    # Pattern 6: Two -1's to reduce radius further
    # -1 on top and bottom at same position
    for pos in range(m):
        top = np.ones(m, dtype=int)
        top[pos] = -1
        bottom = np.ones(m, dtype=int)
        bottom[pos] = -1
        match = np.ones(m, dtype=int)
        try_candidate(make_signs(top, bottom, match))
    
    # -1 on top and matching
    for pos in range(m):
        top = np.ones(m, dtype=int)
        top[pos] = -1
        bottom = np.ones(m, dtype=int)
        match = np.ones(m, dtype=int)
        match[pos] = -1
        try_candidate(make_signs(top, bottom, match))
    
    # -1 on bottom and matching
    for pos in range(m):
        top = np.ones(m, dtype=int)
        bottom = np.ones(m, dtype=int)
        bottom[pos] = -1
        match = np.ones(m, dtype=int)
        match[pos] = -1
        try_candidate(make_signs(top, bottom, match))

    # Pattern 7: Exhaustive search over small cycle space if time permits
    # Define spanning tree to enumerate switching classes
    tree_indices = []
    free_indices = []
    for idx in range(n_edges):
        etype, pos = edge_map[idx]
        is_in_tree = False
        if etype == 'match' and pos == 0:
            is_in_tree = True
        elif etype == 'top' and pos < m - 1:
            is_in_tree = True
        elif etype == 'bottom' and pos < m - 1:
            is_in_tree = True
        if is_in_tree:
            tree_indices.append(idx)
        else:
            free_indices.append(idx)
    
    n_free = len(free_indices)
    
    # Only do exhaustive if n_free is small enough
    if n_free <= 12:  # 4096 max
        num_classes = 1 << n_free
        for mask in range(num_classes):
            if time.time() > DEADLINE:
                break
            signs = np.ones(n_edges, dtype=int)
            for i, idx in enumerate(free_indices):
                if mask & (1 << i):
                    signs[idx] = -1
            r = evaluate_signs(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()

    if best_signs is None:
        # Fallback: all +1 with one -1
        best_signs = np.ones(n_edges, dtype=int)
        best_signs[0] = -1
    
    return best_signs
# EVOLVE-BLOCK-END
