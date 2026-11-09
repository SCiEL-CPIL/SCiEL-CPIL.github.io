"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

# Precompute all K4s
_N = 16
_K4S = list(combinations(range(_N), 4))
_K4_EDGES = []
for quad in _K4S:
    edges = []
    for a in range(4):
        for b in range(a + 1, 4):
            i, j = quad[a], quad[b]
            edges.append((i, j))
    _K4_EDGES.append(edges)

# Precompute for each edge, which K4s contain it
_EDGE_TO_K4S = [[] for _ in range(_N * (_N - 1) // 2)]
def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (_N - 1) - i * (i + 1) // 2 + (j - i - 1)

for k4_idx, edges in enumerate(_K4_EDGES):
    for (i, j) in edges:
        _EDGE_TO_K4S[_edge_idx(i, j)].append(k4_idx)

# For each K4, precompute the 6 edge indices
_K4_EDGE_IDXS = []
for edges in _K4_EDGES:
    _K4_EDGE_IDXS.append([_edge_idx(i, j) for (i, j) in edges])

_NUM_K4S = len(_K4S)
_NUM_EDGES = _N * (_N - 1) // 2  # 120

def _count_bad_k4s(colors):
    """Count K4s with fewer than 5 distinct colors. colors is a numpy array of edge colors."""
    bad = 0
    for k4_idx in range(_NUM_K4S):
        eidxs = _K4_EDGE_IDXS[k4_idx]
        c0 = colors[eidxs[0]]
        c1 = colors[eidxs[1]]
        c2 = colors[eidxs[2]]
        c3 = colors[eidxs[3]]
        c4 = colors[eidxs[4]]
        c5 = colors[eidxs[5]]
        # Count distinct
        distinct = 1
        if c1 != c0: distinct += 1
        if c2 != c0 and c2 != c1: distinct += 1
        if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
        if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
        if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
        if distinct < 5:
            bad += 1
    return bad

def _count_colors_used(colors):
    return len(set(colors.tolist()))

def construct():
    deadline = time.time() + 130  # safety margin
    
    # Start with the known valid 17-coloring
    colors = np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
    
    # Try to reduce to 15 colors first, then 14, then 13
    # We'll do local search to minimize bad K4s, then minimize colors used
    
    best_colors = colors.copy()
    best_score = _count_colors_used(best_colors)  # 17
    
    # Strategy: local search that tries to reduce color count
    # First, try to find a valid coloring with fewer colors
    
    # For each target color count, try to find a valid coloring
    for target_colors in [15, 14, 13]:
        if time.time() > deadline - 30:
            break
        
        # Start from current best or random
        rng = random.Random(target_colors * 1000)
        
        # Initialize with random assignment using target_colors colors
        # But we need to be smart: use structure
        # Try multiple random initializations
        for init_iter in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with target_colors
            c = np.array([rng.randint(0, target_colors - 1) for _ in range(_NUM_EDGES)], dtype=np.int32)
            
            bad = _count_bad_k4s(c)
            
            # Local search
            max_iters = 50000
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                if bad == 0:
                    # Valid coloring! Check color count
                    actual_colors = _count_colors_used(c)
                    if actual_colors < best_score:
                        best_score = actual_colors
                        best_colors = c.copy()
                    break
                
                # Pick a random bad K4 and try to fix it
                # Actually, let's pick a random edge and try changing it
                edge_idx = rng.randint(0, _NUM_EDGES - 1)
                old_color = c[edge_idx]
                
                # Try a new random color
                new_color = rng.randint(0, target_colors - 1)
                if new_color == old_color:
                    continue
                
                # Compute change in bad count
                # Only K4s containing this edge are affected
                affected_k4s = _EDGE_TO_K4S[edge_idx]
                
                delta_bad = 0
                for k4_idx in affected_k4s:
                    eidxs = _K4_EDGE_IDXS[k4_idx]
                    # Before
                    c0 = c[eidxs[0]]
                    c1 = c[eidxs[1]]
                    c2 = c[eidxs[2]]
                    c3 = c[eidxs[3]]
                    c4 = c[eidxs[4]]
                    c5 = c[eidxs[5]]
                    
                    # Count distinct before
                    distinct_before = 1
                    if c1 != c0: distinct_before += 1
                    if c2 != c0 and c2 != c1: distinct_before += 1
                    if c3 != c0 and c3 != c1 and c3 != c2: distinct_before += 1
                    if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct_before += 1
                    if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct_before += 1
                    bad_before = 1 if distinct_before < 5 else 0
                    
                    # After (simulate)
                    # Find which position this edge is in the K4
                    pos = -1
                    for p in range(6):
                        if eidxs[p] == edge_idx:
                            pos = p
                            break
                    
                    new_vals = [c0, c1, c2, c3, c4, c5]
                    new_vals[pos] = new_color
                    nv0, nv1, nv2, nv3, nv4, nv5 = new_vals
                    
                    distinct_after = 1
                    if nv1 != nv0: distinct_after += 1
                    if nv2 != nv0 and nv2 != nv1: distinct_after += 1
                    if nv3 != nv0 and nv3 != nv1 and nv3 != nv2: distinct_after += 1
                    if nv4 != nv0 and nv4 != nv1 and nv4 != nv2 and nv4 != nv3: distinct_after += 1
                    if nv5 != nv0 and nv5 != nv1 and nv5 != nv2 and nv5 != nv3 and nv5 != nv4: distinct_after += 1
                    bad_after = 1 if distinct_after < 5 else 0
                    
                    delta_bad += bad_after - bad_before
                
                if delta_bad <= 0:
                    c[edge_idx] = new_color
                    bad += delta_bad
                else:
                    # With some probability, accept worse (simulated annealing style)
                    if rng.random() < 0.1:
                        c[edge_idx] = new_color
                        bad += delta_bad
            
            if bad == 0:
                actual_colors = _count_colors_used(c)
                if actual_colors < best_score:
                    best_score = actual_colors
                    best_colors = c.copy()
                break  # Found valid for this target, move to next
    
    # Final check: ensure best_colors is valid
    if _count_bad_k4s(best_colors) > 0:
        # Fallback to the original 17-coloring
        return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]
    
    # Return the best coloring found
    # The score is the number of colors used
    return best_colors.tolist()
# EVOLVE-BLOCK-END
