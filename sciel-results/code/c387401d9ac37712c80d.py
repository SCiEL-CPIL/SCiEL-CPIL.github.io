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

# For each K4, precompute the 6 edge indices and edge positions
_K4_EDGE_IDXS = []
_EDGE_POS_IN_K4 = {}  # (k4_idx, edge_idx) -> position in the K4's edge list
for k4_idx, edges in enumerate(_K4_EDGES):
    idxs = []
    for pos, (i, j) in enumerate(edges):
        ei = _edge_idx(i, j)
        idxs.append(ei)
        _EDGE_POS_IN_K4[(k4_idx, ei)] = pos
    _K4_EDGE_IDXS.append(idxs)

_NUM_K4S = len(_K4S)
_NUM_EDGES = _N * (_N - 1) // 2  # 120

# Precompute for each edge, the list of (k4_idx, position) pairs
_EDGE_K4_POSITIONS = []
for ei in range(_NUM_EDGES):
    pairs = []
    for k4_idx in _EDGE_TO_K4S[ei]:
        pos = _EDGE_POS_IN_K4[(k4_idx, ei)]
        pairs.append((k4_idx, pos))
    _EDGE_K4_POSITIONS.append(pairs)

def _count_distinct_6(c0, c1, c2, c3, c4, c5):
    d = 1
    if c1 != c0: d += 1
    if c2 != c0 and c2 != c1: d += 1
    if c3 != c0 and c3 != c1 and c3 != c2: d += 1
    if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: d += 1
    if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: d += 1
    return d

def _count_bad_k4s(colors):
    """Count K4s with fewer than 5 distinct colors."""
    bad = 0
    for k4_idx in range(_NUM_K4S):
        eidxs = _K4_EDGE_IDXS[k4_idx]
        c0 = colors[eidxs[0]]
        c1 = colors[eidxs[1]]
        c2 = colors[eidxs[2]]
        c3 = colors[eidxs[3]]
        c4 = colors[eidxs[4]]
        c5 = colors[eidxs[5]]
        if _count_distinct_6(c0, c1, c2, c3, c4, c5) < 5:
            bad += 1
    return bad

def _count_colors_used(colors):
    return len(set(colors.tolist()))

def _delta_bad_for_edge_change(c, edge_idx, new_color, bad_k4_flags):
    """Compute change in bad K4 count when changing edge_idx to new_color.
    bad_k4_flags is a numpy bool array of length _NUM_K4S where True means bad."""
    old_color = c[edge_idx]
    if old_color == new_color:
        return 0
    
    delta = 0
    for k4_idx, pos in _EDGE_K4_POSITIONS[edge_idx]:
        eidxs = _K4_EDGE_IDXS[k4_idx]
        c0 = c[eidxs[0]]
        c1 = c[eidxs[1]]
        c2 = c[eidxs[2]]
        c3 = c[eidxs[3]]
        c4 = c[eidxs[4]]
        c5 = c[eidxs[5]]
        
        was_bad = bad_k4_flags[k4_idx]
        
        # Compute new distinct count
        if pos == 0: nv = (new_color, c1, c2, c3, c4, c5)
        elif pos == 1: nv = (c0, new_color, c2, c3, c4, c5)
        elif pos == 2: nv = (c0, c1, new_color, c3, c4, c5)
        elif pos == 3: nv = (c0, c1, c2, new_color, c4, c5)
        elif pos == 4: nv = (c0, c1, c2, c3, new_color, c5)
        else: nv = (c0, c1, c2, c3, c4, new_color)
        
        is_bad = _count_distinct_6(*nv) < 5
        
        if was_bad and not is_bad:
            delta -= 1
        elif not was_bad and is_bad:
            delta += 1
    
    return delta

def construct():
    deadline = time.time() + 140  # safety margin (150s limit)
    
    # Start with the known valid 17-coloring
    base_colors = np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
    
    best_colors = base_colors.copy()
    best_score = _count_colors_used(best_colors)  # 17
    
    # Try to find valid colorings with fewer colors
    for target_colors in [15, 14, 13]:
        if time.time() > deadline - 20:
            break
        
        rng = random.Random(target_colors * 1000 + 42)
        
        # Multiple random initializations
        for init_iter in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            c = np.array([rng.randint(0, target_colors - 1) for _ in range(_NUM_EDGES)], dtype=np.int32)
            
            bad = _count_bad_k4s(c)
            bad_flags = np.zeros(_NUM_K4S, dtype=bool)
            for k4_idx in range(_NUM_K4S):
                eidxs = _K4_EDGE_IDXS[k4_idx]
                c0, c1, c2, c3, c4, c5 = (c[eidxs[p]] for p in range(6))
                bad_flags[k4_idx] = (_count_distinct_6(c0, c1, c2, c3, c4, c5) < 5)
            
            # Local search with simulated annealing
            max_iters = 200000
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if bad == 0:
                    actual_colors = _count_colors_used(c)
                    if actual_colors < best_score:
                        best_score = actual_colors
                        best_colors = c.copy()
                    break
                
                # Pick a random edge
                edge_idx = rng.randint(0, _NUM_EDGES - 1)
                new_color = rng.randint(0, target_colors - 1)
                
                delta = _delta_bad_for_edge_change(c, edge_idx, new_color, bad_flags)
                
                # Simulated annealing acceptance
                if delta <= 0 or rng.random() < 0.05:
                    old_color = c[edge_idx]
                    c[edge_idx] = new_color
                    bad += delta
                    # Update bad_flags
                    for k4_idx, pos in _EDGE_K4_POSITIONS[edge_idx]:
                        eidxs = _K4_EDGE_IDXS[k4_idx]
                        c0, c1, c2, c3, c4, c5 = (c[eidxs[p]] for p in range(6))
                        bad_flags[k4_idx] = (_count_distinct_6(c0, c1, c2, c3, c4, c5) < 5)
            
            if bad == 0:
                actual_colors = _count_colors_used(c)
                if actual_colors < best_score:
                    best_score = actual_colors
                    best_colors = c.copy()
                break  # Found valid for this target, move to next
    
    # Final check: ensure best_colors is valid
    if _count_bad_k4s(best_colors) > 0:
        # Fallback to the original 17-coloring
        return base_colors.tolist()
    
    return best_colors.tolist()
# EVOLVE-BLOCK-END
