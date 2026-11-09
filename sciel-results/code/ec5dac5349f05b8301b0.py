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

# Precompute for each K4, the 6 edge indices
_K4_EDGE_IDXS = []
for edges in _K4_EDGES:
    _K4_EDGE_IDXS.append([_edge_idx(i, j) for (i, j) in edges])

_NUM_K4S = len(_K4S)
_NUM_EDGES = _N * (_N - 1) // 2  # 120

# Precompute for each K4 and each position in that K4, the other 5 edge indices
_K4_OTHER_EDGES = []
for k4_idx in range(_NUM_K4S):
    eidxs = _K4_EDGE_IDXS[k4_idx]
    others = []
    for pos in range(6):
        others.append(np.array([eidxs[p] for p in range(6) if p != pos], dtype=np.int32))
    _K4_OTHER_EDGES.append(others)

# Precompute for each edge_idx, the list of (k4_idx, pos) pairs
_EDGE_TO_K4_POS = [[] for _ in range(_NUM_EDGES)]
for k4_idx in range(_NUM_K4S):
    eidxs = _K4_EDGE_IDXS[k4_idx]
    for pos in range(6):
        _EDGE_TO_K4_POS[eidxs[pos]].append((k4_idx, pos))

def _distinct_count6(c0, c1, c2, c3, c4, c5):
    d = 1
    if c1 != c0: d += 1
    if c2 != c0 and c2 != c1: d += 1
    if c3 != c0 and c3 != c1 and c3 != c2: d += 1
    if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: d += 1
    if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: d += 1
    return d

def _count_bad_k4s(colors):
    bad = 0
    for k4_idx in range(_NUM_K4S):
        eidxs = _K4_EDGE_IDXS[k4_idx]
        if _distinct_count6(colors[eidxs[0]], colors[eidxs[1]], colors[eidxs[2]],
                           colors[eidxs[3]], colors[eidxs[4]], colors[eidxs[5]]) < 5:
            bad += 1
    return bad

def _count_colors_used(colors):
    return len(set(colors.tolist()))

def _compute_delta_bad(c, edge_idx, new_color):
    delta_bad = 0
    for k4_idx, pos in _EDGE_TO_K4_POS[edge_idx]:
        eidxs = _K4_EDGE_IDXS[k4_idx]
        others = _K4_OTHER_EDGES[k4_idx][pos]
        
        d_before = _distinct_count6(c[eidxs[0]], c[eidxs[1]], c[eidxs[2]],
                                    c[eidxs[3]], c[eidxs[4]], c[eidxs[5]])
        bad_before = 1 if d_before < 5 else 0
        
        if pos == 0:
            d_after = _distinct_count6(new_color, c[others[0]], c[others[1]], c[others[2]], c[others[3]], c[others[4]])
        elif pos == 1:
            d_after = _distinct_count6(c[others[0]], new_color, c[others[1]], c[others[2]], c[others[3]], c[others[4]])
        elif pos == 2:
            d_after = _distinct_count6(c[others[0]], c[others[1]], new_color, c[others[2]], c[others[3]], c[others[4]])
        elif pos == 3:
            d_after = _distinct_count6(c[others[0]], c[others[1]], c[others[2]], new_color, c[others[3]], c[others[4]])
        elif pos == 4:
            d_after = _distinct_count6(c[others[0]], c[others[1]], c[others[2]], c[others[3]], new_color, c[others[4]])
        else:
            d_after = _distinct_count6(c[others[0]], c[others[1]], c[others[2]], c[others[3]], c[others[4]], new_color)
        
        bad_after = 1 if d_after < 5 else 0
        delta_bad += bad_after - bad_before
    
    return delta_bad

def _anneal(c, bad, rng, deadline, target_colors, t_start, cooling_rate):
    temperature = t_start
    iters_since_improve = 0
    best_bad_local = bad
    
    while time.time() < deadline:
        if bad == 0:
            return c, bad, True
        
        edge_idx = rng.randint(0, _NUM_EDGES - 1)
        old_color = c[edge_idx]
        
        new_color = rng.randint(0, target_colors - 1)
        if new_color == old_color:
            continue
        
        delta_bad = _compute_delta_bad(c, edge_idx, new_color)
        
        if delta_bad <= 0:
            c[edge_idx] = new_color
            bad += delta_bad
            if bad < best_bad_local:
                best_bad_local = bad
                iters_since_improve = 0
            else:
                iters_since_improve += 1
        else:
            if rng.random() < np.exp(-delta_bad / temperature):
                c[edge_idx] = new_color
                bad += delta_bad
                iters_since_improve += 1
            else:
                iters_since_improve += 1
        
        temperature *= cooling_rate
        if temperature < 0.01:
            break
    
    return c, bad, (bad == 0)

def _polish_colors(c, bad, rng, deadline):
    if bad != 0:
        return c, bad
        
    # Maintain a count of colors to quickly check if a color becomes unused
    color_counts = np.zeros(120, dtype=np.int32)
    for col in c:
        color_counts[col] += 1
        
    active_colors = set(c.tolist())
    
    while time.time() < deadline:
        if len(active_colors) <= 1:
            break
            
        edge_idx = rng.randint(0, _NUM_EDGES - 1)
        old_color = c[edge_idx]
        
        # Pick a random other active color
        other_colors = [col for col in active_colors if col != old_color]
        if not other_colors:
            continue
        new_color = rng.choice(other_colors)
        
        delta_bad = _compute_delta_bad(c, edge_idx, new_color)
        
        if delta_bad == 0:
            # Update counts
            color_counts[old_color] -= 1
            if color_counts[old_color] == 0:
                active_colors.remove(old_color)
            color_counts[new_color] += 1
            
            c[edge_idx] = new_color
    
    return c, bad

def construct():
    deadline = time.time() + 145
    
    # Vectorized init for base 17-coloring
    i_arr = np.repeat(np.arange(_N), _N - 1 - np.arange(_N))
    j_arr = np.empty(_NUM_EDGES, dtype=np.int32)
    for k in range(_N):
        start = sum(_N - 1 - np.arange(_N)[:k])
        end = start + (_N - 1 - k)
        j_arr[start:end] = np.arange(k + 1, _N)
        
    base_colors = ((i_arr + j_arr) % 17).astype(np.int32)
    
    best_colors = base_colors.copy()
    best_score = _count_colors_used(best_colors)  # 17
    
    rng = random.Random(42)
    
    for target_colors in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
            
        found_valid = False
        
        # Restart loop
        max_restarts = 500
        for init_iter in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            if init_iter % 3 == 0:
                c = rng.randint(0, target_colors - 1, _NUM_EDGES).astype(np.int32)
            elif init_iter % 3 == 1:
                c = base_colors.copy()
                color_map = np.array([rng.randint(0, target_colors - 1) for _ in range(17)], dtype=np.int32)
                c = color_map[c]
            else:
                c = base_colors.copy()
                perm = rng.sample(range(_NUM_EDGES), 20)
                new_vals = rng.randint(0, target_colors - 1, 20).astype(np.int32)
                c[perm] = new_vals
            
            bad = _count_bad_k4s(c)
            
            t_start = 0.5 + rng.random() * 0.5
            cooling_rate = 0.9995
            restart_deadline = time.time() + 2.0
            
            c, bad, converged = _anneal(c, bad, rng, restart_deadline, target_colors, t_start, cooling_rate)
            
            if bad == 0:
                polish_deadline = time.time() + 5.0
                c, bad = _polish_colors(c, bad, rng, polish_deadline)
                
                actual_colors = _count_colors_used(c)
                if actual_colors < best_score:
                    best_score = actual_colors
                    best_colors = c.copy()
                    
                found_valid = True
                break
                
    if _count_bad_k4s(best_colors) > 0:
        return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]
    
    return best_colors.tolist()
# EVOLVE-BLOCK-END
