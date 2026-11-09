"""Edge-colourings of K16 where every K4 sees at least five colours.
Strategy: Vectorized NumPy simulated annealing + color-merging heuristic."""
import time
import random
import numpy as np

_START_TIME = time.time()
DEADLINE = 140.0  # safety margin before 150s kill

# Precompute all K4s and their edge indices
_K4S = []
for a in range(16):
    for b in range(a+1, 16):
        for c in range(b+1, 16):
            for d in range(c+1, 16):
                _K4S.append((a, b, c, d))

def _edge_index(i, j):
    return i * 15 - i * (i - 1) // 2 + j - i - 1

# K4 edge indices as numpy array: shape (1820, 6)
_K4_EDGE_IDXS = np.array([
    [_edge_index(a,b), _edge_index(a,c), _edge_index(a,d),
     _edge_index(b,c), _edge_index(b,d), _edge_index(c,d)]
    for (a,b,c,d) in _K4S
], dtype=np.int32)

# Also store K4 vertex sets for merging heuristic
_K4_VERTS = np.array(_K4S, dtype=np.int32)  # shape (1820, 4)

def _seed_coloring():
    """The (i+j) mod 17 coloring, uses 16 colors."""
    colors = np.zeros(120, dtype=np.int32)
    for i in range(16):
        for j in range(i+1, 16):
            idx = _edge_index(i, j)
            colors[idx] = (i + j) % 17
    return colors

def _distinct_counts(colors):
    """For each K4, compute number of distinct colors among its 6 edges.
    Returns array of shape (1820,) with values 1..6."""
    # Get colors for all K4 edges: shape (1820, 6)
    edge_colors = colors[_K4_EDGE_IDXS]  # (1820, 6)
    # Count distinct: for each row, count unique values
    # Trick: for 6 elements, distinct = 6 - (number of duplicates)
    # More efficient: sort and count transitions
    sorted_ec = np.sort(edge_colors, axis=1)  # (1820, 6)
    # Count changes: 1 + sum(sorted[i] != sorted[i-1])
    changes = np.sum(sorted_ec[:, 1:] != sorted_ec[:, :-1], axis=1) + 1
    return changes  # shape (1820,)

def _violations(colors):
    """Return number of K4s with < 5 distinct colors."""
    dc = _distinct_counts(colors)
    return int(np.sum(dc < 5))

def _is_valid(colors):
    return _violations(colors) == 0

def _try_merge_colors(colors, max_colors, rng):
    """Try to merge colors to reduce total color count.
    Two colors can be merged if no K4 has exactly 2 distinct colors
    (merging would reduce it to 1). Actually, we need to be more careful:
    merging colors A and B is safe if no K4 has exactly {A, B} as its color set
    (i.e., all 6 edges use only A and B, and both appear).
    More generally, merging is safe if for every K4, the number of distinct
    colors doesn't drop below 5.
    
    We try all pairs of used colors and merge if safe."""
    c = colors.copy()
    used = np.unique(c)
    if len(used) <= max_colors:
        return c
    
    # Try to merge colors greedily
    # For each pair of colors (a, b), check if merging is safe
    # Merging a->b: for each K4, if it has both a and b, the distinct count
    # might decrease by 1 (if a and b were the only two colors, or one of them
    # was unique in that K4).
    
    # Efficient check: for each K4, count how many edges have color a and how many have color b.
    # If a K4 has only colors {a, b} (no other colors), merging drops distinct from 2 to 1.
    # If a K4 has colors {a, b, c, d, e} (5 distinct), merging a and b drops to 4.
    # So we need: for every K4 that contains both a and b, it must have >= 6 distinct
    # colors (impossible since max is 6) OR >= 5 other distinct colors besides a and b.
    # Actually: if a K4 has distinct set S with |S| >= 5, and {a,b} ⊆ S,
    # after merging, distinct set is S\{a,b} ∪ {b} = S\{a} (if a≠b), so |S|-1.
    # We need |S|-1 >= 5, i.e., |S| >= 6.
    # So merging a and b is safe iff no K4 has exactly 5 or 6 distinct colors
    # that include both a and b... wait, that's too restrictive.
    # 
    # Let me reconsider: if a K4 has distinct colors {a, b, c, d, e} (5 colors),
    # and we merge a into b, the new distinct set is {b, c, d, e} (4 colors). VIOLATION.
    # If a K4 has distinct colors {a, b, c, d, e, f} (6 colors),
    # merging a into b gives {b, c, d, e, f} (5 colors). OK.
    # If a K4 has distinct colors {a, c, d, e, f} (5 colors, no b),
    # merging a into b gives {b, c, d, e, f} (5 colors). OK.
    # 
    # So: merging a into b is safe iff for every K4 that contains BOTH a and b,
    # the K4 has at least 6 distinct colors (i.e., all 6 edges have different colors
    # and both a and b appear).
    
    # Precompute: for each K4, the set of colors present
    edge_colors = c[_K4_EDGE_IDXS]  # (1820, 6)
    
    # For each pair of used colors, check safety
    merged = False
    for _ in range(50):  # try up to 50 merges
        if time.time() > DEADLINE:
            break
        used = sorted(np.unique(c))
        if len(used) <= max_colors:
            break
        
        # Try random pairs
        improved = False
        for _ in range(100):
            if time.time() > DEADLINE:
                break
            pair = rng.sample(used, 2)
            a, b = pair
            # Check: for each K4, does it contain both a and b?
            # If yes, does it have >= 6 distinct colors?
            has_a = np.any(edge_colors == a, axis=1)
            has_b = np.any(edge_colors == b, axis=1)
            both = has_a & has_b
            if not np.any(both):
                # Safe to merge!
                c[c == a] = b
                edge_colors = c[_K4_EDGE_IDXS]
                improved = True
                merged = True
                break
            else:
                # Check distinct counts for K4s that have both
                dc = _distinct_counts(c)
                if np.all(dc[both] >= 6):
                    c[c == a] = b
                    edge_colors = c[_K4_EDGE_IDXS]
                    improved = True
                    merged = True
                    break
        
        if not improved:
            break
    
    return c

def _simulated_annealing(colors, target_colors, rng, max_steps=50000):
    """Vectorized simulated annealing to find valid coloring with <= target_colors."""
    c = colors.copy()
    # Compress to target_colors
    used = np.unique(c)
    if len(used) > target_colors:
        # Randomly remap
        for u in used[target_colors:]:
            c[c == u] = rng.randint(0, target_colors - 1)
    
    current_viol = _violations(c)
    best_c = c.copy()
    best_viol = current_viol
    
    T = 10.0
    cooling = 0.9997
    
    for step in range(max_steps):
        if time.time() > DEADLINE:
            break
        
        # Pick a random edge and try a random new color
        e = rng.randint(0, 119)
        old_c = c[e]
        new_c = rng.randint(0, target_colors - 1)
        if new_c == old_c:
            continue
        
        c[e] = new_c
        new_viol = _violations(c)
        
        if new_viol <= current_viol:
            current_viol = new_viol
            if new_viol < best_viol:
                best_viol = new_viol
                best_c = c.copy()
        else:
            # Accept with probability exp(-delta/T)
            if rng.random() < np.exp(-(new_viol - current_viol) / T):
                current_viol = new_viol
            else:
                c[e] = old_c
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    if best_viol == 0:
        return best_c
    # Return best found even if not valid (caller checks)
    return best_c if best_viol < 100 else None

def _fast_local_search(colors, target_colors, rng, max_steps=100000):
    """Faster local search: pick violating K4, change one of its edges."""
    c = colors.copy()
    used = np.unique(c)
    if len(used) > target_colors:
        for u in sorted(used)[target_colors:]:
            c[c == u] = rng.randint(0, target_colors - 1)
    
    current_viol = _violations(c)
    best_c = c.copy()
    best_viol = current_viol
    stuck = 0
    
    for step in range(max_steps):
        if time.time() > DEADLINE:
            break
        if current_viol == 0:
            return c
        
        # Find a violating K4
        dc = _distinct_counts(c)
        viol_mask = dc < 5
        if not np.any(viol_mask):
            return c
        
        # Pick a random violating K4
        viol_indices = np.where(viol_mask)[0]
        k4_idx = viol_indices[rng.randint(0, len(viol_indices) - 1)]
        
        # Get the 6 edges of this K4
        edges = _K4_EDGE_IDXS[k4_idx]
        
        # Try changing each edge to each color
        improved = False
        for e in edges:
            old = c[e]
            for new_col in range(target_colors):
                if new_col == old:
                    continue
                c[e] = new_col
                v = _violations(c)
                if v < current_viol:
                    current_viol = v
                    improved = True
                    if v < best_viol:
                        best_viol = v
                        best_c = c.copy()
                    break
                c[e] = old
            if improved:
                break
        
        if not improved:
            stuck += 1
            if stuck > 50:
                # Perturb: change a random edge
                e = rng.randint(0, 119)
                c[e] = rng.randint(0, target_colors - 1)
                current_viol = _violations(c)
                stuck = 0
        else:
            stuck = 0
    
    if best_viol == 0:
        return best_c
    return None

def construct():
    """Return 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order.
    Every set of four vertices must span edges of at least five distinct colours.
    Minimize the number of colours used."""
    rng = random.Random(42)
    
    # Start with seed coloring (16 colors)
    best = _seed_coloring()
    best_colors = int(len(np.unique(best)))  # 16
    
    # Phase 1: Try to reduce via merging (exact, no search needed)
    for target in range(15, 12, -1):
        if time.time() > DEADLINE - 30:
            break
        merged = _try_merge_colors(best, target, rng)
        if merged is not None and _is_valid(merged):
            used = int(len(np.unique(merged)))
            if used < best_colors:
                best = merged
                best_colors = used
            if best_colors <= target:
                break
    
    # Phase 2: Vectorized simulated annealing
    for target in range(min(15, best_colors), 12, -1):
        if time.time() > DEADLINE - 20:
            break
        for trial in range(3):
            if time.time() > DEADLINE - 20:
                break
            result = _simulated_annealing(best, target, rng, max_steps=30000)
            if result is not None and _is_valid(result):
                used = int(len(np.unique(result)))
                if used < best_colors:
                    best = result
                    best_colors = used
                break
    
    # Phase 3: Fast local search focused on violating K4s
    for target in range(min(15, best_colors), 12, -1):
        if time.time() > DEADLINE - 10:
            break
        for trial in range(5):
            if time.time() > DEADLINE - 10:
                break
            result = _fast_local_search(best, target, rng, max_steps=80000)
            if result is not None:
                used = int(len(np.unique(result)))
                if used < best_colors:
                    best = result
                    best_colors = used
                break
    
    # Phase 4: Final merging attempt on best
    if time.time() < DEADLINE - 5:
        for target in range(best_colors - 1, 12, -1):
            if time.time() > DEADLINE - 5:
                break
            merged = _try_merge_colors(best, target, rng)
            if merged is not None and _is_valid(merged):
                used = int(len(np.unique(merged)))
                if used < best_colors:
                    best = merged
                    best_colors = used
    
    # Final: one more SA pass if time remains
    if best_colors > 13 and time.time() < DEADLINE - 5:
        result = _simulated_annealing(best, 13, rng, max_steps=50000)
        if result is not None and _is_valid(result):
            best = result
            best_colors = int(len(np.unique(best)))
    
    # Ensure colors are in 0..119
    result = np.minimum(best, 119).astype(int)
    return result.tolist()

# Precompute at module level
_K4S = []
for a in range(16):
    for b in range(a+1, 16):
        for c in range(b+1, 16):
            for d in range(c+1, 16):
                _K4S.append((a, b, c, d))

def _edge_index(i, j):
    return i * 15 - i * (i - 1) // 2 + j - i - 1

_K4_EDGE_IDXS = np.array([
    [_edge_index(a,b), _edge_index(a,c), _edge_index(a,d),
     _edge_index(b,c), _edge_index(b,d), _edge_index(c,d)]
    for (a,b,c,d) in _K4S
], dtype=np.int32)
