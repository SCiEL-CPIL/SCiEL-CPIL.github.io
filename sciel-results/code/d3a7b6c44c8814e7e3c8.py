"""Edge-colourings of K16 where every K4 sees at least five colours.
Minimise the number of colours used."""
import time
import random
import itertools

_START_TIME = time.time()
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _edge_index(i, j):
    """Map pair (i,j) with i<j to linear index 0..119 in lexicographic order."""
    # Number of edges before vertex i: i*(i-1)/2 ... actually for lex order (i,j):
    # count = sum_{a=0}^{i-1} (15-a) + (j - i - 1)
    # = i*15 - i*(i-1)//2 + j - i - 1
    return i * 15 - i * (i - 1) // 2 + j - i - 1

def _build_edge_list():
    """Return list of (i, j, idx) for all 120 edges in lex order."""
    edges = []
    idx = 0
    for i in range(16):
        for j in range(i + 1, 16):
            edges.append((i, j, idx))
            idx += 1
    return edges

def _all_k4s():
    """Return list of all 4-vertex subsets (as tuples of 4 vertices)."""
    return list(itertools.combinations(range(16), 4))

def _k4_edges(k4):
    """Return the 6 edge indices for a given K4 (4-tuple of vertices)."""
    a, b, c, d = k4
    return [
        _edge_index(a, b), _edge_index(a, c), _edge_index(a, d),
        _edge_index(b, c), _edge_index(b, d), _edge_index(c, d)
    ]

def _count_colors_used(colors):
    """Return the number of distinct colors used."""
    return len(set(colors))

def _is_valid(colors):
    """Check that every K4 spans at least 5 distinct colors."""
    for k4 in _K4S:
        edge_idxs = _K4_EDGE_IDXS[k4]
        cs = set(colors[e] for e in edge_idxs)
        if len(cs) < 5:
            return False
    return True

def _count_violations(colors):
    """Return the number of K4s that span fewer than 5 colors."""
    violations = 0
    for k4 in _K4S:
        edge_idxs = _K4_EDGE_IDXS[k4]
        cs = set(colors[e] for e in edge_idxs)
        if len(cs) < 5:
            violations += 1
    return violations

# Precompute K4s and their edge indices
_EDGES = _build_edge_list()
_K4S = _all_k4s()
_K4_EDGE_IDXS = {k4: _k4_edges(k4) for k4 in _K4S}

def _seed_coloring():
    """The (i+j) mod 17 coloring, which uses 16 colors."""
    colors = [0] * 120
    for i in range(16):
        for j in range(i + 1, 16):
            idx = _edge_index(i, j)
            colors[idx] = (i + j) % 17
    return colors

def _try_reduce_colors(colors, max_colors, rng):
    """Try to recolor edges to use at most max_colors colors.
    Returns a valid coloring using <= max_colors, or None."""
    # Start from colors, but we want to constrain to max_colors
    # Greedy: for each edge, assign a color in [0, max_colors) that doesn't break validity
    # We'll use local search: randomly reassign colors and check validity
    
    n = 120
    c = list(colors)
    # First, map all colors to [0, max_colors)
    for i in range(n):
        c[i] = c[i] % max_colors
    
    # Local search to fix violations
    for step in range(5000):
        if time.time() > DEADLINE:
            return None
        viol = _count_violations(c)
        if viol == 0:
            return c
        # Find a violating K4 and change one of its edges
        # Pick a random edge and try all colors
        for attempt in range(50):
            e = rng.randint(0, n - 1)
            old = c[e]
            # Try each color
            best_c = None
            best_viol = viol
            for new_c in range(max_colors):
                if new_c == old:
                    continue
                c[e] = new_c
                v = _count_violations(c)
                if v < best_viol:
                    best_viol = v
                    best_c = new_c
                c[e] = old
            if best_c is not None:
                c[e] = best_c
                break
        else:
            # Random perturbation
            e = rng.randint(0, n - 1)
            c[e] = rng.randint(0, max_colors - 1)
    return None if _count_violations(c) > 0 else c

def _local_search_reduce(colors, target_colors, rng, max_steps=20000):
    """Local search to reduce color count to target_colors."""
    n = 120
    c = list(colors)
    # First compress to target_colors
    color_map = {}
    next_c = 0
    for i in range(n):
        if c[i] not in color_map:
            if next_c < target_colors:
                color_map[c[i]] = next_c
                next_c += 1
            else:
                color_map[c[i]] = rng.randint(0, target_colors - 1)
        c[i] = color_map[c[i]]
    
    for step in range(max_steps):
        if time.time() > DEADLINE:
            break
        viol = _count_violations(c)
        if viol == 0:
            return c
        # Pick a random edge, try all target_colors
        improved = False
        for attempt in range(20):
            e = rng.randint(0, n - 1)
            old = c[e]
            best_c = None
            best_viol = viol
            for new_c in range(target_colors):
                if new_c == old:
                    continue
                c[e] = new_c
                v = _count_violations(c)
                if v < best_viol:
                    best_viol = v
                    best_c = new_c
                c[e] = old
            if best_c is not None:
                c[e] = best_c
                improved = True
                break
        if not improved:
            # Perturb
            e = rng.randint(0, n - 1)
            c[e] = rng.randint(0, target_colors - 1)
    return c if _count_violations(c) == 0 else None

def _greedy_construct(k, rng):
    """Try to construct a valid coloring using exactly k colors via greedy + repair."""
    n = 120
    c = [0] * n
    # Assign edges in random order, picking a color that keeps validity so far
    order = list(range(n))
    rng.shuffle(order)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = {e: [] for e in range(n)}
    for k4 in _K4S:
        for e in _K4_EDGE_IDXS[k4]:
            edge_to_k4s[e].append(k4)
    
    for e in order:
        if time.time() > DEADLINE:
            return None
        # Try colors 0..k-1, pick one that doesn't create a violation in any K4 containing e
        valid_colors = []
        for col in range(k):
            c[e] = col
            ok = True
            for k4 in edge_to_k4s[e]:
                # Check if this K4 now has < 5 colors
                edge_idxs = _K4_EDGE_IDXS[k4]
                # Only check K4s where all edges are assigned (or just check current)
                # For simplicity, just check if this assignment causes a problem
                # Actually we need to check all 6 edges of the K4
                cs = set()
                all_assigned = True
                for ee in edge_idxs:
                    if c[ee] < 0:
                        all_assigned = False
                        break
                    cs.add(c[ee])
                if all_assigned and len(cs) < 5:
                    ok = False
                    break
            if ok:
                valid_colors.append(col)
        if valid_colors:
            c[e] = rng.choice(valid_colors)
        else:
            c[e] = rng.randint(0, k - 1)
    
    return c if _is_valid(c) else None

def construct():
    """Return 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order.
    Every set of four vertices must span edges of at least five distinct colours.
    Minimize the number of colours used."""
    rng = random.Random(42)
    
    # Start with seed coloring (16 colors)
    best = _seed_coloring()
    best_colors = _count_colors_used(best)  # 16
    
    # Try to reduce to 15, 14, 13 colors
    for target in range(15, 12, -1):
        if time.time() > DEADLINE - 10:
            break
        # Try local search reduction
        result = _local_search_reduce(best, target, rng, max_steps=15000)
        if result is not None:
            used = _count_colors_used(result)
            if used < best_colors:
                best = result
                best_colors = used
            if best_colors == target:
                break
    
    # Try greedy construction with fewer colors
    for target in range(15, 12, -1):
        if time.time() > DEADLINE - 5:
            break
        for trial in range(5):
            if time.time() > DEADLINE - 5:
                break
            result = _greedy_construct(target, rng)
            if result is not None:
                used = _count_colors_used(result)
                if used < best_colors:
                    best = result
                    best_colors = used
                break
    
    # Final attempt: try to reduce further with more aggressive search
    if best_colors > 13 and time.time() < DEADLINE - 10:
        for target in range(best_colors - 1, 12, -1):
            if time.time() > DEADLINE - 5:
                break
            result = _local_search_reduce(best, target, rng, max_steps=30000)
            if result is not None:
                used = _count_colors_used(result)
                if used < best_colors:
                    best = result
                    best_colors = used
    
    # Ensure colors are in 0..119 (they should be, but just in case)
    result = [min(c, 119) for c in best]
    return result

# Precompute at module level for speed
_EDGES = _build_edge_list()
_K4S = _all_k4s()
_K4_EDGE_IDXS = {k4: _k4_edges(k4) for k4 in _K4S}
