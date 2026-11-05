"""
R_4(C6): maximize n with a 4-colouring of K_n with no monochromatic C6.
Combines C6-counting local search (current) with cycle-breaking repair (inspiration).
"""
import time
import random

DEADLINE_MARGIN = 5.0  # seconds before the 150s wall-clock kill to stop


def _build_adj(colors, n):
    """Build adjacency bitmasks for each color class from a flat color list."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj


def _has_c6_bitset(adj, n):
    """Check if the graph defined by adjacency masks 'adj' contains a 6-cycle."""
    for v0 in range(n):
        mask_v1 = adj[v0]
        while mask_v1:
            v1 = (mask_v1 & -mask_v1).bit_length() - 1
            mask_v1 &= mask_v1 - 1

            mask_v2 = adj[v1] & ~(1 << v0)
            while mask_v2:
                v2 = (mask_v2 & -mask_v2).bit_length() - 1
                mask_v2 &= mask_v2 - 1

                mask_v3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                while mask_v3:
                    v3 = (mask_v3 & -mask_v3).bit_length() - 1
                    mask_v3 &= mask_v3 - 1

                    mask_v4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while mask_v4:
                        v4 = (mask_v4 & -mask_v4).bit_length() - 1
                        mask_v4 &= mask_v4 - 1

                        mask_v5 = adj[v4] & adj[v0]
                        used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                        mask_v5 &= ~used_mask
                        if mask_v5:
                            return True
    return False


def _is_valid(colors, n):
    """Check if the coloring is valid (no mono C6)."""
    adj = _build_adj(colors, n)
    for c in range(4):
        if _has_c6_bitset(adj[c], n):
            return False
    return True


def _count_c6s(adj, n):
    """Count total number of monochromatic C6s across all color classes."""
    total = 0
    for c in range(4):
        for v0 in range(n):
            mask_v1 = adj[c][v0]
            while mask_v1:
                v1 = (mask_v1 & -mask_v1).bit_length() - 1
                mask_v1 &= mask_v1 - 1

                mask_v2 = adj[c][v1] & ~(1 << v0)
                while mask_v2:
                    v2 = (mask_v2 & -mask_v2).bit_length() - 1
                    mask_v2 &= mask_v2 - 1

                    mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                    while mask_v3:
                        v3 = (mask_v3 & -mask_v3).bit_length() - 1
                        mask_v3 &= mask_v3 - 1

                        mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while mask_v4:
                            v4 = (mask_v4 & -mask_v4).bit_length() - 1
                            mask_v4 &= mask_v4 - 1

                            mask_v5 = adj[c][v4] & adj[c][v0]
                            used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                            mask_v5 &= ~used_mask
                            if mask_v5:
                                total += 1
    return total


def _find_c6_in_color(adj, n):
    """Find one C6 in a specific color class. Returns (v0..v5) or None."""
    for v0 in range(n):
        mask_v1 = adj[v0]
        while mask_v1:
            v1 = (mask_v1 & -mask_v1).bit_length() - 1
            mask_v1 &= mask_v1 - 1

            mask_v2 = adj[v1] & ~(1 << v0)
            while mask_v2:
                v2 = (mask_v2 & -mask_v2).bit_length() - 1
                mask_v2 &= mask_v2 - 1

                mask_v3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                while mask_v3:
                    v3 = (mask_v3 & -mask_v3).bit_length() - 1
                    mask_v3 &= mask_v3 - 1

                    mask_v4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while mask_v4:
                        v4 = (mask_v4 & -mask_v4).bit_length() - 1
                        mask_v4 &= mask_v4 - 1

                        mask_v5 = adj[v4] & adj[v0]
                        used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                        mask_v5 &= ~used_mask
                        if mask_v5:
                            v5 = (mask_v5 & -mask_v5).bit_length() - 1
                            return (v0, v1, v2, v3, v4, v5)
    return None


def _edge_index(i, j, n):
    """Convert (i,j) with i<j to linear index in lexicographic order."""
    # Pairs before row i: sum_{k=0}^{i-1}(n-1-k)
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)


def _local_search_counting(colors, n, time_limit, seed=42):
    """
    Local search minimizing C6 count. Returns (colors, c6_count) or (None, None).
    """
    rng = random.Random(seed)
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2

    colors = colors[:]
    adj = _build_adj(colors, n)
    current_c6s = _count_c6s(adj, n)

    if current_c6s == 0:
        return colors, 0

    best_colors = colors[:]
    best_c6s = current_c6s

    for iteration in range(200000):
        if time.time() > deadline:
            break
        if best_c6s == 0:
            return best_colors, 0

        # Pick a random edge to flip
        edge_idx = rng.randint(0, num_edges - 1)
        old_color = colors[edge_idx]
        new_color = (old_color + 1 + rng.randint(0, 2)) % 4

        # Update adjacency incrementally
        # Find which edge this is
        # Rebuild is expensive; do incremental update
        # We need to find (i,j) for edge_idx
        # For speed, precompute edge list
        # Actually, let's just rebuild adj each time for correctness
        # But that's O(n^2) per iteration which is too slow for n=17 (136 edges)
        # Let's do incremental: we need to know which (i,j) corresponds to edge_idx
        
        # Precompute edge mapping once
        # This is getting complex; let's just rebuild adj for now but limit iterations
        colors[edge_idx] = new_color
        adj = _build_adj(colors, n)
        new_c6s = _count_c6s(adj, n)

        if new_c6s <= best_c6s:
            best_c6s = new_c6s
            best_colors = colors[:]
            current_c6s = new_c6s
        else:
            colors[edge_idx] = old_color
            adj = _build_adj(colors, n)
            current_c6s = _count_c6s(adj, n)

    if best_c6s == 0:
        return best_colors, 0
    return None, best_c6s


def _cycle_breaking_search(colors, n, time_limit, seed=42):
    """
    Cycle-breaking repair: find C6s and recolor edges to break them.
    Returns (colors, c6_count) or (None, None).
    """
    rng = random.Random(seed)
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2

    colors = colors[:]
    adj = _build_adj(colors, n)
    current_c6s = _count_c6s(adj, n)

    if current_c6s == 0:
        return colors, 0

    best_colors = colors[:]
    best_c6s = current_c6s

    for iteration in range(50000):
        if time.time() > deadline:
            break
        if best_c6s == 0:
            return best_colors, 0

        # Find a C6 in some color class
        found = False
        for c in range(4):
            cyc = _find_c6_in_color(adj[c], n)
            if cyc is not None:
                found = True
                # Try recoloring one of the 6 edges in the cycle
                edges_in_cycle = []
                for k in range(6):
                    a, b = cyc[k], cyc[(k + 1) % 6]
                    if a > b:
                        a, b = b, a
                    edges_in_cycle.append((a, b))

                success = False
                for (i, j) in edges_in_cycle:
                    idx = _edge_index(i, j, n)
                    old_c = colors[idx]
                    for new_c in range(4):
                        if new_c == old_c:
                            continue
                        colors[idx] = new_c
                        adj_new = _build_adj(colors, n)
                        new_c6s = _count_c6s(adj_new, n)
                        if new_c6s < current_c6s:
                            # Accept this move
                            adj = adj_new
                            current_c6s = new_c6s
                            if current_c6s < best_c6s:
                                best_c6s = current_c6s
                                best_colors = colors[:]
                            success = True
                            break
                        else:
                            colors[idx] = old_c
                    if success:
                        break
                break

        if not found:
            # No C6 found - valid!
            return colors, 0

    if best_c6s == 0:
        return best_colors, 0
    return None, best_c6s


def _hybrid_search(n, time_limit, seed=42):
    """
    Combine cycle-breaking and counting-based local search.
    Alternate between them, keeping the best result.
    """
    rng = random.Random(seed)
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2

    best_colors = None
    best_c6s = float('inf')

    # Try multiple random restarts
    max_restarts = 20
    time_per_restart = time_limit / max_restarts

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        colors = [rng.randint(0, 3) for _ in range(num_edges)]

        # Phase 1: Cycle breaking
        c1, c6_1 = _cycle_breaking_search(colors, n, time_per_restart * 0.6, seed=seed + restart)
        if c6_1 is not None and c6_1 < best_c6s:
            best_c6s = c6_1
            best_colors = c1[:] if c1 is not None else colors[:]
            if best_c6s == 0:
                return best_colors

        # Phase 2: Counting-based local search from the best state
        if c1 is not None:
            start_colors = c1
        else:
            start_colors = colors

        c2, c6_2 = _local_search_counting(start_colors, n, time_per_restart * 0.4, seed=seed + restart + 1000)
        if c6_2 is not None and c6_2 < best_c6s:
            best_c6s = c6_2
            best_colors = c2[:] if c2 is not None else start_colors[:]
            if best_c6s == 0:
                return best_colors

    if best_c6s == 0 and best_colors is not None:
        return best_colors
    return None


def _construct_known_17():
    """
    Find a valid 4-coloring of K_17 using hybrid search.
    Falls back to smaller n if not found.
    """
    # Try n=17 first with hybrid search
    result = _hybrid_search(17, time_limit=20.0, seed=42)
    if result is not None:
        return {'n': 17, 'colors': result}

    # Try n=16
    result = _hybrid_search(16, time_limit=15.0, seed=42)
    if result is not None:
        return {'n': 16, 'colors': result}

    # Try n=15
    result = _hybrid_search(15, time_limit=10.0, seed=42)
    if result is not None:
        return {'n': 15, 'colors': result}

    # Try n=14
    result = _hybrid_search(14, time_limit=10.0, seed=42)
    if result is not None:
        return {'n': 14, 'colors': result}

    # Try n=13
    result = _hybrid_search(13, time_limit=10.0, seed=42)
    if result is not None:
        return {'n': 13, 'colors': result}

    # Try n=12
    result = _hybrid_search(12, time_limit=10.0, seed=42)
    if result is not None:
        return {'n': 12, 'colors': result}

    # Try n=11
    result = _hybrid_search(11, time_limit=10.0, seed=42)
    if result is not None:
        return {'n': 11, 'colors': result}

    # Try n=10
    result = _hybrid_search(10, time_limit=10.0, seed=42)
    if result is not None:
        return {'n': 10, 'colors': result}

    # Try n=9
    result = _hybrid_search(9, time_limit=5.0, seed=42)
    if result is not None:
        return {'n': 9, 'colors': result}

    # Try n=8
    result = _hybrid_search(8, time_limit=5.0, seed=42)
    if result is not None:
        return {'n': 8, 'colors': result}

    # Try n=7
    result = _hybrid_search(7, time_limit=5.0, seed=42)
    if result is not None:
        return {'n': 7, 'colors': result}

    # Try n=6
    result = _hybrid_search(6, time_limit=5.0, seed=42)
    if result is not None:
        return {'n': 6, 'colors': result}

    # Try n=5
    result = _hybrid_search(5, time_limit=2.0, seed=42)
    if result is not None:
        return {'n': 5, 'colors': result}

    # n=4 always works
    rng = random.Random(42)
    colors = [rng.randint(0, 3) for _ in range(6)]
    return {'n': 4, 'colors': colors}


def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)

    # Get a valid baseline
    result = _construct_known_17()
    best_n = result['n']
    best_colors = result['colors']

    return {'n': best_n, 'colors': best_colors}
