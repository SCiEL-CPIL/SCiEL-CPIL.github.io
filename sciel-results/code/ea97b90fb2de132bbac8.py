"""Exhaustive search with bitmask acceleration + local search refinement."""
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    deadline = time.time() + 80.0
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']

    # Precompute adjacency as integer bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Fast popcount using bit manipulation
    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c

    # Precompute degrees for all single-vertex masks? No, we need induced subgraph degrees.
    # For a mask, degree of v in induced subgraph = popcount(adj[v] & mask)

    def distinct_degrees(mask):
        """Return (num_distinct, degree_set) for induced subgraph on mask."""
        degs = set()
        m = mask
        while m:
            v = (m & -m).bit_length() - 1
            degs.add(popcount(adj[v] & mask))
            m &= m - 1
        return len(degs), degs

    best_count = -1
    best_mask = (1 << minv) - 1  # default: first minv vertices

    # Phase 1: Exhaustive search over all masks with popcount >= minv
    # Total subsets of 17 with size >= 8: sum(C(17,k) for k=8..17) = 65536 - sum(C(17,k) for k=0..7)
    # C(17,0..7) = 1+17+136+680+2380+6188+12376+19448 = 41226
    # So 65536 - 41226 = 24310 subsets. Very manageable.
    # We iterate over all masks and check popcount.
    # Better: iterate by size descending, stop early if we find max possible.

    # Max possible distinct degrees for k vertices is k (all distinct).
    # So if we find a subset of size k with k distinct degrees, that's optimal for that size.
    # But we want max over all sizes >= minv.

    # Strategy: for each size from n down to minv, check all subsets of that size.
    # If we find one with `size` distinct degrees, we can stop (can't do better for that size).
    # But we still need to check smaller sizes... actually no, if size k has k distinct, that's the max for size k.
    # We want the global max. So if we find size=k with k distinct, the answer is at least k.
    # Can a smaller size have more distinct degrees? No, max distinct for size m is m.
    # So if we find size=k with k distinct, the global max is at least k, and no size < k can exceed k-1 < k.
    # Wait, size < k has max distinct = size < k. So if we find k distinct at size k, that's global max = k.
    # But could a larger size have more? We're going from largest down. If size n has n distinct, done.
    # If size n has n-1 distinct, size n-1 could have n-1 distinct (tie) but not more.
    # So the max possible is the size itself. Going from largest size down:
    #   For size s, max distinct = s.
    #   If we find s distinct at size s, that's the global max (since any smaller size can't exceed s-1... wait no).
    #   Actually smaller size m < s can have at most m < s distinct. So s distinct at size s is global max.
    #   If we don't find s distinct at size s, the best at size s is s-1 or less.
    #   Then we go to size s-1, where max is s-1. If we find s-1 distinct there, that could tie or beat.
    #   So we should track the best found and continue.

    # Let's just do full exhaustive search since 24310 subsets is tiny.
    # But to be safe with time, let's add a deadline check.

    vertices = list(range(n))
    checked = 0

    for size in range(n, minv - 1, -1):
        # Upper bound: size distinct degrees max. If best_count >= size, skip this size.
        if best_count >= size:
            continue
        for combo in itertools.combinations(vertices, size):
            if time.time() > deadline:
                break
            mask = 0
            for v in combo:
                mask |= (1 << v)
            c, _ = distinct_degrees(mask)
            checked += 1
            if c > best_count:
                best_count = c
                best_mask = mask
                # If we achieved size distinct degrees, this is the max for this size.
                # But continue to check other sizes (smaller ones can't exceed their size).
                # Actually if c == size, we can't do better at this size, but smaller sizes have smaller max.
                # So if c == size, global max is at least size. Smaller sizes max is < size.
                # So we can return immediately!
                if c == size:
                    # This is the theoretical max for this size, and no smaller size can beat it.
                    # But wait, we might have already found a larger size with more distinct.
                    # Since we go from largest size down, if we're at size s and found s distinct,
                    # any larger size was already checked and had < its size distinct (otherwise we'd have returned).
                    # So s is the global max.
                    best_mask = mask
                    best_count = c
                    # Convert to list
                    out = [v for v in range(n) if (mask >> v) & 1]
                    return [int(v) for v in out]
        if time.time() > deadline:
            break

    # Phase 2: If exhaustive didn't finish or we want to refine, do local search.
    # But given 24310 subsets, exhaustive should finish well within time.
    # Let's add a local search anyway for robustness.
    if time.time() < deadline:
        current_mask = best_mask
        current_count = best_count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding a vertex
            for v in range(n):
                if (current_mask >> v) & 1:
                    continue
                new_mask = current_mask | (1 << v)
                c, _ = distinct_degrees(new_mask)
                if c > current_count:
                    current_count = c
                    current_mask = new_mask
                    best_count = c
                    best_mask = new_mask
                    improved = True
                    break
            # Try removing a vertex (if size > minv)
            if not improved and popcount(current_mask) > minv:
                m = current_mask
                while m:
                    v = (m & -m).bit_length() - 1
                    new_mask = current_mask & ~(1 << v)
                    if popcount(new_mask) >= minv:
                        c, _ = distinct_degrees(new_mask)
                        if c > current_count:
                            current_count = c
                            current_mask = new_mask
                            best_count = c
                            best_mask = new_mask
                            improved = True
                            break
                    m &= m - 1
            # Try swapping
            if not improved:
                in_mask = ((1 << n) - 1) & ~current_mask
                out_mask = current_mask
                im = in_mask
                while im and not improved:
                    v_in = (im & -im).bit_length() - 1
                    om = out_mask
                    while om and not improved:
                        v_out = (om & -om).bit_length() - 1
                        new_mask = (current_mask & ~(1 << v_out)) | (1 << v_in)
                        c, _ = distinct_degrees(new_mask)
                        if c > current_count:
                            current_count = c
                            current_mask = new_mask
                            best_count = c
                            best_mask = new_mask
                            improved = True
                        om &= om - 1
                    im &= im - 1

    out = [v for v in range(n) if (best_mask >> v) & 1]
    return [int(v) for v in out]
