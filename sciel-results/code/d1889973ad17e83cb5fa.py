"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # All 4-element subsets of {0..10}: C(11,4) = 330
    all_subsets = list(itertools.combinations(range(D), 4))
    n_subsets = len(all_subsets)

    # Two-sign patterns for even parity (4 coords, ±1 each, even number of -1s)
    # There are 8 even-parity sign patterns out of 16 total
    all_signs = list(itertools.product([1, -1], repeat=4))
    even_signs = [s for s in all_signs if sum(1 for x in s if x == -1) % 2 == 0]
    # For maximum packing, use ALL 16 sign patterns
    all_16_signs = list(itertools.product([1, -1], repeat=4))

    # Precompute bitmask for each subset (for intersection testing)
    subset_masks = [0] * n_subsets
    for idx, sub in enumerate(all_subsets):
        m = 0
        for coord in sub:
            m |= (1 << coord)
        subset_masks[idx] = m

    # Precompute intersection table: intersects[i][j] = True if subsets i,j share >= 2 coords
    # This is a 330x330 boolean matrix - 108900 entries, fine
    intersects = np.zeros((n_subsets, n_subsets), dtype=bool)
    for i in range(n_subsets):
        for j in range(i + 1, n_subsets):
            common = subset_masks[i] & subset_masks[j]
            # Count bits
            c = bin(common).count('1')
            if c >= 2:
                intersects[i][j] = True
                intersects[j][i] = True

    # Strategy: greedy packing of subsets with iterative edge-swapping
    # We want to select as many subsets as possible such that no two selected
    # subsets share >= 2 coordinates. Then assign sign patterns.

    # For each subset, we can place up to 16 vectors (all sign patterns)
    # But we need pairwise distance >= max norm among all vectors
    # With norm 2 (each vector has 4 entries of ±1), max norm = 2
    # Distance between two vectors: ||v1 - v2|| >= 2
    # If they share the same subset but different signs: distance^2 = sum of (diff)^2 over 4 coords
    #   If signs differ in k positions: distance^2 = 4k, so distance >= 2 requires k >= 1
    #   Adjacent sign patterns (differ in 1 coord): distance^2 = 4, distance = 2 ✓
    # If they share different subsets:
    #   If subsets are disjoint (0 common): distance^2 = 16, fine
    #   If 1 common coord: distance^2 = 1 + 9 + 16 = depends on signs at common coord
    #     Actually: at common coord, values can be same (diff=0) or different (diff=2)
    #     At 3 non-common coords of v1: each contributes 1 (value ±1 vs 0)
    #     At 3 non-common coords of v2: each contributes 1
    #     So distance^2 = (common_diff)^2 + 3 + 3 where common_diff is 0 or 2
    #     Minimum: 0 + 6 = 6, max: 4 + 6 = 10
    #     distance = sqrt(6) ≈ 2.45 > 2 ✓ always
    #   If 2 common coords: need to check
    #     distance^2 = sum over 2 common coords of (v1_i - v2_i)^2 + 2*1 + 2*1
    #     = (d1)^2 + (d2)^2 + 4 where d1,d2 ∈ {0, 2}
    #     Minimum: 0+0+4 = 4, distance = 2 ✓ (borderline)
    #     So if 2 common coords AND both signs match at those coords, distance = 2 ✓
    #   If 3 common coords (not possible for 4-subsets unless identical, handled above)
    #   If 4 common coords: same subset, already handled

    # So the constraint is: no two selected subsets may share >= 2 coordinates
    # (because with 2 shared coords, worst case distance is exactly 2, which is OK)
    # Wait, let me recheck: with 2 shared coords, minimum distance^2 = 4, distance = 2 = max_norm ✓
    # So sharing exactly 2 coords is FINE.
    # Sharing >= 3 coords: for 4-element subsets, that means 3 or 4 shared.
    #   4 shared = same subset → handled by sign patterns
    #   3 shared: distance^2 = 3*(0 or 4) + 1 + 0 = depends
    #     At 3 common coords: each diff is 0 or 2, so contributes 0 or 4
    #     At 1 non-common coord of each: contributes 1 each
    #     distance^2 = sum_common + 1 + 1 = sum_common + 2
    #     Minimum sum_common = 0 (all signs match): distance^2 = 2, distance ≈ 1.41 < 2 ✗
    # So subsets sharing 3 coords are NOT allowed.
    # Subsets sharing 2 coords: minimum distance^2 = 4, distance = 2 ✓ OK
    # Subsets sharing 1 or 0 coords: always OK

    # Therefore constraint: no two selected 4-subsets may share 3 or more coordinates.
    # Equivalent: pairwise intersection size <= 2.

    # Let's fix the intersects table: mark True if intersection >= 3
    # (we need to FORBID pairs with intersection >= 3)
    forbidden = np.zeros((n_subsets, n_subsets), dtype=bool)
    for i in range(n_subsets):
        for j in range(i + 1, n_subsets):
            common = subset_masks[i] & subset_masks[j]
            c = bin(common).count('1')
            if c >= 3:
                forbidden[i][j] = True
                forbidden[j][i] = True

    # Greedy + local search to maximize number of selected subsets
    # Upper bound: each coordinate can appear in at most... let's think
    # If coord x is in k selected subsets, then those k subsets must have pairwise intersection <= 2
    # which is automatically satisfied since they all contain x (intersection includes x)
    # But two subsets both containing x can share at most 2 other coords total... no
    # They share x plus up to 2 others = 3 total, which is forbidden
    # So if coord x is in subsets A and B, |A ∩ B| <= 2 means they share x and at most 1 other
    # This is a constraint on the packing

    # Greedy: try to pack as many subsets as possible
    best_count = 0
    best_selection = set()

    # Multiple random restarts
    rng = np.random.default_rng(42)

    for restart in range(200):
        if time.time() > deadline - 10:
            break

        # Random order
        order = list(range(n_subsets))
        rng.shuffle(order)

        selected = []
        selected_set = set()

        for idx in order:
            # Check against all already selected
            compatible = True
            for s in selected:
                if forbidden[idx, s]:
                    compatible = False
                    break
            if compatible:
                selected.append(idx)
                selected_set.add(idx)

        count = len(selected)
        if count > best_count:
            best_count = count
            best_selection = set(selected)

    # Local search: try to add more by removing conflicting ones
    if best_selection:
        if time.time() < deadline - 20:
            # Try to add unselected subsets by removing one conflicting
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                current = list(best_selection)
                remaining = [i for i in range(n_subsets) if i not in best_selection]

                for add_idx in remaining:
                    if time.time() > deadline - 10:
                        break
                    # Find conflicts
                    conflicts = [s for s in current if forbidden[add_idx, s]]
                    if len(conflicts) <= 1:
                        # Remove at most 1 conflict, add new
                        new_set = set(current)
                        for c in conflicts:
                            new_set.discard(c)
                        new_set.add(add_idx)
                        if len(new_set) > len(best_selection):
                            best_selection = new_set
                            improved = True
                            current = list(new_set)
                            remaining = [i for i in range(n_subsets) if i not in best_selection]
                            break

    best_count = len(best_selection)

    # Now build vectors: for each selected subset, use all 16 sign patterns
    vectors = []
    for idx in best_selection:
        sub = all_subsets[idx]
        for signs in all_16_signs:
            v = [0] * D
            for k, coord in enumerate(sub):
                v[coord] = signs[k]
            vectors.append(v)

    if not vectors:
        # Fallback: return seed-style 2-coordinate vectors
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)

    result = np.array(vectors, dtype=np.int64)

    # Verify: minimum pairwise distance >= maximum norm
    # For safety, do a quick check on a sample if time permits
    if time.time() < deadline - 2 and len(result) > 1:
        norms_sq = np.sum(result * result, axis=1)
        max_norm = np.sqrt(np.max(norms_sq))
        # Check a random sample of pairs
        n = len(result)
        sample_size = min(5000, n * (n - 1) // 2)
        idx_pairs = np.random.default_rng(0).choice(n, size=(sample_size, 2), replace=False)
        idx_pairs = idx_pairs[idx_pairs[:, 0] < idx_pairs[:, 1]]
        if len(idx_pairs) > 0:
            diffs = result[idx_pairs[:, 0]] - result[idx_pairs[:, 1]]
            dists_sq = np.sum(diffs * diffs, axis=1)
            min_dist = np.sqrt(np.min(dists_sq))
            if min_dist < max_norm - 1e-9:
                # Invalid! Fall back to safe construction
                vectors = []
                for i, j in itertools.combinations(range(D), 2):
                    for si in (1, -1):
                        for sj in (1, -1):
                            v = [0] * D
                            v[i], v[j] = si, sj
                            vectors.append(v)
                result = np.array(vectors, dtype=np.int64)

    return result
# EVOLVE-BLOCK-END
