import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    d = 11
    t_start = time.time()
    DEADLINE = 625.0  # safety margin

    # ---- Weight-6 vectors: 6-subsets of {0..10} with all signs ----
    # For two weight-6 vectors u, v:
    #   ||u - v||^2 = 12 - 2*(u·v)
    #   u·v = (# shared positions with same sign) - (# shared positions with opposite sign)
    #   Let s = # shared positions. Then u·v = (s - o) where o = # opposite-sign overlaps.
    #   Actually: if s positions overlap, u·v = (number matching sign) - (number opposite sign)
    #   = (s - 2 * opposite_count). So u·v = s - 2*opp.
    #   ||u-v||^2 = 12 - 2(s - 2*opp) = 12 - 2s + 4*opp.
    #   We need ||u-v||^2 >= 12 (max norm sq = 12 since all have norm^2 = 6).
    #   12 - 2s + 4*opp >= 12  =>  -2s + 4*opp >= 0  =>  2*opp >= s  =>  opp >= s/2.
    #
    # So for each pair of 6-subsets, with s = |intersection|, we need at least ceil(s/2)
    # of the overlapping positions to have opposite signs.
    #
    # This is a constraint on the sign assignment given the supports.
    #
    # Strategy:
    # 1. Find a large collection of 6-subsets of {0,...,10} (call it C).
    # 2. For each subset in C, assign a sign vector (each coordinate in {-1,1}) such that
    #    for every pair (A, B) in C with intersection s, the number of opposite signs on
    #    A ∩ B is at least ceil(s/2).
    # 3. The total number of points is |C| (one sign assignment per support).
    #    Actually, we can also include multiple sign assignments per support if they're
    #    mutually compatible. But let's first try one per support for simplicity.
    #    Actually, if we have one sign assignment per support, k = |C|.
    #    But we could potentially get more by having multiple compatible sign assignments.
    #    Let's first maximize |C|, then consider sign multiplicity.
    #
    # For step 1: We need 6-subsets with the property that for any two, the sign constraint
    # is satisfiable. The constraint depends only on the intersection size s:
    #   - s = 0: no constraint (no overlap)
    #   - s = 1: need opp >= 1, i.e., the one shared coordinate must have opposite signs.
    #   - s = 2: need opp >= 1, i.e., at least 1 of 2 shared coords has opposite signs.
    #   - s = 3: need opp >= 2, i.e., at least 2 of 3 shared coords have opposite signs.
    #   - s = 4: need opp >= 2, i.e., at least 2 of 4 shared coords have opposite signs.
    #   - s = 5: need opp >= 3, i.e., at least 3 of 5 shared coords have opposite signs.
    #   - s = 6: same subset, but we only have one sign per subset, so N/A.
    #
    # The sign assignment is a CSP. Let's use backtracking.
    #
    # First, let's find a good set of 6-subsets.
    # Total 6-subsets of 11: C(11,6) = 462.
    # We want a large subset where the sign CSP is solvable.
    #
    # A key insight: if we restrict to 6-subsets with pairwise intersection <= 3,
    # then the constraint is: for s <= 3, need opp >= ceil(s/2).
    #   s=0: always ok
    #   s=1: need opposite sign on that coordinate
    #   s=2: need at least 1 opposite
    #   s=3: need at least 2 opposite
    #
    # Max number of 6-subsets of 11 with pairwise intersection <= 3:
    # This is a packing problem. Let's just try to find a large set via greedy/backtracking.
    #
    # Let me think about this differently. Let's use the complement: 6-subsets of 11
    # correspond to 5-subsets (complements). Intersection of 6-subsets A, B:
    #   |A ∩ B| = 11 - |A^c ∪ B^c| = 11 - (5 + 5 - |A^c ∩ B^c|) = 1 + |A^c ∩ B^c|.
    # So |A ∩ B| <= 3  <=>  |A^c ∩ B^c| <= 2.
    # So we need 5-subsets with pairwise intersection <= 2.
    # This is equivalent to finding a 3-(11,5,?) design or just a large set.
    #
    # Let's use a backtracking approach to find a large set of 5-subsets with pairwise
    # intersection <= 2, then solve the sign CSP.

    # Generate all 5-subsets of range(11)
    all_5subsets = list(itertools.combinations(range(11), 5))
    random.shuffle(all_5subsets)
    # Convert to 6-subsets (complements)
    all_6subsets = [tuple(sorted(set(range(11)) - set(s))) for s in all_5subsets]

    # Greedily build a set of 6-subsets with pairwise intersection <= 3
    # (i.e., complement 5-subsets have pairwise intersection <= 2)
    selected_5 = []
    selected_6 = []
    for s5 in all_5subsets:
        ok = True
        for t5 in selected_5:
            inter = len(set(s5) & set(t5))
            if inter > 2:
                ok = False
                break
        if ok:
            selected_5.append(s5)
            selected_6.append(tuple(sorted(set(range(11)) - set(s5))))
            if time.time() - t_start > DEADLINE - 120:
                break

    # Now solve the sign CSP for the selected 6-subsets.
    # Each subset gets a sign vector of length 6 (for its 6 coordinates).
    # For a pair (A, B) with intersection s, we need >= ceil(s/2) opposite signs on A∩B.

    n_sel = len(selected_6)
    if n_sel < 2:
        # Fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    # Precompute intersections
    intersections = []
    for i in range(n_sel):
        row = []
        for j in range(i+1, n_sel):
            inter = set(selected_6[i]) & set(selected_6[j])
            row.append((j, tuple(inter)))
        intersections.append(row)

    # Backtracking with forward checking
    # State: signs[i] = sign vector for subset i, or None if unassigned
    # Each sign vector is a tuple of 6 values in {-1, 1}
    # We assign subsets one by one. When assigning subset i, we check against all
    # previously assigned subsets j < i.

    # For speed, precompute for each pair (i, j) the list of shared coordinates
    # (as indices within the 6-element subsets)
    # Actually, let's just do it directly.

    def check_pair(signs_i, subset_i, signs_j, subset_j):
        """Check if two sign assignments are compatible."""
        si = set(subset_i)
        sj = set(subset_j)
        inter = si & sj
        if not inter:
            return True
        opp = 0
        for coord in inter:
            # Find position of coord in subset_i and subset_j
            pi = subset_i.index(coord)
            pj = subset_j.index(coord)
            if signs_i[pi] != signs_j[pj]:
                opp += 1
        s = len(inter)
        return opp >= (s + 1) // 2  # ceil(s/2)

    def backtrack(idx, signs, assignments):
        """
        idx: current index to assign
        signs: dict from index to sign tuple
        assignments: list of sign tuples in order
        Returns True if a complete assignment is found.
        """
        if time.time() - t_start > DEADLINE - 10:
            return False

        if idx == n_sel:
            return True

        subset = selected_6[idx]
        # Generate all 2^6 = 64 sign assignments
        # But we can prune: try to find one that works with all previous
        # For speed, shuffle the sign assignments
        sign_options = list(itertools.product([-1, 1], repeat=6))
        random.shuffle(sign_options)

        for sv in sign_options:
            # Check against all previously assigned
            ok = True
            for j in range(idx):
                if not check_pair(sv, subset, assignments[j], selected_6[j]):
                    ok = False
                    break
            if ok:
                assignments[idx] = sv
                if backtrack(idx + 1, signs, assignments):
                    return True
                # Undo
                assignments[idx] = None
                if time.time() - t_start > DEADLINE - 10:
                    return False

        return False

    # Try backtracking
    assignments = [None] * n_sel
    found = backtrack(0, {}, assignments)

    if not found:
        # Try with a smaller subset: use first min(n_sel, 200) subsets
        # Or just use whatever we have from a simpler greedy sign assignment
        # Let's try a simple greedy: assign signs randomly, keep if compatible
        greedy_assignments = []
        greedy_subsets = []
        for i in range(n_sel):
            subset = selected_6[i]
            best_sv = None
            for _ in range(200):  # try 200 random sign assignments
                sv = tuple(random.choice([-1, 1]) for _ in range(6))
                ok = True
                for j, prev_sv in enumerate(greedy_assignments):
                    if not check_pair(sv, subset, prev_sv, greedy_subsets[j]):
                        ok = False
                        break
                if ok:
                    best_sv = sv
                    break
            if best_sv is None:
                # Try all 64
                for sv in itertools.product([-1, 1], repeat=6):
                    ok = True
                    for j, prev_sv in enumerate(greedy_assignments):
                        if not check_pair(sv, subset, prev_sv, greedy_subsets[j]):
                            ok = False
                            break
                    if ok:
                        best_sv = sv
                        break
            if best_sv is not None:
                greedy_assignments.append(best_sv)
                greedy_subsets.append(subset)
            if time.time() - t_start > DEADLINE - 10:
                break

        if len(greedy_assignments) >= 2:
            n_sel = len(greedy_subsets)
            selected_6 = greedy_subsets
            assignments = greedy_assignments
        else:
            # Absolute fallback
            return np.array([[1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    # Build the result array
    k = n_sel
    result = np.zeros((k, d), dtype=np.int64)
    for i in range(k):
        subset = selected_6[i]
        sv = assignments[i]
        for pos, coord in enumerate(subset):
            result[i, coord] = sv[pos]

    # Verify
    norms = np.sum(result**2, axis=1)
    max_norm = int(np.max(norms))
    valid = True
    for i in range(k):
        if norms[i] == 0:
            valid = False
            break
        for j in range(i+1, k):
            diff = result[i] - result[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < max_norm:
                valid = False
                break
        if not valid:
            break

    if not valid or k < 2:
        # Fallback: use weight-2 vectors (known 220)
        cands = []
        for i in range(d):
            for j in range(i+1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        cands.append(v)
        return np.array(cands, dtype=np.int64)

    return np.around(result).astype(np.int64)
