import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0

    # LCM of all divisors
    lcm_all = reduce(_lcm, divisors)  # 30030

    best_val = float('inf')
    best_assign = None

    # For each window width W (starting from minimum possible = 9),
    # search for a window [L, L+W] where each divisor has a multiple,
    # and we can assign distinct multiples.
    #
    # Due to LCM periodicity, we only need to check L in [0, lcm_all-1]
    # for each W. But W itself can be large, so we need to be smart.
    #
    # Actually, the key observation: for a given W, the set of L values
    # where a particular divisor d has a multiple in [L, L+W] is periodic
    # with period d. The combined pattern is periodic with period lcm_all.
    #
    # So for each W, we check L in range(1, lcm_all) (or a subset).
    #
    # But W can go up to... we need to bound it. Let's start small.

    def find_multiple_in_window(d, L, W):
        """Return the smallest multiple of d in [L, L+W], or None."""
        # Smallest multiple of d >= L
        k = (L + d - 1) // d
        m = k * d
        if m <= L + W:
            return m
        return None

    def solve_window(L, W):
        """Try to assign distinct multiples to each divisor within [L, L+W].
        Returns assignment list or None."""
        # For each divisor, find all multiples in [L, L+W]
        candidates = []
        for d in divisors:
            k = (L + d - 1) // d
            m = k * d
            opts = []
            while m <= L + W:
                if m <= MAX_V:
                    opts.append(m)
                m += d
            if not opts:
                return None
            candidates.append(opts)

        # Backtracking: assign distinct values
        # Order divisors by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        assigned = [None] * n
        used = set()

        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    assigned[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.discard(val)
                    assigned[i] = None
            return False

        if backtrack(0):
            return assigned
        return None

    # Search W from minimum upward
    # Minimum W is 9 (need 10 distinct integers)
    # But let's start from a smarter lower bound.
    # The largest divisor is 39. In a window of width W, we need a multiple of 39.
    # That's always possible for W >= 0 (a single point can be a multiple of 39).
    # The real constraint is having ALL divisors covered AND distinct values.

    # Let's search W from 9 upward, but limit total work.
    # For each W, we check L in a range. Due to periodicity with lcm_all=30030,
    # we check L in [1, lcm_all] but that's 30030 values per W, which is too many.
    #
    # Better: for each W, the valid L values are those where every divisor has
    # at least one multiple. We can precompute for each L mod d whether d is covered.
    #
    # Actually, let's be more targeted. For a given W, the condition that d has
    # a multiple in [L, L+W] means L mod d is in some set. Specifically,
    # d has a multiple in [L, L+W] iff there exists k such that L <= kd <= L+W,
    # i.e., L/d - W/d <= k <= L/d, so k = ceil(L/d) and we need ceil(L/d)*d <= L+W.
    #
    # Let me just iterate L in a smart way. For small W, the number of valid L
    # per period is limited.

    # Alternative approach: iterate over possible L values directly.
    # For each L, the window [L, L+W] for various W.
    #
    # Let's try: for W from 9 to some max, for L in a reduced set.
    #
    # Key optimization: for a fixed W, the condition "d has a multiple in [L,L+W]"
    # depends only on L mod d. So the combined condition depends on L mod lcm_all.
    # We can precompute which residues mod lcm_all work for each W.
    #
    # But lcm_all = 30030, so for each W we'd check 30030 residues. That's fine
    # if W range is small.

    # Let's try a different approach: iterate L from 1 to lcm_all, and for each L,
    # find the minimum W such that all divisors have a multiple in [L, L+W]
    # and we can assign distinct values.

    # For a fixed L, the minimum multiple of d >= L is ceil(L/d)*d.
    # So the minimum W needed for d is ceil(L/d)*d - L.
    # The minimum W to cover all divisors is max over d of (ceil(L/d)*d - L).
    # But we also need distinct values, which may require a larger W.

    # Strategy: for each L in [1, lcm_all], compute the "natural" assignment
    # (smallest multiple of each d >= L), check if they're distinct, and if not,
    # try to adjust.

    # Actually, let's just do a direct search:
    # For W from 9 upward:
    #   For L in a set of candidate starts:
    #     Check if all divisors have multiples in [L, L+W]
    #     If yes, try to assign distinct values via backtracking
    #     If successful, record and break

    # To limit L search: for a given W, L only matters mod lcm_all.
    # And for small W, only a fraction of L values work.
    #
    # Let's precompute for each residue r in [0, lcm_all):
    #   For each divisor d, the gap from r to next multiple of d (i.e., (d - r%d) % d)
    #   A window [L, L+W] with L%lcm_all = r covers d iff gap_d <= W
    #
    # So for a given W, valid residues are those where max_d gap_d <= W.
    # Then for each valid residue, L = r + k*lcm_all for k=0,1,...
    # and we need L+W <= MAX_V.

    # Precompute gaps for each residue
    # This is 30030 * 10 = 300300 operations, very fast.

    gaps = [[0]*n for _ in range(lcm_all)]
    for r in range(lcm_all):
        for i, d in enumerate(divisors):
            # gap from r to next multiple of d
            rem = r % d
            if rem == 0:
                gaps[r][i] = 0
            else:
                gaps[r][i] = d - rem

    # For each W, find residues where max gap <= W
    # Then for each such residue, try L values.

    # But we also need distinctness. Let's first find the minimum W
    # where there EXISTS a residue with all gaps <= W.

    min_W_cover = min(max(gaps[r]) for r in range(lcm_all))

    # Now search W from max(9, min_W_cover) upward
    # For each W, get valid residues, then for each residue try L values.

    # To keep it fast, limit the number of L values tried per residue.
    # Since the pattern repeats every lcm_all, and we need L+W <= MAX_V,
    # the number of L values per residue is about MAX_V / lcm_all ≈ 33.

    # Total work per W: (num valid residues) * 33 * (backtracking cost)
    # Backtracking cost is small (10 divisors, few candidates each for small W).

    # Let's start searching.
    search_W = max(9, min_W_cover)
    # But min_W_cover might be large. Let's check.
    # Actually, let's just start from 9 and go up.

    # For efficiency, precompute for each W the valid residues.
    # But W can go up to... let's set a limit.

    # Let's try W from 9 to 200, which should be enough.
    # For each W, find valid residues (max gap <= W).
    # For each valid residue r, try L = r + k*lcm_all for k such that L >= 1 and L+W <= MAX_V.
    #   Actually L must be >= 1, so L = max(1, r) + k*lcm_all... no.
    #   L values with L % lcm_all == r and L >= 1: L = r if r >= 1, else lcm_all, then + lcm_all each time.
    #   But r is in [0, lcm_all-1]. If r == 0, L = lcm_all, 2*lcm_all, ...
    #   If r > 0, L = r, r+lcm_all, ...

    for W in range(9, 300):
        if time.time() > deadline:
            break

        # Find valid residues
        valid_residues = []
        for r in range(lcm_all):
            if max(gaps[r]) <= W:
                valid_residues.append(r)

        if not valid_residues:
            continue

        # For each valid residue, try L values
        for r in valid_residues:
            if time.time() > deadline:
                break

            # L values with L % lcm_all == r, L >= 1, L + W <= MAX_V
            if r == 0:
                L_start = lcm_all
            else:
                L_start = r

            if L_start + W > MAX_V:
                continue

            # Try L = L_start, L_start + lcm_all, ...
            # But we only need to try a few; the assignment feasibility
            # depends on the actual multiples, which shift by lcm_all.
            # Since lcm_all is a multiple of all divisors, the relative
            # positions of multiples within the window are the same for
            # all L with the same residue! So we only need to try ONE L
            # per residue (the smallest valid one).

            L = L_start
            if L < 1:
                L = L_start + lcm_all
            if L + W > MAX_V:
                continue

            result = solve_window(L, W)
            if result is not None:
                span = max(result) - min(result)
                if span < best_val:
                    best_val = span
                    best_assign = result[:]
                # Found a solution for this W; can break inner loops
                # since we're searching W in increasing order
                break

        # If we found a solution at this W, we could break since W is increasing
        # but we want the MINIMUM W, so break.
        if best_assign is not None and best_val <= W:
            # We found a solution with span = best_val, and we're at W = best_val
            # (since span <= W always). If span == W, this is optimal for this W.
            # But could a smaller W have been missed? No, we search in order.
            # Actually, span could be < W if the actual max-min is less than W.
            # But we're using W as the window width, and span <= W.
            # Since we search W in increasing order, the first W where we find
            # a solution gives the minimum possible span.
            break

    if best_assign is None:
        # Fallback: use the divisors themselves (which is valid but not optimal)
        return divisors[:]

    return best_assign
