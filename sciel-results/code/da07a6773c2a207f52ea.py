import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin under 90s

    def is_valid(coeffs):
        if len(coeffs) > 41 or len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    def count_sq_nonzero(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    # Initial best: [1]*12 gives 23 nonzero terms in square
    best_poly = [1] * 12
    best_score = count_sq_nonzero(best_poly)

    def try_poly(coeffs):
        nonlocal best_poly, best_score
        if not is_valid(coeffs):
            return
        s = count_sq_nonzero(coeffs)
        if s < best_score:
            best_score = s
            best_poly = list(coeffs)

    # KEY INSIGHT: For a FIXED support (set of 12 exponents), the number of
    # distinct positions in the square is DETERMINED by the support alone.
    # The square has a term at position p_i + p_j for each pair (i,j).
    # The number of DISTINCT positions is the "structural" minimum.
    # Coefficients can only REDUCE this count (via cancellation to zero),
    # never increase it.
    #
    # So: minimize structural distinct positions first, then optimize coefficients.
    #
    # For 12 exponents, there are C(12,2)+12 = 78 pairs (i<=j).
    # The number of distinct sums p_i+p_j is at most 78 but typically much less
    # due to collisions. We want supports with MANY collisions.
    #
    # Strategy: Branch-and-bound over ordered 12-subsets of {0,...,40},
    # pruning when the partial number of distinct sums already exceeds best.

    # Precompute: for a partial support (sorted list of k exponents),
    # compute the set of distinct pair-sums among them.
    # If |distinct sums| >= best_score, prune.

    # But C(41,12) is ~8.7 billion. We need aggressive pruning.
    #
    # Alternative: Use the fact that for arithmetic progressions,
    # the number of distinct sums is minimized. For an AP with difference d,
    # the sums form a contiguous range of length 2*11*d, giving 22 distinct
    # positions. That's the theoretical minimum for 12 terms!
    #
    # Actually: for 12 terms in AP: positions are a, a+d, ..., a+11d
    # Sums: 2a, 2a+d, ..., 2a+22d → 23 distinct positions.
    # With all coefficients = 1, the square has 23 nonzero terms.
    # Can we do better with non-AP supports?
    #
    # The minimum number of distinct sums for 12 integers is 23
    # (achieved by any AP, by the additive combinatorics lower bound
    # |A+A| >= 2|A|-1 for finite sets of integers).
    #
    # So the STRUCTURAL minimum is 23. We need cancellations to go below 23.
    #
    # For an AP support with all +1 coefficients: square has 23 terms (all nonzero).
    # With signs: can we cancel some? The coefficient at position 2a+kd is
    # sum of s_i*s_j where i+j=k. For the AP, this is a "convolution of signs with itself".
    #
    # Let's focus on AP supports and find sign patterns that maximize cancellations.
    # For AP: support = {0, 1, 2, ..., 11} (wlog, shift doesn't matter for count)
    # Square coefficient at degree k (0<=k<=22): sum_{i=0}^{k} s_i * s_{k-i}
    # where s_i is the sign at position i.
    # We want as many of these 23 sums to be zero.

    # For the standard AP {0,1,...,11}:
    # Coeff at k: sum_{i=max(0,k-11)}^{min(k,11)} s_i * s_{k-i}
    # For k=0: s_0^2 = 1 (always nonzero)
    # For k=22: s_11^2 = 1 (always nonzero)
    # So minimum possible is 2 (the endpoints). Can we get all interior to zero?
    # That would require a "perfect" sign sequence where all autocorrelations vanish.
    # This is related to Barker sequences / Golay complementary sequences.
    #
    # For length 12, a Barker sequence doesn't exist (max length is 13 for binary).
    # But we can get close. Let's enumerate all 2^12 sign patterns for the AP.

    def ap_sign_search():
        """Find best sign pattern for support {0,1,...,11}."""
        best_local = float('inf')
        best_signs = None
        # For AP {0,...,11}, coeff at k = sum_{i=0}^{k} s_i*s_{k-i} for 0<=k<=22
        # Precompute the pairs for each k
        pairs_for_k = []
        for k in range(23):
            pairs = []
            for i in range(12):
                j = k - i
                if 0 <= j < 12 and j >= i:  # count each pair once, but with factor 2 if i!=j
                    if i == j:
                        pairs.append((i, j, 1))
                    else:
                        pairs.append((i, j, 2))
            pairs_for_k.append(pairs)

        for mask in range(4096):
            signs = [1 if ((mask >> i) & 1) == 0 else -1 for i in range(12)]
            nonzero = 0
            for k in range(23):
                val = 0
                for (i, j, mult) in pairs_for_k[k]:
                    val += mult * signs[i] * signs[j]
                if val != 0:
                    nonzero += 1
                    if nonzero >= best_local:
                        break
            if nonzero < best_local:
                best_local = nonzero
                best_signs = signs[:]
                if nonzero <= 2:
                    break

        return best_local, best_signs

    ap_score, ap_signs = ap_sign_search()
    if ap_signs:
        coeffs = [0] * 12
        for i in range(12):
            coeffs[i] = ap_signs[i]
        try_poly(coeffs)

    # Now try shifted APs and other structured supports
    # For support {a, a+d, ..., a+11d}, the square structure is the same
    # (just shifted/scaled), so the sign pattern that works for {0,...,11}
    # also works for any AP with the same relative structure.
    # The COUNT is the same. So all APs give the same score.

    # Try non-AP supports with high collision density.
    # Key: supports where many pairs (i,j) give the same sum.
    # E.g., symmetric supports: {0,1,2,3,4,5,16,17,18,19,20,21}
    # Or: two interleaved APs.

    # Let's do a focused search over "near-AP" supports:
    # Take AP {0,...,11} and perturb 1-3 elements.

    random.seed(42)
    base_ap = list(range(12))

    def eval_support(support):
        """Given 12 sorted positions, find best sign pattern and return (score, signs)."""
        if len(support) != 12 or len(set(support)) != 12:
            return float('inf'), None
        if min(support) < 0 or max(support) > 40:
            return float('inf'), None

        # Build pair-position map
        pair_pos = {}
        for i in range(12):
            for j in range(i, 12):
                pos = support[i] + support[j]
                if pos not in pair_pos:
                    pair_pos[pos] = []
                if i == j:
                    pair_pos[pos].append((i, j, 1))
                else:
                    pair_pos[pos].append((i, j, 2))

        best_local = float('inf')
        best_signs = None
        for mask in range(4096):
            signs = [1 if ((mask >> i) & 1) == 0 else -1 for i in range(12)]
            nonzero = 0
            for pos, pairs in pair_pos.items():
                val = 0
                for (i, j, mult) in pairs:
                    val += mult * signs[i] * signs[j]
                if val != 0:
                    nonzero += 1
                    if nonzero >= best_local:
                        break
            if nonzero < best_local:
                best_local = nonzero
                best_signs = signs[:]
                if nonzero <= 2:
                    break

        return best_local, best_signs

    # Phase 2: Search near-AP supports
    tried = set()
    tried.add(tuple(base_ap))

    # Perturbations of AP
    for _ in range(3000):
        if time.time() > deadline - 10:
            break
        support = base_ap[:]
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, 11)
            support[idx] += random.choice([-2, -1, 1, 2])
        support = sorted(set(support))
        if len(support) != 12:
            continue
        if support[0] < 0 or support[11] > 40:
            continue
        key = tuple(support)
        if key in tried:
            continue
        tried.add(key)

        score, signs = eval_support(support)
        if signs:
            max_deg = max(support)
            coeffs = [0] * (max_deg + 1)
            for i in range(12):
                coeffs[support[i]] = signs[i]
            try_poly(coeffs)

    # Phase 3: Try supports with "paired" structure for extra cancellation
    # If support = {a, a+1, b, b+1, c, c+1, d, d+1, e, e+1, f, f+1}
    # with a,b,c,d,e,f well-separated, the cross terms between pairs
    # are at distinct positions (no cancellation between pairs),
    # but within each pair, (x^a + s*x^(a+1))^2 = x^(2a) + 2s*x^(2a+1) + x^(2a+2)
    # The 2s term can be made to cancel with another pair's term if positions align.

    # Better: use the "rectangle" construction.
    # P = (x^a + x^b)(x^c + x^d)(x^e + x^f)
    # This has at most 8 terms (if all products distinct).
    # To get 12 terms: P = (x^a + x^b + x^c)(x^d + x^e + x^f)
    # This has at most 9 terms. Still not 12.
    # P = (x^a + x^b)(x^c + x^d + x^e + x^f) : 8 terms max.
    # Need 12 terms: P = sum of 12 monomials.

    # Phase 4: Magnitude optimization on best supports found
    # For the best sign pattern on a support, try varying magnitudes
    # to create additional cancellations.

    if time.time() < deadline - 5:
        # Re-run eval on the current best support with magnitude search
        # Extract support from best_poly
        best_support = [i for i, c in enumerate(best_poly) if c != 0]
        if len(best_support) == 12:
            best_signs_cur = [1 if best_poly[i] > 0 else -1 for i in best_support]
            best_mags_cur = [abs(best_poly[i]) for i in best_support]

            # Try magnitude variations
            for trial in range(5000):
                if time.time() > deadline - 3:
                    break
                mags = list(best_mags_cur)
                n_changes = random.randint(1, 4)
                for _ in range(n_changes):
                    idx = random.randint(0, 11)
                    mags[idx] = random.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64])

                # Also try flipping 0-2 signs
                signs = list(best_signs_cur)
                n_flips = random.randint(0, 2)
                for _ in range(n_flips):
                    idx = random.randint(0, 11)
                    signs[idx] = -signs[idx]

                max_deg = max(best_support)
                coeffs = [0] * (max_deg + 1)
                for i in range(12):
                    coeffs[best_support[i]] = signs[i] * mags[i]
                try_poly(coeffs)

    # Phase 5: Broader random search with structured generation
    if time.time() < deadline - 2:
        random.seed(99999)
        for _ in range(8000):
            if time.time() > deadline - 1:
                break
            # Generate support: start with AP, random perturb
            d = random.choice([1, 1, 1, 2, 3])
            a = random.randint(0, 40 - 11 * d)
            support = [a + i * d for i in range(12)]
            # Perturb
            for _ in range(random.randint(0, 2)):
                idx = random.randint(0, 11)
                support[idx] += random.randint(-2, 2)
            support = sorted(set(support))
            if len(support) != 12 or support[0] < 0 or support[11] > 40:
                continue

            # Random signs and small magnitudes
            signs = [random.choice([-1, 1]) for _ in range(12)]
            mags = [random.choice([1, 1, 1, 2, 3]) for _ in range(12)]

            max_deg = max(support)
            coeffs = [0] * (max_deg + 1)
            for i in range(12):
                coeffs[support[i]] = signs[i] * mags[i]
            try_poly(coeffs)

    # Ensure valid result
    if not is_valid(best_poly):
        best_poly = [1] * 12

    return best_poly
