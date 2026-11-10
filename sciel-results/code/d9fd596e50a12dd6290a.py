import time
import random
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin under 90s

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

    def poly_from_terms(terms):
        """terms: list of (degree, coeff). Returns coefficient list."""
        if not terms:
            return [0]
        max_deg = max(d for d, c in terms if c != 0)
        coeffs = [0] * (max_deg + 1)
        for d, c in terms:
            if c != 0:
                coeffs[d] = c
        # Trim trailing zeros
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    def poly_mul(a, b):
        """Multiply two polynomials given as coefficient lists."""
        return list(np.convolve(a, b))

    # ============================================================
    # CORE STRATEGY: Factor decomposition P = A * B
    # |A| * |B| = 12  =>  (|A|,|B|) in {(2,6),(3,4),(4,3),(6,2)}
    # P^2 = A^2 * B^2
    # We search for A (m terms) and B (n terms) such that:
    #   1. A*B has exactly 12 nonzero terms (no accidental cancellation in product)
    #   2. (A*B)^2 has minimal nonzero count
    #
    # Key insight: If A^2 has few nonzero terms AND B^2 has few nonzero terms,
    # then A^2 * B^2 will have few nonzero terms (product of sparse polys).
    #
    # For m=2: A = a*x^p + b*x^q. A^2 = a^2*x^(2p) + 2ab*x^(p+q) + b^2*x^(2q)
    #   => 3 terms (always, unless a or b is 0)
    # For m=3: A = sum of 3 monomials. A^2 has up to 6 terms, but can have cancellations.
    #
    # We enumerate supports and coefficients for A and B, compute P = A*B,
    # check validity, and count P^2 nonzero terms.
    # ============================================================

    # Phase 1: (2,6) decomposition
    # A has 2 terms: a*x^p + b*x^q, with a,b in [-64,64], a*b != 0
    # B has 6 terms.
    # P = A*B has at most 12 terms. For exactly 12, no cancellation in A*B.
    # P^2 = A^2 * B^2. A^2 has 3 terms. B^2 has up to 21 terms (but can be less).
    # |P^2| <= 3 * |B^2|, but typically less due to structure.

    # Phase 2: (3,4) decomposition
    # A has 3 terms, B has 4 terms.
    # A^2 has up to 6 terms, B^2 has up to 10 terms.
    # P^2 = A^2 * B^2 can have fewer terms if there's alignment.

    def search_2x6():
        """Search for best (2,6) factorization."""
        best_local = float('inf')
        best_A = None
        best_B = None

        # A = a*x^p + b*x^q
        # WLOG: p=0 (shift invariance for counting), q > 0
        # a, b: small integers to keep P coefficients in [-64,64]
        # B: 6 terms with degrees and coefficients

        # For A^2 * B^2 to be sparse, we want B^2 to be sparse.
        # B with 6 terms: B^2 has up to 21 terms.
        # If B is an AP of 6 terms: B^2 has 11 terms (2*6-1).
        # If B has more structure, B^2 can be sparser.

        # Try B as AP: {0,1,2,3,4,5} with signs
        # B^2 for AP of 6: 11 positions, coefficients are autocorrelation of signs.

        # For A = a*x^0 + b*x^q:
        # P = a*B + b*x^q*B
        # P has 12 terms iff no overlap between B and x^q*B, i.e., q > 5 (max degree of B)
        # So q >= 6.

        # P^2 = (a*B + b*x^q*B)^2 = a^2*B^2 + 2ab*x^q*B^2 + b^2*x^(2q)*B^2
        # The three B^2 blocks are at positions [0,10], [q, q+10], [2q, 2q+10]
        # For no overlap between blocks: q > 10, i.e., q >= 11.
        # Then |P^2| = 3 * |B^2| = 3 * 11 = 33. That's worse than 23!

        # So we NEED overlap between the B^2 blocks to get cancellation.
        # If q = 6: blocks at [0,10], [6,16], [12,22]. Overlap at [6,10] and [12,16].
        # If q = 7: blocks at [0,10], [7,17], [14,24]. Overlap at [7,10] and [14,17].
        # If q = 8: blocks at [0,10], [8,18], [16,26]. Overlap at [8,10] and [16,18].
        # If q = 9: blocks at [0,10], [9,19], [18,28]. Overlap at [9,10] and [18,19].
        # If q = 10: blocks at [0,10], [10,20], [20,30]. Overlap at {10} and {20}.

        # The more overlap, the more potential for cancellation.
        # q=6 gives max overlap (5 positions in each overlap region).

        # For q=6, P^2 coefficient at position k:
        # From a^2*B^2: a^2 * b2[k] for k in [0,10]
        # From 2ab*x^6*B^2: 2ab * b2[k-6] for k in [6,16]
        # From b^2*x^12*B^2: b^2 * b2[k-12] for k in [12,22]

        # Let b2 = B^2 coefficients (length 11, indices 0..10)
        # P^2[k] = a^2*b2[k] + 2ab*b2[k-6] + b^2*b2[k-12]  (where defined)

        # For k=0..5: only a^2*b2[k]
        # For k=6..10: a^2*b2[k] + 2ab*b2[k-6]
        # For k=11: 2ab*b2[5]
        # For k=12..16: 2ab*b2[k-6] + b^2*b2[k-12]
        # For k=17..22: b^2*b2[k-12]

        # To maximize cancellation, we want:
        # a^2*b2[k] = -2ab*b2[k-6] for k=6..10
        # => b2[k]/b2[k-6] = -2b/a for all k=6..10
        # This requires b2[6..10] / b2[0..4] = constant = -2b/a
        # Similarly for the other overlap.

        # This is a strong constraint. Let's just enumerate.

        # B support: {0,1,2,3,4,5} (AP), signs s0..s5
        # B^2[k] = sum_{i=max(0,k-5)}^{min(k,5)} s_i * s_{k-i}  for k=0..10

        # Try all 2^6 = 64 sign patterns for B, and a few (a,b) pairs.
        for mask in range(64):
            if time.time() > deadline - 15:
                return best_local, best_A, best_B
            signs_b = [1 if ((mask >> i) & 1) == 0 else -1 for i in range(6)]

            # Compute B^2
            b2 = [0] * 11
            for k in range(11):
                val = 0
                for i in range(6):
                    j = k - i
                    if 0 <= j < 6:
                        val += signs_b[i] * signs_b[j]
                b2[k] = val

            # Try various (a, b, q)
            for a in [1, 2, 3, 4, 5, 8, 10, 16]:
                for b in [1, 2, 3, 4, 5, 8, 10, 16]:
                    if a * b == 0:
                        continue
                    for q in [6, 7, 8, 9, 10]:
                        # Build P = a*x^0*B + b*x^q*B
                        # P has terms at degrees: 0,1,2,3,4,5 and q,q+1,q+2,q+3,q+4,q+5
                        # Check for overlap: if q <= 5, there's overlap in P itself
                        if q <= 5:
                            continue

                        max_deg = q + 5
                        p_coeffs = [0] * (max_deg + 1)
                        for i in range(6):
                            p_coeffs[i] = a * signs_b[i]
                            p_coeffs[q + i] = b * signs_b[i]

                        # Check P has exactly 12 nonzero terms
                        if sum(1 for c in p_coeffs if c != 0) != 12:
                            continue
                        if any(c < -64 or c > 64 for c in p_coeffs):
                            continue

                        sq = np.convolve(p_coeffs, p_coeffs)
                        nz = int(np.count_nonzero(sq))

                        if nz < best_local:
                            best_local = nz
                            best_A = [a, 0, 0, 0, 0, 0, b]  # placeholder
                            best_B = signs_b[:]
                            if nz <= 15:
                                return best_local, best_A, best_B

        return best_local, best_A, best_B

    # Phase 3: (3,4) decomposition
    def search_3x4():
        """Search for best (3,4) factorization."""
        best_local = float('inf')
        best_result = None

        # A has 3 terms: a*x^p + b*x^q + c*x^r
        # B has 4 terms: d*x^s + e*x^t + f*x^u + g*x^v
        # P = A*B has up to 12 terms.

        # For P^2 = A^2 * B^2 to be sparse:
        # A^2 has up to 6 terms, B^2 has up to 10 terms.
        # Product can have up to 60 terms, but structure reduces this.

        # Key: if A and B are both "flat" (AP-like), their squares have
        # structured coefficients that may align for cancellation in the product.

        # Strategy: Fix A support as {0, 1, d} (3 terms, one gap)
        # Fix B support as {0, 1, 2, 3} (AP of 4)
        # P = A * B. Check validity and count P^2.

        # A = a + b*x + c*x^d
        # B = d0 + d1*x + d2*x^2 + d3*x^3
        # P = a*d0 + (a*d1+b*d0)*x + (a*d2+b*d1+c*d0*x^(d-1))*x^2 + ...

        # This gets complex. Let's use a more systematic approach:
        # Enumerate A supports (3 elements from {0,...,20}) and B supports (4 elements from {0,...,20})
        # with small coefficients, compute P, check validity, count P^2.

        # To keep search tractable, use structured supports:
        # A: {0, 1, k} for k in {2,3,4,5,6,7,8}
        # B: {0, 1, 2, 3} (AP) or {0, 1, 3, 4} or {0, 2, 3, 5} etc.

        a_supports = [
            [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6],
            [0, 2, 4], [0, 2, 3], [0, 3, 6], [1, 2, 3], [0, 1, 7], [0, 1, 8]
        ]
        b_supports = [
            [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4],
            [0, 1, 2, 5], [0, 1, 3, 5], [0, 2, 3, 5], [1, 2, 3, 4],
            [0, 1, 2, 6], [0, 1, 4, 5], [0, 2, 4, 6]
        ]

        # Coefficient choices: small integers
        coeff_choices = [1, -1, 2, -2, 3, -3]

        # For each (A_support, B_support), try coefficient assignments
        # 6^3 * 6^4 = 216 * 1296 = 280k combinations per support pair
        # ~120 support pairs => 33.6M total. Too many.

        # Reduce: fix A coefficients to {1,1,1} or {1,1,-1} etc. (sign patterns)
        # And B similarly. 2^3 * 2^4 = 128 per support pair.
        # 120 * 128 = 15k. Very tractable.

        for a_sup in a_supports:
            if time.time() > deadline - 15:
                return best_local, best_result
            for b_sup in b_supports:
                # Try all sign patterns for A (8) and B (16)
                for a_mask in range(8):
                    for b_mask in range(16):
                        a_coeffs = [1 if ((a_mask >> i) & 1) == 0 else -1 for i in range(3)]
                        b_coeffs = [1 if ((b_mask >> i) & 1) == 0 else -1 for i in range(4)]

                        # Build A and B polynomials
                        a_poly = [0] * (max(a_sup) + 1)
                        for i, deg in enumerate(a_sup):
                            a_poly[deg] = a_coeffs[i]

                        b_poly = [0] * (max(b_sup) + 1)
                        for i, deg in enumerate(b_sup):
                            b_poly[deg] = b_coeffs[i]

                        # P = A * B
                        p_coeffs = poly_mul(a_poly, b_poly)

                        # Check P has exactly 12 nonzero terms
                        if sum(1 for c in p_coeffs if c != 0) != 12:
                            continue
                        if len(p_coeffs) > 41:
                            continue
                        if any(c < -64 or c > 64 for c in p_coeffs):
                            continue

                        sq = np.convolve(p_coeffs, p_coeffs)
                        nz = int(np.count_nonzero(sq))

                        if nz < best_local:
                            best_local = nz
                            best_result = list(p_coeffs)
                            if nz <= 15:
                                return best_local, best_result

        return best_local, best_result

    # Run factor searches
    s26, A26, B26 = search_2x6()
    if A26 and B26:
        # Reconstruct P from A26 and B26
        # A26 is placeholder, need to rebuild
        pass  # The try_poly calls inside would have updated best

    # Actually, let me restructure: the searches above should call try_poly directly.
    # Let me redo this more cleanly.

    # ============================================================
    # CLEAN RESTRUCTURED FACTOR SEARCH
    # ============================================================

    def factor_search():
        """Systematic factor decomposition search."""
        nonlocal best_poly, best_score

        def evaluate_P(p_coeffs):
            nonlocal best_poly, best_score
            if not is_valid(p_coeffs):
                return
            s = count_sq_nonzero(p_coeffs)
            if s < best_score:
                best_score = s
                best_poly = list(p_coeffs)

        # --- (2,6) decomposition ---
        # A = a + b*x^q (2 terms), B = 6-term polynomial
        # P = a*B + b*x^q*B
        # For 12 terms: q > 5 (no overlap in P)
        # P^2 = a^2*B^2 + 2ab*x^q*B^2 + b^2*x^(2q)*B^2

        # B support: try APs and near-APs of 6 terms
        b_supports_6 = [
            [0, 1, 2, 3, 4, 5],      # AP d=1
            [0, 1, 2, 3, 4, 6],      # near AP
            [0, 1, 2, 3, 5, 6],
            [0, 1, 2, 4, 5, 6],
            [0, 1, 3, 4, 5, 6],
            [0, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 7],
            [0, 1, 2, 3, 5, 7],
            [0, 1, 2, 4, 5, 7],
            [0, 1, 3, 4, 5, 7],
        ]

        for b_sup in b_supports_6:
            if time.time() > deadline - 20:
                return
            for b_mask in range(64):  # 2^6 sign patterns
                b_coeffs = [1 if ((b_mask >> i) & 1) == 0 else -1 for i in range(6)]

                # Build B polynomial
                max_b_deg = max(b_sup)
                b_poly = [0] * (max_b_deg + 1)
                for i, deg in enumerate(b_sup):
                    b_poly[deg] = b_coeffs[i]

                # Try (a, b_coeff, q) combinations
                for a_val in [1, 2, 3, 4]:
                    for b_val in [1, 2, 3, 4]:
                        for q in [6, 7, 8, 9, 10, 11]:
                            # P = a*B + b_val*x^q*B
                            max_p_deg = q + max_b_deg
                            if max_p_deg > 40:
                                continue
                            p_coeffs = [0] * (max_p_deg + 1)
                            for i in range(len(b_poly)):
                                if b_poly[i] != 0:
                                    p_coeffs[i] += a_val * b_poly[i]
                                    p_coeffs[q + i] += b_val * b_poly[i]

                            # Check exactly 12 nonzero terms
                            nz_p = sum(1 for c in p_coeffs if c != 0)
                            if nz_p != 12:
                                continue
                            if any(c < -64 or c > 64 for c in p_coeffs):
                                continue

                            evaluate_P(p_coeffs)

        # --- (3,4) decomposition ---
        # A = 3-term, B = 4-term
        a_supports_3 = [
            [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6],
            [0, 2, 3], [0, 2, 4], [0, 3, 6], [1, 2, 3], [0, 1, 7],
            [0, 2, 5], [0, 3, 5], [1, 3, 5]
        ]
        b_supports_4 = [
            [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4],
            [0, 1, 2, 5], [0, 1, 3, 5], [0, 2, 3, 5], [1, 2, 3, 4],
            [0, 1, 2, 6], [0, 1, 4, 5], [0, 2, 4, 6], [0, 1, 3, 6],
            [0, 2, 3, 6], [1, 2, 4, 5]
        ]

        for a_sup in a_supports_3:
            if time.time() > deadline - 15:
                return
            for b_sup in b_supports_4:
                # All sign patterns: 2^3 * 2^4 = 128
                for a_mask in range(8):
                    for b_mask in range(16):
                        a_coeffs = [1 if ((a_mask >> i) & 1) == 0 else -1 for i in range(3)]
                        b_coeffs = [1 if ((b_mask >> i) & 1) == 0 else -1 for i in range(4)]

                        max_a_deg = max(a_sup)
                        max_b_deg = max(b_sup)

                        a_poly = [0] * (max_a_deg + 1)
                        for i, deg in enumerate(a_sup):
                            a_poly[deg] = a_coeffs[i]

                        b_poly = [0] * (max_b_deg + 1)
                        for i, deg in enumerate(b_sup):
                            b_poly[deg] = b_coeffs[i]

                        p_coeffs = list(np.convolve(a_poly, b_poly))

                        # Check exactly 12 nonzero terms
                        if len(p_coeffs) > 41:
                            continue
                        nz_p = sum(1 for c in p_coeffs if c != 0)
                        if nz_p != 12:
                            continue
                        if any(c < -64 or c > 64 for c in p_coeffs):
                            continue

                        evaluate_P(p_coeffs)

        # --- (4,3) decomposition (symmetric to 3,4 but different supports) ---
        # Already covered by 3,4 since multiplication is commutative
        # but the support enumeration is different. Skip for time.

        # --- (6,2) decomposition (symmetric to 2,6) ---
        # Also covered. Skip.

    factor_search()

    # Phase 4: Local refinement on best found
    if time.time() < deadline - 5:
        best_support = [i for i, c in enumerate(best_poly) if c != 0]
        if len(best_support) == 12:
            # Try small perturbations of coefficients
            for trial in range(2000):
                if time.time() > deadline - 3:
                    break
                coeffs = list(best_poly)
                n_changes = random.randint(1, 3)
                for _ in range(n_changes):
                    idx = random.choice(best_support)
                    delta = random.choice([-2, -1, 1, 2, 3, -3])
                    new_val = coeffs[idx] + delta
                    if -64 <= new_val <= 64 and new_val != 0:
                        coeffs[idx] = new_val

                # Ensure still 12 nonzero terms
                if sum(1 for c in coeffs if c != 0) != 12:
                    continue
                if any(c < -64 or c > 64 for c in coeffs):
                    continue

                s = count_sq_nonzero(coeffs)
                if s < best_score:
                    best_score = s
                    best_poly = coeffs

    # Phase 5: Broader random search with factor-inspired structures
    if time.time() < deadline - 2:
        random.seed(12345)
        for _ in range(5000):
            if time.time() > deadline - 1:
                break
            # Generate a 12-term polynomial with structured support
            # Use two or three interleaved APs
            d1 = random.choice([1, 2, 3])
            d2 = random.choice([1, 2, 3, 4])
            a1 = random.randint(0, 5)
            a2 = random.randint(8, 15)

            support = set()
            for i in range(6):
                support.add(a1 + i * d1)
            for i in range(6):
                support.add(a2 + i * d2)

            if len(support) != 12:
                continue
            support = sorted(support)
            if support[0] < 0 or support[11] > 40:
                continue

            # Random signs and small magnitudes
            coeffs = [0] * (max(support) + 1)
            for deg in support:
                coeffs[deg] = random.choice([-3, -2, -1, 1, 2, 3])

            if sum(1 for c in coeffs if c != 0) != 12:
                continue
            if any(c < -64 or c > 64 for c in coeffs):
                continue

            s = count_sq_nonzero(coeffs)
            if s < best_score:
                best_score = s
                best_poly = coeffs

    # Ensure valid result
    if not is_valid(best_poly):
        best_poly = [1] * 12

    return best_poly
