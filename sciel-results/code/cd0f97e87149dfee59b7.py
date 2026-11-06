import time
import random
import numpy as np
from itertools import combinations, product

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    fallback = [1] * 12
    fallback_score = 21
    best_score = fallback_score
    best_candidate = fallback

    def compute_square_nonzeros_fast(exps, coeffs):
        n = len(exps)
        max_deg = int(max(exps) * 2)
        square = np.zeros(max_deg + 1, dtype=np.int64)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                square[ei + ej] += 2 * ci * cj
        return int(np.count_nonzero(square))

    def try_candidate(exps, coeffs):
        nonlocal best_score, best_candidate
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40:
            return
        if any(abs(c) > 64 for c in coeffs):
            return
        if any(c == 0 for c in coeffs):
            return
        score = compute_square_nonzeros_fast(exps, coeffs)
        if score < best_score:
            best_score = score
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if dense[-1] != 0:
                best_candidate = dense

    def optimize_signs(exps, max_iter=300):
        if time.time() > deadline:
            return
        random.seed(42)
        best_local_score = float('inf')
        best_local_signs = [1] * 12
        for restart in range(10):
            if time.time() > deadline:
                break
            if restart == 0:
                signs = [1] * 12
            else:
                signs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
            current_score = compute_square_nonzeros_fast(exps, signs)
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(12):
                    old = signs[i]
                    signs[i] = -signs[i]
                    new_score = compute_square_nonzeros_fast(exps, signs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        signs[i] = old
                if not improved and restart > 0:
                    for i in range(12):
                        for j in range(i + 1, 12):
                            old_i, old_j = signs[i], signs[j]
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
                            new_score = compute_square_nonzeros_fast(exps, signs)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                            else:
                                signs[i], signs[j] = old_i, old_j
                            if improved:
                                break
                        if improved:
                            break
                if not improved:
                    break
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_signs = signs[:]
        if best_local_score < best_score:
            try_candidate(exps, best_local_signs)

    def optimize_coeffs(exps, max_iter=150):
        if time.time() > deadline:
            return
        random.seed(99)
        best_local_score = float('inf')
        best_local_coeffs = [1] * 12
        for restart in range(8):
            if time.time() > deadline:
                break
            if restart == 0:
                coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
            else:
                coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            current_score = compute_square_nonzeros_fast(exps, coeffs)
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                i = random.randint(0, 11)
                old_val = coeffs[i]
                new_val = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_val == old_val:
                    continue
                coeffs[i] = new_val
                new_score = compute_square_nonzeros_fast(exps, coeffs)
                if new_score < current_score:
                    current_score = new_score
                else:
                    coeffs[i] = old_val
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_coeffs = coeffs[:]
        if best_local_score < best_score:
            try_candidate(exps, best_local_coeffs)

    def count_pair_sums(exps):
        """Count how many times each sum appears in the multiset of pair sums."""
        from collections import Counter
        sums = Counter()
        n = len(exps)
        for i in range(n):
            sums[2 * exps[i]] += 1
            for j in range(i + 1, n):
                sums[exps[i] + exps[j]] += 2
        return sums

    def try_A_plus_B():
        """
        Structural search: P = A + B, each 6-term.
        Strategy: choose A exps and B exps to minimize total collisions.
        Score = |A^2 nonzero| + |B^2 nonzero| + |cross nonzero|
        where cross terms are 2*a_i*b_j at positions a_i+b_j.
        We optimize signs and small coeffs to cancel.
        """
        if time.time() > deadline:
            return

        # Candidate A exponent patterns (6 terms)
        a_patterns = []
        for start in range(0, 20):
            for step in range(1, 6):
                exps = [start + i * step for i in range(6)]
                if max(exps) <= 40:
                    a_patterns.append(tuple(exps))
        # Quadratic
        for start in range(0, 10):
            exps = [start + i * i for i in range(6)]
            if max(exps) <= 40:
                a_patterns.append(tuple(exps))
        # Symmetric
        for center in range(3, 38):
            for offset in range(1, 8):
                exps = [center - 5*offset, center - 3*offset, center - offset,
                        center + offset, center + 3*offset, center + 5*offset]
                if min(exps) >= 0 and max(exps) <= 40 and len(set(exps)) == 6:
                    a_patterns.append(tuple(sorted(exps)))

        # Candidate B exponent patterns (6 terms)
        b_patterns = []
        for start in range(0, 35):
            for step in range(1, 6):
                exps = [start + i * step for i in range(6)]
                if max(exps) <= 40:
                    b_patterns.append(tuple(exps))
        for start in range(0, 15):
            exps = [start + i * i for i in range(6)]
            if max(exps) <= 40:
                b_patterns.append(tuple(exps))
        for center in range(15, 40):
            for offset in range(1, 8):
                exps = [center - 5*offset, center - 3*offset, center - offset,
                        center + offset, center + 3*offset, center + 5*offset]
                if min(exps) >= 0 and max(exps) <= 40 and len(set(exps)) == 6:
                    b_patterns.append(tuple(sorted(exps)))

        # Deduplicate
        a_patterns = list(set(a_patterns))
        b_patterns = list(set(b_patterns))

        # Precompute self-collision counts for each pattern
        def self_nonzero_count(exps):
            """Minimum possible nonzero terms in square of 6-term poly with these exps."""
            sums = count_pair_sums(list(exps))
            return len(sums)

        # Sort patterns by self-collision count (lower is better)
        a_patterns.sort(key=lambda e: self_nonzero_count(e))
        b_patterns.sort(key=lambda e: self_nonzero_count(e))

        # Search: try top patterns for A and B
        max_a_tries = 80
        max_b_tries = 80
        a_limit = min(max_a_tries, len(a_patterns))
        b_limit = min(max_b_tries, len(b_patterns))

        for ai in range(a_limit):
            if time.time() > deadline:
                return
            a_exps = list(a_patterns[ai])
            for bi in range(b_limit):
                if time.time() > deadline:
                    return
                b_exps = list(b_patterns[bi])
                all_exps = a_exps + b_exps
                if len(set(all_exps)) != 12:
                    continue
                if max(all_exps) > 40:
                    continue

                # Quick collision estimate
                a_sums = count_pair_sums(a_exps)
                b_sums = count_pair_sums(b_exps)
                cross_sums = {}
                for ea in a_exps:
                    for eb in b_exps:
                        s = ea + eb
                        cross_sums[s] = cross_sums.get(s, 0) + 1

                # Total positions = |A^2| + |B^2| + |cross|, but overlaps reduce count
                all_positions = set(a_sums.keys()) | set(b_sums.keys()) | set(cross_sums.keys())
                # Lower bound on nonzero terms (ignoring sign cancellation)
                lower_bound = len(all_positions)

                if lower_bound >= best_score:
                    continue

                # Try sign optimization
                best_local = float('inf')
                best_local_coeffs = [1] * 12

                # Try a few sign patterns
                random.seed(7 + ai + bi)
                for trial in range(12):
                    if time.time() > deadline:
                        return
                    if trial == 0:
                        coeffs = [1] * 12
                    else:
                        coeffs = [1 if random.random() > 0.5 else -1 for _ in range(12)]

                    score = compute_square_nonzeros_fast(all_exps, coeffs)
                    if score < best_local:
                        best_local = score
                        best_local_coeffs = coeffs[:]

                    # Local sign flip
                    for _ in range(50):
                        if time.time() > deadline:
                            return
                        improved = False
                        for i in range(12):
                            old = coeffs[i]
                            coeffs[i] = -coeffs[i]
                            new_score = compute_square_nonzeros_fast(all_exps, coeffs)
                            if new_score < best_local:
                                best_local = new_score
                                best_local_coeffs = coeffs[:]
                                improved = True
                            else:
                                coeffs[i] = old
                        if not improved:
                            break

                if best_local < best_score:
                    try_candidate(all_exps, best_local_coeffs)

    def try_separated_blocks():
        """
        Two well-separated blocks: A in [0, k-1], B in [k+gap, ...].
        Cross terms occupy [gap+1, 2k-1+gap], self terms A^2 in [0, 2k-2], B^2 in [2k+2gap, ...].
        Choose gap large enough to avoid overlap between A^2 and cross, and cross and B^2.
        """
        if time.time() > deadline:
            return

        for k in range(5, 8):  # block size 5-7, other block 7-5
            other = 12 - k
            for gap in range(1, 15):
                if time.time() > deadline:
                    return

                # A: k terms in [0, 2k-2]
                # B: other terms starting at k + gap
                # A^2: [0, 2k-2], cross: [gap+1, 2k-1+gap], B^2: [2(k+gap), 2(k+gap)+2(other-1)]
                # Need: 2k-2 < gap+1 => gap >= 2k-1
                # Need: 2k-1+gap < 2(k+gap) => gap < 2k-1+gap => always true
                # Also B^2 max: 2(k+gap) + 2(other-1) <= 40

                if gap < 2 * k - 1:
                    continue

                b_start = k + gap
                b_max = 2 * b_start + 2 * (other - 1)
                if b_max > 40:
                    continue

                # Generate A patterns: k terms in [0, 2k-2]
                a_candidates = []
                for start in range(0, 3):
                    for step in range(1, 4):
                        exps = [start + i * step for i in range(k)]
                        if max(exps) < 2 * k - 1:
                            a_candidates.append(tuple(exps))
                if not a_candidates:
                    a_candidates = [(i for i in range(k))]

                # Generate B patterns: other terms in [b_start, b_start + 2*other - 2]
                b_candidates = []
                for start in range(b_start, b_start + 3):
                    for step in range(1, 4):
                        exps = [start + i * step for i in range(other)]
                        if max(exps) <= b_max:
                            b_candidates.append(tuple(exps))
                if not b_candidates:
                    b_candidates = [(b_start + i for i in range(other))]

                for a_exps in a_candidates[:10]:
                    if time.time() > deadline:
                        return
                    for b_exps in b_candidates[:10]:
                        if time.time() > deadline:
                            return
                        all_exps = list(a_exps) + list(b_exps)
                        if len(set(all_exps)) != 12:
                            continue
                        if max(all_exps) > 40:
                            continue

                        # Optimize signs
                        best_local = float('inf')
                        random.seed(11)
                        for trial in range(8):
                            if time.time() > deadline:
                                return
                            if trial == 0:
                                coeffs = [1] * 12
                            else:
                                coeffs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
                            score = compute_square_nonzeros_fast(all_exps, coeffs)
                            if score < best_local:
                                best_local = score
                                best_local_coeffs = coeffs[:]
                            for _ in range(30):
                                if time.time() > deadline:
                                    return
                                improved = False
                                for i in range(12):
                                    old = coeffs[i]
                                    coeffs[i] = -coeffs[i]
                                    new_score = compute_square_nonzeros_fast(all_exps, coeffs)
                                    if new_score < best_local:
                                        best_local = new_score
                                        best_local_coeffs = coeffs[:]
                                        improved = True
                                    else:
                                        coeffs[i] = old
                                if not improved:
                                    break

                        if best_local < best_score:
                            try_candidate(all_exps, best_local_coeffs)

    def try_sparse_exponents():
        """
        Use widely-spaced exponents to minimize collisions.
        E.g., powers of 2, or Fibonacci-like sequences.
        """
        if time.time() > deadline:
            return

        # Powers of 2: 1,2,4,8,16,32,... but 2^5=32, need 12 terms up to 40
        # So use: 0,1,2,4,8,16,32 and fill in some others
        base_patterns = [
            [0, 1, 2, 4, 8, 16, 32, 33, 34, 35, 36, 37],
            [0, 1, 3, 6, 12, 24, 33, 34, 35, 36, 37, 38],
            [0, 2, 4, 8, 16, 32, 33, 34, 35, 36, 37, 38],
            [1, 2, 4, 8, 16, 32, 33, 34, 35, 36, 37, 39],
            [0, 1, 2, 4, 8, 16, 32, 34, 36, 38, 39, 40],
        ]

        for exps in base_patterns:
            if time.time() > deadline:
                return
            if len(set(exps)) != 12 or max(exps) > 40 or min(exps) < 0:
                continue
            optimize_signs(sorted(exps))
            if best_score <= 18:
                optimize_coeffs(sorted(exps))

        # Random sparse sets
        random.seed(200)
        for _ in range(200):
            if time.time() > deadline:
                return
            exps = sorted(random.sample(range(0, 41), 12))
            # Ensure reasonable spacing
            if min(exps) > 10:
                exps[0] = 0
            optimize_signs(exps)

    # --- Main search ---

    # Phase 1: Well-separated blocks (most structural, likely best)
    try_separated_blocks()

    if time.time() > deadline:
        return best_candidate

    # Phase 2: A+B decomposition with collision-aware pattern selection
    try_A_plus_B()

    if time.time() > deadline:
        return best_candidate

    # Phase 3: Sparse exponent patterns
    try_sparse_exponents()

    if time.time() > deadline:
        return best_candidate

    # Phase 4: Systematic AP search
    seen = set()
    for start in range(0, 12):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) > 40:
                continue
            key = tuple(exps)
            if key in seen:
                continue
            seen.add(key)
            if time.time() > deadline:
                return best_candidate
            optimize_signs(exps)
            if best_score <= 16 and time.time() < deadline - 10:
                optimize_coeffs(exps)

    return best_candidate

# EVOLVE-BLOCK-END
