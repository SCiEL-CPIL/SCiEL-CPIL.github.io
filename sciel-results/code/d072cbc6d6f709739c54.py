import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 82.0

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def _update_best(exps, coeffs, score):
        nonlocal best_score, best_candidate
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    def compute_score(exps, coeffs):
        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c
        q = np.convolve(p, p)
        return int(np.count_nonzero(q))

    def vectorized_sign_search(exps, magnitudes):
        """Try all 2048 sign patterns for fixed exponents and magnitudes."""
        n = len(exps)
        if n != 12:
            return float('inf'), None

        max_deg = max(exps)
        if max_deg > 40:
            return float('inf'), None

        base = np.zeros(max_deg + 1, dtype=np.int64)
        for e, m in zip(exps, magnitudes):
            if abs(m) > 64:
                return float('inf'), None
            base[e] = m

        # Generate all 2048 sign patterns as a (2048, 12) array
        # signs[i][j] = (-1)^bit_j of i
        all_signs = np.array([
            [1 if (i >> j) & 1 else -1 for j in range(12)]
            for i in range(2048)
        ], dtype=np.int64)

        best_local = float('inf')
        best_signs = None

        # Process in batches
        for batch_start in range(0, 2048, 256):
            if time.time() > deadline - 2:
                break
            batch_end = min(batch_start + 256, 2048)
            for idx in range(batch_start, batch_end):
                signs = all_signs[idx]
                p = base.copy()
                for j in range(12):
                    p[exps[j]] = base[exps[j]] * signs[j]
                q = np.convolve(p, p)
                count = int(np.count_nonzero(q))
                if count < best_local:
                    best_local = count
                    best_signs = signs

        if best_signs is None:
            return float('inf'), None

        coeffs = [int(best_signs[j] * magnitudes[j]) for j in range(12)]
        return best_local, coeffs

    def generate_exponent_structures():
        """Generate diverse exponent structures with high collision potential."""
        structures = []

        # 1. Two-cluster: k consecutive + (12-k) consecutive, separated by gap
        for k in range(1, 12):
            for a in range(0, 8):
                for gap in range(1, 15):
                    b = a + k + gap
                    block1 = [a + i for i in range(k)]
                    block2 = [b + i for i in range(12 - k)]
                    if max(block2) <= 40:
                        exps = sorted(block1 + block2)
                        if len(set(exps)) == 12:
                            structures.append(exps)

        # 2. Three-cluster: 4+4+4 with gaps
        for a in range(0, 5):
            for g1 in range(1, 10):
                for g2 in range(1, 10):
                    b = a + 4 + g1
                    c = b + 4 + g2
                    exps = list(range(a, a+4)) + list(range(b, b+4)) + list(range(c, c+4))
                    if max(exps) <= 40:
                        structures.append(sorted(exps))

        # 3. Four-cluster: 3+3+3+3
        for a in range(0, 4):
            for g1 in range(1, 8):
                for g2 in range(1, 8):
                    for g3 in range(1, 8):
                        b = a + 3 + g1
                        c = b + 3 + g2
                        d = c + 3 + g3
                        exps = list(range(a, a+3)) + list(range(b, b+3)) + list(range(c, c+3)) + list(range(d, d+3))
                        if max(exps) <= 40:
                            structures.append(sorted(exps))

        # 4. Arithmetic progressions
        for a in range(0, 10):
            for d in range(1, 5):
                exps = [a + i * d for i in range(12)]
                if max(exps) <= 40:
                    structures.append(exps)

        # 5. Consecutive
        for a in range(0, 30):
            exps = list(range(a, a + 12))
            if max(exps) <= 40:
                structures.append(exps)

        # 6. Symmetric patterns around a center
        for center in range(6, 35):
            for spread in range(1, 15):
                # Try symmetric: center-spread, center-spread+1, ..., center+spread-1
                # Pick 12 from a symmetric range
                lo = center - spread
                hi = center + spread + 1
                if lo >= 0 and hi <= 41:
                    candidates = list(range(lo, hi))
                    if len(candidates) >= 12:
                        # Take 12 most central
                        mid = len(candidates) // 2
                        start = max(0, mid - 6)
                        exps = candidates[start:start+12]
                        if len(exps) == 12:
                            structures.append(sorted(exps))

        # 7. Random structured sets
        random.seed(12345)
        for _ in range(100):
            # Generate with some structure: pick a base pattern and perturb
            base = sorted(random.sample(range(0, 41), 12))
            structures.append(base)

        # 8. Specific patterns known to create collisions
        # Pattern: exponents where many pairs sum to same value
        # e.g., {0,1,2,3, 6,7,8,9, 12,13,14,15} - three blocks of 4
        structures.append([0,1,2,3,6,7,8,9,12,13,14,15])
        structures.append([0,1,2,3,5,6,7,8,10,11,12,13])
        structures.append([0,1,2,3,4,5,8,9,10,11,12,13])
        structures.append([0,1,2,4,5,6,8,9,10,12,13,14])
        structures.append([0,2,4,6,8,10,12,14,16,18,20,22])
        structures.append([0,1,3,4,6,7,9,10,12,13,15,16])
        structures.append([0,2,3,5,6,8,9,11,12,14,15,17])
        structures.append([0,1,4,5,8,9,12,13,16,17,20,21])
        structures.append([0,3,4,7,8,11,12,15,16,19,20,23])

        # Deduplicate
        seen = set()
        unique = []
        for s in structures:
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                unique.append(s)

        return unique

    def collision_count(exps):
        """Count how many pairwise sums collide (higher = more cancellation potential)."""
        sums = {}
        for i in range(12):
            for j in range(i, 12):
                s = exps[i] + exps[j]
                sums[s] = sums.get(s, 0) + 1
        # Total pairs = 78. If all sums distinct, 78 sums. Collisions = 78 - len(sums)
        return 78 - len(sums)

    def local_sign_search(exps, coeffs, iterations=2000):
        """Hill-climbing on signs for fixed exponents."""
        current = coeffs[:]
        current_score = compute_score(exps, current)
        T = 3.0
        T_min = 0.3
        cooling = 0.995

        for it in range(iterations):
            if time.time() > deadline - 2:
                break
            i = random.randint(0, 11)
            new_coeffs = current[:]
            new_coeffs[i] = -new_coeffs[i]
            new_score = compute_score(exps, new_coeffs)
            if new_score < current_score:
                current = new_coeffs
                current_score = new_score
            else:
                delta = new_score - current_score
                if T > T_min:
                    if random.random() < np.exp(-delta / T):
                        current = new_coeffs
                        current_score = new_score
            T = max(T * cooling, T_min)

        return current_score, current

    # --- Main search ---
    random.seed(42)

    # Generate and sort exponent structures by collision potential
    all_structures = generate_exponent_structures()

    # Score by collision count (higher = more potential for cancellation)
    scored_structures = []
    for exps in all_structures:
        cc = collision_count(exps)
        scored_structures.append((cc, exps))

    # Sort: most collisions first
    scored_structures.sort(key=lambda x: -x[0])

    # Phase 1: Exhaustive sign search on top structures
    num_tried = 0
    for cc, exps in scored_structures:
        if time.time() > deadline - 8:
            break
        if max(exps) > 40:
            continue

        score, coeffs = vectorized_sign_search(exps, [1] * 12)
        if score < float('inf') and coeffs is not None:
            _update_best(exps, coeffs, score)
            num_tried += 1

        if best_score <= 13:
            break

    # Phase 2: Local search refinement on best structures
    if time.time() < deadline - 5 and best_score > 13:
        # Take top structures and do local search
        for cc, exps in scored_structures[:30]:
            if time.time() > deadline - 3:
                break
            if max(exps) > 40:
                continue
            score, coeffs = local_sign_search(exps, [1] * 12, iterations=500)
            if score < best_score:
                _update_best(exps, coeffs, score)
            if best_score <= 13:
                break

    # Phase 3: Try with non-unit magnitudes on best structures
    if time.time() < deadline - 5 and best_score > 13:
        mag_patterns = [
            [1]*12,
            [1,1,1,1,2,2,2,2,1,1,1,1],
            [1,2,1,2,1,2,1,2,1,2,1,2],
            [1,1,2,1,1,1,2,1,1,1,2,1],
            [2,1,1,1,1,1,1,1,1,1,1,2],
        ]
        for cc, exps in scored_structures[:20]:
            if time.time() > deadline - 3:
                break
            for mag in mag_patterns:
                if time.time() > deadline - 2:
                    break
                score, coeffs = vectorized_sign_search(exps, mag)
                if score < float('inf') and coeffs is not None:
                    _update_best(exps, coeffs, score)
            if best_score <= 13:
                break

    # Phase 4: Random search with sign optimization
    if time.time() < deadline - 5 and best_score > 13:
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            exps = sorted(random.sample(range(0, 41), 12))
            score, coeffs = vectorized_sign_search(exps, [1] * 12)
            if score < float('inf') and coeffs is not None:
                _update_best(exps, coeffs, score)

    # Phase 5: Final local search on the best candidate
    if time.time() < deadline - 2 and best_score > 13:
        exps = [i for i, c in enumerate(best_candidate) if c != 0]
        coeffs = [c for c in best_candidate if c != 0]
        if len(exps) == 12:
            score, new_coeffs = local_sign_search(exps, coeffs, iterations=1000)
            if score < best_score:
                _update_best(exps, new_coeffs, score)

    # Phase 6: Try to modify exponents of the best candidate
    if time.time() < deadline - 2 and best_score > 13:
        exps = [i for i, c in enumerate(best_candidate) if c != 0]
        coeffs = [c for c in best_candidate if c != 0]
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            new_exps = exps[:]
            i = random.randint(0, 11)
            delta = random.choice([-2, -1, 1, 2])
            new_e = new_exps[i] + delta
            if 0 <= new_e <= 40 and new_e not in new_exps:
                new_exps[i] = new_e
                new_exps.sort()
                idxs = np.argsort(new_exps)
                new_exps = [new_exps[i] for i in idxs]
                new_coeffs = [coeffs[i] for i in idxs]
                score = compute_score(new_exps, new_coeffs)
                if score < best_score:
                    _update_best(new_exps, new_coeffs, score)

    return best_candidate

# EVOLVE-BLOCK-END
