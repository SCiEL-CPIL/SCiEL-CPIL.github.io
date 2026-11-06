import time
import numpy as np
import random
from itertools import product

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def compute_score_fast(exps, coeffs):
        """
        Fast computation of nonzero terms in P(x)^2.
        exps: list of 12 exponents (sorted)
        coeffs: list of 12 coefficients
        Returns count of nonzero terms, or inf if invalid.
        """
        if not exps or not coeffs:
            return float('inf')

        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')

        # Build dense array
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c

        q = np.convolve(p, p)
        count = int(np.count_nonzero(q))
        return count

    def _update_best(exps, coeffs, score):
        """Update the global best candidate if improved."""
        nonlocal best_score, best_candidate
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    def exhaustive_sign_search(exps, coeff_magnitudes):
        """
        For a fixed set of exponents and coefficient magnitudes,
        try all 2^12 sign patterns to find the one minimizing nonzero terms in P^2.
        """
        nonlocal best_score, best_candidate

        n = len(exps)
        if n != 12:
            return

        # Build the polynomial once with all positive coefficients
        max_deg = max(exps)
        if max_deg > 40:
            return

        base_p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeff_magnitudes):
            if abs(c) > 64:
                return
            base_p[e] = c

        best_local_score = float('inf')
        best_local_signs = None

        # Try all sign patterns
        for sign_tuple in product([-1, 1], repeat=12):
            if time.time() > deadline - 2:
                break

            p = base_p.copy()
            for i in range(12):
                p[exps[i]] = base_p[exps[i]] * sign_tuple[i]

            q = np.convolve(p, p)
            score = int(np.count_nonzero(q))

            if score < best_local_score:
                best_local_score = score
                best_local_signs = sign_tuple

        if best_local_signs is not None and best_local_score < best_score:
            # Reconstruct
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for i, (e, c) in enumerate(zip(exps, coeff_magnitudes)):
                dense[e] = c * best_local_signs[i]
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = best_local_score
                best_candidate = dense

    def generate_exponent_sets():
        """
        Generate a diverse set of exponent configurations to try.
        """
        sets = []

        # 1. Consecutive blocks
        for start in range(0, 30):
            exps = list(range(start, start + 12))
            if max(exps) <= 40:
                sets.append(exps)

        # 2. Arithmetic progressions with small gaps
        for gap in [2, 3, 4, 5, 6]:
            for start in range(0, 5):
                exps = [start + i * gap for i in range(12)]
                if max(exps) <= 40:
                    sets.append(exps)

        # 3. Block patterns: e.g., 3 consecutive, gap, 3 consecutive, gap, 3, gap, 3
        for block_size in [3, 4]:
            for gap in [1, 2, 3, 4]:
                n_blocks = 12 // block_size
                for start in range(0, 5):
                    exps = []
                    for b in range(n_blocks):
                        base = start + b * (block_size + gap)
                        for i in range(block_size):
                            exps.append(base + i)
                    if len(exps) == 12 and max(exps) <= 40:
                        sets.append(exps)

        # 4. Power-of-2 like patterns
        for start in range(0, 3):
            exps = [start + i * 3 for i in range(12)]
            if max(exps) <= 40:
                sets.append(exps)

        # 5. Random sets
        random.seed(42)
        for _ in range(200):
            exps = sorted(random.sample(range(0, 41), 12))
            sets.append(exps)

        # Remove duplicates
        seen = set()
        unique_sets = []
        for s in sets:
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                unique_sets.append(s)

        return unique_sets

    def factor_product_score(factors):
        """
        Given 3 factors, each a list of (exp, coeff) pairs (4 each),
        compute the number of nonzero terms in P(x)^2 where P = F1*F2*F3.
        """
        max_exp = max(max(e for e, c in f) for f in factors)

        if max_exp > 13:
            return float('inf')

        arrays = []
        for f in factors:
            arr = np.zeros(max_exp + 1, dtype=np.int64)
            for e, c in f:
                if 0 <= e <= max_exp and abs(c) <= 64 and c != 0:
                    arr[e] = c
                else:
                    return float('inf')
            arrays.append(arr)

        p = np.convolve(np.convolve(arrays[0], arrays[1]), arrays[2])

        if len(p) > 41:
            return float('inf')
        if p[-1] == 0:
            while len(p) > 1 and p[-1] == 0:
                p = p[:-1]
            if p[-1] == 0:
                return float('inf')

        nonzero_in_p = np.count_nonzero(p)
        if nonzero_in_p != 12:
            return float('inf')

        if any(abs(c) > 64 for c in p):
            return float('inf')

        q = np.convolve(p, p)
        return int(np.count_nonzero(q))

    def optimize_factor_structure():
        """
        Search over 3-factor decompositions.
        """
        nonlocal best_score, best_candidate

        # Generate candidate factor structures
        base_structures = []

        # Structure 1: Consecutive small exponents
        for s1 in range(0, 3):
            for s2 in range(0, 3):
                for s3 in range(0, 3):
                    f1 = [(s1 + i, 1) for i in range(4)]
                    f2 = [(s2 + i, 1) for i in range(4)]
                    f3 = [(s3 + i, 1) for i in range(4)]
                    base_structures.append((f1, f2, f3))

        # Structure 2: Gapped exponents
        for gap in [2, 3, 4]:
            for offset in range(0, 5):
                f1 = [(offset + i * gap, 1) for i in range(4)]
                f2 = [(offset + i * gap, 1) for i in range(4)]
                f3 = [(offset + i * gap, 1) for i in range(4)]
                if max(e for e, c in f3) <= 13:
                    base_structures.append((f1, f2, f3))

        # Structure 3: Mixed
        f1 = [(0, 1), (1, 1), (2, 1), (4, 1)]
        f2 = [(0, 1), (1, 1), (2, 1), (3, 1)]
        f3 = [(0, 1), (1, 1), (2, 1), (3, 1)]
        base_structures.append((f1, f2, f3))

        # For each structure, try different coefficient patterns
        for f1, f2, f3 in base_structures:
            if time.time() > deadline - 15:
                break

            for signs in product([-1, 1], repeat=12):
                if time.time() > deadline - 10:
                    break

                sf1 = [(e, c * signs[i]) for i, (e, c) in enumerate(f1)]
                sf2 = [(e, c * signs[i + 4]) for i, (e, c) in enumerate(f2)]
                sf3 = [(e, c * signs[i + 8]) for i, (e, c) in enumerate(f3)]

                score = factor_product_score([sf1, sf2, sf3])
                if score < float('inf') and score < best_score:
                    max_exp = max(max(e for e, c in f) for f in [sf1, sf2, sf3])
                    arrs = []
                    for f in [sf1, sf2, sf3]:
                        arr = np.zeros(max_exp + 1, dtype=np.int64)
                        for e, c in f:
                            arr[e] = c
                        arrs.append(arr)
                    p = np.convolve(np.convolve(arrs[0], arrs[1]), arrs[2])
                    while len(p) > 1 and p[-1] == 0:
                        p = p[:-1]
                    if len(p) <= 41 and p[-1] != 0:
                        dense = list(p)
                        best_score = score
                        best_candidate = dense

        # Random factor structures
        if time.time() < deadline - 10:
            for _ in range(50):
                if time.time() > deadline - 5:
                    break

                f1_exps = sorted(random.sample(range(0, 8), 4))
                f2_exps = sorted(random.sample(range(0, 8), 4))
                f3_exps = sorted(random.sample(range(0, 8), 4))

                f1 = [(e, random.choice([-1, 1])) for e in f1_exps]
                f2 = [(e, random.choice([-1, 1])) for e in f2_exps]
                f3 = [(e, random.choice([-1, 1])) for e in f3_exps]

                score = factor_product_score([f1, f2, f3])
                if score < float('inf') and score < best_score:
                    max_exp = max(max(e for e, c in f) for f in [f1, f2, f3])
                    arrs = []
                    for f in [f1, f2, f3]:
                        arr = np.zeros(max_exp + 1, dtype=np.int64)
                        for e, c in f:
                            arr[e] = c
                        arrs.append(arr)
                    p = np.convolve(np.convolve(arrs[0], arrs[1]), arrs[2])
                    while len(p) > 1 and p[-1] == 0:
                        p = p[:-1]
                    if len(p) <= 41 and p[-1] != 0:
                        dense = list(p)
                        best_score = score
                        best_candidate = dense

    def local_search_from_candidate():
        """
        Given the current best candidate, do local search by modifying
        the underlying factor structure or directly modifying P.
        """
        nonlocal best_score, best_candidate

        if time.time() > deadline - 5:
            return

        exps = [i for i, c in enumerate(best_candidate) if c != 0]
        coeffs = [c for c in best_candidate if c != 0]

        if len(exps) != 12:
            return

        T = 5.0
        T_min = 0.5
        cooling = 0.99

        current_score = compute_score_fast(exps, coeffs)

        for it in range(3000):
            if time.time() > deadline - 3:
                break

            i = random.randint(0, 11)
            new_coeffs = coeffs[:]
            new_coeffs[i] = -new_coeffs[i]

            new_score = compute_score_fast(exps, new_coeffs)

            if new_score < current_score:
                coeffs = new_coeffs
                current_score = new_score
                _update_best(exps, coeffs, current_score)
            else:
                delta = new_score - current_score
                if T > T_min:
                    if random.random() < np.exp(-delta / T):
                        coeffs = new_coeffs
                        current_score = new_score

            T = max(T * cooling, T_min)

    def genetic_algorithm(pop_size=20, generations=50):
        """
        Genetic algorithm with crossover and mutation to explore the space.
        """
        nonlocal best_score, best_candidate

        population = []

        def generate_random_individual():
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1, -2, 2, -3, 3]) for _ in range(12)]
            return exps, coeffs

        structured_candidates = []
        for start in range(0, 5):
            exps = [start + i for i in range(12)]
            if max(exps) <= 40:
                for pattern in [
                    [1] * 12,
                    [1 if i % 2 == 0 else -1 for i in range(12)],
                    [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
                    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
                ]:
                    structured_candidates.append((exps[:], pattern[:]))

        for _ in range(pop_size - len(structured_candidates)):
            structured_candidates.append(generate_random_individual())

        population = structured_candidates[:pop_size]

        for exps, coeffs in population:
            score = compute_score_fast(exps, coeffs)
            _update_best(exps, coeffs, score)

        for gen in range(generations):
            if time.time() > deadline - 5:
                break

            scored_population = []
            for exps, coeffs in population:
                score = compute_score_fast(exps, coeffs)
                scored_population.append((score, exps, coeffs))

            scored_population.sort(key=lambda x: x[0])

            elite_count = max(1, pop_size // 2)
            new_population = []
            for _, exps, coeffs in scored_population[:elite_count]:
                new_population.append((exps[:], coeffs[:]))

            while len(new_population) < pop_size:
                if time.time() > deadline - 5:
                    break

                def tournament_select(pop, k=3):
                    candidates = random.sample(pop, min(k, len(pop)))
                    return min(candidates, key=lambda x: x[0])

                parent1 = tournament_select(scored_population)
                parent2 = tournament_select(scored_population)

                child_exps = []
                child_coeffs = []
                for i in range(12):
                    if random.random() < 0.5:
                        child_exps.append(parent1[1][i])
                        child_coeffs.append(parent1[2][i])
                    else:
                        child_exps.append(parent2[1][i])
                        child_coeffs.append(parent2[2][i])

                if random.random() < 0.3:
                    idx = random.randint(0, 11)
                    child_coeffs[idx] = random.choice([-1, 1, -2, 2, -3, 3])

                if random.random() < 0.2:
                    idx = random.randint(0, 11)
                    delta = random.choice([-1, 1])
                    new_e = child_exps[idx] + delta
                    if 0 <= new_e <= 40 and new_e not in child_exps:
                        child_exps[idx] = new_e
                        child_exps.sort()
                        idxs = np.argsort(child_exps)
                        child_exps = [child_exps[i] for i in idxs]
                        child_coeffs = [child_coeffs[i] for i in idxs]

                new_population.append((child_exps, child_coeffs))

            population = new_population

        for exps, coeffs in population:
            score = compute_score_fast(exps, coeffs)
            _update_best(exps, coeffs, score)

    # --- Main search loop ---

    random.seed(42)

    # Phase 1: Exhaustive sign search over structured exponent sets
    # This is the key improvement: systematically try all 2^12 sign patterns
    # for each candidate exponent configuration
    exponent_sets = generate_exponent_sets()

    # Sort by "promisingness" - prefer denser, more regular patterns
    def promisingness(exps):
        # Prefer consecutive or arithmetic progressions
        diffs = [exps[i+1] - exps[i] for i in range(11)]
        if all(d == 1 for d in diffs):
            return 0
        if all(d == diffs[0] for d in diffs):
            return 1
        # Prefer smaller gaps
        return sum(diffs)

    exponent_sets.sort(key=promisingness)

    # Try each exponent set with unit coefficients first (most common case)
    for exps in exponent_sets:
        if time.time() > deadline - 5:
            break
        if max(exps) > 40:
            continue
        exhaustive_sign_search(exps, [1] * 12)

    # Phase 2: Try some with mixed coefficient magnitudes
    if time.time() < deadline - 10:
        mixed_mag_sets = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
            [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4],
        ]
        for exps in exponent_sets[:50]:
            if time.time() > deadline - 5:
                break
            if max(exps) > 40:
                continue
            for mags in mixed_mag_sets:
                if time.time() > deadline - 5:
                    break
                if any(abs(m) > 64 for m in mags):
                    continue
                exhaustive_sign_search(exps, mags)

    # Phase 3: Factor structure search
    if time.time() < deadline - 10:
        optimize_factor_structure()

    # Phase 4: Genetic algorithm
    if time.time() < deadline - 10:
        genetic_algorithm(pop_size=20, generations=50)

    # Phase 5: Local search on the result
    local_search_from_candidate()

    # Phase 6: If we haven't found something great, try direct random search
    if best_score > 15 and time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            score = compute_score_fast(exps, coeffs)
            if score < best_score:
                max_deg = max(exps)
                dense = [0] * (max_deg + 1)
                for e, c in zip(exps, coeffs):
                    dense[e] = c
                if len(dense) <= 41 and dense[-1] != 0:
                    best_score = score
                    best_candidate = dense

    return best_candidate

# EVOLVE-BLOCK-END
