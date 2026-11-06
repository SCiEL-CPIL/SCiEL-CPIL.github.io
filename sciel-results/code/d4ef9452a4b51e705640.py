"""Rewritten program: sparse squares of 12-term integer polynomials."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before 90s kill

    best_poly = None
    best_nz = float('inf')

    def count_nonzero(arr):
        return int(np.count_nonzero(arr))

    def square_nonzero(exponents, coeffs):
        """Fast exact square nonzero count via numpy."""
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        # (12,12) outer
        degs = e[:, None] + e[None, :]
        vals = c[:, None] * c[None, :]
        # diagonal (i==j) counted once, off-diagonal pairs counted twice
        # We can just sum over all pairs and then halve the off-diagonal... 
        # Actually: P^2 = sum_{i,j} c_i c_j x^{e_i+e_j}
        # Each pair (i,j) with i!=j appears twice (once as (i,j) and once as (j,i))
        # So just sum all 144 entries grouped by degree
        result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=82)
        return int(np.count_nonzero(result))

    def validate(poly_coeff_list):
        if not poly_coeff_list or len(poly_coeff_list) > 41:
            return False
        if poly_coeff_list[-1] == 0:
            return False
        for c in poly_coeff_list:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if sum(1 for c in poly_coeff_list if c != 0) != 12:
            return False
        return True

    def try_candidate(exponents, coeffs):
        nonlocal best_poly, best_nz
        if len(exponents) != 12 or len(set(exponents)) != 12:
            return
        if any(e < 0 or e >= 41 for e in exponents):
            return
        if any(c == 0 or abs(c) > 64 for c in coeffs):
            return
        nz = square_nonzero(exponents, coeffs)
        if nz < best_nz:
            best_nz = nz
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                poly[e] = c
            best_poly = poly

    # Phase 1: Structured AP exponents with many coefficient patterns
    sign_patterns = []
    for period in [1, 2, 3, 4, 6, 12]:
        for phase in range(period):
            pat = [1 if ((i + phase) % (2 * period)) < period else -1 for i in range(12)]
            sign_patterns.append(pat)
    # Additional structured patterns
    extra_patterns = [
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
        [1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1],
        [2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2],
        [1, 1, -2, -1, 1, 1, -2, -1, 1, 1, -2, -1],
        [3, 1, -1, -3, -1, 1, 3, 1, -1, -3, -1, 1],
        [1, 3, -1, -3, 1, 3, -1, -3, 1, 3, -1, -3],
    ]
    all_patterns = sign_patterns + extra_patterns

    for d in range(1, 8):
        if time.time() > deadline - 30:
            break
        max_a = min(40 - 11 * d, 15)
        for a in range(0, max_a + 1):
            if time.time() > deadline - 30:
                break
            exponents = [a + i * d for i in range(12)]
            if exponents[-1] >= 41:
                continue
            for cp in all_patterns:
                if time.time() > deadline - 30:
                    break
                try_candidate(exponents, cp)

    # Phase 2: Two-AP union structures (6+6 terms)
    if time.time() < deadline - 25:
        for d1 in range(1, 5):
            for d2 in range(1, 5):
                if time.time() > deadline - 25:
                    break
                for a1 in range(0, 12):
                    for gap in range(3, 22):
                        if time.time() > deadline - 25:
                            break
                        a2 = a1 + gap
                        exp1 = [a1 + i * d1 for i in range(6)]
                        exp2 = [a2 + i * d2 for i in range(6)]
                        exponents = sorted(exp1 + exp2)
                        if len(set(exponents)) != 12 or any(e >= 41 for e in exponents):
                            continue
                        # Try sign patterns for the two groups
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                if time.time() > deadline - 25:
                                    break
                                # Uniform signs per group
                                coeffs = [s1] * 6 + [s2] * 6
                                try_candidate(exponents, coeffs)
                                # Alternating within each group
                                c_alt = [s1 * (1 if i % 2 == 0 else -1) for i in range(6)] + \
                                        [s2 * (1 if i % 2 == 0 else -1) for i in range(6)]
                                try_candidate(exponents, c_alt)
                                # Blocks of 2 within groups
                                c_blk = [s1 * (1 if (i // 2) % 2 == 0 else -1) for i in range(6)] + \
                                        [s2 * (1 if (i // 2) % 2 == 0 else -1) for i in range(6)]
                                try_candidate(exponents, c_blk)

    # Phase 3: Symmetric exponent structures
    if time.time() < deadline - 20:
        for center in range(6, 35):
            if time.time() > deadline - 20:
                break
            for offsets in [
                [1, 2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5, 7],
                [1, 2, 3, 4, 6, 8],
                [1, 2, 3, 5, 7, 9],
                [1, 3, 5, 7, 9, 11],
                [2, 4, 6, 8, 10, 12],
                [1, 2, 4, 5, 7, 8],
                [1, 3, 4, 6, 7, 9],
                [1, 2, 4, 6, 8, 10],
                [2, 3, 4, 5, 6, 7],
            ]:
                if time.time() > deadline - 20:
                    break
                exponents = sorted([center - o for o in offsets] + [center + o for o in offsets])
                if any(e < 0 or e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                for cp in all_patterns[:20]:
                    if time.time() > deadline - 20:
                        break
                    try_candidate(exponents, cp)

    # Phase 4: Randomized search with diverse strategies
    random.seed(42)

    def random_search(num_trials=200000):
        nonlocal best_poly, best_nz
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                break

            r = random.random()
            if r < 0.20:
                # AP-like exponents
                d = random.randint(1, 6)
                max_a = max(0, 40 - 11 * d)
                a = random.randint(0, max_a)
                exponents = [a + i * d for i in range(12)]
                if exponents[-1] >= 41:
                    continue
            elif r < 0.35:
                # Two-cluster: 6+6 in two AP groups
                d1 = random.randint(1, 4)
                d2 = random.randint(1, 4)
                a1 = random.randint(0, 10)
                gap = random.randint(4, 18)
                a2 = a1 + gap
                exponents = [a1 + i * d1 for i in range(6)] + [a2 + i * d2 for i in range(6)]
                if any(e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                exponents.sort()
            elif r < 0.50:
                # Symmetric around center
                center = random.randint(6, 34)
                offsets = sorted(random.sample(range(1, 20), 6))
                exponents = [center - o for o in offsets] + [center + o for o in offsets]
                if any(e < 0 or e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                exponents.sort()
            elif r < 0.65:
                # Clustered in a small window
                window = random.randint(14, 28)
                base = random.randint(0, max(0, 40 - window))
                exponents = sorted(random.sample(range(base, min(base + window, 41)), 12))
            elif r < 0.80:
                # 3+4+5 cluster
                d1 = random.randint(1, 3)
                d2 = random.randint(1, 3)
                d3 = random.randint(1, 3)
                a1 = random.randint(0, 5)
                g1 = random.randint(3, 8)
                a2 = a1 + g1
                g2 = random.randint(3, 8)
                a3 = a2 + g2
                exponents = ([a1 + i * d1 for i in range(3)] +
                             [a2 + i * d2 for i in range(4)] +
                             [a3 + i * d3 for i in range(5)])
                if any(e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                exponents.sort()
            else:
                # Fully random
                exponents = sorted(random.sample(range(41), 12))

            # Choose coefficients
            cstyle = random.random()
            if cstyle < 0.30:
                coeffs = [random.choice([-1, 1]) for _ in range(12)]
            elif cstyle < 0.50:
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            elif cstyle < 0.70:
                pat = random.choice(all_patterns)
                coeffs = list(pat)
            elif cstyle < 0.85:
                coeffs = [random.randint(-3, 3) for _ in range(12)]
                for i in range(12):
                    while coeffs[i] == 0:
                        coeffs[i] = random.randint(1, 3) * random.choice([-1, 1])
            else:
                coeffs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]

            if all(abs(c) <= 64 and c != 0 for c in coeffs):
                try_candidate(exponents, coeffs)

    random_search(200000)

    # Phase 5: Local search / hill climbing from best
    if best_poly is not None and time.time() < deadline - 5:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]

        for restart in range(8):
            if time.time() > deadline - 3:
                break

            cur_exp = list(best_exp)
            cur_coef = list(best_coef)
            cur_nz = square_nonzero(cur_exp, cur_coef)

            for it in range(5000):
                if time.time() > deadline - 3:
                    break

                exp_copy = list(cur_exp)
                coef_copy = list(cur_coef)

                r = random.random()
                if r < 0.40:
                    # Change a coefficient
                    idx = random.randint(0, 11)
                    new_c = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 12])
                    if new_c != coef_copy[idx]:
                        coef_copy[idx] = new_c
                elif r < 0.70:
                    # Move an exponent
                    idx = random.randint(0, 11)
                    shift = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
                    new_e = exp_copy[idx] + shift
                    if 0 <= new_e < 41 and new_e not in exp_copy:
                        exp_copy[idx] = new_e
                        exp_copy.sort()
                    else:
                        continue
                elif r < 0.80:
                    # Swap two exponents
                    i, j = random.sample(range(12), 2)
                    exp_copy[i], exp_copy[j] = exp_copy[j], exp_copy[i]
                    exp_copy.sort()
                elif r < 0.90:
                    # Change two coefficients
                    i, j = random.sample(range(12), 2)
                    coef_copy[i] = random.choice([-3, -2, -1, 1, 2, 3, 4])
                    coef_copy[j] = random.choice([-3, -2, -1, 1, 2, 3, 4])
                else:
                    # Move two exponents
                    i, j = random.sample(range(12), 2)
                    si = random.choice([-2, -1, 1, 2, 3])
                    sj = random.choice([-2, -1, 1, 2, 3])
                    new_ei = exp_copy[i] + si
                    new_ej = exp_copy[j] + sj
                    if (0 <= new_ei < 41 and 0 <= new_ej < 41 and
                            new_ei not in exp_copy and new_ej not in exp_copy and
                            new_ei != new_ej):
                        exp_copy[i] = new_ei
                        exp_copy[j] = new_ej
                        exp_copy.sort()
                    else:
                        continue

                if len(set(exp_copy)) != 12:
                    continue
                if any(e < 0 or e >= 41 for e in exp_copy):
                    continue
                if any(c == 0 or abs(c) > 64 for c in coef_copy):
                    continue

                nz = square_nonzero(exp_copy, coef_copy)
                if nz < cur_nz:
                    cur_nz = nz
                    cur_exp = list(exp_copy)
                    cur_coef = list(coef_copy)

                    if nz < best_nz:
                        best_nz = nz
                        max_deg = max(cur_exp)
                        poly_new = [0] * (max_deg + 1)
                        for e, c in zip(cur_exp, cur_coef):
                            poly_new[e] = c
                        best_poly = poly_new
                        best_exp = list(cur_exp)
                        best_coef = list(cur_coef)

    # Final fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    # Final validation
    if not validate(best_poly):
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return best_poly

# EVOLVE-BLOCK-END
