"""Sparse squares of 12-term integer polynomials – DFT spectral search."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 85.0

    best_poly = None
    best_nz = float('inf')

    N = 8192  # 2^13
    # Precompute 2^13-th roots of unity
    omega = np.exp(2j * np.pi / N)

    def square_nonzero(exponents, coeffs):
        """Exact square nonzero count."""
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        degs = e[:, None] + e[None, :]
        vals = c[:, None] * c[None, :]
        result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=82)
        return int(np.count_nonzero(result))

    def spectral_l1(exponents, coeffs):
        """
        Compute L1 norm of |P(z_k)^2| over 2^13-th roots of unity.
        This upper-bounds the time-domain L1 norm of the square.
        """
        c = np.asarray(coeffs, dtype=np.float64)
        e = np.asarray(exponents, dtype=np.int64)
        # P(z_k) = sum_i c_i * z_k^{e_i} = sum_i c_i * omega^{k*e_i}
        # Compute P at all N points efficiently
        # P_k = sum_i c_i * omega^{k * e_i}
        # Use matrix: M[k, i] = omega^{k * e_i}
        # For speed, compute powers for each exponent
        # P = M @ c
        # But N=8192, 12 exponents -> 8192x12 matrix, fine
        # omega^{k*e_i} for k=0..N-1, i=0..11
        # Precompute omega^{e_i} for each i, then omega^{k*e_i} = (omega^{e_i})^k
        pw = omega ** e  # shape (12,)
        # P[k, i] = pw[i]^k for k=0..N-1
        # P = pw^k for each k -> use outer: for each i, pw[i]**arange(N)
        # Actually P[k] = sum_i c[i] * pw[i]**k
        # Vectorize: for each i, pw[i]**np.arange(N) is a geometric sequence
        # P = sum_i c[i] * (pw[i] ** arange(N))
        arange_N = np.arange(N)
        # Shape: (12, N) -> sum over axis 0 with weights c
        # pw[i] ** arange_N[k] = exp(k * log(pw[i]))
        # But pw[i] is complex root of unity, so pw[i]**k cycles
        # Direct: P = np.zeros(N, dtype=complex)
        # for i in range(12): P += c[i] * (pw[i] ** arange_N)
        # Vectorized:
        # log_pw = np.log(pw)  # but roots of unity, log is multi-valued
        # Better: just compute directly
        # pw ** arange_N[:,None] would be (N, 12)
        # Actually: result[k] = sum_i c[i] * pw[i]**k
        # = np.sum(c * (pw ** arange_N[None,:]), axis=0)  -- no
        # pw ** arange_N[:, None] has shape (N, 12) where entry [k,i] = pw[i]**k
        # Then P = (pw ** arange_N[:, None]) @ c  -> shape (N,)
        P_vals = (pw ** arange_N[:, None]) @ c  # shape (N,)
        # Square: |P(z_k)|^2
        sq = np.abs(P_vals) ** 2
        return float(np.sum(sq))

    def spectral_l1_grad(exponents, coeffs):
        """
        Compute gradient of spectral L1 w.r.t. coefficients.
        P(z_k) = sum_i c_i * w_i^k where w_i = omega^{e_i}
        |P|^2 = P * conj(P)
        d|P|^2/dc_i = conj(P) * w_i^k + P * conj(w_i^k)  ... for each k
        Sum over k: grad_i = sum_k (conj(P_k) * w_i^k + P_k * conj(w_i^k))
        """
        c = np.asarray(coeffs, dtype=np.float64)
        e = np.asarray(exponents, dtype=np.int64)
        pw = omega ** e  # (12,)
        arange_N = np.arange(N)
        P_vals = (pw ** arange_N[:, None]) @ c  # (N,)
        conj_P = np.conj(P_vals)
        # grad_i = sum_k (conj_P_k * pw_i^k + P_k * conj(pw_i)^k)
        # = sum_k conj_P_k * pw_i^k + sum_k P_k * conj(pw_i)^k
        # First term: (conj_P @ (pw_i^k for k)) = conj_P dot (pw_i ** arange_N)
        # pw_i ** arange_N for each i: shape (12, N)
        pw_powers = pw[None, :] ** arange_N[:, None]  # (N, 12)
        # First term: sum_k conj_P[k] * pw_powers[k, i] = (conj_P * pw_powers).sum(axis=0)
        term1 = (conj_P[:, None] * pw_powers).sum(axis=0)  # (12,)
        term2 = (P_vals[:, None] * np.conj(pw_powers)).sum(axis=0)  # (12,)
        grad = term1 + term2
        return np.real(grad)  # gradient w.r.t. real coefficients

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

    # ---------- Phase 1: DFT-guided gradient descent on coefficients ----------
    # Start from structured exponent patterns, then optimize coefficients via spectral gradient
    random.seed(42)

    # Generate diverse exponent structures
    def gen_exponents_ap(d, a):
        return [a + i * d for i in range(12)]

    def gen_exponents_two_ap(d1, a1, d2, a2):
        e1 = [a1 + i * d1 for i in range(6)]
        e2 = [a2 + i * d2 for i in range(6)]
        e = sorted(e1 + e2)
        return e if len(set(e)) == 12 and all(0 <= x < 41 for x in e) else None

    def gen_exponents_symmetric(center, offsets):
        e = sorted([center - o for o in offsets] + [center + o for o in offsets])
        return e if len(set(e)) == 12 and all(0 <= x < 41 for x in e) else None

    # Build a pool of candidate exponent structures
    exp_pool = []
    # AP structures
    for d in range(1, 8):
        for a in range(0, min(40 - 11 * d + 1, 15)):
            e = gen_exponents_ap(d, a)
            if e and all(x < 41 for x in e):
                exp_pool.append(e)
    # Two-AP
    for d1 in range(1, 5):
        for d2 in range(1, 5):
            for a1 in range(0, 10):
                for gap in range(3, 20):
                    a2 = a1 + gap
                    e = gen_exponents_two_ap(d1, a1, d2, a2)
                    if e:
                        exp_pool.append(e)
    # Symmetric
    offset_sets = [
        [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 6, 8],
        [1, 2, 3, 5, 7, 9], [1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12],
        [1, 2, 4, 5, 7, 8], [1, 3, 4, 6, 7, 9], [1, 2, 4, 6, 8, 10],
    ]
    for center in range(6, 35, 2):
        for off in offset_sets:
            e = gen_exponents_symmetric(center, off)
            if e:
                exp_pool.append(e)

    # Deduplicate
    exp_pool = list(set(tuple(e) for e in exp_pool))
    exp_pool = [list(e) for e in exp_pool]

    # Initial coefficient patterns
    init_patterns = [
        [1] * 12,
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
        [1, 1, -2, -1, 1, 1, -2, -1, 1, 1, -2, -1],
    ]

    # For each exponent structure, do spectral gradient descent on coefficients
    # Use continuous relaxation: allow real coefficients, then round
    for exp_struct in exp_pool[:80]:
        if time.time() > deadline - 40:
            break
        for init_pat in init_patterns:
            if time.time() > deadline - 40:
                break
            # Start from integer init
            c = np.array(init_pat, dtype=np.float64)
            # Gradient descent on spectral L1
            lr = 0.5
            for step in range(50):
                if time.time() > deadline - 40:
                    break
                grad = spectral_l1_grad(exp_struct, c)
                # Normalize gradient
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-6:
                    break
                c -= lr * grad / gnorm
                # Project to [-64, 64]
                c = np.clip(c, -64, 64)
                # Round to nearest integer for evaluation
                c_int = np.round(c).astype(int)
                # Ensure nonzero
                for i in range(12):
                    if c_int[i] == 0:
                        c_int[i] = 1 if c[i] >= 0 else -1
                if all(abs(x) <= 64 for x in c_int):
                    try_candidate(exp_struct, list(c_int))
            # Also try the rounded continuous solution
            c_final = np.round(c).astype(int)
            for i in range(12):
                if c_final[i] == 0:
                    c_final[i] = 1
            if all(abs(x) <= 64 for x in c_final):
                try_candidate(exp_struct, list(c_final))

    # ---------- Phase 2: Randomized search with spectral scoring ----------
    def spectral_score(exponents, coeffs):
        """Quick spectral L1 for ranking."""
        return spectral_l1(exponents, coeffs)

    def random_search(num_trials=100000):
        nonlocal best_poly, best_nz
        for trial in range(num_trials):
            if time.time() > deadline - 25:
                break

            r = random.random()
            if r < 0.20:
                d = random.randint(1, 6)
                max_a = max(0, 40 - 11 * d)
                a = random.randint(0, max_a)
                exponents = [a + i * d for i in range(12)]
                if exponents[-1] >= 41:
                    continue
            elif r < 0.35:
                d1 = random.randint(1, 5)
                d2 = random.randint(1, 5)
                a1 = random.randint(0, 12)
                gap = random.randint(4, 20)
                a2 = a1 + gap
                exponents = [a1 + i * d1 for i in range(6)] + [a2 + i * d2 for i in range(6)]
                if any(e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                exponents.sort()
            elif r < 0.50:
                center = random.randint(6, 34)
                offsets = sorted(random.sample(range(1, 20), 6))
                exponents = [center - o for o in offsets] + [center + o for o in offsets]
                if any(e < 0 or e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                exponents.sort()
            elif r < 0.65:
                window = random.randint(14, 28)
                base = random.randint(0, max(0, 40 - window))
                exponents = sorted(random.sample(range(base, min(base + window, 41)), 12))
            elif r < 0.80:
                d1 = random.randint(1, 3)
                d2 = random.randint(1, 3)
                a1 = random.randint(0, 5)
                g1 = random.randint(3, 8)
                a2 = a1 + g1
                exponents = [a1 + i * d1 for i in range(3)] + [a2 + i * d2 for i in range(9)]
                if any(e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                exponents.sort()
            else:
                exponents = sorted(random.sample(range(41), 12))

            cstyle = random.random()
            if cstyle < 0.25:
                coeffs = [random.choice([-1, 1]) for _ in range(12)]
            elif cstyle < 0.45:
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            elif cstyle < 0.65:
                coeffs = [random.randint(-4, 4) for _ in range(12)]
                for i in range(12):
                    while coeffs[i] == 0:
                        coeffs[i] = random.randint(1, 4) * random.choice([-1, 1])
            elif cstyle < 0.85:
                coeffs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]
            else:
                coeffs = [random.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 6]) for _ in range(12)]

            if all(abs(c) <= 64 and c != 0 for c in coeffs):
                try_candidate(exponents, coeffs)

    random_search(100000)

    # ---------- Phase 3: DFT-guided local search on coefficients ----------
    if best_poly is not None and time.time() < deadline - 10:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]

        for restart in range(10):
            if time.time() > deadline - 8:
                break

            cur_exp = list(best_exp)
            cur_coef = np.array(best_coef, dtype=np.float64)
            cur_nz = square_nonzero(cur_exp, cur_coef)

            # Spectral gradient descent with integer rounding
            lr = 0.3
            for step in range(200):
                if time.time() > deadline - 8:
                    break
                grad = spectral_l1_grad(cur_exp, cur_coef)
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-6:
                    break
                cur_coef -= lr * grad / gnorm
                cur_coef = np.clip(cur_coef, -64, 64)
                c_int = np.round(cur_coef).astype(int)
                for i in range(12):
                    if c_int[i] == 0:
                        c_int[i] = 1 if cur_coef[i] >= 0 else -1
                if all(abs(x) <= 64 for x in c_int):
                    nz = square_nonzero(cur_exp, c_int)
                    if nz < cur_nz:
                        cur_nz = nz
                        cur_coef = c_int.astype(np.float64)
                        if nz < best_nz:
                            best_nz = nz
                            max_deg = max(cur_exp)
                            poly_new = [0] * (max_deg + 1)
                            for e, c in zip(cur_exp, c_int):
                                poly_new[e] = c
                            best_poly = poly_new
                            best_exp = list(cur_exp)
                            best_coef = c_int

            # Try exponent perturbations guided by spectral score
            for _ in range(500):
                if time.time() > deadline - 8:
                    break
                exp_copy = list(cur_exp)
                idx = random.randint(0, 11)
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                new_e = exp_copy[idx] + shift
                if 0 <= new_e < 41 and new_e not in exp_copy:
                    exp_copy[idx] = new_e
                    exp_copy.sort()
                    c_int = np.round(cur_coef).astype(int)
                    for i in range(12):
                        if c_int[i] == 0:
                            c_int[i] = 1
                    if all(abs(x) <= 64 for x in c_int):
                        nz = square_nonzero(exp_copy, c_int)
                        if nz < cur_nz:
                            cur_nz = nz
                            cur_exp = exp_copy
                            cur_coef = c_int.astype(np.float64)
                            if nz < best_nz:
                                best_nz = nz
                                max_deg = max(cur_exp)
                                poly_new = [0] * (max_deg + 1)
                                for e, c in zip(cur_exp, c_int):
                                    poly_new[e] = c
                                best_poly = poly_new
                                best_exp = list(cur_exp)
                                best_coef = c_int

    # ---------- Phase 4: Exhaustive small-neighborhood search on best ----------
    if best_poly is not None and time.time() < deadline - 3:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]
        cur_nz = square_nonzero(best_exp, best_coef)

        # Try all single coefficient changes
        for idx in range(12):
            if time.time() > deadline - 3:
                break
            for new_c in range(-64, 65):
                if new_c == 0 or new_c == best_coef[idx]:
                    continue
                test_coef = list(best_coef)
                test_coef[idx] = new_c
                nz = square_nonzero(best_exp, test_coef)
                if nz < cur_nz:
                    cur_nz = nz
                    best_coef = test_coef
                    if nz < best_nz:
                        best_nz = nz
                        max_deg = max(best_exp)
                        poly_new = [0] * (max_deg + 1)
                        for e, c in zip(best_exp, best_coef):
                            poly_new[e] = c
                        best_poly = poly_new

    # Final fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    if not validate(best_poly):
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return best_poly

# EVOLVE-BLOCK-END
