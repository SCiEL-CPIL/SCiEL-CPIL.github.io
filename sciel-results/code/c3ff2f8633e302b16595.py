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

    # Phase 0: Structural Factorization Search
    # We look for P(x) = Q(x) * R(x) where Q and R are sparse,
    # and the product has exactly 12 non-zero terms.
    # We specifically target Q(x) = 1+x-x^2-x^3 (4 terms) and R(x) with 3 terms.
    # This creates 12 terms if supports are disjoint.
    
    def get_poly_from_factorization(q_exp, q_coef, r_exp, r_coef):
        """Compute product Q*R and return exponents/coeffs if valid."""
        p_exp = []
        p_coef = []
        for i, eq in enumerate(q_exp):
            for j, er in enumerate(r_exp):
                e = eq + er
                c = q_coef[i] * r_coef[j]
                # Check if this exponent already exists in p_exp
                found = False
                for k, pe in enumerate(p_exp):
                    if pe == e:
                        p_coef[k] += c
                        found = True
                        break
                if not found:
                    p_exp.append(e)
                    p_coef.append(c)
        
        # Remove zeros
        valid_exp = []
        valid_coef = []
        for e, c in zip(p_exp, p_coef):
            if c != 0:
                valid_exp.append(e)
                valid_coef.append(c)
                
        if len(valid_exp) != 12:
            return None, None
        if any(abs(c) > 64 for c in valid_coef):
            return None, None
        if any(e >= 41 for e in valid_exp):
            return None, None
        if len(set(valid_exp)) != 12:
            return None, None
            
        return valid_exp, valid_coef

    # Define Q(x) = 1 + x - x^2 - x^3
    q_exp_base = [0, 1, 2, 3]
    q_coef_base = [1, 1, -1, -1]
    
    # Search for R(x) with 3 terms such that Q*R has 12 terms
    # R(x) = c1*x^a + c2*x^b + c3*x^c
    # To ensure disjoint supports (or minimal overlap that cancels to 0 or stays non-zero),
    # we can try R(x) with large gaps.
    
    if time.time() < deadline - 60:
        # Try R(x) patterns
        r_patterns = [
            [1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 2, 1], [1, -2, 1], [2, 1, 1]
        ]
        for rc in r_patterns:
            for a in range(0, 10):
                for b in range(a + 2, 20):
                    for c in range(b + 2, 30):
                        if time.time() > deadline - 55:
                            break
                        r_exp = [a, b, c]
                        r_coef = rc
                        pe, pc = get_poly_from_factorization(q_exp_base, q_coef_base, r_exp, r_coef)
                        if pe is not None:
                            try_candidate(pe, pc)

    # Also try Q(x) with 3 terms and R(x) with 4 terms
    q3_exp = [0, 2, 4] # 1 + x^2 + x^4
    q3_coef = [1, 1, 1]
    # And variations
    q3_variants = [
        ([0, 1, 2], [1, 1, 1]),
        ([0, 1, 2], [1, -1, 1]),
        ([0, 2, 3], [1, 1, 1]),
        ([0, 3, 6], [1, 1, 1]),
    ]
    
    if time.time() < deadline - 45:
        for qe, qc in q3_variants:
            for rc in [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, 2, 1, 2]]:
                for a in range(0, 8):
                    for b in range(a + 2, 15):
                        for c in range(b + 2, 22):
                            for d in range(c + 2, 30):
                                if time.time() > deadline - 40:
                                    break
                                r_exp = [a, b, c, d]
                                r_coef = rc
                                pe, pc = get_poly_from_factorization(qe, qc, r_exp, r_coef)
                                if pe is not None:
                                    try_candidate(pe, pc)

    # Phase 1: Structured AP exponents (retained from previous, but shorter)
    sign_patterns = []
    for period in [1, 2, 3, 4, 6, 12]:
        for phase in range(period):
            pat = [1 if ((i + phase) % (2 * period)) < period else -1 for i in range(12)]
            sign_patterns.append(pat)
            
    for d in range(1, 5):
        if time.time() > deadline - 30:
            break
        max_a = min(40 - 11 * d, 15)
        for a in range(0, max_a + 1):
            if time.time() > deadline - 30:
                break
            exponents = [a + i * d for i in range(12)]
            if exponents[-1] >= 41:
                continue
            for cp in sign_patterns[:10]:
                if time.time() > deadline - 30:
                    break
                try_candidate(exponents, cp)

    # Phase 2: Randomized search
    random.seed(42)

    def random_search(num_trials=100000):
        nonlocal best_poly, best_nz
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                break

            r = random.random()
            if r < 0.30:
                # Factorization-like random
                q_len = random.choice([3, 4])
                r_len = 12 // q_len if 12 % q_len == 0 else 0
                if r_len == 0: continue
                
                q_exp = sorted(random.sample(range(0, 10), q_len))
                q_coef = [random.choice([-1, 1]) for _ in range(q_len)]
                
                r_exp = sorted(random.sample(range(5, 35), r_len))
                r_coef = [random.choice([-1, 1]) for _ in range(r_len)]
                
                pe, pc = get_poly_from_factorization(q_exp, q_coef, r_exp, r_coef)
                if pe is not None:
                    try_candidate(pe, pc)
            else:
                # Standard random
                exponents = sorted(random.sample(range(41), 12))
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                try_candidate(exponents, coeffs)

    random_search(100000)

    # Phase 3: Local search
    if best_poly is not None and time.time() < deadline - 5:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]

        for restart in range(5):
            if time.time() > deadline - 3:
                break

            cur_exp = list(best_exp)
            cur_coef = list(best_coef)
            cur_nz = square_nonzero(cur_exp, cur_coef)

            for it in range(2000):
                if time.time() > deadline - 3:
                    break

                exp_copy = list(cur_exp)
                coef_copy = list(cur_coef)

                idx = random.randint(0, 11)
                if random.random() < 0.5:
                    shift = random.choice([-2, -1, 1, 2])
                    new_e = exp_copy[idx] + shift
                    if 0 <= new_e < 41 and new_e not in exp_copy:
                        exp_copy[idx] = new_e
                        exp_copy.sort()
                    else:
                        continue
                else:
                    new_c = random.choice([-2, -1, 1, 2, 3])
                    if new_c != coef_copy[idx]:
                        coef_copy[idx] = new_c

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
