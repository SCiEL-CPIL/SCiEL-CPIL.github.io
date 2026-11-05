import time
import random
import math

def _count_sq_nonzero(coeff_list):
    """Count nonzero terms in the square of polynomial given as coefficient list."""
    n = len(coeff_list)
    max_deg = 0
    for i, c in enumerate(coeff_list):
        if c != 0:
            max_deg = i
    if max_deg == 0:
        sq = [coeff_list[0] * coeff_list[0]]
        return sum(1 for c in sq if c != 0)
    sq = [0] * (2 * max_deg + 1)
    terms = [(i, c) for i, c in enumerate(coeff_list) if c != 0]
    for idx_i, (i, ci) in enumerate(terms):
        sq[i + i] += ci * ci
        for idx_j in range(idx_i + 1, len(terms)):
            j, cj = terms[idx_j]
            sq[i + j] += 2 * ci * cj
    return sum(1 for c in sq if c != 0)

def _validate(coeff_list):
    if len(coeff_list) > 41:
        return False
    if len(coeff_list) < 1:
        return False
    if coeff_list[-1] == 0:
        return False
    nz = sum(1 for c in coeff_list if c != 0)
    if nz != 12:
        return False
    for c in coeff_list:
        if c < -64 or c > 64:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85

    best_sq = 999
    best_coeffs = None

    def _time_check():
        return time.time() > deadline

    def _eval_exps_coefs(exps, coefs):
        """Fast evaluation: exps is sorted list of 12 exponents, coefs is list of 12 ints."""
        max_e = exps[-1]
        sq = [0] * (2 * max_e + 1)
        for idx_i in range(12):
            i = exps[idx_i]
            ci = coefs[idx_i]
            sq[i + i] += ci * ci
            for idx_j in range(idx_i + 1, 12):
                j = exps[idx_j]
                cj = coefs[idx_j]
                sq[i + j] += 2 * ci * cj
        return sum(1 for c in sq if c != 0)

    def _to_full(exps, coefs):
        max_e = exps[-1]
        if max_e >= 41:
            return None
        full = [0] * (max_e + 1)
        for e, c in zip(exps, coefs):
            full[e] = c
        if full[-1] == 0:
            return None
        nz = sum(1 for c in full if c != 0)
        if nz != 12:
            return None
        if any(c < -64 or c > 64 for c in full):
            return None
        return full

    def _try_update(count, exps, coefs):
        nonlocal best_sq, best_coeffs
        if count < best_sq:
            full = _to_full(exps, coefs)
            if full is not None:
                best_sq = count
                best_coeffs = list(full)
                return True
        return False

    # Phase 1: Structured AP search with alternating signs
    # For AP with step k, exponents = a + i*k for i=0..11
    # Try coefficients c_i = (-1)^i * m where m in {1, 2, 3}
    # This should give square support at even multiples of k
    for k in range(1, 4):
        if _time_check():
            break
        max_a = 40 - 11 * k
        for a in range(0, max_a + 1):
            if _time_check():
                break
            exps = [a + i * k for i in range(12)]
            # Try alternating signs with magnitude 1, 2, 3
            for m in [1, 2, 3]:
                coefs = [((-1) ** i) * m for i in range(12)]
                count = _eval_exps_coefs(exps, coefs)
                _try_update(count, exps, coefs)
                if best_sq == 12:
                    return best_coeffs

    # Phase 2: Local search / simulated annealing from best AP structure
    # Start from the best found, or from a good AP default
    if best_coeffs is not None:
        # Extract exps and coefs from best_coeffs
        best_terms = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        start_exps = [e for e, c in best_terms]
        start_coefs = [c for e, c in best_terms]
    else:
        # Default: AP step 1, a=0, alternating signs
        start_exps = list(range(12))
        start_coefs = [(-1) ** i for i in range(12)]

    random.seed(42)

    # Simulated annealing on coefficients (exponents fixed at best structure)
    T = 1.0
    current_exps = list(start_exps)
    current_coefs = list(start_coefs)
    current_sq = _eval_exps_coefs(current_exps, current_coefs)
    _try_update(current_sq, current_exps, current_coefs)

    no_improve = 0
    max_iter_per_restart = 3000
    restarts = 0
    max_restarts = 30

    while not _time_check() and restarts < max_restarts:
        no_improve = 0
        for _ in range(max_iter_per_restart):
            if _time_check():
                break

            idx = random.randint(0, 11)
            old_val = current_coefs[idx]

            # Try small integer perturbations
            new_val = random.choice([-3, -2, -1, 1, 2, 3])
            if new_val == old_val:
                continue

            current_coefs[idx] = new_val
            new_sq = _eval_exps_coefs(current_exps, current_coefs)
            delta = new_sq - current_sq

            if delta < 0:
                current_sq = new_sq
                no_improve = 0
                _try_update(current_sq, current_exps, current_coefs)
            else:
                if random.random() < math.exp(-delta / max(T, 0.01)):
                    current_sq = new_sq
                    no_improve += 1
                else:
                    current_coefs[idx] = old_val
                    no_improve += 1

            if no_improve > 150:
                break

        T *= 0.85
        if T < 0.05:
            # Restart with new random coefficients on same exponents
            restarts += 1
            T = 1.0
            current_coefs = [random.choice([-1, 1]) for _ in range(12)]
            current_sq = _eval_exps_coefs(current_exps, current_coefs)
            _try_update(current_sq, current_exps, current_coefs)

    # Phase 3: Try different AP structures with local search
    # Explore more AP parameter combinations
    for k in range(1, 5):
        if _time_check():
            break
        max_a = 40 - 11 * k
        # Sample a subset of 'a' values
        a_samples = list(range(0, max_a + 1, max(1, max_a // 10)))
        random.shuffle(a_samples)
        for a in a_samples[:5]:
            if _time_check():
                break
            exps = [a + i * k for i in range(12)]
            # Start with alternating signs
            coefs = [((-1) ** i) for i in range(12)]
            count = _eval_exps_coefs(exps, coefs)
            _try_update(count, exps, coefs)

            # Short local search on this structure
            T_local = 0.5
            for _ in range(500):
                if _time_check():
                    break
                idx = random.randint(0, 11)
                old_val = coefs[idx]
                new_val = random.choice([-3, -2, -1, 1, 2, 3])
                if new_val == old_val:
                    continue
                coefs[idx] = new_val
                new_count = _eval_exps_coefs(exps, coefs)
                if new_count < count:
                    count = new_count
                    _try_update(count, exps, coefs)
                else:
                    if random.random() < math.exp(-(new_count - count) / T_local):
                        count = new_count
                    else:
                        coefs[idx] = old_val
            T_local *= 0.8

    # Phase 4: Random non-AP structures with local search
    while not _time_check():
        exps = sorted(random.sample(range(0, 41), 12))
        coefs = [random.choice([-1, 1]) for _ in range(12)]
        count = _eval_exps_coefs(exps, coefs)
        _try_update(count, exps, coefs)

        # Brief local search
        T_local = 0.3
        for _ in range(200):
            if _time_check():
                break
            idx = random.randint(0, 11)
            old_val = coefs[idx]
            new_val = random.choice([-3, -2, -1, 1, 2, 3])
            if new_val == old_val:
                continue
            coefs[idx] = new_val
            new_count = _eval_exps_coefs(exps, coefs)
            if new_count < count:
                count = new_count
                _try_update(count, exps, coefs)
            else:
                if random.random() < math.exp(-(new_count - count) / T_local):
                    count = new_count
                else:
                    coefs[idx] = old_val

    if best_coeffs is not None:
        return best_coeffs

    # Fallback
    return [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1] + [0] * 29
