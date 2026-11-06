"""Sparse squares of 12-term integer polynomials – simulated annealing approach."""
import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s kill

    def square_nonzero_fast(exponents, coeffs):
        """Compute nonzero count of P^2 using numpy convolution."""
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        # Build full coefficient array
        max_deg = int(e.max())
        arr = np.zeros(max_deg + 1, dtype=np.int64)
        arr[e] = c
        # Convolve with itself
        sq = np.convolve(arr, arr)
        return int(np.count_nonzero(sq))

    def square_nonzero_exact(exponents, coeffs):
        """Exact square nonzero count using bincount."""
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        degs = e[:, None] + e[None, :]
        vals = c[:, None] * c[None, :]
        result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=82)
        return int(np.count_nonzero(result))

    def is_valid(exp, coef):
        if len(exp) != 12 or len(coef) != 12:
            return False
        if len(set(exp)) != 12:
            return False
        if any(e < 0 or e >= 41 for e in exp):
            return False
        if any(c == 0 or abs(c) > 64 for c in coef):
            return False
        return True

    def make_poly(exp, coef):
        max_deg = max(exp)
        poly = [0] * (max_deg + 1)
        for e, c in zip(exp, coef):
            poly[e] = int(c)
        return poly

    # Initial candidate structures
    def generate_initial_candidates():
        candidates = []
        random.seed(42)

        # Arithmetic progressions with various signs
        for d in range(1, 7):
            max_a = max(0, 40 - 11 * d)
            for a in range(0, min(max_a + 1, 10)):
                exp = [a + i * d for i in range(12)]
                # All ones
                candidates.append((exp, [1] * 12))
                # Alternating
                candidates.append((exp, [1 if i % 2 == 0 else -1 for i in range(12)]))
                # Random signs
                candidates.append((exp, [random.choice([-1, 1]) for _ in range(12)]))
                # Random small coeffs
                candidates.append((exp, [random.choice([-2, -1, 1, 2]) for _ in range(12)]))

        # Symmetric structures
        for center in range(6, 35, 3):
            for _ in range(5):
                offsets = sorted(random.sample(range(1, 20), 6))
                exp = sorted([center - o for o in offsets] + [center + o for o in offsets])
                if is_valid(exp, [1] * 12):
                    candidates.append((exp, [random.choice([-1, 1]) for _ in range(12)]))
                    candidates.append((exp, [random.choice([-2, -1, 1, 2]) for _ in range(12)]))

        # Two-AP structures
        for _ in range(20):
            d1 = random.randint(1, 4)
            d2 = random.randint(1, 4)
            a1 = random.randint(0, 8)
            gap = random.randint(4, 18)
            a2 = a1 + gap
            exp = sorted([a1 + i * d1 for i in range(6)] + [a2 + i * d2 for i in range(6)])
            if is_valid(exp, [1] * 12):
                candidates.append((exp, [random.choice([-1, 1]) for _ in range(12)]))

        # Random structures
        for _ in range(30):
            exp = sorted(random.sample(range(41), 12))
            candidates.append((exp, [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]))

        return candidates

    def simulated_annealing(initial_exp, initial_coef, max_iters, t_start, t_end, deadline):
        """Run simulated annealing from given initial state."""
        cur_exp = list(initial_exp)
        cur_coef = list(initial_coef)
        cur_nz = square_nonzero_exact(cur_exp, cur_coef)

        best_exp = list(cur_exp)
        best_coef = list(cur_coef)
        best_nz = cur_nz

        n = 12
        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Temperature schedule
            progress = it / max(max_iters - 1, 1)
            temp = t_start * (t_end / t_start) ** progress

            # Generate move
            r = random.random()
            new_exp = list(cur_exp)
            new_coef = list(cur_coef)
            valid_move = True

            if r < 0.30:
                # Change one coefficient
                idx = random.randint(0, n - 1)
                new_c = random.choice([-8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8])
                if new_c == cur_coef[idx]:
                    continue
                new_coef[idx] = new_c
            elif r < 0.50:
                # Change two coefficients
                i, j = random.sample(range(n), 2)
                new_coef[i] = random.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 6])
                new_coef[j] = random.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 6])
            elif r < 0.65:
                # Shift one exponent
                idx = random.randint(0, n - 1)
                shift = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                new_e = cur_exp[idx] + shift
                if 0 <= new_e < 41 and new_e not in cur_exp:
                    new_exp[idx] = new_e
                    new_exp.sort()
                else:
                    continue
            elif r < 0.78:
                # Shift two exponents
                i, j = random.sample(range(n), 2)
                si = random.choice([-3, -2, -1, 1, 2, 3])
                sj = random.choice([-3, -2, -1, 1, 2, 3])
                new_ei = cur_exp[i] + si
                new_ej = cur_exp[j] + sj
                if (0 <= new_ei < 41 and 0 <= new_ej < 41 and
                        new_ei not in cur_exp and new_ej not in cur_exp and
                        new_ei != new_ej):
                    new_exp[i] = new_ei
                    new_exp[j] = new_ej
                    new_exp.sort()
                else:
                    continue
            elif r < 0.88:
                # Swap two exponents
                i, j = random.sample(range(n), 2)
                new_exp[i], new_exp[j] = new_exp[j], new_exp[i]
                new_exp.sort()
            else:
                # Large random perturbation
                idx = random.randint(0, n - 1)
                new_exp[idx] = random.randint(0, 40)
                new_exp.sort()
                if len(set(new_exp)) != n:
                    continue
                # Also randomize some coefficients
                for _ in range(random.randint(1, 4)):
                    ci = random.randint(0, n - 1)
                    new_coef[ci] = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])

            if not is_valid(new_exp, new_coef):
                continue

            new_nz = square_nonzero_exact(new_exp, new_coef)

            # Acceptance criterion
            delta = new_nz - cur_nz
            if delta <= 0:
                # Better or equal - always accept
                cur_exp = new_exp
                cur_coef = new_coef
                cur_nz = new_nz
            else:
                # Worse - accept with probability exp(-delta/temp)
                if temp > 0 and random.random() < np.exp(-delta / max(temp, 0.01)):
                    cur_exp = new_exp
                    cur_coef = new_coef
                    cur_nz = new_nz

            # Update best
            if cur_nz < best_nz:
                best_nz = cur_nz
                best_exp = list(cur_exp)
                best_coef = list(cur_coef)

        return best_exp, best_coef, best_nz

    # Phase 1: Evaluate initial candidates
    candidates = generate_initial_candidates()
    best_exp = None
    best_coef = None
    best_nz = float('inf')

    for exp, coef in candidates:
        if time.time() > deadline - 40:
            break
        if not is_valid(exp, coef):
            continue
        nz = square_nonzero_exact(exp, coef)
        if nz < best_nz:
            best_nz = nz
            best_exp = list(exp)
            best_coef = list(coef)

    # Phase 2: Simulated annealing from best and diverse starts
    if best_exp is not None and time.time() < deadline - 5:
        # SA from best
        e, c, nz = simulated_annealing(best_exp, best_coef, 15000, 5.0, 0.01, deadline - 3)
        if nz < best_nz:
            best_nz = nz
            best_exp = e
            best_coef = c

    # SA from diverse initial points
    diverse_starts = []
    random.seed(99)
    for _ in range(8):
        # Different structural families
        r = random.random()
        if r < 0.3:
            d = random.randint(1, 5)
            a = random.randint(0, max(0, 40 - 11 * d))
            exp = [a + i * d for i in range(12)]
        elif r < 0.6:
            center = random.randint(8, 32)
            offsets = sorted(random.sample(range(1, 18), 6))
            exp = sorted([center - o for o in offsets] + [center + o for o in offsets])
        else:
            exp = sorted(random.sample(range(41), 12))

        if is_valid(exp, [1] * 12):
            coef = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            diverse_starts.append((exp, coef))

    for exp, coef in diverse_starts:
        if time.time() > deadline - 3:
            break
        e, c, nz = simulated_annealing(exp, coef, 8000, 4.0, 0.01, deadline - 3)
        if nz < best_nz:
            best_nz = nz
            best_exp = e
            best_coef = c

    # Phase 3: Focused local refinement around best
    if best_exp is not None and time.time() < deadline - 2:
        cur_exp = list(best_exp)
        cur_coef = list(best_coef)
        cur_nz = best_nz

        for it in range(20000):
            if time.time() > deadline - 1:
                break

            new_exp = list(cur_exp)
            new_coef = list(cur_coef)
            r = random.random()

            if r < 0.4:
                idx = random.randint(0, 11)
                new_c = random.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
                if new_c != cur_coef[idx]:
                    new_coef[idx] = new_c
            elif r < 0.7:
                idx = random.randint(0, 11)
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                new_e = cur_exp[idx] + shift
                if 0 <= new_e < 41 and new_e not in cur_exp:
                    new_exp[idx] = new_e
                    new_exp.sort()
                else:
                    continue
            else:
                i, j = random.sample(range(12), 2)
                si = random.choice([-2, -1, 1, 2])
                sj = random.choice([-2, -1, 1, 2])
                new_ei = cur_exp[i] + si
                new_ej = cur_exp[j] + sj
                if (0 <= new_ei < 41 and 0 <= new_ej < 41 and
                        new_ei not in cur_exp and new_ej not in cur_exp and
                        new_ei != new_ej):
                    new_exp[i] = new_ei
                    new_exp[j] = new_ej
                    new_exp.sort()
                else:
                    continue

            if not is_valid(new_exp, new_coef):
                continue

            new_nz = square_nonzero_exact(new_exp, new_coef)
            if new_nz < cur_nz:
                cur_nz = new_nz
                cur_exp = new_exp
                cur_coef = new_coef
                if cur_nz < best_nz:
                    best_nz = cur_nz
                    best_exp = list(cur_exp)
                    best_coef = list(cur_coef)

    # Final validation and packaging
    def validate(poly):
        if not poly or len(poly) > 41:
            return False
        if poly[-1] == 0:
            return False
        for c in poly:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if sum(1 for c in poly if c != 0) != 12:
            return False
        return True

    if best_exp is None or not is_valid(best_exp, best_coef):
        # Fallback
        best_exp = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 40]
        best_exp = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39, 40]
        best_coef = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    poly = make_poly(best_exp, best_coef)

    if not validate(poly):
        poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return poly
