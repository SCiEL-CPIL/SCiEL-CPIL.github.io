import random
import time
import numpy as np

def construct():
    """
    Finds a polynomial with 12 non-zero terms that minimizes the number of 
    non-zero terms in its exact square.
    """
    start_time = time.time()
    timeout = 80.0  # Safety margin before 90s wall-clock kill

    # Best found so far
    best_count = 23  # Max possible is 2*12 - 1 = 23
    best_sparse = [(i, 1) for i in range(12)]  # x^0 + x^1 + ... + x^11

    def sparse_to_dense(sparse, max_deg=40):
        """Convert list of (exponent, coeff) to dense coefficient list."""
        coeffs = [0] * (max_deg + 1)
        for exp, c in sparse:
            if 0 <= exp <= max_deg:
                coeffs[exp] = c
        # Trim trailing zeros but keep at least one element
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    def count_nonzero_square(sparse):
        """Count nonzero terms in exact square using NumPy convolution."""
        dense = sparse_to_dense(sparse)
        c = np.array(dense, dtype=np.int64)
        sq = np.convolve(c, c)
        return int(np.count_nonzero(sq))

    def validate(sparse):
        """Check constraints: 12 terms, exponents in [0,40], coeffs in [-64,64], nonzero final."""
        if len(sparse) != 12:
            return False
        exps = [e for e, c in sparse]
        if len(set(exps)) != 12:  # distinct exponents
            return False
        if any(e < 0 or e > 40 for e in exps):
            return False
        if any(c == 0 for e, c in sparse):
            return False
        if any(c < -64 or c > 64 for e, c in sparse):
            return False
        # Nonzero final coefficient: the highest-degree term must be nonzero (it is by construction)
        return True

    # --- Structured initialization ---
    random.seed(12345)

    structured_inits = []

    # 1. Simple arithmetic progression
    for d in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        base = random.randint(0, max(0, 40 - 11 * d))
        exps = [base + i * d for i in range(12)]
        if max(exps) <= 40:
            coeffs = [1] * 12
            structured_inits.append([(exps[i], coeffs[i]) for i in range(12)])

    # 2. Alternating signs in AP
    for d in [1, 2, 3, 4, 5, 6, 7, 8]:
        base = random.randint(0, max(0, 40 - 11 * d))
        exps = [base + i * d for i in range(12)]
        if max(exps) <= 40:
            coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
            structured_inits.append([(exps[i], coeffs[i]) for i in range(12)])

    # 3. Two-group structure: 6 terms in low range, 6 in high range
    for d1 in [1, 2, 3, 4]:
        for d2 in [1, 2, 3, 4]:
            base1 = random.randint(0, 4)
            base2 = random.randint(20, 30)
            exps1 = [base1 + i * d1 for i in range(6)]
            exps2 = [base2 + i * d2 for i in range(6)]
            exps = exps1 + exps2
            if max(exps) <= 40 and len(set(exps)) == 12:
                coeffs = [1] * 6 + [-1] * 6
                structured_inits.append([(exps[i], coeffs[i]) for i in range(12)])

    # 4. Symmetric pairs: (e, S-e) for 6 pairs
    for S in range(10, 41):
        half = S // 2
        exps = []
        for i in range(6):
            e = half - 3 + i
            if 0 <= e <= 40:
                exps.append(e)
                if e != S - e and 0 <= S - e <= 40:
                    exps.append(S - e)
        if len(exps) >= 12:
            exps = exps[:12]
            if len(set(exps)) == 12 and max(exps) <= 40:
                coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
                structured_inits.append([(exps[i], coeffs[i]) for i in range(12)])

    # 5. Random small coefficients
    for _ in range(200):
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        structured_inits.append([(exps[i], coeffs[i]) for i in range(12)])

    # 6. Composition-inspired: P(x) = Q(x^d) with Q having 12 terms
    #    This forces exponents to be multiples of d, creating gaps
    for d in [2, 3, 4, 5]:
        max_q_deg = 40 // d
        if max_q_deg >= 11:
            for _ in range(50):
                q_exps = sorted(random.sample(range(0, max_q_deg + 1), 12))
                p_exps = [e * d for e in q_exps]
                if max(p_exps) <= 40 and len(set(p_exps)) == 12:
                    coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                    structured_inits.append([(p_exps[i], coeffs[i]) for i in range(12)])

    # Evaluate all structured inits
    for sparse in structured_inits:
        if not validate(sparse):
            continue
        cnt = count_nonzero_square(sparse)
        if cnt < best_count:
            best_count = cnt
            best_sparse = sparse[:]

    # --- Local search with sparse representation ---
    current_sparse = best_sparse[:]
    current_count = best_count

    max_steps = 200000
    coeff_choices = [-4, -3, -2, -1, 1, 2, 3, 4]
    stall_count = 0
    max_stall = 5000

    for step in range(max_steps):
        if time.time() - start_time > timeout:
            break

        # Pick a move type
        move_type = random.choices(
            ['flip_sign', 'shift_exp', 'change_coeff', 'shift_pair', 'coeff_pair', 'swap_exps'],
            weights=[25, 25, 15, 15, 10, 10]
        )[0]

        trial_sparse = [(e, c) for e, c in current_sparse]
        valid = True

        if move_type == 'flip_sign':
            idx = random.randrange(12)
            e, c = trial_sparse[idx]
            trial_sparse[idx] = (e, -c)

        elif move_type == 'shift_exp':
            idx = random.randrange(12)
            e, c = trial_sparse[idx]
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_e = e + shift
            if 0 <= new_e <= 40:
                existing_exps = set(x[0] for x in trial_sparse if x[0] != e)
                if new_e not in existing_exps:
                    trial_sparse[idx] = (new_e, c)
                else:
                    valid = False
            else:
                valid = False

        elif move_type == 'change_coeff':
            idx = random.randrange(12)
            e, c = trial_sparse[idx]
            new_c = random.choice(coeff_choices)
            if new_c != c:
                trial_sparse[idx] = (e, new_c)
            else:
                valid = False

        elif move_type == 'shift_pair':
            i, j = random.sample(range(12), 2)
            ei, ci = trial_sparse[i]
            ej, cj = trial_sparse[j]
            shift_i = random.choice([-2, -1, 1, 2])
            shift_j = random.choice([-2, -1, 1, 2])
            new_ei = ei + shift_i
            new_ej = ej + shift_j
            if 0 <= new_ei <= 40 and 0 <= new_ej <= 40:
                other_exps = set(x[0] for k, x in enumerate(trial_sparse) if k not in (i, j))
                if new_ei not in other_exps and new_ej not in other_exps and new_ei != new_ej:
                    trial_sparse[i] = (new_ei, ci)
                    trial_sparse[j] = (new_ej, cj)
                else:
                    valid = False
            else:
                valid = False

        elif move_type == 'coeff_pair':
            i, j = random.sample(range(12), 2)
            ei, ci = trial_sparse[i]
            ej, cj = trial_sparse[j]
            new_ci = random.choice(coeff_choices)
            new_cj = random.choice(coeff_choices)
            if new_ci != ci or new_cj != cj:
                trial_sparse[i] = (ei, new_ci)
                trial_sparse[j] = (ej, new_cj)
            else:
                valid = False

        elif move_type == 'swap_exps':
            i, j = random.sample(range(12), 2)
            ei, ci = trial_sparse[i]
            ej, cj = trial_sparse[j]
            trial_sparse[i] = (ej, ci)
            trial_sparse[j] = (ei, cj)

        if not valid:
            continue

        if not validate(trial_sparse):
            continue

        trial_count = count_nonzero_square(trial_sparse)

        if trial_count < current_count:
            current_sparse = trial_sparse
            current_count = trial_count
            stall_count = 0
            if trial_count < best_count:
                best_count = trial_count
                best_sparse = trial_sparse[:]
        else:
            stall_count += 1
            # Accept worse moves with small probability for exploration
            if random.random() < 0.03:
                current_sparse = trial_sparse
                current_count = trial_count

        # Periodic reset to best to escape local minima
        if step % 5000 == 0 and stall_count > 1000:
            current_sparse = best_sparse[:]
            current_count = best_count
            stall_count = 0

    # Convert best_sparse to dense coefficient list
    best_dense = sparse_to_dense(best_sparse, max_deg=40)
    # Ensure length <= 41
    if len(best_dense) > 41:
        best_dense = best_dense[:41]
    # Ensure nonzero final coefficient
    if best_dense[-1] == 0:
        best_dense[-1] = 1

    return best_dense
