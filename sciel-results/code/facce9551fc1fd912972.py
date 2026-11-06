import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

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

    def simulated_annealing(exps, coeffs, max_iters=5000):
        """
        Simulated annealing with diverse moves.
        """
        nonlocal best_score, best_candidate

        # Ensure sorted
        idx = np.argsort(exps)
        exps = [exps[i] for i in idx]
        coeffs = [coeffs[i] for i in idx]

        current_score = compute_score_fast(exps, coeffs)
        _update_best(exps, coeffs, current_score)

        if current_score <= 12:
            return

        # Annealing parameters
        T = 10.0
        T_min = 0.1
        cooling = 0.995

        for it in range(max_iters):
            if time.time() > deadline - 2:
                break

            # Choose move type
            move_type = random.random()
            new_exps = exps[:]
            new_coeffs = coeffs[:]

            if move_type < 0.3:
                # Flip sign of one coefficient
                i = random.randint(0, 11)
                new_coeffs[i] = -new_coeffs[i]
            elif move_type < 0.5:
                # Shift one exponent by small amount
                i = random.randint(0, 11)
                delta = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                new_e = new_exps[i] + delta
                if 0 <= new_e <= 40 and new_e not in new_exps:
                    new_exps[i] = new_e
                else:
                    continue
            elif move_type < 0.7:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                new_exps[i], new_exps[j] = new_exps[j], new_exps[i]
                # Re-sort to maintain consistency for future moves? 
                # Actually, swapping values in a sorted list breaks sort order.
                # We should sort after the move.
                nidx = np.argsort(new_exps)
                new_exps = [new_exps[i] for i in nidx]
                new_coeffs = [new_coeffs[i] for i in nidx]
            elif move_type < 0.85:
                # Jump: random reposition of one exponent
                i = random.randint(0, 11)
                new_e = random.randint(0, 40)
                if new_e not in new_exps:
                    new_exps[i] = new_e
                    nidx = np.argsort(new_exps)
                    new_exps = [new_exps[i] for i in nidx]
                    new_coeffs = [new_coeffs[i] for i in nidx]
                else:
                    continue
            else:
                # Change coefficient magnitude
                i = random.randint(0, 11)
                old_c = new_coeffs[i]
                new_c = random.choice([-8, -4, -2, -1, 1, 2, 4, 8])
                if new_c != old_c and abs(new_c) <= 64:
                    new_coeffs[i] = new_c
                else:
                    continue

            # Validate
            if len(set(new_exps)) != 12:
                continue
            if max(new_exps) > 40:
                continue
            if any(abs(c) > 64 for c in new_coeffs):
                continue
            if any(c == 0 for c in new_coeffs):
                continue

            new_score = compute_score_fast(new_exps, new_coeffs)

            # Accept or reject
            if new_score < current_score:
                exps = new_exps
                coeffs = new_coeffs
                current_score = new_score
                _update_best(exps, coeffs, current_score)
                if current_score <= 12:
                    break
            else:
                # Simulated annealing acceptance
                delta_score = new_score - current_score
                if T > T_min:
                    accept_prob = np.exp(-delta_score / T)
                    if random.random() < accept_prob:
                        exps = new_exps
                        coeffs = new_coeffs
                        current_score = new_score

            T = max(T * cooling, T_min)

        return exps, coeffs, current_score

    # --- Main search loop ---

    random.seed(42)

    # Initial candidates
    initial_candidates = []

    # 1. Consecutive exponents with various sign patterns
    for start in range(0, 5):
        exps = [start + i for i in range(12)]
        if max(exps) <= 40:
            # Alternating signs
            coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
            initial_candidates.append((exps[:], coeffs[:]))
            # All positive
            initial_candidates.append((exps[:], [1]*12))
            # Random signs
            for _ in range(5):
                c = [random.choice([-1, 1]) for _ in range(12)]
                initial_candidates.append((exps[:], c))

    # 2. Gapped exponents (e.g., two blocks)
    # Block 1: 0..3, Block 2: 10..13
    exps = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]
    if max(exps) <= 40:
        for _ in range(5):
            c = [random.choice([-1, 1]) for _ in range(12)]
            initial_candidates.append((exps[:], c))

    # 3. Random sparse
    for _ in range(20):
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        initial_candidates.append((exps, coeffs))

    # Evaluate and run SA on best initial candidates
    scored_candidates = []
    for exps, coeffs in initial_candidates:
        score = compute_score_fast(exps, coeffs)
        if score < float('inf'):
            scored_candidates.append((score, exps, coeffs))
    
    scored_candidates.sort(key=lambda x: x[0])

    # Run SA on the top 5 initial candidates
    for score, exps, coeffs in scored_candidates[:5]:
        if time.time() > deadline - 10:
            break
        simulated_annealing(exps, coeffs, max_iters=2000)
        if best_score <= 12:
            break

    # If still not good, run SA on random restarts
    if best_score > 12 and time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            simulated_annealing(exps, coeffs, max_iters=1000)
            if best_score <= 12:
                break

    return best_candidate

# EVOLVE-BLOCK-END
