import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 118.0  # safety margin before 150s kill

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute balls
    nb_count = 1 + n * (q - 1)
    balls = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        balls[idx] = nb_list

    # Precompute ball bitsets for fast coverage
    ball_bits = np.zeros((space_size, space_size), dtype=np.int8)
    for c in range(space_size):
        ball_bits[c, balls[c]] = 1

    # Precompute "who covers me" (transpose of ball_bits)
    covers_me = ball_bits.T  # covers_me[word] = bitmask of codewords covering it

    rng = np.random.default_rng(42)

    # --- Initial construction: greedy set cover ---
    uncovered = np.ones(space_size, dtype=np.int8)
    code_indices = []
    selected_mask = np.zeros(space_size, dtype=bool)

    while int(np.sum(uncovered)) > 0 and (time.time() - t_start) < deadline - 60.0:
        scores = ball_bits @ uncovered.astype(np.int32)
        scores[selected_mask] = -1
        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            break
        code_indices.append(best_idx)
        selected_mask[best_idx] = True
        uncovered[balls[best_idx]] = 0

    # Ensure full coverage
    while int(np.sum(uncovered)) > 0 and (time.time() - t_start) < deadline - 55.0:
        scores = ball_bits @ uncovered.astype(np.int32)
        scores[selected_mask] = -1
        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            break
        code_indices.append(best_idx)
        selected_mask[best_idx] = True
        uncovered[balls[best_idx]] = 0

    # Greedy removal
    def compute_cov(indices):
        c = np.zeros(space_size, dtype=np.int32)
        for idx in indices:
            c[balls[idx]] += 1
        return c

    def greedy_remove(indices):
        if not indices:
            return indices
        cov = compute_cov(indices)
        active = [True] * len(indices)
        changed = True
        while changed and (time.time() - t_start) < deadline - 50.0:
            changed = False
            for i in range(len(indices)):
                if not active[i]:
                    continue
                idx = indices[i]
                if np.min(cov[balls[idx]]) >= 2:
                    active[i] = False
                    cov[balls[idx]] -= 1
                    changed = True
        return [indices[i] for i in range(len(indices)) if active[i]]

    best_code = greedy_remove(code_indices)
    best_size = len(best_code)

    # --- Fixed-size ILS ---
    # We'll work with a fixed pool size M. Start with M = best_size.
    # Objective: minimize (uncovered * WEIGHT + total_surplus)
    # where total_surplus = sum over words of max(0, cov[w] - 1)
    # A valid solution has uncovered=0.

    WEIGHT = 100  # heavy penalty for uncovered

    def objective(indices):
        if not indices:
            return space_size * WEIGHT
        cov = compute_cov(indices)
        unc = int(np.sum(cov == 0))
        surplus = int(np.sum(np.maximum(cov - 1, 0)))
        return unc * WEIGHT + surplus

    def is_valid(indices):
        if not indices:
            return False
        cov = compute_cov(indices)
        return int(np.sum(cov == 0)) == 0

    # ILS: fixed-size M, try to find a valid configuration of size M
    # Then try smaller M.

    # Start from best_size, try to reduce
    current_M = best_size

    # Build initial permutation of size M
    # We need exactly M distinct codewords
    # Use best_code as seed, pad with random if needed
    cur_set = list(best_code)
    # Ensure exactly M elements, distinct
    seen = set()
    cur_perm = []
    for c in cur_set:
        if c not in seen:
            seen.add(c)
            cur_perm.append(c)
    # Pad to M
    while len(cur_perm) < M:
        new_c = int(rng.integers(0, space_size))
        if new_c not in seen:
            seen.add(new_c)
            cur_perm.append(new_c)
    cur_perm = cur_perm[:M]

    cur_obj = objective(cur_perm)
    best_obj = cur_obj
    best_perm = cur_perm[:]

    T_start = 5.0
    T_end = 0.05
    sa_budget = deadline - (time.time() - t_start) - 15.0
    if sa_budget < 30.0:
        sa_budget = 30.0

    step = 0
    stall_count = 0
    max_stall = 500

    while (time.time() - t_start) < deadline - 15.0:
        elapsed_frac = 1.0 - (deadline - 15.0 - (time.time() - t_start)) / sa_budget
        elapsed_frac = max(0.0, min(1.0, elapsed_frac))
        T = T_start * (T_end / T_start) ** elapsed_frac

        r = rng.random()
        improved = False

        if r < 0.40:
            # 2-opt swap: replace one codeword with another
            if M < 2:
                step += 1
                continue
            pos = int(rng.integers(0, M))
            old_c = cur_perm[pos]
            new_c = int(rng.integers(0, space_size))
            if new_c == old_c or new_c in cur_perm:
                step += 1
                continue
            cur_perm[pos] = new_c
            new_obj = objective(cur_perm)
            delta = new_obj - cur_obj
            if delta <= 0 or (T > 0.01 and rng.random() < np.exp(-delta / T)):
                cur_obj = new_obj
                improved = True
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_perm = cur_perm[:]
                    # If valid, try to reduce M
                    if int(np.sum(compute_cov(cur_perm) == 0)) == 0:
                        # Valid! Try greedy removal to find smaller size
                        reduced = greedy_remove(cur_perm)
                        if len(reduced) < best_size:
                            best_size = len(reduced)
                            best_code = reduced
                            # Shrink M
                            new_M = best_size
                            # Rebuild permutation
                            seen2 = set()
                            cur_perm = []
                            for c in reduced:
                                if c not in seen2:
                                    seen2.add(c)
                                    cur_perm.append(c)
                            while len(cur_perm) < new_M:
                                nc = int(rng.integers(0, space_size))
                                if nc not in seen2:
                                    seen2.add(nc)
                                    cur_perm.append(nc)
                            cur_perm = cur_perm[:new_M]
                            M = new_M
                            cur_obj = objective(cur_perm)
                            best_obj = cur_obj
                            best_perm = cur_perm[:]
                            stall_count = 0
            else:
                cur_perm[pos] = old_c

        elif r < 0.70:
            # 2-swap: replace two codewords
            if M < 4:
                step += 1
                continue
            pos_a = int(rng.integers(0, M))
            pos_b = int(rng.integers(0, M))
            if pos_a == pos_b:
                step += 1
                continue
            old_a = cur_perm[pos_a]
            old_b = cur_perm[pos_b]
            new_a = int(rng.integers(0, space_size))
            new_b = int(rng.integers(0, space_size))
            if new_a == new_b or new_a in cur_perm or new_b in cur_perm:
                step += 1
                continue
            cur_perm[pos_a] = new_a
            cur_perm[pos_b] = new_b
            new_obj = objective(cur_perm)
            delta = new_obj - cur_obj
            if delta <= 0 or (T > 0.01 and rng.random() < np.exp(-delta / T)):
                cur_obj = new_obj
                improved = True
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_perm = cur_perm[:]
                    if int(np.sum(compute_cov(cur_perm) == 0)) == 0:
                        reduced = greedy_remove(cur_perm)
                        if len(reduced) < best_size:
                            best_size = len(reduced)
                            best_code = reduced
                            new_M = best_size
                            seen2 = set()
                            cur_perm = []
                            for c in reduced:
                                if c not in seen2:
                                    seen2.add(c)
                                    cur_perm.append(c)
                            while len(cur_perm) < new_M:
                                nc = int(rng.integers(0, space_size))
                                if nc not in seen2:
                                    seen2.add(nc)
                                    cur_perm.append(nc)
                            cur_perm = cur_perm[:new_M]
                            M = new_M
                            cur_obj = objective(cur_perm)
                            best_obj = cur_obj
                            best_perm = cur_perm[:]
                            stall_count = 0
            else:
                cur_perm[pos_a] = old_a
                cur_perm[pos_b] = old_b

        elif r < 0.85:
            # Kick: replace k random codewords
            k = max(1, min(3, M // 4))
            positions = rng.choice(M, size=k, replace=False)
            old_vals = [cur_perm[p] for p in positions]
            for p in positions:
                new_c = int(rng.integers(0, space_size))
                while new_c in cur_perm:
                    new_c = int(rng.integers(0, space_size))
                cur_perm[p] = new_c
            new_obj = objective(cur_perm)
            delta = new_obj - cur_obj
            if delta <= 0 or (T > 0.01 and rng.random() < np.exp(-delta / (T * 2))):
                cur_obj = new_obj
                improved = True
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_perm = cur_perm[:]
                    if int(np.sum(compute_cov(cur_perm) == 0)) == 0:
                        reduced = greedy_remove(cur_perm)
                        if len(reduced) < best_size:
                            best_size = len(reduced)
                            best_code = reduced
                            new_M = best_size
                            seen2 = set()
                            cur_perm = []
                            for c in reduced:
                                if c not in seen2:
                                    seen2.add(c)
                                    cur_perm.append(c)
                            while len(cur_perm) < new_M:
                                nc = int(rng.integers(0, space_size))
                                if nc not in seen2:
                                    seen2.add(nc)
                                    cur_perm.append(nc)
                            cur_perm = cur_perm[:new_M]
                            M = new_M
                            cur_obj = objective(cur_perm)
                            best_obj = cur_obj
                            best_perm = cur_perm[:]
                            stall_count = 0
            else:
                for p, ov in zip(positions, old_vals):
                    cur_perm[p] = ov

        else:
            # Local improvement: greedy removal + re-pad
            reduced = greedy_remove(cur_perm)
            if len(reduced) < M:
                new_M = len(reduced)
                seen2 = set()
                cur_perm = []
                for c in reduced:
                    if c not in seen2:
                        seen2.add(c)
                        cur_perm.append(c)
                while len(cur_perm) < new_M:
                    nc = int(rng.integers(0, space_size))
                    if nc not in seen2:
                        seen2.add(nc)
                        cur_perm.append(nc)
                cur_perm = cur_perm[:new_M]
                M = new_M
                cur_obj = objective(cur_perm)
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_perm = cur_perm[:]
                    if int(np.sum(compute_cov(cur_perm) == 0)) == 0:
                        if M < best_size:
                            best_size = M
                            best_code = reduced
                improved = True
                stall_count = 0

        if not improved:
            stall_count += 1
            if stall_count > max_stall:
                # Force kick
                k = max(1, min(4, M // 3))
                positions = rng.choice(M, size=k, replace=False)
                for p in positions:
                    new_c = int(rng.integers(0, space_size))
                    while new_c in cur_perm:
                        new_c = int(rng.integers(0, space_size))
                    cur_perm[p] = new_c
                cur_obj = objective(cur_perm)
                stall_count = 0

        step += 1

    # --- Final: use best valid code found ---
    final_indices = best_code[:]

    # Ensure validity
    if not final_indices:
        final_indices = list(range(space_size))

    final_cov = compute_cov(final_indices)
    if int(np.sum(final_cov == 0)) > 0:
        # Fallback: add uncovered
        unc = np.where(final_cov == 0)[0]
        for u in unc:
            final_indices.append(int(u))
        final_cov = compute_cov(final_indices)

    # Final greedy removal
    final_indices = greedy_remove(final_indices)
    final_cov = compute_cov(final_indices)

    # Ensure distinctness
    seen = set()
    unique = []
    for c in final_indices:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    if not unique:
        unique = list(range(space_size))

    return [idx_to_word(i) for i in unique]
