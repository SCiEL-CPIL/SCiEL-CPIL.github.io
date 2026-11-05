import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 148.0

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbors (radius=1: self + all 1-change neighbors)
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    def build_cov(code_indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov[neighbors[c]] += 1
        return cov

    def objective(cov, code_size):
        # Penalize uncovered words heavily, then minimize code size
        uncovered = np.sum(cov == 0)
        return code_size + 100.0 * uncovered

    def greedy_cover(seed=0):
        """Standard greedy set cover for initialization."""
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 30.0:
            gains = np.sum(uncovered[neighbors], axis=1)
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    def simulated_annealing(initial_code, max_time, seed=0):
        """
        Simulated annealing on codeword sets.
        Moves: add a random codeword, or remove a random codeword.
        Objective: code_size + 100 * uncovered_count.
        """
        rng = np.random.default_rng(seed)

        cur_code = list(initial_code)
        cur_cov = build_cov(cur_code)
        cur_obj = objective(cur_cov, len(cur_code))

        best_code = cur_code[:]
        best_cov = cur_cov.copy()
        best_obj = cur_obj

        # Track membership for O(1) lookup
        in_set = np.zeros(space_size, dtype=bool)
        for c in cur_code:
            in_set[c] = True

        # Cooling schedule
        t0 = time.time()
        t_end = t0 + max_time
        start_temp = 50.0
        end_temp = 0.1
        # Exponential cooling
        cool_rate = (end_temp / start_temp) ** (1.0 / max(1, int(max_time * 100)))

        temp = start_temp

        while (time.time() - t_start) < t_end and temp > end_temp:
            # Choose move: add or remove
            if len(cur_code) == 0:
                # Must add
                move_type = 'add'
            else:
                # Bias towards removal when we have many codewords
                if len(cur_code) > space_size // 4:
                    move_type = rng.choice(['add', 'remove'], p=[0.3, 0.7])
                else:
                    move_type = rng.choice(['add', 'remove'], p=[0.5, 0.5])

            if move_type == 'add':
                # Pick a random codeword not in set
                # For efficiency, sample a few candidates
                n_samples = min(10, space_size - len(cur_code))
                if n_samples <= 0:
                    continue
                candidates = rng.choice(space_size, size=n_samples, replace=False)
                # Find first not in set
                c_add = None
                for c in candidates:
                    c = int(c)
                    if not in_set[c]:
                        c_add = c
                        break
                if c_add is None:
                    continue

                # Compute delta objective
                # Adding c_add: code_size increases by 1
                # Uncovered count may decrease
                new_uncovered = np.sum(cur_cov == 0)
                # After adding, nodes in neighbors[c_add] get +1 coverage
                # Nodes that were uncovered (cov==0) and in neighbors[c_add] become covered
                newly_covered = np.sum((cur_cov[neighbors[c_add]] == 0))
                new_uncovered_after = new_uncovered - newly_covered
                delta_obj = 1.0 + 100.0 * (new_uncovered_after - new_uncovered)
                # delta_obj = 1.0 - 100.0 * newly_covered

                # Accept?
                if delta_obj <= 0 or rng.random() < np.exp(-delta_obj / temp):
                    # Apply move
                    cur_code.append(c_add)
                    cur_cov[neighbors[c_add]] += 1
                    in_set[c_add] = True
                    cur_obj = objective(cur_cov, len(cur_code))
                    if cur_obj < best_obj:
                        best_obj = cur_obj
                        best_code = cur_code[:]
                        best_cov = cur_cov.copy()

            else:
                # Remove a random codeword
                if len(cur_code) == 0:
                    continue
                c_remove = cur_code[int(rng.integers(0, len(cur_code)))]

                # Compute delta objective
                # Removing c_remove: code_size decreases by 1
                # Uncovered count may increase
                # Nodes in neighbors[c_remove] lose 1 coverage
                # Nodes that had cov==1 and are in neighbors[c_remove] become uncovered
                newly_uncovered = np.sum(cur_cov[neighbors[c_remove]] == 1)
                delta_obj = -1.0 + 100.0 * newly_uncovered

                # Accept?
                if delta_obj <= 0 or rng.random() < np.exp(-delta_obj / temp):
                    # Apply move
                    cur_code.remove(c_remove)
                    cur_cov[neighbors[c_remove]] -= 1
                    in_set[c_remove] = False
                    cur_obj = objective(cur_cov, len(cur_code))
                    if cur_obj < best_obj:
                        best_obj = cur_obj
                        best_code = cur_code[:]
                        best_cov = cur_cov.copy()

            # Cool
            temp *= cool_rate
            # Safety: if temp gets too small, reset with a kick
            if temp < end_temp * 2:
                # Kick: randomly remove a few codewords and re-heat
                if len(cur_code) > 5:
                    k = min(3, len(cur_code) // 10 + 1)
                    kick_idx = rng.choice(len(cur_code), size=k, replace=False)
                    kick_set = set(int(ri) for ri in kick_idx)
                    cur_code = [c for i, c in enumerate(cur_code) if i not in kick_set]
                    in_set = np.zeros(space_size, dtype=bool)
                    for c in cur_code:
                        in_set[c] = True
                    cur_cov = build_cov(cur_code)
                    cur_obj = objective(cur_cov, len(cur_code))
                temp = start_temp * 0.5

        return best_code, best_cov, best_obj

    # Phase 1: Generate multiple greedy initial solutions
    best_code = None
    best_obj = float('inf')

    for seed in range(6):
        if (time.time() - t_start) > deadline - 60.0:
            break

        code = greedy_cover(seed)
        cov = build_cov(code)
        obj = objective(cov, len(code))

        if obj < best_obj:
            best_code = code[:]
            best_obj = obj

    # Phase 2: Simulated annealing on the best greedy solution
    if best_code is not None and (time.time() - t_start) < deadline - 20.0:
        sa_time = min(40.0, deadline - 25.0 - (time.time() - t_start))
        code, cov, obj = simulated_annealing(best_code, sa_time, seed=42)
        if obj < best_obj:
            best_code = code
            best_obj = obj

    # Phase 3: SA on a few other greedy solutions
    for seed in range(2, 6):
        if (time.time() - t_start) > deadline - 30.0:
            break

        code = greedy_cover(seed)
        cov = build_cov(code)
        obj = objective(cov, len(code))

        if obj < best_obj:
            best_code = code[:]
            best_obj = obj

        if (time.time() - t_start) < deadline - 15.0:
            sa_time = min(15.0, deadline - 20.0 - (time.time() - t_start))
            code2, cov2, obj2 = simulated_annealing(code, sa_time, seed=100 + seed)
            if obj2 < best_obj:
                best_code = code2
                best_obj = obj2

    # Final verification
    if best_code is None or len(best_code) == 0:
        best_code = greedy_cover(0)

    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        fc = greedy_cover(0)
        fcov = build_cov(fc)
        if np.all(fcov > 0):
            best_code = fc
        else:
            best_code = list(range(space_size))

    return [idx_to_word(c) for c in best_code]
