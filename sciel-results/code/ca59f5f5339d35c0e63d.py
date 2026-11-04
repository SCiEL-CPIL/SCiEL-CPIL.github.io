import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 118.0  # safety margin before 120s limit

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

    # Precompute neighbors (radius=1: self + all 1-change neighbors)
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
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

    def is_valid(cov):
        return np.all(cov > 0)

    def greedy_repair(code_set, cov, max_time=5.0):
        """Add codewords to cover all uncovered points."""
        code_list = list(code_set)
        uncovered_mask = cov == 0
        while np.any(uncovered_mask) and (time.time() - t_start) < max_time:
            # Find codeword covering most uncovered points
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code_list.append(int(best))
            cov[neighbors[best]] += 1
            uncovered_mask = cov == 0
        return code_list, cov

    # --- Phase 1: Generate diverse starting codes ---
    def greedy_cover():
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 60.0:
            gains = np.sum(uncovered[neighbors], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    def random_cover(target_size):
        rng = np.random.default_rng(42)
        code = []
        cov = np.zeros(space_size, dtype=np.int32)
        attempts = 0
        while not np.all(cov > 0) and attempts < target_size * 50 and (time.time() - t_start) < deadline - 55.0:
            c = rng.integers(0, space_size)
            if c not in code:
                code.append(int(c))
                cov[neighbors[c]] += 1
            attempts += 1
        # Repair if needed
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 50.0)
        return code

    def direct_sum_cover():
        """Constructive upper bound: product of smaller codes."""
        # For q=3, n=7: use direct sum of K_3(1,1)=3 and K_3(3,1)=? 
        # Simpler: just use greedy as fallback for diverse starts
        return greedy_cover()

    # Generate initial code
    init_code = greedy_cover()
    init_cov = build_cov(init_code)
    # Ensure valid
    if not np.all(init_cov > 0):
        init_code, init_cov = greedy_repair(init_code, init_cov, deadline - 50.0)

    best_code = init_code[:]
    best_cov = init_cov.copy()
    best_size = len(best_code)

    # --- Phase 2: Multi-start local search with multi-kick removal + greedy repair ---
    rng = np.random.default_rng(123)

    # Start from different initializations
    starts = [init_code[:]]
    # Add a random-based start
    if (time.time() - t_start) < deadline - 40.0:
        rand_code = random_cover(200)
        if len(rand_code) < best_size + 30:  # not too far off
            starts.append(rand_code)

    for start_idx, start_code in enumerate(starts):
        if (time.time() - t_start) > deadline - 5.0:
            break

        cur_code = start_code[:]
        cur_cov = build_cov(cur_code)
        if not np.all(cur_cov > 0):
            cur_code, cur_cov = greedy_repair(cur_code, cur_cov, deadline - 40.0)

        # Local search loop
        improved = True
        while improved and (time.time() - t_start) < deadline - 5.0:
            improved = False

            # Try multi-kick removal: remove k random codewords, then repair
            max_k = min(3, len(cur_code) - 1)
            found_improvement = False

            for k in range(1, max_k + 1):
                if (time.time() - t_start) > deadline - 5.0:
                    break

                # Try multiple random kicks at this k
                for trial in range(20):
                    if (time.time() - t_start) > deadline - 5.0:
                        break

                    # Pick k random distinct codewords to remove
                    if len(cur_code) <= k:
                        break
                    remove_indices = rng.choice(len(cur_code), size=k, replace=False)
                    remove_set = set(int(ri) for ri in remove_indices)

                    # Build new code without removed
                    new_code = [c for i, c in enumerate(cur_code) if i not in remove_set]
                    new_cov = build_cov(new_code)

                    # Repair
                    new_code, new_cov = greedy_repair(new_code, new_cov, deadline - 10.0)

                    if np.all(new_cov > 0) and len(new_code) < len(cur_code):
                        cur_code = new_code
                        cur_cov = new_cov
                        found_improvement = True
                        improved = True
                        break  # restart kick search with new state

                    if found_improvement:
                        break

                if found_improvement:
                    break

            # Single-removal pass (deterministic)
            if not found_improvement and (time.time() - t_start) < deadline - 5.0:
                for i in range(len(cur_code)):
                    c = cur_code[i]
                    if np.min(cur_cov[neighbors[c]]) >= 2:
                        cur_code.pop(i)
                        cur_cov[neighbors[c]] -= 1
                        improved = True
                        break

        # Update global best
        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_cov = cur_cov.copy()
            best_size = len(best_code)

    # --- Phase 3: Final deterministic removal pass on best ---
    if (time.time() - t_start) < deadline - 2.0:
        changed = True
        while changed and (time.time() - t_start) < deadline - 2.0:
            changed = False
            for i in range(len(best_code)):
                c = best_code[i]
                if np.min(best_cov[neighbors[c]]) >= 2:
                    best_code.pop(i)
                    best_cov[neighbors[c]] -= 1
                    changed = True
                    break

    # Final verification
    if len(best_code) == 0:
        return [idx_to_word(i) for i in range(space_size)]

    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        # Fallback: rebuild from greedy
        fc = greedy_cover()
        fcov = build_cov(fc)
        if np.all(fcov > 0):
            return [idx_to_word(c) for c in fc]
        return [idx_to_word(i) for i in range(space_size)]

    return [idx_to_word(c) for c in best_code]
