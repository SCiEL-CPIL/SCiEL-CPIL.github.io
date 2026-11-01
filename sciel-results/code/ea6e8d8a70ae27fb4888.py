"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    rng = np.random.default_rng(42)

    def get_candidates(weights):
        candidates = []
        for w in weights:
            if w == 1:
                for pos in range(D):
                    for sign in [1, -1]:
                        v = [0] * D
                        v[pos] = 2 * sign
                        candidates.append(v)
            elif w == 2:
                for positions in itertools.combinations(range(D), 2):
                    for signs in itertools.product([1, -1], repeat=2):
                        v = [0] * D
                        for pos, s in zip(positions, signs):
                            v[pos] = s
                        candidates.append(v)
            elif w == 3:
                for positions in itertools.combinations(range(D), 3):
                    for signs in itertools.product([1, -1], repeat=3):
                        v = [0] * D
                        for pos, s in zip(positions, signs):
                            v[pos] = s
                        candidates.append(v)
            elif w == 4:
                for positions in itertools.combinations(range(D), 4):
                    for signs in itertools.product([1, -1], repeat=4):
                        v = [0] * D
                        for pos, s in zip(positions, signs):
                            v[pos] = s
                        candidates.append(v)
        return np.array(candidates, dtype=np.int64)

    def solve_independent_set(candidates, R2, max_time=30.0, n_restarts=60):
        if len(candidates) == 0:
            return np.array([], dtype=np.int64).reshape(0, D), 0

        N = len(candidates)
        norms_sq = np.sum(candidates ** 2, axis=1)
        dots = candidates @ candidates.T

        # Conflict if dist^2 < R2  =>  n1^2 + n2^2 - 2*dot < R2  =>  2*dot > n1^2+n2^2-R2
        threshold = (norms_sq[:, None] + norms_sq[None, :] - R2) / 2.0
        conflicts = dots > threshold
        np.fill_diagonal(conflicts, False)

        neighbors = [np.where(conflicts[i])[0] for i in range(N)]
        neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

        def greedy_min_degree():
            in_set = np.zeros(N, dtype=bool)
            alive = np.ones(N, dtype=bool)
            degrees = neighbor_counts.copy().astype(np.float64)
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                min_deg = np.min(degrees[alive_indices])
                cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]
                if len(cands) > 1:
                    scores = np.zeros(len(cands))
                    for idx, c in enumerate(cands):
                        nbrs_c = neighbors[c]
                        alive_nbrs = nbrs_c[alive[nbrs_c]]
                        if len(alive_nbrs) > 0:
                            scores[idx] = np.sum(degrees[alive_nbrs])
                    chosen = cands[np.argmin(scores)]
                else:
                    chosen = cands[0]
                in_set[chosen] = True
                nbrs = neighbors[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
            return in_set

        def sa_local_search(in_set_init, max_time=10.0):
            in_set = in_set_init.copy()
            current_size = int(in_set.sum())
            local_best_size = current_size
            local_best_in_set = in_set.copy()

            count_conflicts = np.zeros(N, dtype=np.int32)
            for i in np.where(in_set)[0]:
                count_conflicts[neighbors[i]] += 1

            T = 4.0
            cooling = 0.9993
            min_T = 0.05
            sa_deadline = time.time() + max_time

            while time.time() < sa_deadline:
                not_in_set = np.where(~in_set)[0]
                in_set_indices = np.where(in_set)[0]
                if len(not_in_set) == 0 and len(in_set_indices) == 0:
                    break

                move_type = rng.random()

                if move_type < 0.45 and len(not_in_set) > 0:
                    # ADD move
                    counts = count_conflicts[not_in_set]
                    min_count = np.min(counts)
                    min_cands = not_in_set[counts == min_count]
                    v = min_cands[rng.integers(len(min_cands))]
                    k = min_count
                    delta = 1 - k
                    if delta > 0:
                        accept = True
                    elif delta == 0:
                        accept = rng.random() < 0.4
                    else:
                        accept = rng.random() < np.exp(delta / T)
                    if accept:
                        to_remove = np.where(conflicts[v] & in_set)[0]
                        if len(to_remove) > 0:
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                        in_set[v] = True
                        current_size += delta
                        count_conflicts[neighbors[v]] += 1
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                elif move_type < 0.85 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                    # SWAP / KICK-OUT move
                    out_conf_counts = count_conflicts[not_in_set]
                    min_out_count = np.min(out_conf_counts)
                    min_out_cands = not_in_set[out_conf_counts == min_out_count]
                    v_add = min_out_cands[rng.integers(len(min_out_cands))]
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]

                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]

                    if len(other_conflicts) == 0:
                        # Clean swap: size unchanged, but may open up future adds
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        count_conflicts[neighbors[v_remove]] -= 1
                        count_conflicts[neighbors[v_add]] += 1
                        # Re-add conflict counts for neighbors of v_remove that lost a conflict source
                        # (neighbors[v_remove] already decremented; v_remove no longer in set)
                        # But neighbors that are in_set and also neighbors of v_remove: their count dropped correctly.
                        # However, v_remove itself might be a neighbor of some in-set vertex; that's handled by neighbors[v_remove] decrement.
                    else:
                        # Kick-out: remove v_remove + other_conflicts, add v_add
                        to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                        delta = 1 - len(to_remove)
                        if delta > 0:
                            accept = True
                        elif delta == 0:
                            accept = rng.random() < 0.3
                        else:
                            accept = rng.random() < np.exp(delta / T)
                        if accept:
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                            in_set[v_add] = True
                            current_size += delta
                            count_conflicts[neighbors[v_add]] += 1
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()
                else:
                    # REMOVE move (rare): remove a random in-set vertex to diversify
                    if len(in_set_indices) > 0:
                        v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                        in_set[v_remove] = False
                        current_size -= 1
                        count_conflicts[neighbors[v_remove]] -= 1

                T = max(min_T, T * cooling)

            # Final greedy polish: add any zero-conflict vertex
            changed = True
            while changed and time.time() < sa_deadline:
                changed = False
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                zero_conf = not_in_set[count_conflicts[not_in_set] == 0]
                if len(zero_conf) > 0:
                    v = zero_conf[0]
                    in_set[v] = True
                    current_size += 1
                    count_conflicts[neighbors[v]] += 1
                    changed = True

            return local_best_in_set, local_best_size

        best_size = 0
        best_in_set = np.zeros(N, dtype=bool)
        local_deadline = start_time + max_time
        n_restarts = 0

        while time.time() < local_deadline and n_restarts < n_restarts:
            n_restarts += 1
            in_set = greedy_min_degree()
            current_size = int(in_set.sum())
            if current_size > best_size:
                best_size = current_size
                best_in_set = in_set.copy()

            # Per-restart SA budget shrinks as we approach the local deadline
            remaining = local_deadline - time.time()
            sa_budget = max(0.5, min(12.0, remaining * 0.25))
            ls_result, ls_size = sa_local_search(in_set, max_time=sa_budget)
            if ls_size > best_size:
                best_size = ls_size
                best_in_set = ls_result.copy()

        final_indices = np.where(best_in_set)[0]
        return candidates[final_indices], best_size

    best_result = np.array([], dtype=np.int64).reshape(0, D)
    best_size = 0

    # Case 1: R2 = 2, Weight 2 only (small, fast)
    if time.time() < deadline - 60:
        cands_w2 = get_candidates([2])
        res1, size1 = solve_independent_set(cands_w2, 2, max_time=25.0, n_restarts=30)
        if size1 > best_size:
            best_result, best_size = res1, size1

    # Case 2: R2 = 3, Weight 2, 3
    if time.time() < deadline - 40:
        cands_w23 = get_candidates([2, 3])
        res2, size2 = solve_independent_set(cands_w23, 3, max_time=35.0, n_restarts=50)
        if size2 > best_size:
            best_result, best_size = res2, size2

    # Case 3: R2 = 4, Weight 1, 2, 3, 4 (the productive regime; give it the most time)
    if time.time() < deadline - 10:
        cands_all = get_candidates([1, 2, 3, 4])
        remaining = deadline - 5 - time.time()
        res3, size3 = solve_independent_set(cands_all, 4, max_time=remaining, n_restarts=200)
        if size3 > best_size:
            best_result, best_size = res3, size3

    return best_result

# EVOLVE-BLOCK-END
