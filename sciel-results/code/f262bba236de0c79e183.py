"""Kissing number in dimension 11 - Optimized Independent Set Search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Generate W4 vectors: norm²=4, with ±1 in exactly 4 positions
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)

    # Generate W1 vectors: norm²=4, with ±2 in exactly 1 position
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)

    all_v4 = np.array(v4_list, dtype=np.int64)
    all_v1 = np.array(v1_list, dtype=np.int64)
    N4 = len(all_v4)
    N1 = len(all_v1)

    if N4 == 0:
        return all_v1

    # Compute conflict matrix: two W4 vectors conflict if dot > 2
    dots = all_v4 @ all_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists and degrees
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    # W1 vectors never conflict with W4 vectors (dot is at most 2)
    # So we can always add all W1 vectors

    def greedy_init(seed=42):
        """Greedy minimum-degree independent set initialization."""
        rng_local = np.random.default_rng(seed)
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        deg = degrees.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            chosen = cands[rng_local.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def local_search(in_set, max_time=10.0, seed=42):
        """Multi-pass local search with various swap moves."""
        rng_local = np.random.default_rng(seed)
        in_set = in_set.copy()
        current_size = int(in_set.sum())
        best_size = current_size
        best_in_set = in_set.copy()

        # Track conflict counts: conflict_count[v] = # neighbors of v in in_set
        conflict_count = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1

        ls_deadline = time.time() + max_time
        stuck = 0

        while time.time() < ls_deadline:
            improved = False
            not_in_set = np.where(~in_set)[0]

            if len(not_in_set) > 0:
                counts = conflict_count[not_in_set]

                # Move 1: Add 0-conflict vector
                zero_conf = not_in_set[counts == 0]
                if len(zero_conf) > 0:
                    v = zero_conf[rng_local.integers(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                    stuck = 0
                    continue

                # Move 2: 1-for-2 swap (remove 1, add 2 non-conflicting)
                one_conf = not_in_set[counts == 1]
                if len(one_conf) > 0:
                    groups = defaultdict(list)
                    for v in one_conf:
                        nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                        if len(nbrs_in_set) == 1:
                            u = nbrs_in_set[0]
                            groups[u].append(v)

                    swap_found = False
                    for u, vs in groups.items():
                        if len(vs) >= 2:
                            # Find two non-conflicting vectors in vs
                            if len(vs) < 30:
                                for i in range(len(vs)):
                                    for j in range(i + 1, len(vs)):
                                        v1, v2 = vs[i], vs[j]
                                        if not conflicts[v1, v2]:
                                            in_set[u] = False
                                            in_set[v1] = True
                                            in_set[v2] = True
                                            conflict_count[neighbors[u]] -= 1
                                            conflict_count[neighbors[v1]] += 1
                                            conflict_count[neighbors[v2]] += 1
                                            current_size += 1
                                            improved = True
                                            swap_found = True
                                            if current_size > best_size:
                                                best_size = current_size
                                                best_in_set = in_set.copy()
                                            break
                                    if swap_found:
                                        break
                            else:
                                for _ in range(30):
                                    i = rng_local.integers(len(vs))
                                    j = rng_local.integers(len(vs))
                                    if i == j:
                                        continue
                                    v1, v2 = vs[i], vs[j]
                                    if not conflicts[v1, v2]:
                                        in_set[u] = False
                                        in_set[v1] = True
                                        in_set[v2] = True
                                        conflict_count[neighbors[u]] -= 1
                                        conflict_count[neighbors[v1]] += 1
                                        conflict_count[neighbors[v2]] += 1
                                        current_size += 1
                                        improved = True
                                        swap_found = True
                                        if current_size > best_size:
                                            best_size = current_size
                                            best_in_set = in_set.copy()
                                        break
                            if swap_found:
                                break
                    if swap_found:
                        stuck = 0
                        continue

                # Move 3: 1-for-3 swap (remove 1, add 3 mutually non-conflicting)
                if not improved and len(one_conf) > 0:
                    groups = defaultdict(list)
                    for v in one_conf:
                        nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                        if len(nbrs_in_set) == 1:
                            u = nbrs_in_set[0]
                            groups[u].append(v)

                    swap_found = False
                    for u, vs in groups.items():
                        if len(vs) >= 3:
                            chosen_vs = []
                            for v in vs:
                                if all(not conflicts[v, cv] for cv in chosen_vs):
                                    chosen_vs.append(v)
                                    if len(chosen_vs) == 3:
                                        break
                            if len(chosen_vs) == 3:
                                in_set[u] = False
                                for v in chosen_vs:
                                    in_set[v] = True
                                conflict_count[neighbors[u]] -= 1
                                for v in chosen_vs:
                                    conflict_count[neighbors[v]] += 1
                                current_size += 2
                                improved = True
                                swap_found = True
                                if current_size > best_size:
                                    best_size = current_size
                                    best_in_set = in_set.copy()
                                break
                        if swap_found:
                            break
                    if swap_found:
                        stuck = 0
                        continue

                # Move 4: 2-for-2 swap (remove 2, add 2 non-conflicting)
                if not improved and stuck > 3:
                    two_conf = not_in_set[counts == 2]
                    if len(two_conf) > 0:
                        pairs = defaultdict(list)
                        for v in two_conf:
                            nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                            if len(nbrs_in_set) == 2:
                                key = (min(nbrs_in_set), max(nbrs_in_set))
                                pairs[key].append(v)

                        swap_found = False
                        for (u1, u2), vs in pairs.items():
                            if len(vs) >= 2:
                                if len(vs) < 30:
                                    for i in range(len(vs)):
                                        for j in range(i + 1, len(vs)):
                                            v1, v2 = vs[i], vs[j]
                                            if not conflicts[v1, v2]:
                                                in_set[u1] = False
                                                in_set[u2] = False
                                                in_set[v1] = True
                                                in_set[v2] = True
                                                conflict_count[neighbors[u1]] -= 1
                                                conflict_count[neighbors[u2]] -= 1
                                                conflict_count[neighbors[v1]] += 1
                                                conflict_count[neighbors[v2]] += 1
                                                improved = True
                                                swap_found = True
                                                if current_size > best_size:
                                                    best_size = current_size
                                                    best_in_set = in_set.copy()
                                                break
                                        if swap_found:
                                            break
                                else:
                                    for _ in range(30):
                                        i = rng_local.integers(len(vs))
                                        j = rng_local.integers(len(vs))
                                        if i == j:
                                            continue
                                        v1, v2 = vs[i], vs[j]
                                        if not conflicts[v1, v2]:
                                            in_set[u1] = False
                                            in_set[u2] = False
                                            in_set[v1] = True
                                            in_set[v2] = True
                                            conflict_count[neighbors[u1]] -= 1
                                            conflict_count[neighbors[u2]] -= 1
                                            conflict_count[neighbors[v1]] += 1
                                            conflict_count[neighbors[v2]] += 1
                                            improved = True
                                            swap_found = True
                                            if current_size > best_size:
                                                best_size = current_size
                                                best_in_set = in_set.copy()
                                            break
                                if swap_found:
                                    break
                        if swap_found:
                            stuck = 0
                            continue

                # Move 5: 2-for-3 swap (remove 2, add 3 mutually non-conflicting)
                if not improved and stuck > 5:
                    two_conf = not_in_set[counts == 2]
                    if len(two_conf) > 0:
                        pairs = defaultdict(list)
                        for v in two_conf:
                            nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                            if len(nbrs_in_set) == 2:
                                key = (min(nbrs_in_set), max(nbrs_in_set))
                                pairs[key].append(v)

                        swap_found = False
                        for (u1, u2), vs in pairs.items():
                            if len(vs) >= 3:
                                # Try to find 3 mutually non-conflicting vectors in vs
                                # that also don't conflict with each other
                                # Note: vs are vectors that ONLY conflict with u1, u2.
                                # So if we pick v1, v2, v3 from vs, they don't conflict with the rest of the set.
                                # We just need them to not conflict with each other.
                                chosen = []
                                for v in vs:
                                    if all(not conflicts[v, cv] for cv in chosen):
                                        chosen.append(v)
                                        if len(chosen) == 3:
                                            break
                                if len(chosen) == 3:
                                    in_set[u1] = False
                                    in_set[u2] = False
                                    for v in chosen:
                                        in_set[v] = True
                                    conflict_count[neighbors[u1]] -= 1
                                    conflict_count[neighbors[u2]] -= 1
                                    for v in chosen:
                                        conflict_count[neighbors[v]] += 1
                                    current_size += 1
                                    improved = True
                                    swap_found = True
                                    if current_size > best_size:
                                        best_size = current_size
                                        best_in_set = in_set.copy()
                                    break
                        if swap_found:
                            stuck = 0
                            continue

            if not improved:
                stuck += 1
                if stuck > 10:
                    # Aggressive perturbation: remove 2 random in-set vectors
                    # and then immediately try to add back as many as possible
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) >= 2:
                        to_remove = rng_local.choice(in_set_indices, size=2, replace=False)
                        for u in to_remove:
                            in_set[u] = False
                            current_size -= 1
                            conflict_count[neighbors[u]] -= 1
                        
                        # Try to add back any 0-conflict vectors
                        not_in_set_now = np.where(~in_set)[0]
                        if len(not_in_set_now) > 0:
                            counts_now = conflict_count[not_in_set_now]
                            zero_conf_now = not_in_set_now[counts_now == 0]
                            for v in zero_conf_now:
                                in_set[v] = True
                                current_size += 1
                                conflict_count[neighbors[v]] += 1
                        stuck = 0

        return best_in_set, best_size

    def simulated_annealing(in_set_init, max_time=8.0, seed=42):
        """Simulated annealing with add/remove/swap moves."""
        rng_local = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 5.0
        cooling = 0.9995
        min_T = 0.05

        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            move_type = rng_local.random()

            if move_type < 0.45 and len(not_in_set) > 0:
                # ADD move
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng_local.integers(len(min_cands))]

                k = min_count
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng_local.random() < 0.3
                else:
                    accept = rng_local.random() < np.exp(delta / T)

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
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.65 and len(in_set_indices) > 0:
                # REMOVE move
                v = in_set_indices[rng_local.integers(len(in_set_indices))]
                delta = -1
                accept = rng_local.random() < np.exp(delta / T)

                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                    stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.85 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move
                v_remove = in_set_indices[rng_local.integers(len(in_set_indices))]
                v_add_candidates = not_in_set[(count_conflicts[not_in_set] <= 1)]
                if len(v_add_candidates) == 0:
                    v_add_candidates = not_in_set
                v_add = v_add_candidates[rng_local.integers(len(v_add_candidates))]

                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]

                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                    stuck_counter += 1
                else:
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng_local.random() < np.exp(delta / T))
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
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
            else:
                # 2-for-1 move
                if len(in_set_indices) >= 2 and len(not_in_set) > 0:
                    v_add = not_in_set[rng_local.integers(len(not_in_set))]
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    if len(conflicts_with) == 2:
                        to_remove = conflicts_with
                        delta = 1 - 2
                        accept = rng_local.random() < np.exp(delta / T)
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
                                stuck_counter = 0
                            else:
                                stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            T = max(min_T, T * cooling)

            if stuck_counter > 60:
                T = 5.0
                stuck_counter = 0

        return local_best_in_set, local_best_size

    def final_polish(in_set):
        """Aggressively add all possible 0-conflict vectors repeatedly."""
        in_set = in_set.copy()
        conflict_count = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1

        changed = True
        while changed:
            changed = False
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            counts = conflict_count[not_in_set]
            zero_conf = not_in_set[counts == 0]
            for v in zero_conf:
                in_set[v] = True
                conflict_count[neighbors[v]] += 1
                changed = True
            if changed:
                not_in_set = np.where(~in_set)[0]

        return in_set

    # Main search loop
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    # Initial greedy
    in_set = greedy_init()
    size = int(in_set.sum())
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()

    # Run local search on initial result
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 3:
        ls_time = min(remaining_time * 0.35, 12.0)
        ls_result, ls_size = local_search(best_in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    # Run SA on best result
    remaining_time = deadline - 5 - time.time()
    if remaining_time > 3:
        sa_time = min(remaining_time * 0.4, 10.0)
        sa_result, sa_size = simulated_annealing(best_in_set, max_time=sa_time)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

    # Multiple restarts
    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 40:
        n_restarts += 1
        in_set = greedy_init(seed=100 + n_restarts)
        size = int(in_set.sum())
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()

        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break

        # Alternate between LS and SA
        if n_restarts % 3 == 0:
            ls_time = min(remaining_time * 0.25, 7.0)
            ls_result, ls_size = local_search(in_set, max_time=ls_time)
            if ls_size > best_size:
                best_size = ls_size
                best_in_set = ls_result.copy()
        elif n_restarts % 3 == 1:
            sa_time = min(remaining_time * 0.25, 7.0)
            sa_result, sa_size = simulated_annealing(in_set, max_time=sa_time)
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()
        else:
            # Run both LS then SA
            ls_time = min(remaining_time * 0.15, 5.0)
            ls_result, ls_size = local_search(in_set, max_time=ls_time)
            if ls_size > best_size:
                best_size = ls_size
                best_in_set = ls_result.copy()

            remaining_time = deadline - 5 - time.time()
            if remaining_time > 2:
                sa_time = min(remaining_time * 0.15, 5.0)
                sa_result, sa_size = simulated_annealing(best_in_set, max_time=sa_time)
                if sa_size > best_size:
                    best_size = sa_size
                    best_in_set = sa_result.copy()

    # Final polish
    if time.time() < deadline - 2:
        best_in_set = final_polish(best_in_set)
        best_size = int(best_in_set.sum())

    # Combine W4 independent set with ALL W1 vectors
    final_indices = np.where(best_in_set)[0]
    result_v4 = all_v4[final_indices]

    if len(result_v4) > 0:
        result = np.vstack([result_v4, all_v1])
    else:
        result = all_v1

    return result

# EVOLVE-BLOCK-END
