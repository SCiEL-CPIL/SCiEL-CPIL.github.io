import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors (weight 4, entries in {-1, 0, 1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2, 0, 2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists as arrays for faster SA updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def local_search_fast(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        """Optimized SA with O(1) not_in_set updates, swap moves, and position tracking."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 3.0
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1

        # Maintain not_in_set as a list with position tracking for O(1) removal
        not_in_set = list(np.where(~in_set)[0])
        pos_in_not_in_set = {}
        for idx, v in enumerate(not_in_set):
            pos_in_not_in_set[v] = idx

        # Maintain in_set list for swap moves
        in_set_list = list(np.where(in_set)[0])
        pos_in_set = {}
        for idx, v in enumerate(in_set_list):
            pos_in_set[v] = idx

        def remove_from_not_in_set(v):
            """O(1) removal from not_in_set list."""
            if v in pos_in_not_in_set:
                idx = pos_in_not_in_set[v]
                last = not_in_set[-1]
                not_in_set[idx] = last
                pos_in_not_in_set[last] = idx
                not_in_set.pop()
                del pos_in_not_in_set[v]

        def add_to_not_in_set(v):
            """O(1) addition to not_in_set list."""
            if v not in pos_in_not_in_set:
                pos_in_not_in_set[v] = len(not_in_set)
                not_in_set.append(v)

        def remove_from_in_set_list(v):
            """O(1) removal from in_set_list."""
            if v in pos_in_set:
                idx = pos_in_set[v]
                last = in_set_list[-1]
                in_set_list[idx] = last
                pos_in_set[last] = idx
                in_set_list.pop()
                del pos_in_set[v]

        def add_to_in_set_list(v):
            """O(1) addition to in_set_list."""
            if v not in pos_in_set:
                pos_in_set[v] = len(in_set_list)
                in_set_list.append(v)

        while time.time() < sa_deadline:
            if len(not_in_set) == 0:
                break

            # Choose move type: add, swap, or remove
            r = rng.random()
            if r < 0.55:
                move_type = 'add'
            elif r < 0.85 and len(in_set_list) > 0:
                move_type = 'swap'
            else:
                move_type = 'remove'

            if move_type == 'add':
                # Sample a subset if not_in_set is large
                if len(not_in_set) > 300:
                    sample_size = min(300, len(not_in_set))
                    sample_indices = rng.choice(len(not_in_set), size=sample_size, replace=False)
                    sample = np.array([not_in_set[i] for i in sample_indices])
                else:
                    sample = np.array(not_in_set)

                counts = count_conflicts[sample]
                min_count = int(np.min(counts))

                if min_count == 0:
                    zero_mask = counts == 0
                    if np.any(zero_mask):
                        zero_cands = sample[zero_mask]
                        v = int(zero_cands[rng.integers(len(zero_cands))])
                    else:
                        v = int(not_in_set[rng.integers(len(not_in_set))])
                else:
                    one_mask = counts == 1
                    if np.any(one_mask) and rng.random() < 0.7:
                        one_cands = sample[one_mask]
                        v = int(one_cands[rng.integers(len(one_cands))])
                    else:
                        min_mask = counts == min_count
                        min_cands = sample[min_mask]
                        v = int(min_cands[rng.integers(len(min_cands))])

                k = int(count_conflicts[v])
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    if delta < -20 * T:
                        accept = False
                    else:
                        accept = rng.random() < np.exp(delta / T)

                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                            add_to_not_in_set(int(u))
                            remove_from_in_set_list(int(u))

                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                    remove_from_not_in_set(v)
                    add_to_in_set_list(v)

                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type == 'swap':
                # Pick a random in-set node and a random not-in-set node
                if len(in_set_list) == 0 or len(not_in_set) == 0:
                    stuck_counter += 1
                    continue

                u = int(in_set_list[rng.integers(len(in_set_list))])
                if len(not_in_set) > 300:
                    sample_size = min(300, len(not_in_set))
                    sample_indices = rng.choice(len(not_in_set), size=sample_size, replace=False)
                    sample = np.array([not_in_set[i] for i in sample_indices])
                else:
                    sample = np.array(not_in_set)

                v = int(sample[rng.integers(len(sample))])

                # Compute delta for swap: remove u, add v
                # After removing u: u's neighbors lose one conflict count
                # After adding v: v must not conflict with remaining in_set
                # Net delta = 1 - (number of in_set neighbors of v that are NOT u)
                # = 1 - (count_conflicts[v] - (1 if u is neighbor of v else 0))
                u_is_nbr_of_v = bool(conflicts[v, u])
                effective_conflicts_v = count_conflicts[v] - (1 if u_is_nbr_of_v else 0)
                # But we also need to check: after removing u, some of v's neighbors
                # that were only conflicted via u might become free
                # Actually the proper delta:
                # Remove u: size -1, then add v: size +1 if v has no conflicts with remaining in_set
                # v conflicts with in_set minus {u} = count_conflicts[v] - (1 if u~v else 0)
                # But wait: when we remove u, the in_set changes. Let's be precise:
                # New in_set = old_in_set - {u} + {v} (if valid)
                # v is valid if v has no neighbors in (old_in_set - {u})
                # = count_conflicts[v] - (1 if u is neighbor of v else 0) == 0
                # If valid: delta = 0 (remove one, add one)
                # If not valid: we'd need to remove additional nodes, making it complex
                # For simplicity, only accept swaps where v is compatible after removing u
                if effective_conflicts_v == 0:
                    # Valid swap, delta = 0
                    accept = rng.random() < 0.5
                else:
                    # Try swap with removals: remove u and all of v's other in_set neighbors
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    # u might or might not be in to_remove
                    n_remove = len(to_remove)
                    delta = 1 - n_remove
                    if delta > 0:
                        accept = True
                    elif delta == 0:
                        accept = rng.random() < 0.3
                    else:
                        if delta < -20 * T:
                            accept = False
                        else:
                            accept = rng.random() < np.exp(delta / T)

                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for w in to_remove:
                            count_conflicts[neighbors[w]] -= 1
                            add_to_not_in_set(int(w))
                            remove_from_in_set_list(int(w))

                    in_set[v] = True
                    current_size += (1 - len(to_remove))
                    count_conflicts[neighbors[v]] += 1
                    remove_from_not_in_set(v)
                    add_to_in_set_list(v)

                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            else:
                # Remove a node from in_set (to allow re-exploration)
                if len(in_set_list) == 0:
                    stuck_counter += 1
                    continue

                u = int(in_set_list[rng.integers(len(in_set_list))])
                # Only remove if it doesn't reduce size (i.e., we plan to re-add later)
                # For now, just do it with low probability
                if rng.random() < 0.1:
                    in_set[u] = False
                    current_size -= 1
                    count_conflicts[neighbors[u]] -= 1
                    add_to_not_in_set(u)
                    remove_from_in_set_list(u)
                    stuck_counter += 1
                else:
                    stuck_counter += 1

            if stuck_counter > 600:
                T = max(T, 2.5)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def exact_polish(in_set_init, max_time=15.0):
        """High-precision polish: iterative greedy add + swap optimization."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        polish_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1

        # Phase 1: Greedy add all zero-conflict nodes repeatedly
        while time.time() < polish_deadline - 2:
            changed = False
            # Find all zero-conflict nodes not in set
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            zero_cands = not_in[count_conflicts[not_in] == 0]
            if len(zero_cands) > 0:
                for v in zero_cands:
                    in_set[v] = True
                    current_size += 1
                    count_conflicts[neighbors[v]] += 1
                changed = True
            if not changed:
                break

        # Phase 2: Iterative swap optimization
        while time.time() < polish_deadline:
            improved = False
            in_set_list = list(np.where(in_set)[0])
            not_in_set_list = list(np.where(~in_set)[0])

            if len(in_set_list) == 0 or len(not_in_set_list) == 0:
                break

            # Try each in-set node for potential swap
            for u in in_set_list:
                if time.time() >= polish_deadline:
                    break

                # Find not-in-set nodes that would be compatible if we remove u
                # v is compatible if count_conflicts[v] - (1 if u~v else 0) == 0
                # and v is not in set
                u_nbrs_in_not = np.where(conflicts[u] & ~in_set)[0]
                # For each v in not_in_set, effective conflicts = count_conflicts[v] - (1 if u~v)
                # We want effective == 0, i.e., count_conflicts[v] == (1 if u~v else 0)
                # So v must have count_conflicts[v] <= 1, and if count_conflicts[v]==1, u must be its only neighbor in set

                for v in not_in_set_list:
                    if time.time() >= polish_deadline:
                        break

                    cc = int(count_conflicts[v])
                    if cc == 0:
                        # v can be added without removing anything (shouldn't happen after Phase 1)
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                        improved = True
                        break
                    elif cc == 1:
                        # v has exactly one conflict in set; check if it's u
                        if conflicts[v, u]:
                            # Swap u for v: net change = 0 in size, but might enable future adds
                            # Actually this is a neutral swap; do it to explore
                            in_set[u] = False
                            in_set[v] = True
                            # Update counts
                            count_conflicts[neighbors[u]] -= 1
                            count_conflicts[neighbors[v]] += 1
                            # Fix: v's count was 1 (u), now 0; u's neighbors lose u
                            # But v might conflict with other in_set nodes? No, cc==1 and that was u
                            improved = True
                            break
                    # Also try: remove u, see if we can add multiple nodes
                    # This is expensive; skip for now

                if improved:
                    break

            if not improved:
                # Try 2-for-1 swaps: remove u, add v1 and v2 that were only blocked by u
                found_2for1 = False
                for u in in_set_list:
                    if time.time() >= polish_deadline:
                        break

                    # Nodes that have u as their only conflict
                    candidates = []
                    for v in not_in_set_list:
                        if time.time() >= polish_deadline:
                            break
                        cc = int(count_conflicts[v])
                        if cc == 1 and conflicts[v, u]:
                            candidates.append(v)

                    if len(candidates) >= 2:
                        # Check if candidates are mutually non-conflicting
                        valid_pair = False
                        for i in range(len(candidates)):
                            for j in range(i+1, len(candidates)):
                                v1, v2 = candidates[i], candidates[j]
                                if not conflicts[v1, v2]:
                                    # Valid 2-for-1 swap
                                    in_set[u] = False
                                    in_set[v1] = True
                                    in_set[v2] = True
                                    current_size += 1
                                    count_conflicts[neighbors[u]] -= 1
                                    count_conflicts[neighbors[v1]] += 1
                                    count_conflicts[neighbors[v2]] += 1
                                    # Update: v1 and v2 each had 1 conflict (u), now 0
                                    # Other nodes' counts change
                                    found_2for1 = True
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break

                if not improved:
                    # No more improvements possible
                    break

        return in_set, current_size

    # Main Loop
    n_restarts = 0
    max_restarts = 200

    while time.time() < deadline - 25 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = greedy_min_degree()
        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        remaining_time = deadline - 25 - time.time()
        if remaining_time < 2:
            break

        sa_time = min(remaining_time * 0.5, 14.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = local_search_fast(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

    # Final high-precision polish pass
    remaining_polish = deadline - 5 - time.time()
    if remaining_polish > 3:
        polished, polished_size = exact_polish(best_in_set, max_time=remaining_polish)
        if polished_size > best_size:
            best_size = polished_size
            best_in_set = polished

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
