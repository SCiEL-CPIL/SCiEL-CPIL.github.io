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

    # Bit-pack conflict matrix for O(1) neighborhood operations
    W = (N4 + 63) // 64  # number of 64-bit words
    N_words = W * 64     # padded size
    conflict_bits = np.zeros((N4, W), dtype=np.uint64)
    for i in range(N4):
        row = conflicts[i]
        for j in np.where(row)[0]:
            conflict_bits[i, j // 64] |= (np.uint64(1) << np.uint64(j % 64))

    # Precompute neighbor lists (for exact_polish and greedy)
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    # --- Bit-packed helper functions ---
    def popcount_uint64(arr):
        """Popcount for uint64 array."""
        x = arr.copy()
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        x = x + (x >> np.uint64(8))
        x = x + (x >> np.uint64(16))
        x = x + (x >> np.uint64(32))
        return (x & np.uint64(63)).astype(np.int32)

    def get_in_set_bits(in_set):
        """Convert boolean in_set array to bit-packed uint64 array."""
        bits = np.zeros(W, dtype=np.uint64)
        set_indices = np.where(in_set)[0]
        for idx in set_indices:
            bits[idx // 64] |= (np.uint64(1) << np.uint64(idx % 64))
        return bits

    def bits_to_in_set(bits):
        """Convert bit-packed uint64 array to boolean in_set array."""
        in_set = np.zeros(N4, dtype=bool)
        for w in range(W):
            if bits[w] != 0:
                base = w * 64
                for b in range(64):
                    if (bits[w] >> np.uint64(b)) & np.uint64(1):
                        idx = base + b
                        if idx < N4:
                            in_set[idx] = True
        return in_set

    def count_conflicts_from_bits(in_set_bits):
        """For each node, count how many of its neighbors are in the set."""
        counts = np.zeros(N4, dtype=np.int32)
        for i in range(N4):
            counts[i] = popcount_uint64(conflict_bits[i] & in_set_bits).sum()
        return counts

    def find_addable(in_set_bits):
        """Find all nodes not in set with 0 conflicts with set."""
        addable = np.zeros(N4, dtype=bool)
        for i in range(N4):
            if not (in_set_bits[i // 64] >> np.uint64(i % 64) & np.uint64(1)):
                if np.bitwise_and(conflict_bits[i], in_set_bits).any() == 0:
                    addable[i] = True
        return addable

    # --- Greedy initialization ---
    def greedy_min_degree(seed=None):
        """Greedy independent set with min-degree heuristic."""
        local_rng = np.random.default_rng(seed) if seed is not None else rng
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[local_rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def greedy_randomized(top_k=5, seed=None):
        """Greedy with randomization among top-k min-degree nodes."""
        local_rng = np.random.default_rng(seed) if seed is not None else rng
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            top = min(top_k, len(cands))
            chosen = cands[local_rng.integers(top)]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    # --- Bit-packed SA local search ---
    def sa_bitpacked(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        """Fast SA using bit-packed conflict representation."""
        in_set = in_set_init.copy()
        in_set_bits = get_in_set_bits(in_set)
        current_size = in_set.sum()

        T = 3.0
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        # Precompute addable mask and conflict counts
        addable = find_addable(in_set_bits)
        counts = count_conflicts_from_bits(in_set_bits)

        step = 0
        while time.time() < sa_deadline:
            step += 1

            # Try to add a 0-conflict node
            if addable.any():
                zero_cands = np.where(addable)[0]
                v = zero_cands[rng.integers(len(zero_cands))]

                in_set[v] = True
                in_set_bits[v // 64] |= (np.uint64(1) << np.uint64(v % 64))
                current_size += 1

                # Update addable: remove v and any nodes that now conflict
                addable[v] = False
                for u in neighbors[v]:
                    if in_set[u]:
                        addable[u] = False
                    else:
                        addable[u] = False  # u now conflicts with v which is in set

                # Actually, need to recompute addable properly
                # Nodes not in set with no conflict with set
                new_addable = np.zeros(N4, dtype=bool)
                for i in range(N4):
                    if not in_set[i]:
                        if np.bitwise_and(conflict_bits[i], in_set_bits).any() == 0:
                            new_addable[i] = True
                addable = new_addable

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue

            # No addable nodes: try a move
            # Pick node with minimum conflict count among not-in-set
            not_in_set_mask = ~in_set
            if not not_in_set_mask.any():
                break

            not_in_set_idx = np.where(not_in_set_mask)[0]
            c = counts[not_in_set_idx]
            min_c = np.min(c)
            min_cands = not_in_set_idx[c == min_c]
            v = min_cands[rng.integers(len(min_cands))]

            k = min_c
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
                # Remove all neighbors of v that are in set
                to_remove = neighbors[v]
                removed_any = False
                for u in to_remove:
                    if in_set[u]:
                        in_set[u] = False
                        in_set_bits[u // 64] &= ~(np.uint64(1) << np.uint64(u % 64))
                        removed_any = True

                if removed_any:
                    current_size -= len([u for u in to_remove if in_set[u]])
                    # Actually count properly
                    # Recompute size
                    current_size = in_set.sum()

                in_set[v] = True
                in_set_bits[v // 64] |= (np.uint64(1) << np.uint64(v % 64))
                current_size = in_set.sum()

                # Recompute addable and counts
                addable = find_addable(in_set_bits)
                counts = count_conflicts_from_bits(in_set_bits)

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1

            if stuck_counter > 2000:
                T = max(T, 2.0)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    # --- Larger swap perturbation for escaping local optima ---
    def large_swap_perturb(in_set, k_remove=3, k_add=2):
        """Remove k random nodes, add up to k_add new nodes."""
        in_set_new = in_set.copy()
        set_idx = np.where(in_set_new)[0]
        if len(set_idx) < k_remove:
            return in_set_new, in_set_new.sum()

        to_remove = rng.choice(set_idx, size=k_remove, replace=False)
        for i in to_remove:
            in_set_new[i] = False

        # Find addable nodes
        in_set_bits = get_in_set_bits(in_set_new)
        addable = find_addable(in_set_bits)
        addable_idx = np.where(addable)[0]

        if len(addable_idx) >= k_add:
            to_add = rng.choice(addable_idx, size=k_add, replace=False)
        elif len(addable_idx) > 0:
            to_add = addable_idx
        else:
            to_add = np.array([], dtype=int)

        for i in to_add:
            in_set_new[i] = True

        return in_set_new, in_set_new.sum()

    # --- Exact polish: try all single additions and 1-swap moves ---
    def exact_polish(in_set_init, max_time=5.0):
        """Exact local search: try all single additions and 1-swap moves."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        polish_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1

        improved = True
        while improved and time.time() < polish_deadline:
            improved = False

            # Try adding nodes with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                zero_cands = not_in_set[counts == 0]
                if len(zero_cands) > 0:
                    for v in zero_cands:
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                        improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()

            # Try 1-swap: remove one node i, add node v
            in_set_indices = np.where(in_set)[0]
            for i in in_set_indices:
                if time.time() >= polish_deadline:
                    break
                # Temporarily remove i
                in_set[i] = False
                current_size -= 1
                count_conflicts[neighbors[i]] -= 1

                not_in_set_now = np.where(~in_set)[0]
                if len(not_in_set_now) > 0:
                    counts_now = count_conflicts[not_in_set_now]
                    addable = not_in_set_now[counts_now == 0]
                    if len(addable) > 0:
                        # Score candidates by how many new nodes they unblock
                        best_v = addable[0]
                        best_score = -1
                        for v in addable:
                            v_nbrs = neighbors[v]
                            score = np.sum(~in_set[v_nbrs])
                            if score > best_score:
                                best_score = score
                                best_v = v
                        v = best_v
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                        improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                        continue
                # No swap possible, restore i
                in_set[i] = True
                current_size += 1
                count_conflicts[neighbors[i]] += 1

        return local_best_in_set, local_best_size

    # --- Redundant set elimination: remove nodes whose removal doesn't reduce max addable ---
    def eliminate_redundant(in_set, max_time=3.0):
        """Remove nodes that are redundant (their removal allows adding at least one node)."""
        in_set = in_set.copy()
        current_size = in_set.sum()
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        elim_deadline = time.time() + max_time

        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1

        improved = True
        while improved and time.time() < elim_deadline:
            improved = False
            in_set_indices = np.where(in_set)[0]
            for i in in_set_indices:
                if time.time() >= elim_deadline:
                    break
                # Check if removing i allows adding a node
                in_set[i] = False
                current_size -= 1
                count_conflicts[neighbors[i]] -= 1

                not_in_set_now = np.where(~in_set)[0]
                if len(not_in_set_now) > 0:
                    counts_now = count_conflicts[not_in_set_now]
                    addable = not_in_set_now[counts_now == 0]
                    if len(addable) > 0:
                        # Removing i is beneficial or neutral
                        # Add the best addable node
                        best_v = addable[0]
                        best_score = -1
                        for v in addable:
                            v_nbrs = neighbors[v]
                            score = np.sum(~in_set[v_nbrs])
                            if score > best_score:
                                best_score = score
                                best_v = v
                        v = best_v
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                        improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                        continue

                # No addable, restore i
                in_set[i] = True
                current_size += 1
                count_conflicts[neighbors[i]] += 1

        return local_best_in_set, local_best_size

    # --- Main search loop ---
    n_restarts = 0
    max_restarts = 200

    while time.time() < deadline - 12 and n_restarts < max_restarts:
        n_restarts += 1

        # 1. Greedy Init with varying strategy
        if n_restarts % 4 == 0:
            in_set = greedy_min_degree(seed=n_restarts * 7919)
        elif n_restarts % 4 == 1:
            in_set = greedy_randomized(top_k=3, seed=n_restarts * 104729)
        elif n_restarts % 4 == 2:
            in_set = greedy_randomized(top_k=7, seed=n_restarts * 15485863)
        else:
            in_set = greedy_min_degree(seed=n_restarts * 7919)

        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # 2. SA Local Search (bit-packed, fast)
        remaining_time = deadline - 12 - time.time()
        if remaining_time < 2:
            break

        sa_time = min(remaining_time * 0.4, 10.0)
        if sa_time < 1.5:
            sa_time = 1.5

        sa_result, sa_size = sa_bitpacked(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # 3. Large swap perturbation + short SA
        remaining_time = deadline - 10 - time.time()
        if remaining_time > 3:
            perturbed, pert_size = large_swap_perturb(sa_result, k_remove=3, k_add=2)
            if pert_size >= sa_size - 1:
                pert_time = min(remaining_time * 0.2, 4.0)
                if pert_time > 1:
                    pert_result, pert_result_size = sa_bitpacked(perturbed, max_time=pert_time, min_T=0.1, cooling=0.999)
                    if pert_result_size > best_size:
                        best_size = pert_result_size
                        best_in_set = pert_result.copy()

        # 4. Exact polish
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            polish_time = min(remaining_time * 0.4, 5.0)
            if polish_time > 1:
                polish_result, polish_size = exact_polish(best_in_set, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    # Final polish on best solution
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1 and best_size > 0:
        for pass_num in range(8):
            remaining_time = deadline - 1 - pass_num - time.time()
            if remaining_time < 0.5:
                break
            polish_time = min(remaining_time * 0.4, 3.0)
            if polish_time < 0.5:
                break

            # Alternate between exact polish and redundant elimination
            if pass_num % 2 == 0:
                polish_result, polish_size = exact_polish(best_in_set, max_time=polish_time)
            else:
                polish_result, polish_size = eliminate_redundant(best_in_set, max_time=polish_time)

            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
            else:
                # Try a short SA with low temperature
                remaining_time = deadline - 1 - time.time()
                if remaining_time > 1:
                    sa_time = min(remaining_time * 0.4, 2.0)
                    if sa_time > 0.5:
                        sa_result, sa_size = sa_bitpacked(
                            best_in_set, max_time=sa_time, min_T=0.01, cooling=0.9999
                        )
                        if sa_size > best_size:
                            best_size = sa_size
                            best_in_set = sa_result.copy()
                        break

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
