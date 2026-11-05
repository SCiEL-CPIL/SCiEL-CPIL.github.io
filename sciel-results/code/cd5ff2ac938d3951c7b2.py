import itertools
import time
import numpy as np
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under the 120 s kill

    # ---------- Precompute all t-subsets ----------
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}

    # ---------- Precompute all k-subsets (blocks) ----------
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Map block -> list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)

    # Map t-subset -> list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)

    C_k_t = math.comb(k, t)
    block_sizes = np.array([len(subs) for subs in block_to_tsubs], dtype=np.int32)
    n_tsub = n_t_subs

    # ---------- Bitmask representation for fast coverage checks ----------
    # Each block covers C(k,t) = C(7,4) = 35 four-subsets
    # We represent the set of covered t-subsets as a bitmask (Python int)
    block_bitmasks = []
    for subs in block_to_tsubs:
        mask = 0
        for s in subs:
            mask |= (1 << s)
        block_bitmasks.append(mask)

    # Precompute: for each block, which other blocks can replace it (share all critical subsets)
    # This is computed on-the-fly during search

    # ---------- Greedy construction ----------
    def greedy_init(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        uncovered = np.ones(n_t_subs, dtype=np.int8)
        total_uncovered = n_t_subs
        solution_set = set()
        solution_list = []
        uncovered_count_per_block = block_sizes.copy()

        while total_uncovered > 0:
            if time.time() > deadline:
                break
            best_b_idx = int(np.argmax(uncovered_count_per_block))
            if uncovered_count_per_block[best_b_idx] == 0:
                break
            solution_set.add(best_b_idx)
            solution_list.append(best_b_idx)
            for s in block_to_tsubs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = 0
                    total_uncovered -= 1
                    for b_idx in tsub_to_blocks[s]:
                        uncovered_count_per_block[b_idx] -= 1
        return solution_list, solution_set

    # ---------- Simulated Annealing Local Search ----------
    def sa_local_search(solution_list, solution_set, max_time=5.0):
        """
        Simulated annealing to minimize number of blocks.
        Moves: delete redundant block, swap block, 2-for-1 replacement.
        Uses bitmask for fast coverage verification.
        """
        if not solution_list:
            return solution_list, solution_set

        # Compute coverage count using bitmask approach
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1

        # Current energy = number of blocks
        current_energy = len(solution_list)
        best_energy = current_energy
        best_solution_list = list(solution_list)
        best_solution_set = set(solution_set)

        # Temperature schedule
        T0 = 10.0
        T_end = 0.01
        cooling_rate = 0.995
        T = T0

        rng = random.Random()

        sa_start = time.time()

        while time.time() - sa_start < max_time and T > T_end:
            T *= cooling_rate

            # Generate a random move
            move_type = rng.random()
            n = len(solution_list)

            if n == 0:
                break

            if move_type < 0.3 and n > 1:
                # Try deletion: remove a random block
                idx = rng.randint(0, n - 1)
                b_idx = solution_list[idx]
                # Check if redundant
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    # Accept deletion
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_list.pop(idx)
                    solution_set.remove(b_idx)
                    current_energy -= 1
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_solution_list = list(solution_list)
                        best_solution_set = set(solution_set)
                else:
                    # Reject (would break coverage)
                    pass

            elif move_type < 0.7 and n > 0:
                # Try swap: replace a random block with another
                idx = rng.randint(0, n - 1)
                b_idx = solution_list[idx]
                # Find critical subsets (only covered by this block)
                critical = [s for s in block_to_tsubs[b_idx] if cover_count[s] == 1]

                if not critical:
                    # Block is redundant, remove it
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_list.pop(idx)
                    solution_set.remove(b_idx)
                    current_energy -= 1
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_solution_list = list(solution_list)
                        best_solution_set = set(solution_set)
                else:
                    # Find candidates that cover all critical subsets
                    # Start with the most constrained critical subset
                    min_idx = min(range(len(critical)),
                                   key=lambda j: len(tsub_to_blocks[critical[j]]))
                    candidates = set(tsub_to_blocks[critical[min_idx]])
                    for s in critical:
                        if s != critical[min_idx]:
                            candidates.intersection_update(tsub_to_blocks[s])
                            if not candidates:
                                break

                    if candidates:
                        # Filter out blocks already in solution
                        candidates = [c for c in candidates if c not in solution_set]
                        if candidates:
                            # Pick a random candidate
                            new_b_idx = rng.choice(candidates)
                            # Perform swap
                            for s in block_to_tsubs[b_idx]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[new_b_idx]:
                                cover_count[s] += 1
                            solution_list[idx] = new_b_idx
                            solution_set.remove(b_idx)
                            solution_set.add(new_b_idx)
                            # Energy unchanged (still same number of blocks)
                            # But might enable future deletions
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_solution_list = list(solution_list)
                                best_solution_set = set(solution_set)

            else:
                # Try 2-for-1: replace two blocks with one
                if n >= 2:
                    idx1 = rng.randint(0, n - 1)
                    idx2 = rng.randint(0, n - 1)
                    if idx1 == idx2:
                        idx2 = (idx2 + 1) % n

                    b1 = solution_list[idx1]
                    b2 = solution_list[idx2]

                    # Find critical subsets for both blocks
                    crit1 = [s for s in block_to_tsubs[b1] if cover_count[s] == 1]
                    crit2 = [s for s in block_to_tsubs[b2] if cover_count[s] == 1]

                    all_crit = set(crit1) | set(crit2)

                    if len(all_crit) <= C_k_t:
                        # Find a block covering all critical subsets
                        min_crit = min(all_crit, key=lambda s: len(tsub_to_blocks[s]))
                        candidates = set(tsub_to_blocks[min_crit])
                        for s in all_crit:
                            if s != min_crit:
                                candidates.intersection_update(tsub_to_blocks[s])
                                if not candidates:
                                    break

                        if candidates:
                            candidates = [c for c in candidates if c not in solution_set]
                            if candidates:
                                new_b_idx = rng.choice(candidates)
                                # Perform 2-for-1
                                for s in block_to_tsubs[b1]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[b2]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[new_b_idx]:
                                    cover_count[s] += 1

                                # Remove both blocks, add new one
                                solution_list.pop(max(idx1, idx2))
                                solution_list.pop(min(idx1, idx2))
                                solution_list.append(new_b_idx)
                                solution_set.remove(b1)
                                solution_set.remove(b2)
                                solution_set.add(new_b_idx)
                                current_energy -= 1
                                if current_energy < best_energy:
                                    best_energy = current_energy
                                    best_solution_list = list(solution_list)
                                    best_solution_set = set(solution_set)

        # Return best solution found
        return best_solution_list, best_solution_set

    # ---------- Fast local search (deletion + swap only) ----------
    def fast_local_search(solution_list, solution_set):
        """Quick pass to remove redundant blocks and perform beneficial swaps."""
        if not solution_list:
            return solution_list, solution_set

        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1

        improved = True
        max_passes = 5

        while improved and time.time() < deadline - 2.0:
            improved = False

            # Deletion pass
            i = 0
            while i < len(solution_list):
                b_idx = solution_list[i]
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_list.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                else:
                    i += 1

            # Swap pass
            if not improved:
                i = 0
                while i < len(solution_list):
                    b_idx = solution_list[i]
                    critical = [s for s in block_to_tsubs[b_idx] if cover_count[s] == 1]
                    if not critical:
                        for s in block_to_tsubs[b_idx]:
                            cover_count[s] -= 1
                        solution_list.pop(i)
                        solution_set.remove(b_idx)
                        improved = True
                    else:
                        min_idx = min(range(len(critical)),
                                       key=lambda j: len(tsub_to_blocks[critical[j]]))
                        candidates = set(tsub_to_blocks[critical[min_idx]])
                        for s in critical:
                            if s != critical[min_idx]:
                                candidates.intersection_update(tsub_to_blocks[s])
                                if not candidates:
                                    break
                        if candidates:
                            for c in candidates:
                                if c not in solution_set:
                                    for s in block_to_tsubs[b_idx]:
                                        cover_count[s] -= 1
                                    for s in block_to_tsubs[c]:
                                        cover_count[s] += 1
                                    solution_list[i] = c
                                    solution_set.remove(b_idx)
                                    solution_set.add(c)
                                    improved = True
                                    break
                    i += 1

        return solution_list, solution_set

    # ---------- Perturbation: remove low-criticality blocks ----------
    def perturb(solution_list, solution_set, n_remove):
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1

        criticality = []
        for i, b_idx in enumerate(solution_list):
            crit_count = 0
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    crit_count += 1
            criticality.append(crit_count)

        if len(solution_list) <= n_remove:
            return greedy_init()

        indices_by_crit = sorted(range(len(solution_list)), key=lambda i: criticality[i])
        indices_to_remove = indices_by_crit[:n_remove]
        indices_to_remove_set = set(indices_to_remove)

        new_solution_set = set(solution_set)
        for idx in indices_to_remove:
            new_solution_set.remove(solution_list[idx])
        new_solution_list = [b for i, b in enumerate(solution_list) if i not in indices_to_remove_set]

        return new_solution_list, new_solution_set

    # ---------- Repair: add blocks to cover uncovered subsets ----------
    def repair(solution_list, solution_set):
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1

        uncovered_indices = [i for i in range(n_t_subs) if cover_count[i] == 0]

        if not uncovered_indices:
            return solution_list, solution_set

        to_cover = set(uncovered_indices)

        while to_cover and time.time() < deadline - 3.0:
            # Find the uncovered subset with fewest available candidates
            best_uncovered = None
            min_candidates = None
            min_len = float('inf')
            for s in to_cover:
                avail = [b for b in tsub_to_blocks[s] if b not in solution_set]
                if not avail:
                    break
                if len(avail) < min_len:
                    min_len = len(avail)
                    min_candidates = avail
                    best_uncovered = s
                if min_len == 1:
                    break

            if not min_candidates:
                break

            # Pick candidate covering most to_cover subsets
            best_b_idx = min_candidates[0]
            max_cov = -1
            for b_idx in min_candidates:
                cnt = 0
                for s in block_to_tsubs[b_idx]:
                    if s in to_cover:
                        cnt += 1
                if cnt > max_cov:
                    max_cov = cnt
                    best_b_idx = b_idx
                if cnt == len(to_cover):
                    break

            solution_set.add(best_b_idx)
            solution_list.append(best_b_idx)
            for s in block_to_tsubs[best_b_idx]:
                if s in to_cover:
                    to_cover.discard(s)

        return solution_list, solution_set

    # ---------- Main SA-based ILS Loop ----------
    # Initial solution
    best_solution_list, best_solution_set = greedy_init()
    best_solution_list, best_solution_set = fast_local_search(best_solution_list, best_solution_set)
    best_len = len(best_solution_list)

    current_solution_list = list(best_solution_list)
    current_solution_set = set(best_solution_set)

    stagnant = 0
    iter_count = 0

    while time.time() < deadline - 10.0:
        iter_count += 1

        # Adaptive perturbation
        if stagnant == 0:
            n_remove = random.randint(2, 5)
        elif stagnant < 5:
            n_remove = random.randint(4, 8)
        else:
            n_remove = random.randint(6, 12)

        # Perturb
        pert_list, pert_set = perturb(current_solution_list, current_solution_set, n_remove)

        # Repair
        rep_list, rep_set = repair(pert_list, pert_set)

        # SA optimization
        opt_list, opt_set = sa_local_search(rep_list, rep_set, max_time=3.0)

        # Fast local search
        opt_list, opt_set = fast_local_search(opt_list, opt_set)

        if len(opt_list) < best_len:
            best_len = len(opt_list)
            best_solution_list = opt_list
            best_solution_set = opt_set
            current_solution_list = opt_list
            current_solution_set = opt_set
            stagnant = 0
        else:
            stagnant += 1
            if len(opt_list) < len(current_solution_list):
                current_solution_list = opt_list
                current_solution_set = opt_set
            elif random.random() < 0.2:
                current_solution_list = opt_list
                current_solution_set = opt_set

            if stagnant > 8:
                # Restart from best
                current_solution_list = list(best_solution_list)
                current_solution_set = set(best_solution_set)
                stagnant = 0

                # Occasionally try fresh greedy
                if random.random() < 0.4 and time.time() < deadline - 15.0:
                    fresh_list, fresh_set = greedy_init(seed=iter_count)
                    fresh_list, fresh_set = fast_local_search(fresh_list, fresh_set)
                    if len(fresh_list) < best_len:
                        best_len = len(fresh_list)
                        best_solution_list = fresh_list
                        best_solution_set = fresh_set
                        current_solution_list = fresh_list
                        current_solution_set = fresh_set

    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_solution_list]
    return final_blocks
