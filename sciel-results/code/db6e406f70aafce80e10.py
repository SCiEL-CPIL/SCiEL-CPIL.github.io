import itertools
import time
import numpy as np
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under the 120 s kill

    # ---------- Precompute all t-subsets and their indices ----------
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
            # Find the block covering the most currently-uncovered subsets.
            # Vectorized argmax over the current uncovered mask.
            # For speed, compute the count via a matrix-free approach:
            # count[b] = number of uncovered subsets in block_to_tsubs[b]
            # We maintain this incrementally below, so just argmax.
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

    # ---------- Full-sweep local search (Deletion, Swap, 2-for-1) ----------
    def local_search(solution_list, solution_set):
        if not solution_list:
            return solution_list, solution_set
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1

        improved = True
        max_passes = 30
        pass_count = 0

        while improved and time.time() < deadline - 1.0 and pass_count < max_passes:
            improved = False
            pass_count += 1
            n = len(solution_list)

            # Pass 1: Deletion — full sweep, remove any redundant block
            i = 0
            while i < len(solution_list):
                if time.time() > deadline - 1.0:
                    break
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
                    # don't increment i, re-check same index
                else:
                    i += 1

            # Pass 2: Swap — full sweep
            if not improved:
                i = 0
                while i < len(solution_list):
                    if time.time() > deadline - 1.0:
                        break
                    b_idx = solution_list[i]
                    critical_subsets = [s for s in block_to_tsubs[b_idx] if cover_count[s] == 1]
                    if not critical_subsets:
                        # redundant, remove
                        for s in block_to_tsubs[b_idx]:
                            cover_count[s] -= 1
                        solution_list.pop(i)
                        solution_set.remove(b_idx)
                        improved = True
                    else:
                        # Find a candidate block covering all critical subsets
                        if len(critical_subsets) == 1:
                            candidates = tsub_to_blocks[critical_subsets[0]]
                        else:
                            # Start from the most constrained critical subset
                            min_idx = min(range(len(critical_subsets)),
                                           key=lambda j: len(tsub_to_blocks[critical_subsets[j]]))
                            candidates = set(tsub_to_blocks[critical_subsets[min_idx]])
                            for s in critical_subsets:
                                if s == critical_subsets[min_idx]:
                                    continue
                                candidates.intersection_update(tsub_to_blocks[s])
                                if not candidates:
                                    break
                        found = False
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
                                    found = True
                                    break
                        if found:
                            # don't increment i, re-check
                            continue
                    i += 1

            # Pass 3: 2-for-1 replacement — full sweep
            if not improved:
                i = 0
                while i < len(solution_list):
                    if time.time() > deadline - 1.0:
                        break
                    b1 = solution_list[i]
                    crit1 = [s for s in block_to_tsubs[b1] if cover_count[s] == 1]
                    if not crit1:
                        # b1 is redundant, remove it
                        for s in block_to_tsubs[b1]:
                            cover_count[s] -= 1
                        solution_list.pop(i)
                        solution_set.remove(b1)
                        improved = True
                        continue
                    j = i + 1
                    found2 = False
                    while j < len(solution_list):
                        if time.time() > deadline - 1.0:
                            break
                        b2 = solution_list[j]
                        crit2 = [s for s in block_to_tsubs[b2] if cover_count[s] == 1]
                        if not crit2:
                            # b2 redundant, remove
                            for s in block_to_tsubs[b2]:
                                cover_count[s] -= 1
                            solution_list.pop(j)
                            solution_set.remove(b2)
                            improved = True
                            # j stays, re-check
                            continue
                        all_crit = set(crit1) | set(crit2)
                        if len(all_crit) <= C_k_t:
                            min_crit = min(all_crit, key=lambda s: len(tsub_to_blocks[s]))
                            candidates = set(tsub_to_blocks[min_crit])
                            for s in all_crit:
                                if s == min_crit:
                                    continue
                                candidates.intersection_update(tsub_to_blocks[s])
                                if not candidates:
                                    break
                            if candidates:
                                for c in candidates:
                                    if c not in solution_set:
                                        for s in block_to_tsubs[b1]:
                                            cover_count[s] -= 1
                                        for s in block_to_tsubs[b2]:
                                            cover_count[s] -= 1
                                        for s in block_to_tsubs[c]:
                                            cover_count[s] += 1
                                        solution_list.pop(j)
                                        solution_list.pop(i)
                                        solution_list.append(c)
                                        solution_set.remove(b1)
                                        solution_set.remove(b2)
                                        solution_set.add(c)
                                        improved = True
                                        found2 = True
                                        break
                        if found2:
                            break
                        j += 1
                    if not found2:
                        i += 1

        return solution_list, solution_set

    # ---------- Perturbation and Repair ----------
    def perturb_and_repair(solution_list, solution_set, n_remove):
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

        cover_count_new = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in new_solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count_new[s] += 1
        uncovered_mask = (cover_count_new == 0)
        uncovered_indices = np.where(uncovered_mask)[0]

        if len(uncovered_indices) == 0:
            return new_solution_list, new_solution_set

        to_cover = set(uncovered_indices.tolist())

        while to_cover:
            if time.time() > deadline - 2.0:
                break
            # Find the uncovered subset with the fewest candidate blocks
            best_uncovered = None
            min_candidates = None
            min_len = float('inf')
            for s in to_cover:
                cands = tsub_to_blocks[s]
                # Filter to blocks not already in solution
                avail = [b for b in cands if b not in new_solution_set]
                if not avail:
                    break
                if len(avail) < min_len:
                    min_len = len(avail)
                    min_candidates = avail
                    best_uncovered = s
                if min_len == 1:
                    break
            if min_candidates is None or not min_candidates:
                break

            # Pick the candidate that covers the most to_cover subsets
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

            new_solution_set.add(best_b_idx)
            new_solution_list.append(best_b_idx)
            for s in block_to_tsubs[best_b_idx]:
                if s in to_cover:
                    to_cover.discard(s)

        return new_solution_list, new_solution_set

    # ---------- Main ILS Loop ----------
    best_solution_list, best_solution_set = greedy_init()
    best_solution_list, best_solution_set = local_search(best_solution_list, best_solution_set)
    best_len = len(best_solution_list)
    current_solution_list = list(best_solution_list)
    current_solution_set = set(best_solution_set)

    stagnant = 0
    iter_count = 0

    while time.time() < deadline - 5.0:
        iter_count += 1
        # Adaptive perturbation strength
        if stagnant == 0:
            n_remove = random.randint(2, 4)
        elif stagnant < 3:
            n_remove = random.randint(3, 6)
        else:
            n_remove = random.randint(4, 8)

        pert_list, pert_set = perturb_and_repair(current_solution_list, current_solution_set, n_remove)
        opt_list, opt_set = local_search(pert_list, pert_set)

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
            elif random.random() < 0.15:
                current_solution_list = opt_list
                current_solution_set = opt_set
            if stagnant > 10:
                # Restart from best
                current_solution_list = list(best_solution_list)
                current_solution_set = set(best_solution_set)
                stagnant = 0
                # Occasionally try a fresh greedy from a different seed
                if random.random() < 0.3 and time.time() < deadline - 10.0:
                    fresh_list, fresh_set = greedy_init(seed=iter_count)
                    fresh_list, fresh_set = local_search(fresh_list, fresh_set)
                    if len(fresh_list) < best_len:
                        best_len = len(fresh_list)
                        best_solution_list = fresh_list
                        best_solution_set = fresh_set
                        current_solution_list = fresh_list
                        current_solution_set = fresh_set

    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_solution_list]
    return final_blocks
