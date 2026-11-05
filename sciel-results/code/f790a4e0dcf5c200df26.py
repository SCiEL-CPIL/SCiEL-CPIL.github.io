import itertools
import time
import random
import heapq
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}

    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Map block index to list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)

    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)

    # Compute frequency of each t-subset (how many blocks contain it)
    sub_freq = np.zeros(n_t_subs, dtype=np.int32)
    for subs in block_to_tsubs:
        for s in subs:
            sub_freq[s] += 1

    # Weights for weighted greedy: inverse frequency
    weights = 1.0 / np.maximum(sub_freq, 1).astype(np.float64)

    # --- Phase 1: Weighted Greedy Construction ---
    uncovered = np.ones(n_t_subs, dtype=np.bool_)
    total_uncovered = n_t_subs
    solution_blocks = []
    solution_set = set()

    # gain[b] = sum of weights of uncovered t-subsets in block b
    gain = np.array([sum(weights[s] for s in block_to_tsubs[b]) for b in range(n_blocks)], dtype=np.float64)

    while total_uncovered > 0 and time.time() < deadline - 5:
        # Find block with max gain
        best_idx = int(np.argmax(gain))
        best_gain = gain[best_idx]

        if best_gain <= 1e-12:
            break

        solution_blocks.append(best_idx)
        solution_set.add(best_idx)

        # Update uncovered and gains
        for s in block_to_tsubs[best_idx]:
            if uncovered[s]:
                uncovered[s] = False
                total_uncovered -= 1
                # Subtract weight from all blocks covering s
                for b_idx in tsub_to_blocks[s]:
                    gain[b_idx] -= weights[s]

    # --- Phase 2: Local Search ---
    # cover_count[s] = number of blocks in solution covering t-subset s
    cover_count = np.zeros(n_t_subs, dtype=np.int32)
    for b_idx in solution_blocks:
        cover_count[block_to_tsubs[b_idx]] += 1

    best_solution = list(solution_blocks)
    best_len = len(solution_blocks)

    def is_valid(sol):
        if not sol:
            return n_t_subs == 0
        cc = np.zeros(n_t_subs, dtype=np.int32)
        for b in sol:
            cc[block_to_tsubs[b]] += 1
        return np.all(cc > 0)

    # Ensure initial solution is valid
    if not is_valid(best_solution):
        for s in range(n_t_subs):
            if cover_count[s] == 0:
                b = tsub_to_blocks[s][0]
                if b not in solution_set:
                    solution_blocks.append(b)
                    solution_set.add(b)
                    cover_count[block_to_tsubs[b]] += 1
        best_solution = list(solution_blocks)
        best_len = len(solution_blocks)

    no_improve_count = 0
    max_no_improve = 80

    while time.time() < deadline - 10:
        improved = False
        current_len = len(solution_blocks)

        # Pass 1: Try to remove redundant blocks
        if current_len > 0:
            for i in range(current_len):
                b_idx = solution_blocks[i]
                if np.all(cover_count[block_to_tsubs[b_idx]] > 1):
                    cover_count[block_to_tsubs[b_idx]] -= 1
                    solution_blocks.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    no_improve_count = 0
                    if len(solution_blocks) < best_len:
                        best_len = len(solution_blocks)
                        best_solution = list(solution_blocks)
                    break

        # Pass 2: Try single-block swap
        if not improved and time.time() < deadline - 10:
            n_sol = len(solution_blocks)
            if n_sol > 0:
                # Find block with fewest critical subsets
                min_crit = float('inf')
                target_idx = -1
                target_crit = []

                for i in range(n_sol):
                    b_idx = solution_blocks[i]
                    crit = [s for s in block_to_tsubs[b_idx] if cover_count[s] == 1]
                    if len(crit) < min_crit:
                        min_crit = len(crit)
                        target_idx = i
                        target_crit = crit
                        if min_crit == 0:
                            break

                if target_idx != -1 and 0 < min_crit <= 5:  # Only swap if few critical subsets
                    b_idx = solution_blocks[target_idx]
                    # Find replacement covering all critical subsets
                    if target_crit:
                        candidates = set(tsub_to_blocks[target_crit[0]])
                        for s in target_crit[1:]:
                            candidates.intersection_update(tsub_to_blocks[s])
                            if not candidates:
                                break

                        if candidates:
                            # Pick candidate not in solution that maximizes overlap
                            b_subs = set(block_to_tsubs[b_idx])
                            best_cand = None
                            max_ov = -1
                            for c in candidates:
                                if c not in solution_set:
                                    ov = len(b_subs & set(block_to_tsubs[c]))
                                    if ov > max_ov:
                                        max_ov = ov
                                        best_cand = c

                            if best_cand is not None:
                                cover_count[block_to_tsubs[b_idx]] -= 1
                                cover_count[block_to_tsubs[best_cand]] += 1
                                solution_blocks[target_idx] = best_cand
                                solution_set.remove(b_idx)
                                solution_set.add(best_cand)
                                improved = True
                                no_improve_count = 0
                                if len(solution_blocks) < best_len:
                                    best_len = len(solution_blocks)
                                    best_solution = list(solution_blocks)

        # Pass 3: Try 2-block removal (if no single removal worked)
        if not improved and time.time() < deadline - 10:
            n_sol = len(solution_blocks)
            if n_sol >= 2:
                # Try pairs of blocks
                found = False
                for i in range(n_sol):
                    if found:
                        break
                    b1 = solution_blocks[i]
                    crit1 = [s for s in block_to_tsubs[b1] if cover_count[s] == 1]
                    if len(crit1) > 3:
                        continue
                    for j in range(i + 1, n_sol):
                        b2 = solution_blocks[j]
                        crit2 = [s for s in block_to_tsubs[b2] if cover_count[s] == 1]
                        if len(crit2) > 3:
                            continue

                        # Check if removing both leaves any subset uncovered
                        all_crit = set(crit1) | set(crit2)
                        ok = True
                        for s in all_crit:
                            if cover_count[s] <= 1:
                                # This subset is only covered by b1 or b2
                                # Check if other block covers it
                                if s in block_to_tsubs[b1] and s not in block_to_tsubs[b2]:
                                    ok = False
                                    break
                                elif s in block_to_tsubs[b2] and s not in block_to_tsubs[b1]:
                                    ok = False
                                    break
                                else:
                                    # Covered by both, after removal count becomes 0
                                    # unless another block covers it
                                    if cover_count[s] == 1:
                                        ok = False
                                        break

                        if ok:
                            # Both can be removed
                            cover_count[block_to_tsubs[b1]] -= 1
                            cover_count[block_to_tsubs[b2]] -= 1
                            # Remove in reverse order to preserve indices
                            if j > i:
                                solution_blocks.pop(j)
                                solution_blocks.pop(i)
                            else:
                                solution_blocks.pop(i)
                                solution_blocks.pop(j)
                            solution_set.remove(b1)
                            solution_set.remove(b2)
                            improved = True
                            no_improve_count = 0
                            if len(solution_blocks) < best_len:
                                best_len = len(solution_blocks)
                                best_solution = list(solution_blocks)
                            found = True
                            break

        if not improved:
            no_improve_count += 1
            # Perturbation
            if no_improve_count > max_no_improve:
                no_improve_count = 0
                if len(solution_blocks) > 15:
                    # Remove 3-5 random blocks
                    num_remove = random.randint(3, 5)
                    for _ in range(num_remove):
                        if not solution_blocks:
                            break
                        idx = random.randint(0, len(solution_blocks) - 1)
                        b_idx = solution_blocks[idx]
                        cover_count[block_to_tsubs[b_idx]] -= 1
                        solution_blocks.pop(idx)
                        solution_set.remove(b_idx)

                    # Re-greedy fill for uncovered subsets
                    max_iter = 50
                    for _ in range(max_iter):
                        if time.time() > deadline - 10:
                            break
                        uncov = np.where(cover_count == 0)[0]
                        if len(uncov) == 0:
                            break
                        s = int(uncov[random.randint(0, len(uncov) - 1)])
                        cands = [b for b in tsub_to_blocks[s] if b not in solution_set]
                        if not cands:
                            cands = tsub_to_blocks[s]
                        b = random.choice(cands)
                        solution_blocks.append(b)
                        solution_set.add(b)
                        cover_count[block_to_tsubs[b]] += 1

                    if len(solution_blocks) < best_len:
                        best_len = len(solution_blocks)
                        best_solution = list(solution_blocks)

    # Final Polish: Remove redundant blocks from best_solution
    if best_solution:
        final_cc = np.zeros(n_t_subs, dtype=np.int32)
        for b in best_solution:
            final_cc[block_to_tsubs[b]] += 1

        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for i in range(len(best_solution)):
                b = best_solution[i]
                if np.all(final_cc[block_to_tsubs[b]] > 1):
                    final_cc[block_to_tsubs[b]] -= 1
                    best_solution.pop(i)
                    changed = True
                    break

    # Final Validation
    if not is_valid(best_solution):
        final_cc = np.zeros(n_t_subs, dtype=np.int32)
        for b in best_solution:
            final_cc[block_to_tsubs[b]] += 1
        for s in range(n_t_subs):
            if final_cc[s] == 0:
                b = tsub_to_blocks[s][0]
                if b not in set(best_solution):
                    best_solution.append(b)
                    final_cc[block_to_tsubs[b]] += 1

    return [list(all_blocks[b_idx]) for b_idx in best_solution]
