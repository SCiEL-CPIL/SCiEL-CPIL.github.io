import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}

    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Build binary coverage matrix M (n_blocks x n_t_subs)
    # M[b, s] = 1 if block b covers t-subset s
    M = np.zeros((n_blocks, n_t_subs), dtype=np.uint8)
    for b_idx, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            s_idx = t_sub_index[sub]
            M[b_idx, s_idx] = 1

    # Also build list of t-subset indices for each block (for fast iteration when needed)
    block_tsubs = []
    for b_idx in range(n_blocks):
        subs = np.where(M[b_idx] == 1)[0].tolist()
        block_tsubs.append(subs)

    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx in range(n_blocks):
        for s in block_tsubs[b_idx]:
            tsub_to_blocks[s].append(b_idx)

    # Helper: Greedy construction with vectorized gains
    def greedy_init():
        uncovered = np.ones(n_t_subs, dtype=np.uint8)
        solution_list = []
        solution_set = set()

        while True:
            if time.time() > deadline:
                break

            # Vectorized gain calculation
            gains = M @ uncovered
            best_b_idx = int(np.argmax(gains))
            if gains[best_b_idx] == 0:
                break

            solution_set.add(best_b_idx)
            solution_list.append(best_b_idx)

            # Update uncovered status
            uncovered[M[best_b_idx] == 1] = 0

            if not np.any(uncovered):
                break

        return solution_list, solution_set

    # Helper: Local Search (Deletion and Swap) - vectorized
    def local_search(solution_list, solution_set):
        if not solution_list:
            return solution_list, solution_set

        # Calculate cover counts using matrix multiplication
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            cover_count += M[b_idx]

        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False

            # Pass 1: Deletion - find redundant blocks
            for i in range(len(solution_list)):
                b_idx = solution_list[i]
                # Check if any t-subset covered by b_idx has cover_count == 1
                covered = M[b_idx] == 1
                if not np.any(cover_count[covered] == 1):
                    # Redundant
                    cover_count -= M[b_idx]
                    solution_list.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break

            # Pass 2: Swap - try to replace a block with a better one
            if not improved:
                idx_list = list(solution_set)
                random.shuffle(idx_list)

                for b_idx in idx_list:
                    if time.time() > deadline - 3.0:
                        break

                    covered = M[b_idx] == 1
                    critical_mask = cover_count == 1
                    critical_indices = np.where(covered & critical_mask)[0]

                    if len(critical_indices) == 0:
                        # Should have been removed in deletion pass
                        cover_count -= M[b_idx]
                        solution_set.remove(b_idx)
                        improved = True
                        break

                    if len(critical_indices) > 30:
                        # Too many constraints, unlikely to find a single block
                        continue

                    # Find candidates that cover all critical indices
                    # candidates_mask = M[:, critical_indices].all(axis=1)
                    # This is expensive, so we filter first
                    candidates = set()
                    for c_idx in critical_indices:
                        candidates.update(tsub_to_blocks[c_idx])
                    candidates -= solution_set

                    if not candidates:
                        continue

                    # Pick the candidate that covers the most other t-subsets in the solution
                    best_cand = -1
                    best_score = -1
                    for c_idx in candidates:
                        # Score: number of t-subsets covered by c_idx that are also covered by other blocks
                        # (i.e., not critical for other blocks)
                        c_covered = M[c_idx] == 1
                        score = np.sum(cover_count[c_covered] > 1)
                        if score > best_score:
                            best_score = score
                            best_cand = c_idx

                    if best_cand != -1:
                        # Perform swap
                        cover_count -= M[b_idx]
                        cover_count += M[best_cand]
                        solution_set.remove(b_idx)
                        solution_set.add(best_cand)
                        # Update solution_list
                        solution_list = list(solution_set)
                        improved = True
                        break

        return solution_list, solution_set

    # Helper: Perturbation and Repair
    def perturb_and_repair(solution_list, solution_set):
        n_remove = random.randint(3, 7)
        if len(solution_list) <= n_remove:
            return greedy_init()

        indices_to_remove = random.sample(range(len(solution_list)), n_remove)
        indices_to_remove.sort(reverse=True)

        new_solution_set = set(solution_set)
        for idx in indices_to_remove:
            b_idx = solution_list[idx]
            new_solution_set.remove(b_idx)

        new_solution_list = [b for i, b in enumerate(solution_list) if i not in indices_to_remove]

        # Calculate uncovered subsets
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in new_solution_list:
            cover_count += M[b_idx]

        uncovered_mask = (cover_count == 0)
        uncovered_indices = np.where(uncovered_mask)[0]

        if len(uncovered_indices) == 0:
            return new_solution_list, new_solution_set

        # Greedy Repair with vectorized gains
        uncovered = np.zeros(n_t_subs, dtype=np.uint8)
        uncovered[uncovered_indices] = 1

        while True:
            if time.time() > deadline:
                break

            gains = M @ uncovered
            # Only consider blocks not in new_solution_set
            for b_idx in range(n_blocks):
                if b_idx in new_solution_set:
                    gains[b_idx] = 0

            best_b_idx = int(np.argmax(gains))
            if gains[best_b_idx] == 0:
                break

            new_solution_set.add(best_b_idx)
            new_solution_list.append(best_b_idx)

            # Update uncovered
            uncovered[M[best_b_idx] == 1] = 0

            if not np.any(uncovered):
                break

        return new_solution_list, new_solution_set

    # Main ILS Loop
    best_solution_list, best_solution_set = greedy_init()
    best_solution_list, best_solution_set = local_search(best_solution_list, best_solution_set)
    best_len = len(best_solution_list)
    current_solution_list = best_solution_list
    current_solution_set = best_solution_set

    random.seed(42)

    while time.time() < deadline - 5.0:
        # Perturb
        pert_list, pert_set = perturb_and_repair(current_solution_list, current_solution_set)

        # Local Search on perturbed
        opt_list, opt_set = local_search(pert_list, pert_set)

        # Update
        if len(opt_list) < best_len:
            best_len = len(opt_list)
            best_solution_list = opt_list
            best_solution_set = opt_set
            current_solution_list = opt_list
            current_solution_set = opt_set
        else:
            if len(opt_list) < len(current_solution_list):
                current_solution_list = opt_list
                current_solution_set = opt_set

    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_solution_list]
    return final_blocks
