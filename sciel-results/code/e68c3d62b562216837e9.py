import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # 1. Map t-subsets to IDs
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_map = {sub: i for i, sub in enumerate(t_subs)}
    num_t = len(t_subs)

    # 2. Pre-compute block -> t-subset IDs
    k_subs = list(itertools.combinations(range(v), k))
    num_k = len(k_subs)

    t_in_k = list(itertools.combinations(range(k), t))
    num_t_in_k = len(t_in_k)

    block_t_ids = np.zeros((num_k, num_t_in_k), dtype=np.int32)

    for i, block in enumerate(k_subs):
        for j, t_indices in enumerate(t_in_k):
            global_t_sub = tuple(sorted(block[x] for x in t_indices))
            block_t_ids[i, j] = t_sub_map[global_t_sub]

    # Block index mapping
    block_to_idx = {block: i for i, block in enumerate(k_subs)}

    # 3. Greedy Cover to get initial solution
    covered = np.zeros(num_t, dtype=bool)
    used_blocks = set()
    solution = []

    while not covered.all():
        if time.time() > deadline - 30.0:
            break

        G = np.sum(~covered[block_t_ids], axis=1)
        G[list(used_blocks)] = -1
        best_idx = np.argmax(G)

        if G[best_idx] <= 0:
            break

        used_blocks.add(best_idx)
        solution.append(list(k_subs[best_idx]))
        covered[block_t_ids[best_idx]] = True

    # 4. Local Search: Remove redundant blocks
    def remove_redundant(sol):
        if not sol:
            return sol, False
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in sol:
            idx = block_to_idx[tuple(b)]
            cover_count[block_t_ids[idx]] += 1

        new_sol = []
        changed = False
        for b in sol:
            idx = block_to_idx[tuple(b)]
            min_count = np.min(cover_count[block_t_ids[idx]])
            if min_count > 1:
                changed = True
                cover_count[block_t_ids[idx]] -= 1
            else:
                new_sol.append(b)
        return new_sol, changed

    for _ in range(10):
        if time.time() > deadline - 30.0:
            break
        solution, changed = remove_redundant(solution)
        if not changed:
            break

    # 5. WalkSAT-style local search with 2-swap moves
    included = np.zeros(num_k, dtype=bool)
    for b in solution:
        included[block_to_idx[tuple(b)]] = True

    # Coverage count
    cover_count = np.zeros(num_t, dtype=np.int32)
    for i in range(num_k):
        if included[i]:
            cover_count[block_t_ids[i]] += 1

    num_uncovered = np.sum(cover_count == 0)
    best_num_uncovered = num_uncovered
    best_included = included.copy()
    best_block_count = np.sum(included)

    # Build t_to_blocks mapping for swap search
    t_to_blocks = defaultdict(set)
    for i in range(num_k):
        for tid in block_t_ids[i]:
            t_to_blocks[tid].add(i)

    # Parameters
    noise = 0.2
    max_steps = 20000

    step = 0
    no_improve_steps = 0
    max_no_improve = 500

    while step < max_steps and time.time() < deadline - 15.0:
        step += 1

        # Pick a block to flip
        if random.random() < noise:
            # Random flip
            i = random.randint(0, num_k - 1)
        else:
            # Pick block that maximizes improvement
            delta = np.zeros(num_k, dtype=np.int32)

            # For included blocks: removing it may increase uncovered
            inc_mask = included
            if np.any(inc_mask):
                inc_block_tids = block_t_ids[inc_mask]
                delta[inc_mask] = np.sum(cover_count[inc_block_tids] == 1, axis=1)

            # For not included blocks: adding it may decrease uncovered
            not_inc_mask = ~included
            if np.any(not_inc_mask):
                not_inc_block_tids = block_t_ids[not_inc_mask]
                delta[not_inc_mask] = -np.sum(cover_count[not_inc_block_tids] == 0, axis=1)

            best_i = np.argmin(delta)
            i = int(best_i)

        # Flip block i
        if included[i]:
            cover_count[block_t_ids[i]] -= 1
            included[i] = False
        else:
            cover_count[block_t_ids[i]] += 1
            included[i] = True

        num_uncovered = np.sum(cover_count == 0)

        # Track best
        if num_uncovered < best_num_uncovered:
            best_num_uncovered = num_uncovered
            best_included = included.copy()
            no_improve_steps = 0
            if best_num_uncovered == 0:
                # Try to reduce block count
                for _ in range(50):
                    if time.time() > deadline - 10.0:
                        break
                    # Remove redundant
                    inc_list = np.where(included)[0]
                    found_removable = False
                    for _ in range(20):
                        if len(inc_list) == 0:
                            break
                        idx = inc_list[random.randint(0, len(inc_list)-1)]
                        tids = block_t_ids[idx]
                        if np.all(cover_count[tids] > 1):
                            cover_count[tids] -= 1
                            included[idx] = False
                            inc_list = np.where(included)[0]
                            found_removable = True
                            best_block_count = np.sum(included)
                            best_included = included.copy()
                    if not found_removable:
                        break
                break
        else:
            no_improve_steps += 1

        # If stuck, try 2-swap moves
        if no_improve_steps > max_no_improve:
            no_improve_steps = 0
            # Try 2-swap: remove one included, add one not included
            inc_list = np.where(included)[0]
            not_inc_list = np.where(~included)[0]

            if len(inc_list) > 0 and len(not_inc_list) > 0:
                # Pick a random included block to potentially remove
                A = int(inc_list[random.randint(0, len(inc_list)-1)])
                A_tids = block_t_ids[A]
                critical = A_tids[cover_count[A_tids] == 1]

                if len(critical) == 0:
                    # Can remove freely
                    cover_count[A_tids] -= 1
                    included[A] = False
                    best_block_count = np.sum(included)
                    best_included = included.copy()
                else:
                    # Need to find a block that covers all critical t-subsets
                    # Find candidates that cover the first critical t-subset
                    first_critical = int(critical[0])
                    candidates = [int(x) for x in t_to_blocks[first_critical]]
                    candidates = [c for c in candidates if not included[c] and c != A]

                    for B in candidates[:10]:
                        B_tids = block_t_ids[B]
                        # Check if B covers all critical t-subsets
                        covers_all = True
                        for ct in critical:
                            if ct not in B_tids:
                                covers_all = False
                                break
                        if covers_all:
                            # Do the swap
                            cover_count[A_tids] -= 1
                            cover_count[B_tids] += 1
                            included[A] = False
                            included[B] = True
                            num_uncovered = np.sum(cover_count == 0)
                            if num_uncovered <= best_num_uncovered:
                                if num_uncovered < best_num_uncovered:
                                    best_num_uncovered = num_uncovered
                                best_included = included.copy()
                            else:
                                # Undo
                                included[A] = True
                                included[B] = False
                                cover_count[A_tids] += 1
                                cover_count[B_tids] -= 1
                            break

    # Build solution from best_included
    best_solution = []
    for i in range(num_k):
        if best_included[i]:
            best_solution.append(list(k_subs[i]))

    # 6. Final redundancy removal
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        best_solution, changed = remove_redundant(best_solution)
        if not changed:
            break

    # Use the better of greedy and walksat solutions
    if len(best_solution) < len(solution):
        solution = best_solution

    return solution
