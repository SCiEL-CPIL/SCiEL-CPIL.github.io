import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start_time = time.time()
    limit = 115.0
    deadline = start_time + limit

    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}

    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    # Map each block to the indices of the t-subsets it covers
    block_covers = []
    sub_to_blocks = [[] for _ in range(n_t)]

    for i, block in enumerate(blocks):
        covers = []
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            covers.append(idx)
            sub_to_blocks[idx].append(i)
        block_covers.append(np.array(covers, dtype=np.int32))

    if not _deadline_ok(deadline):
        return []

    # 2. Weighted Greedy Initialization
    # Weight each t-subset by 1/(number of blocks covering it) initially
    # This makes rare subsets more valuable
    sub_weights = np.array([1.0 / max(1, len(sub_to_blocks[i])) for i in range(n_t)], dtype=np.float64)

    uncovered_mask = np.ones(n_t, dtype=bool)
    # gain[i] = sum of weights of uncovered subsets covered by block i
    gains = np.zeros(n_blocks, dtype=np.float64)
    for i in range(n_blocks):
        gains[i] = np.sum(sub_weights[block_covers[i]])

    selected = []
    selected_set = set()
    rng = random.Random(42)

    while np.any(uncovered_mask):
        if not _deadline_ok(deadline):
            break

        max_gain = np.max(gains)
        if max_gain == 0:
            break

        # Random tie-breaking among top gains to diversify
        top_threshold = max_gain * 0.95
        candidates = np.where(gains >= top_threshold)[0]
        candidates = candidates[candidates not in selected_set]
        if len(candidates) == 0:
            candidates = np.where(gains == max_gain)[0]
            candidates = candidates[candidates not in selected_set]
        if len(candidates) == 0:
            break

        best_block = int(rng.choice(candidates))
        selected.append(best_block)
        selected_set.add(best_block)

        # Update coverage and gains
        covers_b = block_covers[best_block]
        for s_idx in covers_b:
            if uncovered_mask[s_idx]:
                uncovered_mask[s_idx] = False
                w = sub_weights[s_idx]
                for b_prime in sub_to_blocks[s_idx]:
                    gains[b_prime] -= w

        gains[best_block] = -1

    if not selected:
        return []

    # 3. Build counts and prune
    def rebuild_counts(block_list):
        c = np.zeros(n_t, dtype=np.int32)
        for b in block_list:
            c[block_covers[b]] += 1
        return c

    def prune_redundant(selected_list, counts, deadline):
        changed = True
        while changed and _deadline_ok(deadline):
            changed = False
            for i in range(len(selected_list) - 1, -1, -1):
                b = selected_list[i]
                covers_b = block_covers[b]
                if np.all(counts[covers_b] > 1):
                    selected_list.pop(i)
                    counts[covers_b] -= 1
                    changed = True
        return changed

    selected_list = list(selected)
    counts = rebuild_counts(selected_list)
    prune_redundant(selected_list, counts, deadline)

    best_size = len(selected_list)
    best_solution = selected_list.copy()
    best_counts = counts.copy()

    # 4. Local Search: Add-Remove with weighted selection
    current_set = set(selected_list)
    all_block_indices = list(range(n_blocks))

    # Precompute a "centrality" score for each block based on how many
    # weakly-covered subsets it touches. Update periodically.
    def compute_block_priority(counts, current_set):
        # Priority = number of subsets covered by block that have count == 1
        # (i.e., subsets that would become uncovered if we removed the only other block)
        # Actually, for ADD, we want blocks that cover many currently-uncovered-or-weakly-covered subsets.
        # Let's use: for each block, count how many of its subsets have count <= 2
        # This favors blocks that add coverage where it's thin.
        priority = np.zeros(n_blocks, dtype=np.int32)
        for b in range(n_blocks):
            if b in current_set:
                priority[b] = -1
                continue
            covers_b = block_covers[b]
            priority[b] = np.sum(counts[covers_b] <= 2)
        return priority

    # Main local search loop
    iteration = 0
    while _deadline_ok(deadline):
        iteration += 1

        # Periodically recompute priorities
        if iteration % 50 == 0:
            priority = compute_block_priority(counts, current_set)
            # Sort blocks by priority for weighted sampling
            valid_blocks = [b for b in range(n_blocks) if b not in current_set]
            if not valid_blocks:
                break
            prios = [max(0, priority[b]) + 1 for b in valid_blocks]
            total_prio = sum(prios)
        else:
            valid_blocks = [b for b in range(n_blocks) if b not in current_set]
            if not valid_blocks:
                break
            prios = None

        # Pick a block to add
        if prios is not None:
            r = rng.uniform(0, total_prio)
            cumulative = 0
            b_add = valid_blocks[-1]
            for b, p in zip(valid_blocks, prios):
                cumulative += p
                if r <= cumulative:
                    b_add = b
                    break
        else:
            b_add = rng.choice(valid_blocks)

        if b_add in current_set:
            continue

        # Create temp copy
        temp_list = selected_list.copy()
        temp_set = current_set.copy()
        temp_list.append(b_add)
        temp_set.add(b_add)

        temp_counts = counts.copy()
        temp_counts[block_covers[b_add]] += 1

        # Prune redundant blocks
        prune_redundant(temp_list, temp_counts, deadline)

        current_size = len(temp_list)

        if current_size < best_size:
            best_size = current_size
            best_solution = temp_list.copy()
            best_counts = temp_counts.copy()
            selected_list = temp_list
            current_set = temp_set
            counts = temp_counts
        elif current_size < len(selected_list):
            selected_list = temp_list
            current_set = temp_set
            counts = temp_counts

        # Drift control
        if len(selected_list) > best_size + 8:
            selected_list = best_solution.copy()
            current_set = set(best_solution)
            counts = best_counts.copy()

    # 5. Final 2-for-1 exchange pass
    # Try to remove 2 blocks and add 1 block that covers all subsets they covered
    if _deadline_ok(deadline):
        sol = best_solution.copy()
        cnt = best_counts.copy()
        improved = True
        while improved and _deadline_ok(deadline):
            improved = False
            # First try single removals
            for i in range(len(sol) - 1, -1, -1):
                b = sol[i]
                covers_b = block_covers[b]
                if np.all(cnt[covers_b] > 1):
                    sol.pop(i)
                    cnt[covers_b] -= 1
                    improved = True
            # Then try 2-for-1: find two blocks whose union of uniquely-covered subsets
            # can be covered by a single other block
            if len(sol) >= 2:
                sol_set = set(sol)
                # For each pair, check if there's a block covering all their "critical" subsets
                # Critical subsets for block b = subsets where cnt[s] == 1 and s in covers_b
                # Actually, for 2-for-1: remove b1, b2. The subsets that become uncovered are
                # those where cnt[s] == 1 and s in covers_b1 OR cnt[s] == 1 and s in covers_b2
                # But if a subset is in both and cnt[s]==2, removing both makes it 0.
                # So: subsets that become uncovered = {s : s in covers_b1 or s in covers_b2} and cnt[s] <= 2
                # We need a block b3 (not b1, b2) that covers ALL of these.
                # This is expensive to check for all pairs. Let's limit to random samples.
                n_pairs_to_check = min(200, len(sol) * (len(sol) - 1) // 2)
                rng2 = random.Random(iteration + 7)
                pairs_checked = set()
                attempts = 0
                while attempts < n_pairs_to_check and _deadline_ok(deadline):
                    i1 = rng2.randint(0, len(sol) - 1)
                    i2 = rng2.randint(0, len(sol) - 1)
                    if i1 == i2:
                        continue
                    key = (min(i1, i2), max(i1, i2))
                    if key in pairs_checked:
                        continue
                    pairs_checked.add(key)
                    attempts += 1

                    b1 = sol[i1]
                    b2 = sol[i2]
                    covers_b1 = block_covers[b1]
                    covers_b2 = block_covers[b2]

                    # Subsets that would be uncovered if we remove both
                    # Union of covers_b1 and covers_b2 where cnt[s] <= 2
                    union_covers = np.union1d(covers_b1, covers_b2)
                    critical = union_covers[cnt[union_covers] <= 2]
                    if len(critical) == 0:
                        # Both are redundant together, just remove one
                        continue
                    if len(critical) > 30:
                        # Too many to likely cover with one block
                        continue

                    # Find a block that covers all critical subsets
                    # Check blocks not in sol
                    found = -1
                    for b3 in range(n_blocks):
                        if b3 in sol_set:
                            continue
                        covers_b3 = block_covers[b3]
                        # Check if covers_b3 contains all of critical
                        # Use set intersection for speed
                        if len(critical) <= len(covers_b3):
                            # Check if critical is subset of covers_b3
                            # Convert to set for O(1) lookup
                            covers_b3_set = set(covers_b3.tolist())
                            if all(s in covers_b3_set for s in critical):
                                found = b3
                                break

                    if found != -1:
                        # Remove b1, b2 and add found
                        # Remove higher index first
                        if i1 > i2:
                            sol.pop(i1)
                            sol.pop(i2)
                        else:
                            sol.pop(i2)
                            sol.pop(i1)
                        cnt[block_covers[b1]] -= 1
                        cnt[block_covers[b2]] -= 1
                        sol.append(found)
                        cnt[block_covers[found]] += 1
                        sol_set = set(sol)
                        improved = True
                        break  # Restart the loop after a change

        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol.copy()
            best_counts = cnt.copy()

    # 6. Final prune
    final_counts = rebuild_counts(best_solution)
    prune_redundant(best_solution, final_counts, deadline)

    result = [list(blocks[i]) for i in best_solution]
    return result
