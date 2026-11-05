"""Greedy covering design C(v,k,t) with randomized restarts, fast numpy tracking,
and multi-pass local search (removal + 1-exchange + 2-for-1)."""
import itertools
import time
import numpy as np
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Enumerate all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # Enumerate all k-subsets (candidate blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_tsubs = []
    for b in all_blocks:
        subs = [tsub_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_tsubs.append(subs)

    # Precompute subset_to_blocks: for each t-subset index, which block indices contain it
    subset_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, subs in enumerate(block_tsubs):
        for s in subs:
            subset_to_blocks[s].append(b_idx)

    # Precompute a frozenset for each block's t-subsets for fast membership
    block_tsub_sets = [frozenset(subs) for subs in block_tsubs]

    def greedy_cover(rng):
        """Run one greedy pass with randomized tie-breaking."""
        uncovered = np.ones(n_tsubs, dtype=np.int8)
        total_uncovered = n_tsubs
        unc_per_block = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        blocks = []

        while total_uncovered > 0:
            if time.time() > deadline:
                break

            # Find max gain
            max_gain = int(unc_per_block.max())
            if max_gain == 0:
                break

            # Find all blocks with max gain, pick randomly
            candidates = np.where(unc_per_block == max_gain)[0]
            best_b_idx = int(rng.choice(candidates))

            blocks.append(best_b_idx)

            # Update
            for s in block_tsubs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = 0
                    total_uncovered -= 1
                    for b_idx in subset_to_blocks[s]:
                        unc_per_block[b_idx] -= 1

        return blocks

    def local_search(block_indices):
        """Perform local search: removal, 1-exchange, 2-for-1."""
        if not block_indices:
            return block_indices

        # Build cover_count
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            for s in block_tsubs[b_idx]:
                cover_count[s] += 1

        # Removal phase
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(block_indices)):
                b_idx = block_indices[i]
                redundant = True
                for s in block_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_tsubs[b_idx]:
                        cover_count[s] -= 1
                    block_indices.pop(i)
                    improved = True
                    break

        # 1-exchange phase: multiple passes
        for _ in range(5):
            if time.time() > deadline:
                break
            any_swap = False
            for i in range(len(block_indices)):
                if time.time() > deadline:
                    break
                b_idx = block_indices[i]
                # Critical subsets: those covered only by this block
                critical = [s for s in block_tsubs[b_idx] if cover_count[s] == 1]
                if not critical:
                    continue

                # Try to find a replacement block that covers all critical subsets
                best_swap = -1
                best_gain = 0
                # Candidate blocks: intersection of subset_to_blocks for all critical
                # For efficiency, use the critical subset with fewest candidate blocks
                crit_sorted = sorted(critical, key=lambda s: len(subset_to_blocks[s]))
                candidate_set = set(subset_to_blocks[crit_sorted[0]])
                for s in crit_sorted[1:]:
                    candidate_set &= set(subset_to_blocks[s])
                    if not candidate_set:
                        break

                for cand_idx in candidate_set:
                    if cand_idx == b_idx:
                        continue
                    # Check if cand covers all critical subsets
                    cand_set = block_tsub_sets[cand_idx]
                    if all(cs in cand_set for cs in critical):
                        # Gain: number of subsets in cand not in current block
                        cur_set = block_tsub_sets[b_idx]
                        gain = sum(1 for cs in block_tsubs[cand_idx] if cs not in cur_set)
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = cand_idx

                if best_swap != -1:
                    old_subs = block_tsub_sets[b_idx]
                    new_subs = block_tsub_sets[best_swap]
                    for s in old_subs:
                        cover_count[s] -= 1
                    for s in new_subs:
                        cover_count[s] += 1
                    block_indices[i] = best_swap
                    any_swap = True

            if not any_swap:
                break

        # 2-for-1 phase: try removing two blocks and adding one that covers their critical union
        for _ in range(3):
            if time.time() > deadline:
                break
            any_swap = False
            n_bi = len(block_indices)
            for i in range(n_bi):
                if time.time() > deadline:
                    break
                b1 = block_indices[i]
                crit1 = [s for s in block_tsubs[b1] if cover_count[s] == 1]
                for j in range(i + 1, n_bi):
                    if time.time() > deadline:
                        break
                    b2 = block_indices[j]
                    crit2 = [s for s in block_tsubs[b2] if cover_count[s] == 1]
                    # Combined critical: subsets that would become uncovered if we remove both
                    # A subset is critical for the pair if it's covered by exactly one of b1,b2
                    # and that one is being removed
                    combined_critical = []
                    for s in block_tsubs[b1]:
                        if cover_count[s] == 1:
                            combined_critical.append(s)
                        elif cover_count[s] == 2 and s in block_tsub_sets[b2]:
                            combined_critical.append(s)
                    for s in block_tsubs[b2]:
                        if cover_count[s] == 1:
                            if s not in block_tsub_sets[b1]:
                                combined_critical.append(s)
                        elif cover_count[s] == 2 and s in block_tsub_sets[b1]:
                            if s not in combined_critical:
                                combined_critical.append(s)

                    if not combined_critical:
                        continue

                    # Find a block that covers all combined_critical
                    crit_sorted = sorted(combined_critical, key=lambda s: len(subset_to_blocks[s]))
                    candidate_set = set(subset_to_blocks[crit_sorted[0]])
                    for s in crit_sorted[1:]:
                        candidate_set &= set(subset_to_blocks[s])
                        if not candidate_set:
                            break

                    best_swap = -1
                    best_gain = 0
                    for cand_idx in candidate_set:
                        if cand_idx == b1 or cand_idx == b2:
                            continue
                        cand_set = block_tsub_sets[cand_idx]
                        if all(cs in cand_set for cs in combined_critical):
                            # Gain: coverage not already provided by remaining blocks
                            # Simplified: just count new coverage
                            gain = 0
                            for cs in block_tsubs[cand_idx]:
                                if cover_count[cs] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_swap = cand_idx

                    if best_swap != -1:
                        # Remove b1 and b2, add best_swap
                        for s in block_tsub_sets[b1]:
                            cover_count[s] -= 1
                        for s in block_tsub_sets[b2]:
                            cover_count[s] -= 1
                        for s in block_tsub_sets[best_swap]:
                            cover_count[s] += 1
                        block_indices[j] = best_swap
                        block_indices.pop(i)
                        any_swap = True
                        break
                if any_swap:
                    break

            if not any_swap:
                break

        # Final removal pass
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(block_indices)):
                b_idx = block_indices[i]
                redundant = True
                for s in block_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_tsubs[b_idx]:
                        cover_count[s] -= 1
                    block_indices.pop(i)
                    improved = True
                    break

        return block_indices

    # Run multiple restarts
    best_solution = None
    best_len = float('inf')

    # Budget: use ~70% of time for greedy+local, keep 30% margin
    n_restarts = 8
    per_restart_budget = (deadline - start_time) / n_restarts

    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break

        rng = random.Random(42 + r)
        # Greedy
        block_indices = greedy_cover(rng)
        if not block_indices:
            continue

        # Local search
        block_indices = local_search(block_indices)

        if len(block_indices) < best_len:
            best_len = len(block_indices)
            best_solution = block_indices[:]

        # If we already beat the known best (151), we can be more aggressive
        # but for now, continue with restarts

    if best_solution is None:
        # Fallback: simple greedy without local search
        rng = random.Random(0)
        best_solution = greedy_cover(rng)

    result = [list(all_blocks[i]) for i in best_solution]
    return result
