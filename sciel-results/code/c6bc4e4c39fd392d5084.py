import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Precomputation
    all_points = range(v)
    all_t_subsets = list(itertools.combinations(all_points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    n_t_subsets = len(all_t_subsets)

    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(all_blocks)

    # For each block, compute which t-subsets it covers
    block_to_tsubsets = []
    for block in all_blocks:
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(t_subset_to_idx[ts])
        block_to_tsubsets.append(covered)

    # For each t-subset, which blocks cover it
    tsubset_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, covered in enumerate(block_to_tsubsets):
        for ts_idx in covered:
            tsubset_to_blocks[ts_idx].append(b_idx)

    # Greedy initialization to get a valid covering
    def greedy_init(seed):
        rng = random.Random(seed)
        counts = [0] * n_t_subsets
        selected = set()
        blocks_list = []

        while True:
            uncovered = [i for i in range(n_t_subsets) if counts[i] == 0]
            if not uncovered:
                break

            best_block = -1
            best_gain = -1

            # Sample candidates for speed
            candidates = list(selected)
            for b in range(n_blocks):
                if b not in selected:
                    candidates.append(b)
            if len(candidates) > 2000:
                candidates = rng.sample(candidates, 2000)

            for b in candidates:
                gain = 0
                for ts in block_to_tsubsets[b]:
                    if counts[ts] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b

            if best_block == -1 or best_gain <= 0:
                break

            selected.add(best_block)
            blocks_list.append(best_block)
            for ts in block_to_tsubsets[best_block]:
                counts[ts] += 1

        return blocks_list, counts

    # Local search via swaps
    def local_search(initial_blocks, initial_counts, seed, time_budget):
        rng = random.Random(seed)
        current_blocks = initial_blocks[:]
        current_counts = initial_counts[:]
        current_size = len(current_blocks)

        best_blocks = current_blocks[:]
        best_size = current_size

        # Temperature for simulated annealing
        temperature = 1.0
        cooling = 0.9995

        no_improve = 0
        max_no_improve = 50

        while time.time() < time_budget:
            if no_improve > max_no_improve:
                # Perturbation: random restart of part of the solution
                if current_size > 0:
                    # Remove a random block and re-add greedily
                    remove_idx = rng.randint(0, len(current_blocks) - 1)
                    removed_block = current_blocks.pop(remove_idx)
                    for ts in block_to_tsubsets[removed_block]:
                        current_counts[ts] -= 1

                    # Check if still valid
                    invalid = [i for i in range(n_t_subsets) if current_counts[i] == 0]
                    if invalid:
                        # Greedily fix
                        for ts in invalid:
                            candidates = [b for b in tsubset_to_blocks[ts] if b not in current_blocks]
                            if candidates:
                                best_b = rng.choice(candidates)
                                current_blocks.append(best_b)
                                for ts2 in block_to_tsubsets[best_b]:
                                    current_counts[ts2] += 1
                    no_improve = 0
                    continue

            # Try a swap: remove one block, add another
            if len(current_blocks) == 0:
                break

            remove_idx = rng.randint(0, len(current_blocks) - 1)
            remove_block = current_blocks[remove_idx]

            # Compute which t-subsets would be uncovered if we remove this block
            uncovered_by_removal = []
            for ts in block_to_tsubsets[remove_block]:
                if current_counts[ts] == 1:
                    uncovered_by_removal.append(ts)

            if not uncovered_by_removal:
                # Block is redundant, just remove it
                for ts in block_to_tsubsets[remove_block]:
                    current_counts[ts] -= 1
                current_blocks.pop(remove_idx)
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_blocks = current_blocks[:]
                no_improve = 0
                continue

            # Find blocks that cover all uncovered_by_removal
            # Start with blocks covering the rarest uncovered subset
            uncovered_by_removal.sort(key=lambda x: len(tsubset_to_blocks[x]))
            candidate_blocks = set(tsubset_to_blocks[uncovered_by_removal[0]])
            for ts in uncovered_by_removal[1:]:
                if not candidate_blocks:
                    break
                candidate_blocks &= set(tsubset_to_blocks[ts])
            candidate_blocks.discard(remove_block)

            if not candidate_blocks:
                no_improve += 1
                continue

            # Choose a candidate block to add
            candidate_list = list(candidate_blocks)
            if len(candidate_list) > 100:
                candidate_list = rng.sample(candidate_list, 100)

            # Score candidates by how many new subsets they cover
            best_add_block = None
            best_score = -1
            for add_block in candidate_list:
                # Count how many subsets this block covers that are currently uncovered by removal
                score = 0
                for ts in block_to_tsubsets[add_block]:
                    if ts in set(uncovered_by_removal):
                        score += 1
                # Bonus for covering additional subsets
                if score > best_score:
                    best_score = score
                    best_add_block = add_block

            if best_add_block is None:
                no_improve += 1
                continue

            # Perform the swap
            # Remove
            for ts in block_to_tsubsets[remove_block]:
                current_counts[ts] -= 1
            current_blocks.pop(remove_idx)

            # Add
            current_blocks.append(best_add_block)
            for ts in block_to_tsubsets[best_add_block]:
                current_counts[ts] += 1

            # Verify validity
            valid = True
            for i in range(n_t_subsets):
                if current_counts[i] == 0:
                    valid = False
                    break

            if not valid:
                # Rollback
                for ts in block_to_tsubsets[best_add_block]:
                    current_counts[ts] -= 1
                current_blocks.pop()
                current_blocks.insert(remove_idx, remove_block)
                for ts in block_to_tsubsets[remove_block]:
                    current_counts[ts] += 1
                no_improve += 1
                continue

            no_improve = 0
            temperature *= cooling

        return best_blocks, best_size

    # Run greedy init
    init_blocks, init_counts = greedy_init(42)
    init_size = len(init_blocks)

    # Run local search with time budget
    time_budget = deadline - 5
    ls_blocks, ls_size = local_search(init_blocks, init_counts, 42, time_budget)

    # Final cleanup: remove redundant blocks
    final_counts = [0] * n_t_subsets
    for b in ls_blocks:
        for ts in block_to_tsubsets[b]:
            final_counts[ts] += 1

    final_blocks = ls_blocks[:]
    for i in range(len(final_blocks) - 1, -1, -1):
        b = final_blocks[i]
        redundant = True
        for ts in block_to_tsubsets[b]:
            if final_counts[ts] == 1:
                redundant = False
                break
        if redundant:
            for ts in block_to_tsubsets[b]:
                final_counts[ts] -= 1
            final_blocks.pop(i)

    return [list(all_blocks[i]) for i in final_blocks]
