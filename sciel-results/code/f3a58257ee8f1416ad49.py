import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    num_triples = len(all_t_subsets)

    # Precompute: for each point x, the list of triple indices containing x
    # This is used for fast gain computation
    point_triple_indices = []
    for x in range(v):
        indices = []
        for i, triple in enumerate(all_t_subsets):
            if x in triple:
                indices.append(i)
        point_triple_indices.append(indices)

    # For t=3, precompute pairs within a block for faster gain
    def block_triple_indices(block):
        """Return list of triple indices covered by this block."""
        indices = []
        for triple in itertools.combinations(block, t):
            indices.append(triple_to_idx[tuple(triple)])
        return indices

    def compute_gains(block, uncovered, candidates):
        """Compute gain for each candidate point to add to block."""
        gains = np.zeros(len(candidates), dtype=np.int32)
        if t == 3:
            # For t=3, gain(x) = sum over pairs (a,b) in block of uncovered[idx(x,a,b)]
            block_pairs = list(itertools.combinations(block, 2))
            for x_i, x in enumerate(candidates):
                gain = 0
                for a, b in block_pairs:
                    triple = tuple(sorted((a, b, x)))
                    idx = triple_to_idx[triple]
                    gain += uncovered[idx]
                gains[x_i] = gain
        else:
            # General case: gain(x) = sum over (t-1)-subsets S of block of uncovered[idx(S ∪ {x})]
            block_subsets = list(itertools.combinations(block, t - 1))
            for x_i, x in enumerate(candidates):
                gain = 0
                for subset in block_subsets:
                    triple = tuple(sorted(subset + (x,)))
                    idx = triple_to_idx[triple]
                    gain += uncovered[idx]
                gains[x_i] = gain
        return gains

    def random_greedy_cover(rng, use_weighted_seed=True):
        """Greedy cover with optional weighted seed selection."""
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks = []

        while True:
            if time.time() > deadline - 5.0:
                break
            if not np.any(uncovered):
                break

            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break

            if use_weighted_seed:
                # Weight by how many uncovered triples share each point
                # This biases toward "hot" regions
                point_weights = np.zeros(v, dtype=np.int32)
                for idx in uncovered_indices:
                    for p in all_t_subsets[idx]:
                        point_weights[p] += 1
                total_weight = np.sum(point_weights)
                if total_weight == 0:
                    seed_idx = rng.choice(uncovered_indices.tolist())
                else:
                    # Pick a point proportional to weight, then pick a random uncovered triple containing it
                    probs = point_weights / total_weight
                    chosen_point = rng.choices(all_points, weights=probs.tolist())[0]
                    # Find uncovered triples containing chosen_point
                    valid_indices = [idx for idx in uncovered_indices if chosen_point in all_t_subsets[idx]]
                    if valid_indices:
                        seed_idx = rng.choice(valid_indices)
                    else:
                        seed_idx = rng.choice(uncovered_indices.tolist())
            else:
                seed_idx = rng.choice(uncovered_indices.tolist())

            seed_triple = all_t_subsets[seed_idx]
            block = list(seed_triple)
            block_set = set(block)

            # Extend block to size k
            while len(block) < k:
                if time.time() > deadline - 5.0:
                    break

                candidates = [x for x in all_points if x not in block_set]
                if not candidates:
                    break

                gains = compute_gains(block, uncovered, candidates)
                max_gain = int(np.max(gains))
                if max_gain <= 0:
                    # All gains are zero; pick randomly
                    block.append(rng.choice(candidates))
                else:
                    # Soft-max sampling: prefer high gain but allow exploration
                    # Temperature-based: lower temp = more greedy
                    temp = max(1.0, max_gain * 0.3)
                    weights = np.exp(gains / temp)
                    total = np.sum(weights)
                    if total == 0:
                        block.append(rng.choice(candidates))
                    else:
                        probs = weights / total
                        chosen = rng.choices(range(len(candidates)), weights=probs.tolist())[0]
                        block.append(candidates[chosen])

                block_set.add(block[-1])

            # Fill remaining if needed
            if len(block) < k:
                block_set = set(block)
                while len(block) < k:
                    available = [x for x in all_points if x not in block_set]
                    if not available:
                        break
                    block.append(rng.choice(available))
                    block_set.add(block[-1])

            block.sort()
            blocks.append(block)

            # Update uncovered
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                uncovered[idx] = 0

        return blocks

    def remove_redundant_blocks(blocks):
        """Remove redundant blocks iteratively."""
        if not blocks:
            return blocks

        triple_count = np.zeros(num_triples, dtype=np.int32)
        block_indices = []

        for block in blocks:
            indices = block_triple_indices(block)
            block_indices.append(indices)
            triple_count[indices] += 1

        changed = True
        while changed:
            changed = False
            to_remove = []

            for i in range(len(blocks)):
                is_redundant = True
                for idx in block_indices[i]:
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(i)

            if to_remove:
                remove_set = set(to_remove)
                new_blocks = []
                new_block_indices = []
                for i in range(len(blocks)):
                    if i not in remove_set:
                        new_blocks.append(blocks[i])
                        new_block_indices.append(block_indices[i])

                for i in to_remove:
                    triple_count[block_indices[i]] -= 1

                blocks = new_blocks
                block_indices = new_block_indices
                changed = True

        return blocks

    def local_search_swap(blocks, max_iterations=300):
        """Try swaps to enable further deletions. Returns improved blocks."""
        if not blocks:
            return blocks

        triple_count = np.zeros(num_triples, dtype=np.int32)
        block_indices = []

        for block in blocks:
            indices = block_triple_indices(block)
            block_indices.append(indices)
            triple_count[indices] += 1

        improved = False

        for iteration in range(max_iterations):
            if time.time() > deadline - 5.0:
                break

            # Try to delete redundant blocks first
            deleted = False
            for i in range(len(blocks)):
                is_redundant = True
                for idx in block_indices[i]:
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    triple_count[block_indices[i]] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    deleted = True
                    improved = True
                    break

            if deleted:
                continue

            if len(blocks) == 0:
                break

            # Try swaps: pick a random block, try to swap one point for another
            # Score swaps by how many new redundancies they create
            found_swap = False
            best_swap_score = 0
            best_swap_info = None

            for _ in range(50):  # Try up to 50 random swaps per iteration
                if time.time() > deadline - 5.0:
                    break

                idx = random.randint(0, len(blocks) - 1)
                block = blocks[idx]
                old_x = random.choice(block)
                new_x = random.randint(0, v - 1)

                if new_x == old_x or new_x in block:
                    continue

                other_points = [p for p in block if p != old_x]
                new_block = sorted(other_points + [new_x])

                old_triples_set = set(block_triple_indices(block))
                new_triples_set = set(block_triple_indices(new_block))

                lost = old_triples_set - new_triples_set
                gained = new_triples_set - old_triples_set

                # Check validity: all lost triples must have count > 1
                valid = True
                for tri_idx in lost:
                    if triple_count[tri_idx] <= 1:
                        valid = False
                        break

                if not valid:
                    continue

                # Score: number of gained triples that would become redundant (count becomes 1 after this block is removed)
                # Actually, score by: how many triples in 'gained' have count == 1 (they become new single-coverage)
                # And how many in 'lost' have count == 2 (they become single-coverage, making that block redundant)
                score = 0
                for tri_idx in lost:
                    if triple_count[tri_idx] == 2:
                        score += 1  # This triple will have count 1 after removing old coverage, potentially making another block redundant
                for tri_idx in gained:
                    if triple_count[tri_idx] == 0:
                        score -= 1  # Bad: creating a new uncovered triple is not allowed (validity check above prevents this)
                    elif triple_count[tri_idx] == 1:
                        score -= 0.5  # Slightly bad: making a triple have count 2 doesn't help

                if score > best_swap_score:
                    best_swap_score = score
                    best_swap_info = (idx, old_x, new_x, lost, gained, old_triples_set, new_triples_set, new_block)

            if best_swap_info is not None and best_swap_score > 0:
                idx, old_x, new_x, lost, gained, old_triples_set, new_triples_set, new_block = best_swap_info
                triple_count[list(old_triples_set)] -= 1
                triple_count[list(new_triples_set)] += 1
                blocks[idx] = new_block
                block_indices[idx] = block_triple_indices(new_block)
                found_swap = True
                improved = True

            if not found_swap:
                break

        return blocks

    def two_opt_swap(blocks):
        """Try 2-opt style: swap a point between two blocks."""
        if len(blocks) < 2:
            return blocks

        triple_count = np.zeros(num_triples, dtype=np.int32)
        block_indices = []

        for block in blocks:
            indices = block_triple_indices(block)
            block_indices.append(indices)
            triple_count[indices] += 1

        improved = False
        max_iterations = 200

        for iteration in range(max_iterations):
            if time.time() > deadline - 5.0:
                break

            # Try to delete redundant blocks first
            deleted = False
            for i in range(len(blocks)):
                is_redundant = True
                for idx in block_indices[i]:
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    triple_count[block_indices[i]] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    deleted = True
                    improved = True
                    break

            if deleted:
                continue

            if len(blocks) < 2:
                break

            # Try 2-opt: pick two blocks, swap a point from each
            found_swap = False
            best_score = 0
            best_info = None

            for _ in range(30):
                if time.time() > deadline - 5.0:
                    break

                i, j = random.sample(range(len(blocks)), 2)
                x_i = random.choice(blocks[i])
                x_j = random.choice(blocks[j])

                if x_i == x_j:
                    continue

                # New blocks after swap
                block_i_new = sorted([p for p in blocks[i] if p != x_i] + [x_j])
                block_j_new = sorted([p for p in blocks[j] if p != x_j] + [x_i])

                old_i_set = set(block_triple_indices(blocks[i]))
                old_j_set = set(block_triple_indices(blocks[j]))
                new_i_set = set(block_triple_indices(block_i_new))
                new_j_set = set(block_triple_indices(block_j_new))

                old_all = old_i_set | old_j_set
                new_all = new_i_set | new_j_set

                lost = old_all - new_all
                gained = new_all - old_all

                valid = True
                for tri_idx in lost:
                    if triple_count[tri_idx] <= 1:
                        valid = False
                        break

                if not valid:
                    continue

                # Score: count how many lost triples have count == 2 (potential redundancy creation)
                score = 0
                for tri_idx in lost:
                    if triple_count[tri_idx] == 2:
                        score += 1

                if score > best_score:
                    best_score = score
                    best_info = (i, j, x_i, x_j, old_i_set, old_j_set, new_i_set, new_j_set, block_i_new, block_j_new)

            if best_info is not None and best_score > 0:
                i, j, x_i, x_j, old_i_set, old_j_set, new_i_set, new_j_set, block_i_new, block_j_new = best_info
                triple_count[list(old_i_set)] -= 1
                triple_count[list(old_j_set)] -= 1
                triple_count[list(new_i_set)] += 1
                triple_count[list(new_j_set)] += 1
                blocks[i] = block_i_new
                blocks[j] = block_j_new
                block_indices[i] = block_triple_indices(block_i_new)
                block_indices[j] = block_triple_indices(block_j_new)
                found_swap = True
                improved = True

            if not found_swap:
                break

        return blocks

    # Main loop
    best_blocks = None
    best_len = float('inf')

    # Phase 1: Greedy restarts (55% of time)
    greedy_deadline = start_time + (deadline - start_time) * 0.55
    seed = 42

    while time.time() < greedy_deadline:
        rng = random.Random(seed)
        seed += 1
        use_weighted = (seed % 3 != 0)  # Vary strategy
        blocks = random_greedy_cover(rng, use_weighted_seed=use_weighted)
        if not blocks:
            continue

        blocks = remove_redundant_blocks(blocks)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks]

    # Phase 2: Local search refinement (40% of time)
    if best_blocks is not None and time.time() < deadline - 5.0:
        refine_deadline = deadline - 5.0
        current_blocks = [b[:] for b in best_blocks]
        current_len = len(current_blocks)

        for attempt in range(30):
            if time.time() > refine_deadline:
                break

            trial_blocks = [b[:] for b in current_blocks]

            # Alternate between different local search strategies
            if attempt % 3 == 0:
                trial_blocks = local_search_swap(trial_blocks)
            elif attempt % 3 == 1:
                trial_blocks = two_opt_swap(trial_blocks)
            else:
                trial_blocks = local_search_swap(trial_blocks, max_iterations=200)

            trial_blocks = remove_redundant_blocks(trial_blocks)

            if len(trial_blocks) < current_len:
                current_blocks = trial_blocks
                current_len = len(trial_blocks)
                if current_len < best_len:
                    best_len = current_len
                    best_blocks = [b[:] for b in current_blocks]
            else:
                # Perturb: try a fresh greedy + merge
                if time.time() > refine_deadline:
                    break
                rng = random.Random(seed)
                seed += 1
                greedy_blocks = random_greedy_cover(rng, use_weighted_seed=True)
                if greedy_blocks:
                    greedy_blocks = remove_redundant_blocks(greedy_blocks)
                    if len(greedy_blocks) < current_len:
                        current_blocks = greedy_blocks
                        current_len = len(greedy_blocks)
                        if current_len < best_len:
                            best_len = current_len
                            best_blocks = [b[:] for b in current_blocks]

    # Final polish: one more pass of local search on best solution
    if best_blocks is not None and time.time() < deadline - 2.0:
        final_blocks = [b[:] for b in best_blocks]
        final_blocks = local_search_swap(final_blocks, max_iterations=100)
        final_blocks = remove_redundant_blocks(final_blocks)
        if len(final_blocks) < best_len:
            best_blocks = final_blocks
            best_len = len(final_blocks)

    # Fallback: deterministic greedy if nothing found
    if best_blocks is None:
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks = []
        while np.any(uncovered) and time.time() < deadline - 2.0:
            uncovered_indices = np.where(uncovered)[0]
            seed_idx = uncovered_indices[0]
            seed_triple = all_t_subsets[seed_idx]
            block = list(seed_triple)
            block_set = set(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in all_points:
                    if x in block_set:
                        continue
                    gain = 0
                    for subset in itertools.combinations(block, t - 1):
                        triple = tuple(sorted(subset + (x,)))
                        idx = triple_to_idx[triple]
                        gain += uncovered[idx]
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
                block_set.add(best_x)
            if len(block) < k:
                available = [x for x in all_points if x not in block_set]
                while len(block) < k and available:
                    block.append(available.pop())
            block.sort()
            blocks.append(block)
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                uncovered[idx] = 0
        best_blocks = blocks
        best_len = len(best_blocks)

    return best_blocks
