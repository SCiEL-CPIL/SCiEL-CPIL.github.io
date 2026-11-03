import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 110

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute for each block (k-subset) the list of t-subset indices it covers
    # We'll compute this on the fly but cache for repeated blocks

    def block_to_tsub_indices(block):
        """Return list of t-subset indices covered by this block"""
        idxs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                idxs.append(idx)
        return idxs

    def compute_gains(block, uncovered_set):
        """For each candidate point not in block, compute how many uncovered t-subsets it would cover"""
        block_set = set(block)
        gains = {}
        # For each (t-1)-subset of block, for each point x not in block,
        # check if R U {x} is uncovered
        for r in itertools.combinations(block, t - 1):
            r_set = set(r)
            for x in range(v):
                if x in block_set:
                    continue
                ts = tuple(sorted(r_set | {x}))
                if len(ts) == t:
                    idx = tsub_to_idx.get(ts)
                    if idx is not None and idx in uncovered_set:
                        gains[x] = gains.get(x, 0) + 1
        return gains

    def greedy_construct(uncovered, rng, deadline):
        """Greedy set cover with randomization"""
        blocks = []
        order = list(range(n_tsubs))
        rng.shuffle(order)

        while uncovered and time.time() < deadline - 10:
            # Find first uncovered t-subset in random order
            seed_idx = None
            for idx in order:
                if idx in uncovered:
                    seed_idx = idx
                    break
            if seed_idx is None:
                break

            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)

            while len(block) < k:
                if time.time() > deadline - 10:
                    break
                gains = compute_gains(block, uncovered)
                if not gains:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(rng.choice(remaining))
                        block_set.add(block[-1])
                    else:
                        break
                    continue
                max_gain = max(gains.values())
                candidates = [x for x, g in gains.items() if g == max_gain]
                best_point = rng.choice(candidates)
                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    uncovered.discard(idx)

        return blocks

    def remove_redundant(blocks, deadline):
        """Remove blocks that don't cover any unique t-subset"""
        if not blocks:
            return blocks

        block_tsub_indices = [block_to_tsub_indices(b) for b in blocks]
        cover_count = [0] * n_tsubs
        for idxs in block_tsub_indices:
            for idx in idxs:
                cover_count[idx] += 1

        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                is_redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsub_indices.pop(i)
                    changed = True
                    break

        return blocks

    def local_search(blocks, deadline):
        """Local search: try to remove blocks or swap points to reduce count"""
        if not blocks:
            return blocks

        block_tsub_indices = [block_to_tsub_indices(b) for b in blocks]
        cover_count = [0] * n_tsubs
        for idxs in block_tsub_indices:
            for idx in idxs:
                cover_count[idx] += 1

        improved = True
        max_iterations = 50
        iter_count = 0

        while improved and time.time() < deadline - 8 and iter_count < max_iterations:
            improved = False
            iter_count += 1

            # Try to remove a redundant block
            for i in range(len(blocks)):
                if time.time() > deadline - 8:
                    break
                criticals = [idx for idx in block_tsub_indices[i] if cover_count[idx] == 1]
                if not criticals:
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsub_indices.pop(i)
                    improved = True
                    break

            if improved:
                continue

            # Try swap: modify a block to cover a critical t-subset of another block
            for i in range(len(blocks)):
                if time.time() > deadline - 8:
                    break
                criticals = [idx for idx in block_tsub_indices[i] if cover_count[idx] == 1]
                if not criticals:
                    continue
                crit_idx = criticals[0]
                crit_tsub = set(all_tsubs[crit_idx])

                for j in range(len(blocks)):
                    if j == i or time.time() > deadline - 8:
                        continue
                    block_j_set = set(blocks[j])
                    common = block_j_set & crit_tsub
                    if len(common) >= t - 1:
                        missing = crit_tsub - block_j_set
                        if len(missing) == 1:
                            missing_point = list(missing)[0]
                            extra_points = [p for p in blocks[j] if p not in common]
                            if extra_points:
                                for rem_p in extra_points:
                                    new_block = [p if p != rem_p else missing_point for p in blocks[j]]
                                    new_block_set = set(new_block)
                                    if len(new_block_set) != k:
                                        continue
                                    new_block.sort()

                                    old_tsubs_set = set(block_tsub_indices[j])
                                    new_tsubs = block_to_tsub_indices(new_block)
                                    new_tsubs_set = set(new_tsubs)

                                    can_swap = True
                                    for idx in old_tsubs_set:
                                        if idx not in new_tsubs_set and cover_count[idx] == 1:
                                            can_swap = False
                                            break

                                    if can_swap:
                                        for idx in old_tsubs_set:
                                            cover_count[idx] -= 1
                                        for idx in new_tsubs_set:
                                            cover_count[idx] += 1
                                        blocks[j] = new_block
                                        block_tsub_indices[j] = new_tsubs
                                        improved = True
                                        break
                    if improved:
                        break
                if improved:
                    break

        return blocks

    def simulated_annealing(blocks, deadline, rng):
        """Simulated annealing to reduce number of blocks"""
        if not blocks:
            return blocks

        # Start with current blocks
        current_blocks = [list(b) for b in blocks]
        best_blocks = [list(b) for b in blocks]
        best_count = len(current_blocks)

        block_tsub_indices = [block_to_tsub_indices(b) for b in current_blocks]
        cover_count = [0] * n_tsubs
        for idxs in block_tsub_indices:
            for idx in idxs:
                cover_count[idx] += 1

        # Count uncovered
        uncovered_count = sum(1 for c in cover_count if c == 0)

        start_temp = 1.0
        end_temp = 0.01
        cooling_rate = 0.995
        temp = start_temp

        while time.time() < deadline - 8 and temp > end_temp:
            temp *= cooling_rate

            # Generate a move
            move_type = rng.random()
            if move_type < 0.3:
                # Remove a random block
                if len(current_blocks) <= 1:
                    continue
                i = rng.randint(0, len(current_blocks) - 1)
                old_tsubs = block_tsub_indices[i]
                for idx in old_tsubs:
                    cover_count[idx] -= 1
                current_blocks.pop(i)
                block_tsub_indices.pop(i)
                new_uncovered = sum(1 for c in cover_count if c == 0)
                delta = new_uncovered - uncovered_count
                if delta > 0:
                    # Accept with probability
                    if rng.random() < (1.0 / (1.0 + delta)) * temp:
                        uncovered_count = new_uncovered
                    else:
                        # Revert
                        current_blocks.insert(i, None)
                        block_tsub_indices.insert(i, old_tsubs)
                        for idx in old_tsubs:
                            cover_count[idx] += 1
                else:
                    uncovered_count = new_uncovered
                    if len(current_blocks) < best_count:
                        best_count = len(current_blocks)
                        best_blocks = [list(b) for b in current_blocks]

            elif move_type < 0.6:
                # Add a random block
                if len(current_blocks) > 500:
                    continue
                # Pick a random t-subset and extend to a k-block
                idx = rng.randint(0, n_tsubs - 1)
                seed = list(all_tsubs[idx])
                block_set = set(seed)
                while len(block_set) < k:
                    x = rng.randint(0, v - 1)
                    block_set.add(x)
                block = sorted(block_set)
                # Check if already exists
                if block in current_blocks:
                    continue
                new_tsubs = block_to_tsub_indices(block)
                for idx2 in new_tsubs:
                    cover_count[idx2] += 1
                current_blocks.append(block)
                block_tsub_indices.append(new_tsubs)
                new_uncovered = sum(1 for c in cover_count if c == 0)
                # Adding a block never increases uncovered, so always accept
                uncovered_count = new_uncovered

            else:
                # Swap a point in a random block
                if len(current_blocks) <= 1:
                    continue
                i = rng.randint(0, len(current_blocks) - 1)
                block = current_blocks[i]
                pos = rng.randint(0, len(block) - 1)
                old_point = block[pos]
                new_point = rng.randint(0, v - 1)
                if new_point == old_point or new_point in block:
                    continue
                new_block = block[:]
                new_block[pos] = new_point
                new_block.sort()
                if new_block in current_blocks:
                    continue

                old_tsubs = block_tsub_indices[i]
                new_tsubs = block_to_tsub_indices(new_block)

                # Compute delta in uncovered
                old_set = set(old_tsubs)
                new_set = set(new_tsubs)
                delta = 0
                for idx in old_set - new_set:
                    if cover_count[idx] == 1:
                        delta += 1
                for idx in new_set - old_set:
                    if cover_count[idx] == 0:
                        delta -= 1

                if delta <= 0 or rng.random() < (1.0 / (1.0 + max(0, delta))) * temp:
                    # Apply move
                    for idx in old_set:
                        cover_count[idx] -= 1
                    for idx in new_set:
                        cover_count[idx] += 1
                    current_blocks[i] = new_block
                    block_tsub_indices[i] = new_tsubs
                    uncovered_count += delta
                    if len(current_blocks) < best_count and uncovered_count == 0:
                        best_count = len(current_blocks)
                        best_blocks = [list(b) for b in current_blocks]
                else:
                    pass  # Reject

            # Periodically try to remove redundant blocks
            if iter_count % 10 == 0:
                current_blocks = remove_redundant(current_blocks, deadline)
                block_tsub_indices = [block_to_tsub_indices(b) for b in current_blocks]
                cover_count = [0] * n_tsubs
                for idxs in block_tsub_indices:
                    for idx in idxs:
                        cover_count[idx] += 1
                uncovered_count = sum(1 for c in cover_count if c == 0)
                if len(current_blocks) < best_count and uncovered_count == 0:
                    best_count = len(current_blocks)
                    best_blocks = [list(b) for b in current_blocks]

        return best_blocks if uncovered_count == 0 else best_blocks

    best_blocks = None
    best_count = float('inf')

    # Multiple restarts with different strategies
    for restart in range(200):
        if time.time() > deadline - 15:
            break

        rng = random.Random(hash(("greedy", restart)) % (2**32))
        blocks = greedy_construct(set(range(n_tsubs)), rng, deadline)

        if not blocks:
            continue

        # Check if all covered
        all_covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                all_covered.add(ts)
        if len(all_covered) < n_tsubs:
            continue

        # Post-processing
        blocks = remove_redundant(blocks, deadline)
        blocks = local_search(blocks, deadline)
        blocks = remove_redundant(blocks, deadline)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if best_count <= 330:
                break

    # If we have a good solution, try simulated annealing to improve further
    if best_blocks is not None and time.time() < deadline - 20:
        rng = random.Random(hash(("sa", 0)) % (2**32))
        sa_blocks = simulated_annealing(best_blocks, deadline, rng)
        if sa_blocks and len(sa_blocks) < best_count:
            # Verify
            all_covered = set()
            for block in sa_blocks:
                for ts in itertools.combinations(block, t):
                    all_covered.add(ts)
            if len(all_covered) == n_tsubs:
                best_count = len(sa_blocks)
                best_blocks = [list(b) for b in sa_blocks]

    if best_blocks is None:
        # Fallback
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered and time.time() < deadline - 5:
            seed_idx = min(uncovered)
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            while len(block) < k:
                gains = compute_gains(block, uncovered)
                if not gains:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(remaining[0])
                        block_set.add(block[-1])
                    else:
                        break
                    continue
                best_point = max(gains.keys(), key=lambda x: gains[x])
                block.append(best_point)
                block_set.add(best_point)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    uncovered.discard(idx)
        best_blocks = blocks

    return best_blocks
