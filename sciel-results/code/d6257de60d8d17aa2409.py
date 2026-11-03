"""Probabilistic covering design with greedy + simulated annealing for C(19,8,5)."""
import random
import itertools
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t):
    1. Greedy initialization targeting uncovered t-subsets
    2. Simulated annealing with block replacement to minimize uncovered count
    """
    start_time = time.time()
    timeout = 115  # Leave 5s buffer

    # Precompute all t-subsets and index them
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    def get_block_tsubsets(block):
        return [t_subset_index[ts] for ts in itertools.combinations(block, t)]

    # Greedy initialization: repeatedly pick a block that covers the most uncovered t-subsets
    def greedy_cover():
        cov_count = [0] * num_t_subsets
        blocks = []
        uncovered_set = set(range(num_t_subsets))

        while uncovered_set:
            best_block = None
            best_gain = -1

            # Sample candidate blocks to find a good one
            # Try to build blocks around uncovered subsets
            sampled = set()
            attempts = 0
            while len(sampled) < 500 and attempts < 5000:
                attempts += 1
                # Pick a random uncovered subset and build a block around it
                unc_idx = random.choice(list(uncovered_set)) if uncovered_set else 0
                seed_subset = list(all_t_subsets[unc_idx])
                block = list(seed_subset)
                random.shuffle(block)

                while len(block) < k:
                    candidates = [x for x in range(v) if x not in block]
                    random.shuffle(candidates)
                    # Pick the point that maximizes coverage of uncovered t-subsets
                    best_x = None
                    best_local_gain = -1
                    for x in candidates[:15]:
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            idx = t_subset_index.get(ts)
                            if idx is not None and idx in uncovered_set:
                                gain += 1
                        if gain > best_local_gain:
                            best_local_gain = gain
                            best_x = x
                    if best_x is not None:
                        block.append(best_x)

                block.sort()
                block_tup = tuple(block)
                if block_tup in sampled:
                    continue
                sampled.add(block_tup)

                # Compute gain of this block
                gain = 0
                for ts in itertools.combinations(block_tup, t):
                    idx = t_subset_index.get(ts)
                    if idx is not None and idx in uncovered_set:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_block = block_tup

            if best_block is None:
                break

            blocks.append(best_block)
            for ts in itertools.combinations(best_block, t):
                idx = t_subset_index.get(ts)
                if idx is not None:
                    cov_count[idx] += 1
                    if cov_count[idx] == 1:
                        uncovered_set.discard(idx)

        return blocks

    # Initialize with greedy
    blocks = greedy_cover()
    block_tsubs = [get_block_tsubsets(b) for b in blocks]

    # Build coverage counts
    cov = [0] * num_t_subsets
    for tsub_list in block_tsubs:
        for idx in tsub_list:
            cov[idx] += 1

    current_uncovered = sum(1 for c in cov if c == 0)
    best_blocks = list(blocks)
    best_uncovered = current_uncovered

    # Simulated annealing
    temperature = 50.0
    cooling = 0.998
    min_temp = 0.5

    while temperature > min_temp:
        if time.time() - start_time > timeout:
            break

        block_idx = random.randint(0, len(blocks) - 1)
        old_block = blocks[block_idx]
        old_tsubs = block_tsubs[block_idx]

        # Remove old block's coverage
        for idx in old_tsubs:
            cov[idx] -= 1

        # Generate a new block: try to cover uncovered subsets
        # Strategy: pick an uncovered t-subset (if any) and build a block around it
        uncovered_indices = [i for i, c in enumerate(cov) if c == 0]

        if uncovered_indices:
            # Target an uncovered subset
            target_idx = random.choice(uncovered_indices)
            seed = list(all_t_subsets[target_idx])
            block = list(seed)
            random.shuffle(block)
        else:
            # All covered; try to replace with a block that covers well-covered subsets less
            block = list(old_block)
            random.shuffle(block)

        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            random.shuffle(candidates)
            best_x = None
            best_local_gain = -1
            for x in candidates[:12]:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = t_subset_index.get(ts)
                    if idx is not None and cov[idx] == 0:
                        gain += 1
                if gain > best_local_gain:
                    best_local_gain = gain
                    best_x = x
            if best_x is not None:
                block.append(best_x)

        block.sort()
        new_block = tuple(block)

        # Avoid duplicates
        if new_block != old_block:
            dup_found = False
            for i, b in enumerate(blocks):
                if i != block_idx and b == new_block:
                    dup_found = True
                    break
            if dup_found:
                # Restore old block
                for idx in old_tsubs:
                    cov[idx] += 1
                continue

        new_tsubs = get_block_tsubsets(new_block)
        for idx in new_tsubs:
            cov[idx] += 1

        new_uncovered = sum(1 for c in cov if c == 0)
        delta = new_uncovered - current_uncovered

        # Metropolis acceptance
        if delta <= 0 or random.random() < pow(2.718281828, -delta / temperature):
            blocks[block_idx] = new_block
            block_tsubs[block_idx] = new_tsubs
            current_uncovered = new_uncovered

            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_blocks = list(blocks)
        else:
            # Reject: restore
            for idx in new_tsubs:
                cov[idx] -= 1
            for idx in old_tsubs:
                cov[idx] += 1

        temperature *= cooling

    # Final pass: try to replace blocks that cover few unique t-subsets
    # with blocks targeting remaining uncovered subsets
    if time.time() - start_time < timeout - 2:
        cov_final = [0] * num_t_subsets
        for tsub_list in [get_block_tsubsets(b) for b in best_blocks]:
            for idx in tsub_list:
                cov_final[idx] += 1
        uncovered_final = [i for i, c in enumerate(cov_final) if c == 0]

        if uncovered_final:
            # Try to add or replace blocks to cover more
            for _ in range(200):
                if time.time() - start_time > timeout - 1:
                    break
                target_idx = random.choice(uncovered_final)
                seed = list(all_t_subsets[target_idx])
                block = list(seed)
                random.shuffle(block)
                while len(block) < k:
                    candidates = [x for x in range(v) if x not in block]
                    random.shuffle(candidates)
                    best_x = None
                    best_local_gain = -1
                    for x in candidates[:10]:
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            idx = t_subset_index.get(ts)
                            if idx is not None and cov_final[idx] == 0:
                                gain += 1
                        if gain > best_local_gain:
                            best_local_gain = gain
                            best_x = x
                    if best_x is not None:
                        block.append(best_x)
                block.sort()
                new_block = tuple(block)

                # Check if it covers new ground
                new_tsubs = get_block_tsubsets(new_block)
                new_gain = 0
                for idx in new_tsubs:
                    if cov_final[idx] == 0:
                        new_gain += 1

                if new_gain > 0:
                    # Replace a random block or add
                    if len(best_blocks) < 400:
                        best_blocks.append(new_block)
                    else:
                        replace_idx = random.randint(0, len(best_blocks) - 1)
                        best_blocks[replace_idx] = new_block

                    # Update coverage
                    for idx in new_tsubs:
                        cov_final[idx] += 1

                    uncovered_final = [i for i, c in enumerate(cov_final) if c == 0]
                    if not uncovered_final:
                        break

    return [list(b) for b in best_blocks]
