import itertools
import time
import random
import heapq
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(all_points, t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets
    all_ksubs = list(itertools.combinations(all_points, k))
    n_ksubs = len(all_ksubs)

    # For each k-subset, compute its t-subset indices (list of t indices it covers)
    # C(7,3) = 35 per block
    block_tsubs = []
    for ksub in all_ksubs:
        tsubs = [tsub_to_idx[ts] for ts in itertools.combinations(ksub, t)]
        block_tsubs.append(tsubs)

    # Build inverted index: for each t-subset, list of blocks containing it
    # This is critical for fast updates
    tsub_to_blocks = defaultdict(list)
    for bi, tsubs in enumerate(block_tsubs):
        for ti in tsubs:
            tsub_to_blocks[ti].append(bi)

    # Coverage count for each t-subset (how many selected blocks cover it)
    # Initially 0 for all

    best_blocks = None
    best_count = float('inf')

    def greedy_with_lagrangian(seed):
        """
        Greedy set cover with Lagrangian relaxation guidance.
        Dual variables λ_t guide block selection toward under-covered t-subsets.
        """
        rng = random.Random(seed)
        n_selected = 0

        # Coverage count: how many selected blocks cover each t-subset
        cov = [0] * n_tsubs
        # Uncovered set as bitset for quick check
        uncovered_mask = (1 << n_tsubs) - 1

        # Lagrangian dual variables
        lam = [1.0] * n_tsubs
        eta = 0.5  # step size for subgradient

        # For each block, track its marginal gain (uncovered t-subsets it covers)
        # Initialize: all 35 t-subsets are uncovered
        marginal = [len(block_tsubs[i]) for i in range(n_ksubs)]

        # Selected blocks
        selected = []
        selected_set = set()

        # We need an efficient way to find the block with max marginal gain.
        # Use a max-heap. But marginal gains change, so we use lazy deletion.
        # heap entries: (-marginal, block_idx, version)
        version = [0] * n_ksubs
        heap = []
        for i in range(n_ksubs):
            if marginal[i] > 0:
                heapq.heappush(heap, (-marginal[i], i, version[i]))

        max_iter = 500  # cap iterations
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break

            if uncovered_mask == 0:
                break

            # Find best block from heap (lazy deletion)
            best_gain = 0
            best_idx = -1
            while heap:
                neg_gain, idx, ver = heapq.heappop(heap)
                if ver != version[idx]:
                    continue
                if idx in selected_set:
                    continue
                gain = marginal[idx]
                if gain <= 0:
                    continue
                # Apply Lagrangian adjustment: prefer blocks covering under-covered t-subsets
                # Score = gain + small bonus from lambda for rare t-subsets
                # Compute a quick score using the top few t-subsets
                score = gain
                # Sample a few t-subsets from this block to get lambda bonus
                tsubs_i = block_tsubs[idx]
                # Only consider uncovered ones
                bonus = 0.0
                count = 0
                for ti in tsubs_i:
                    if (uncovered_mask >> ti) & 1:
                        bonus += lam[ti]
                        count += 1
                        if count >= 5:  # limit for speed
                            break
                if count > 0:
                    score = gain + 0.1 * (bonus / count)

                if score > best_gain:
                    best_gain = score
                    best_idx = idx
                # Don't break - we want the true max, but for speed we accept first good one
                # Actually, let's just take the first one with good score for speed
                break

            if best_idx == -1:
                break

            # Select this block
            selected.append(best_idx)
            selected_set.add(best_idx)
            n_selected += 1

            # Update: for each t-subset in this block, decrement uncovered
            # and update marginal gains of other blocks sharing those t-subsets
            for ti in block_tsubs[best_idx]:
                if (uncovered_mask >> ti) & 1:
                    uncovered_mask &= ~(1 << ti)
                    cov[ti] += 1
                    # Update marginal gains of all blocks containing ti
                    for other_idx in tsub_to_blocks[ti]:
                        if other_idx != best_idx and other_idx not in selected_set:
                            marginal[other_idx] -= 1
                            if marginal[other_idx] > 0:
                                version[other_idx] += 1
                                heapq.heappush(heap, (-marginal[other_idx], other_idx, version[other_idx]))
                            elif marginal[other_idx] == 0:
                                version[other_idx] += 1

            # Update Lagrangian duals via subgradient
            # λ_t += eta * (1 - indicator(t is covered))
            # For uncovered t-subsets, λ increases (penalize blocks not covering them)
            # For covered t-subsets, λ decreases
            for ti in range(n_tsubs):
                if (uncovered_mask >> ti) & 1:
                    lam[ti] = min(lam[ti] + eta, 10.0)
                else:
                    lam[ti] = max(lam[ti] - eta * 0.1, 0.1)

        if uncovered_mask != 0:
            return None

        return selected

    def blocks_from_indices(indices):
        return [list(all_ksubs[i]) for i in indices]

    def try_remove_redundant(blocks, seed):
        """Remove redundant blocks."""
        rng = random.Random(seed)
        blocks = list(blocks)
        n = len(blocks)

        # Precompute coverage masks for selected blocks
        block_masks = []
        for block in blocks:
            m = 0
            for tsub in itertools.combinations(block, t):
                idx = tsub_to_idx[tsub]
                m |= (1 << idx)
            block_masks.append(m)

        changed = True
        while changed:
            changed = False
            order = list(range(len(blocks)))
            rng.shuffle(order)
            for i in order:
                if i >= len(blocks):
                    continue
                redundant = True
                for tsub in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[tsub]
                    covered = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if (block_masks[j] >> idx) & 1:
                            covered = True
                            break
                    if not covered:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break

        return blocks

    # Main loop: try multiple seeds
    seed = 0
    while time.time() < deadline - 20:
        selected = greedy_with_lagrangian(seed)
        if selected is not None:
            blocks = blocks_from_indices(selected)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)

            # Try to improve with local search
            if time.time() < deadline - 10:
                improved = try_remove_redundant(blocks, seed)
                if len(improved) < best_count:
                    best_count = len(improved)
                    best_blocks = list(improved)

        seed += 1
        if seed >= 50:
            break

    if best_blocks is None:
        # Fallback: simple greedy without Lagrangian
        # Use a basic greedy that scans all blocks
        cov = [0] * n_tsubs
        uncovered_mask = (1 << n_tsubs) - 1
        selected = []
        used = [False] * n_ksubs

        while uncovered_mask and time.time() < deadline - 5:
            best_gain = 0
            best_idx = -1
            for i in range(n_ksubs):
                if used[i]:
                    continue
                gain = 0
                for ti in block_tsubs[i]:
                    if (uncovered_mask >> ti) & 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            used[best_idx] = True
            for ti in block_tsubs[best_idx]:
                if (uncovered_mask >> ti) & 1:
                    uncovered_mask &= ~(1 << ti)

        if uncovered_mask == 0:
            best_blocks = blocks_from_indices(selected)
        else:
            # Emergency: include all blocks (valid but large)
            best_blocks = [list(all_ksubs[i]) for i in range(n_ksubs)]

    return best_blocks
