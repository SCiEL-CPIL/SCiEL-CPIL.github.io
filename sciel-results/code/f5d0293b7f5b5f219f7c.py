import itertools
import random
import time
import heapq
import numpy as np

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build sparse coverage structure
    # For each block, store list of tsub indices it covers
    block_tsubs = []
    for block in all_kblocks:
        subs = [tsub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_tsubs.append(subs)

    # For each tsub, store list of blocks that cover it (for 2-exchange)
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, subs in enumerate(block_tsubs):
        for ts_idx in subs:
            tsub_blocks[ts_idx].append(b_idx)

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    best_solution = None
    best_len = float('inf')

    rng = random.Random(42)

    # Phase 1: Greedy-Set-Union (GSU) with lazy priority queue
    # Each entry: (-gain, block_idx, version)
    # We recompute gains lazily to avoid stale entries
    if time.monotonic() < deadline - 5.0:
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        counts = np.zeros(num_tsubs, dtype=np.int32)
        selected_mask = np.zeros(num_kblocks, dtype=bool)
        selected_indices = []

        # Lazy heap: (-gain, block_idx, version_counter)
        # version_counter increments each time we push; we skip stale entries
        heap = []
        version_counter = [0]

        # Initial gains: all blocks have gain = len(block_tsubs[b]) since nothing is covered
        for b_idx in range(num_kblocks):
            gain = len(block_tsubs[b_idx])
            heapq.heappush(heap, (-gain, b_idx, 0))
        version_counter[0] = 1

        while np.any(uncovered):
            if time.monotonic() > deadline - 10.0:
                break

            # Find the best non-stale, non-selected block
            best_block = -1
            best_gain = 0
            found = False

            while heap:
                neg_gain, b_idx, ver = heapq.heappop(heap)
                gain = -neg_gain

                # Skip if already selected
                if selected_mask[b_idx]:
                    continue

                # Lazy re-evaluation: check if this gain is still valid
                # The gain is valid if no new blocks have been selected since this entry was pushed
                # We use a simpler check: recompute the actual gain and compare
                # If the stored gain >= actual gain, it might be stale (we overestimated)
                # If the stored gain < actual gain, something is wrong
                # Actually, gains only decrease as we select more blocks, so:
                # stored_gain >= actual_gain always. If they differ, it's stale.
                # But recomputing is expensive. Instead, we use the version counter.
                # If version_counter[0] > ver, this entry is potentially stale.
                # We recompute the actual gain for this block.
                if ver < version_counter[0] - 1:
                    # Potentially stale, recompute
                    actual_gain = 0
                    for ts_idx in block_tsubs[b_idx]:
                        if uncovered[ts_idx]:
                            actual_gain += 1
                    if actual_gain > best_gain:
                        best_gain = actual_gain
                        best_block = b_idx
                        found = True
                    # Don't re-push; we'll handle it in the next iteration if needed
                    continue

                # Version is recent, trust the stored gain
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    found = True
                    break

            if not found or best_block == -1:
                break

            if best_gain == 0:
                break

            # Select the best block
            selected_indices.append(best_block)
            selected_mask[best_block] = True

            # Update uncovered and counts
            for ts_idx in block_tsubs[best_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = 0
                counts[ts_idx] += 1

            # Increment version counter to invalidate old heap entries
            version_counter[0] += 1

            # Push updated gains for blocks that might now have different gains
            # Only push blocks that cover at least one newly-covered tsub
            # For efficiency, we push all blocks that share a tsub with the new block
            # But that's expensive. Instead, we just let the lazy evaluation handle it.
            # The key insight: we don't need to push all blocks. The heap already has entries
            # for all blocks. When we pop, we recompute if the version is stale.

        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_solution = selected_indices[:]
            best_counts = counts.copy()
            best_selected_mask = selected_mask.copy()

    # Phase 2: Local Search with 2-exchange and redundancy removal
    if best_solution is not None:
        current_indices = best_solution[:]
        current_counts = best_counts.copy()
        current_mask = best_selected_mask.copy()

        improved = True
        stagnation_counter = 0
        max_stagnation = 30

        while time.monotonic() < deadline - 2.0:
            if not improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            # If stagnating, perform a perturbation
            if stagnation_counter >= max_stagnation and len(current_indices) > 0:
                n_remove = min(3 + rng.randint(0, 4), len(current_indices))
                to_remove = rng.sample(current_indices, n_remove)
                for b_idx in to_remove:
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    for ts_idx in block_tsubs[b_idx]:
                        current_counts[ts_idx] -= 1

                # Greedily re-add to cover gaps
                uncovered_mask = current_counts == 0
                while np.any(uncovered_mask):
                    if time.monotonic() > deadline - 5.0:
                        break
                    # Find block with max gain among uncovered
                    best_idx = -1
                    best_gain = 0
                    for b_idx in range(num_kblocks):
                        if current_mask[b_idx]:
                            continue
                        gain = 0
                        for ts_idx in block_tsubs[b_idx]:
                            if uncovered_mask[ts_idx]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = b_idx
                    if best_gain == 0:
                        break
                    current_mask[best_idx] = True
                    current_indices.append(best_idx)
                    for ts_idx in block_tsubs[best_idx]:
                        current_counts[ts_idx] += 1
                    uncovered_mask = current_counts == 0

                stagnation_counter = 0
                improved = True
                continue

            improved = False
            current_indices_set = set(current_indices)

            # 1. Remove Redundant Blocks
            redundant_found = False
            for b_idx in current_indices[:]:
                all_covered = True
                for ts_idx in block_tsubs[b_idx]:
                    if current_counts[ts_idx] <= 1:
                        all_covered = False
                        break
                if all_covered:
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    for ts_idx in block_tsubs[b_idx]:
                        current_counts[ts_idx] -= 1
                    improved = True
                    redundant_found = True
                    break

            if redundant_found:
                continue

            # 2. Try 1-exchange: replace one block with another
            swap_done = False
            idx_list = current_indices[:]
            rng.shuffle(idx_list)

            for b_idx in idx_list:
                if time.monotonic() > deadline - 2.0:
                    break

                # Find critical tsubs (count == 1)
                critical_indices = []
                for ts_idx in block_tsubs[b_idx]:
                    if current_counts[ts_idx] == 1:
                        critical_indices.append(ts_idx)

                if len(critical_indices) == 0:
                    # Redundant, should have been caught
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    for ts_idx in block_tsubs[b_idx]:
                        current_counts[ts_idx] -= 1
                    improved = True
                    swap_done = True
                    break

                # Find candidate blocks that cover all critical tsubs
                candidates = []
                for c_idx in range(num_kblocks):
                    if current_mask[c_idx]:
                        continue
                    covers_all = True
                    for ts_idx in critical_indices:
                        if c_idx not in tsub_blocks[ts_idx]:
                            covers_all = False
                            break
                    if covers_all:
                        candidates.append(c_idx)

                if len(candidates) > 0:
                    # Pick candidate that maximizes "usefulness" (covers most count==1 tsubs)
                    best_c = candidates[0]
                    best_score = -1
                    for c_idx in candidates[:10]:  # Limit candidates for speed
                        score = 0
                        for ts_idx in block_tsubs[c_idx]:
                            if current_counts[ts_idx] == 1:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_c = c_idx

                    # Perform swap
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    for ts_idx in block_tsubs[b_idx]:
                        current_counts[ts_idx] -= 1

                    current_mask[best_c] = True
                    current_indices.append(best_c)
                    for ts_idx in block_tsubs[best_c]:
                        current_counts[ts_idx] += 1

                    improved = True
                    swap_done = True
                    break

            if swap_done:
                continue

            # 3. Try 2-exchange: remove two blocks, add two new blocks
            if time.monotonic() < deadline - 3.0:
                two_ex_done = False
                # Try a random pair of blocks to remove
                if len(current_indices) >= 2:
                    pair = rng.sample(current_indices, 2)
                    b1, b2 = pair

                    # Check if removing both still leaves coverage
                    # Find tsubs that would become uncovered
                    would_be_uncovered = []
                    for ts_idx in block_tsubs[b1]:
                        if current_counts[ts_idx] == 1:
                            would_be_uncovered.append(ts_idx)
                    for ts_idx in block_tsubs[b2]:
                        if current_counts[ts_idx] == 1 and ts_idx not in block_tsubs[b1]:
                            would_be_uncovered.append(ts_idx)

                    if len(would_be_uncovered) <= 2:
                        # Find two new blocks that cover all would_be_uncovered tsubs
                        # and together cover them
                        found_pair = False
                        for c1 in range(num_kblocks):
                            if current_mask[c1] or c1 in pair:
                                continue
                            covers_c1 = all(c1 in tsub_blocks[ts_idx] for ts_idx in would_be_uncovered)
                            if not covers_c1:
                                continue
                            for c2 in range(num_kblocks):
                                if current_mask[c2] or c2 in pair or c2 == c1:
                                    continue
                                covers_c2 = all(c2 in tsub_blocks[ts_idx] for ts_idx in would_be_uncovered)
                                if not covers_c2:
                                    continue
                                # Check together they cover all
                                all_covered = True
                                for ts_idx in would_be_uncovered:
                                    if c1 not in tsub_blocks[ts_idx] and c2 not in tsub_blocks[ts_idx]:
                                        all_covered = False
                                        break
                                if all_covered:
                                    # Perform 2-exchange
                                    for b_idx in [b1, b2]:
                                        current_mask[b_idx] = False
                                        current_indices.remove(b_idx)
                                        for ts_idx in block_tsubs[b_idx]:
                                            current_counts[ts_idx] -= 1
                                    for c_idx in [c1, c2]:
                                        current_mask[c_idx] = True
                                        current_indices.append(c_idx)
                                        for ts_idx in block_tsubs[c_idx]:
                                            current_counts[ts_idx] += 1
                                    improved = True
                                    two_ex_done = True
                                    found_pair = True
                                    break
                            if found_pair:
                                break

                        if two_ex_done:
                            break

            if two_ex_done:
                continue

            # If we get here, no improvements found in this iteration

    # Final Result
    if best_solution is None:
        return []

    return get_blocks(best_solution)
