import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    # Pre-compute all t-subsets and k-blocks
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_idx = {s: i for i, s in enumerate(all_t_subs)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)

    # For each k-block, compute which t-subsets it covers
    # Use integer bitmasks for fast intersection
    # n_t = C(19,5) = 11628, which fits in a Python int bitmask
    block_t_masks = [0] * n_k
    block_t_list = [[] for _ in range(n_k)]

    for bi, block in enumerate(all_k_blocks):
        mask = 0
        subs = []
        for sub in itertools.combinations(block, t):
            idx = t_sub_idx[sub]
            subs.append(idx)
            mask |= (1 << idx)
        block_t_masks[bi] = mask
        block_t_list[bi] = subs

    def greedy_construct(seed=None, deadline_t=None):
        """Run greedy set cover with optional randomization."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        if deadline_t is None:
            deadline_t = deadline

        # Initial uncovered mask
        uncovered = (1 << n_t) - 1
        selected = []
        selected_set = set()
        cover_count = np.zeros(n_t, dtype=np.int32)

        while uncovered and time.time() < deadline_t - 3:
            best_block = -1
            best_gain = 0

            # Randomize tie-breaking by shuffling candidate order occasionally
            candidates = list(range(n_k))
            if seed is not None:
                rng.shuffle(candidates)

            for bi in candidates:
                if bi in selected_set:
                    continue
                gain = (block_t_masks[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi

            if best_block == -1 or best_gain == 0:
                break

            selected.append(best_block)
            selected_set.add(best_block)
            uncovered &= ~block_t_masks[best_block]

            for s in block_t_list[best_block]:
                cover_count[s] += 1

        return selected, cover_count, selected_set

    def remove_redundant(selected, cover_count, selected_set, deadline_t):
        """Remove blocks that don't cover any unique t-subset."""
        changed = True
        while changed and time.time() < deadline_t - 2:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                bi = selected[i]
                redundant = True
                for s in block_t_list[bi]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_t_list[bi]:
                        cover_count[s] -= 1
                    selected.pop(i)
                    selected_set.discard(bi)
                    changed = True
        return selected, cover_count, selected_set

    def local_search(selected, cover_count, selected_set, deadline_t, max_iters=200):
        """Try to swap or add blocks to reduce total count."""
        if not selected:
            return selected, cover_count, selected_set

        best_selected = list(selected)
        best_count = len(selected)
        best_cover = cover_count.copy()
        best_set = set(selected_set)

        # Try removing one block and adding one different block
        for i in range(len(selected)):
            if time.time() > deadline_t - 5:
                break

            bi = selected[i]
            # Check if bi is redundant
            is_redundant = True
            for s in block_t_list[bi]:
                if cover_count[s] == 1:
                    is_redundant = False
                    break

            if is_redundant:
                # Remove it
                for s in block_t_list[bi]:
                    cover_count[s] -= 1
                selected.pop(i)
                selected_set.discard(bi)
                best_selected = list(selected)
                best_count = len(selected)
                best_cover = cover_count.copy()
                best_set = set(selected_set)
                # Continue trying to remove more
                i -= 1
                continue

        # Try pairwise swaps: remove one, add one that covers more uncovered
        for i in range(len(selected)):
            if time.time() > deadline_t - 5:
                break

            bi = selected[i]
            # Temporarily remove bi
            temp_cover = cover_count.copy()
            for s in block_t_list[bi]:
                temp_cover[s] -= 1

            # Find blocks that become uncovered
            newly_uncovered = set()
            for s in block_t_list[bi]:
                if temp_cover[s] == 0:
                    newly_uncovered.add(s)

            if not newly_uncovered:
                # bi was redundant, just remove it
                for s in block_t_list[bi]:
                    cover_count[s] -= 1
                selected.pop(i)
                selected_set.discard(bi)
                best_selected = list(selected)
                best_count = len(selected)
                best_cover = cover_count.copy()
                best_set = set(selected_set)
                i -= 1
                continue

            # Try to find a block that covers all newly_uncovered and adds few new
            best_new_block = -1
            best_new_gain = -1

            for bj in range(n_k):
                if bj in selected_set or bj == bi:
                    continue

                # Does bj cover all newly_uncovered?
                covers_all = True
                for s in newly_uncovered:
                    if not (block_t_masks[bj] >> s) & 1:
                        covers_all = False
                        break

                if not covers_all:
                    continue

                # Count how many new t-subsets bj covers (not already covered)
                new_gain = 0
                for s in block_t_list[bj]:
                    if temp_cover[s] == 0:
                        new_gain += 1

                # We want to minimize new_gain (ideally 0 means perfect swap)
                # Actually we want to find a block that covers the uncovered ones
                # with minimal additional coverage
                if best_new_gain == -1 or new_gain < best_new_gain:
                    best_new_gain = new_gain
                    best_new_block = bj

            if best_new_block != -1:
                # Perform the swap
                for s in block_t_list[bi]:
                    cover_count[s] -= 1
                for s in block_t_list[best_new_block]:
                    cover_count[s] += 1

                selected[i] = best_new_block
                selected_set.discard(bi)
                selected_set.add(best_new_block)

                # Update best
                if len(selected) < best_count:
                    best_count = len(selected)
                    best_selected = list(selected)
                    best_cover = cover_count.copy()
                    best_set = set(selected_set)

        return best_selected, best_cover, best_set

    # Run multiple greedy restarts with different seeds
    best_blocks = None
    best_len = float('inf')

    # First, run a deterministic greedy
    sel, cc, ss = greedy_construct(seed=0, deadline_t=deadline - 30)
    sel, cc, ss = remove_redundant(sel, cc, ss, deadline - 25)
    sel, cc, ss = local_search(sel, cc, ss, deadline - 10, max_iters=100)
    if len(sel) < best_len:
        best_len = len(sel)
        best_blocks = sel

    # Run randomized restarts
    n_restarts = 0
    while time.time() < deadline - 20:
        seed = random.randint(1, 1000000)
        sel, cc, ss = greedy_construct(seed=seed, deadline_t=deadline - 25)
        if not sel:
            break
        sel, cc, ss = remove_redundant(sel, cc, ss, deadline - 15)
        sel, cc, ss = local_search(sel, cc, ss, deadline - 5, max_iters=50)
        if len(sel) < best_len:
            best_len = len(sel)
            best_blocks = sel
        n_restarts += 1

    if best_blocks is None:
        # Fallback: just return all blocks (shouldn't happen)
        best_blocks = list(range(n_k))

    # Convert to list of lists
    result = [list(all_k_blocks[i]) for i in best_blocks]
    return result
