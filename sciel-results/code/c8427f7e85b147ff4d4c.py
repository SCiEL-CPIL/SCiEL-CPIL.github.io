import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Precomputation
    # Generate all t-subsets and assign each a compact index 0..N-1
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {}
    t_idx_to_mask = []
    for i, sub in enumerate(all_t_subsets):
        t_sub_to_idx[sub] = i
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_idx_to_mask.append(mask)

    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Precompute for each k-block:
    #   block_cover_mask: a Python int bitmask over t-subset indices (bit i set iff t-sub i is in this block)
    #   block_t_indices: list of t-subset indices covered by this block
    block_cover_mask = [0] * n_k
    block_t_indices = [[] for _ in range(n_k)]
    for bi, block in enumerate(all_k_subsets):
        cm = 0
        tlist = []
        for t_sub in itertools.combinations(block, t):
            ti = t_sub_to_idx[t_sub]
            cm |= (1 << ti)
            tlist.append(ti)
        block_cover_mask[bi] = cm
        block_t_indices[bi] = tlist

    # For each t-subset index, list of block indices that cover it
    t_idx_to_blocks = [[] for _ in range(n_t)]
    for bi in range(n_k):
        for ti in block_t_indices[bi]:
            t_idx_to_blocks[ti].append(bi)

    # 2. Greedy with restarts
    best_blocks = None
    best_count = float('inf')

    # full_mask has all n_t bits set
    full_mask = (1 << n_t) - 1

    def greedy_once(seed_rng=None):
        """Run one greedy pass, return list of block indices."""
        rng = seed_rng if seed_rng else random
        uncovered_mask = full_mask
        blocks = []

        while uncovered_mask:
            n_uncovered = uncovered_mask.bit_count()
            r = rng.randrange(n_uncovered)
            temp = uncovered_mask
            for _ in range(r):
                temp &= temp - 1
            target_bit = temp & (-temp)
            target_idx = target_bit.bit_length() - 1

            candidates = t_idx_to_blocks[target_idx]
            if not candidates:
                break

            max_gain = 0
            best_candidates = []

            for bi in candidates:
                gain = (uncovered_mask & block_cover_mask[bi]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [bi]
                elif gain == max_gain:
                    best_candidates.append(bi)

            if not best_candidates:
                break

            chosen_bi = rng.choice(best_candidates)
            blocks.append(chosen_bi)
            uncovered_mask &= ~block_cover_mask[chosen_bi]

        if uncovered_mask:
            return None
        return blocks

    def local_search(block_indices, time_limit=5.0):
        """Try to reduce the number of blocks via removal and swap operations."""
        if block_indices is None:
            return None
        if time.time() > time_limit:
            return block_indices

        # Work with a set for O(1) membership
        block_set = set(block_indices)
        current_count = len(block_set)

        # Build coverage count: for each t-subset, how many blocks in our set cover it
        cover_count = [0] * n_t
        for bi in block_set:
            for ti in block_t_indices[bi]:
                cover_count[ti] += 1

        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Try removing redundant blocks
            for bi in list(block_set):
                # Check if removing bi leaves all t-subsets covered
                removable = True
                for ti in block_t_indices[bi]:
                    if cover_count[ti] == 1:
                        removable = False
                        break
                if removable:
                    # Remove it
                    block_set.remove(bi)
                    for ti in block_t_indices[bi]:
                        cover_count[ti] -= 1
                    current_count -= 1
                    improved = True

            # Try single-block swaps: replace one block with another that covers more
            if not improved and time.time() < time_limit:
                for bi in list(block_set):
                    # For each block, see if we can find a replacement
                    # that covers all unique t-subsets of bi and possibly more
                    unique_to_bi = []
                    for ti in block_t_indices[bi]:
                        if cover_count[ti] == 1:
                            unique_to_bi.append(ti)

                    if not unique_to_bi:
                        continue

                    # Find a block that covers all unique_to_bi
                    # and has higher total coverage
                    best_swap = None
                    best_gain = 0

                    # Candidate blocks: those that cover at least one of unique_to_bi
                    candidate_set = set()
                    for ti in unique_to_bi:
                        for cb in t_idx_to_blocks[ti]:
                            if cb != bi:
                                candidate_set.add(cb)

                    for cb in candidate_set:
                        # Check if cb covers all unique_to_bi
                        covers_all = True
                        for ti in unique_to_bi:
                            if cb not in t_idx_to_blocks[ti]:
                                covers_all = False
                                break
                        if not covers_all:
                            continue

                        # Compute net gain: how many t-subsets does cb cover that bi doesn't,
                        # minus how many bi covers that cb doesn't
                        # Actually, we need: after swap, all previously covered t-subsets
                        # must still be covered.
                        # t-subsets only covered by bi: unique_to_bi (already checked cb covers them)
                        # t-subsets only covered by cb that bi doesn't: these become newly covered
                        # But we need to ensure no t-subset becomes uncovered.
                        # A t-subset ti is uncovered after swap iff:
                        #   cover_count[ti] == 1 and that one block is bi and cb doesn't cover ti
                        # We already handle unique_to_bi (cover_count==1 for bi).
                        # For cb's t-subsets: if cover_count[ti] == 1 and that block is cb,
                        # and bi doesn't cover ti, then ti becomes uncovered.
                        # So we need: for all ti in block_t_indices[cb] where cover_count[ti]==1,
                        # bi must also cover ti.

                        safe = True
                        for ti in block_t_indices[cb]:
                            if cover_count[ti] == 1:
                                # The only block covering ti is cb (since we're considering swapping bi->cb)
                                # Wait, cover_count includes bi. Let me think again.
                                # Currently, ti is covered by cover_count[ti] blocks.
                                # After removing bi and adding cb:
                                #   If ti is in bi: new count = cover_count[ti] - 1 + (1 if ti in cb else 0)
                                #   If ti not in bi: new count = cover_count[ti] + (1 if ti in cb else 0)
                                # We need new count >= 1 for all ti.

                                # If ti in bi and not in cb: new count = cover_count[ti] - 1. Need >= 1, so cover_count[ti] >= 2.
                                # If ti not in bi and in cb: new count = cover_count[ti] + 1 >= 1 always.
                                # If ti in bi and in cb: new count = cover_count[ti] >= 1 always.
                                # If ti not in bi and not in cb: new count = cover_count[ti] >= 1 (unchanged).

                                # So the only risk is: ti in bi, not in cb, and cover_count[ti] == 1.
                                # That's exactly unique_to_bi, which we already checked.
                                pass

                        # So if cb covers all unique_to_bi, the swap is safe.
                        # Gain = |block_t_indices[cb] \ block_t_indices[bi]| - |block_t_indices[bi] \ block_t_indices[cb]|
                        # But since we're keeping same count, gain in terms of "slack" is:
                        # how many t-subsets go from count 1 to count >= 2, minus how many go from count >= 2 to count 1.

                        # Simpler: just check if cb covers all unique_to_bi. If yes, swap.
                        # To prefer better swaps, compute the number of t-subsets where
                        # cover_count increases (more robustness).

                        # For simplicity, just do the swap if valid.
                        best_swap = cb
                        break  # Take first valid swap

                    if best_swap is not None:
                        # Perform swap
                        block_set.remove(bi)
                        for ti in block_t_indices[bi]:
                            cover_count[ti] -= 1
                        block_set.add(best_swap)
                        for ti in block_t_indices[best_swap]:
                            cover_count[ti] += 1
                        improved = True

        return list(block_set)

    while time.time() < deadline:
        # Run greedy
        block_indices = greedy_once()
        if block_indices is None:
            continue

        # Local search refinement
        refined = local_search(block_indices, time_limit=min(deadline - time.time(), 3.0))
        if refined is not None:
            if len(refined) < best_count:
                best_count = len(refined)
                best_blocks = [list(all_k_subsets[bi]) for bi in refined]

    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered_mask = full_mask
        blocks = []
        while uncovered_mask:
            target_bit = uncovered_mask & (-uncovered_mask)
            target_idx = target_bit.bit_length() - 1
            candidates = t_idx_to_blocks[target_idx]
            max_gain = 0
            best_bi = candidates[0]
            for bi in candidates:
                gain = (uncovered_mask & block_cover_mask[bi]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_bi = bi
            blocks.append(all_k_subsets[best_bi])
            uncovered_mask &= ~block_cover_mask[best_bi]
        return [list(b) for b in blocks]

    return best_blocks
