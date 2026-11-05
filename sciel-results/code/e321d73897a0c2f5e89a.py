import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # 1. Precompute all t-subsets and k-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # 2. For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)

    # 3. Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    # Precompute overlap matrix between blocks (how many t-subsets they share)
    # This is expensive but useful for local search. Use sparse representation.
    # For each block, store a set of its t-subset indices for fast intersection
    k_sub_t_sets = [set(t_idxs) for t_idxs in k_sub_t_idxs]

    # 4. Greedy set cover with weighted scoring
    def greedy_cover(seed=0, weight_alpha=1.0):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=np.int32)  # 1 = uncovered, 0 = covered
        # gains[i] = number of currently-uncovered t-subsets in block i
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.float64)
        # Add small random perturbation to break ties differently per seed
        gains += rng.random() * 0.005
        selected = []
        remaining = n_t

        for _ in range(n_k):
            if time.time() > deadline - 30:
                break
            if remaining == 0:
                break

            # Find best block (highest gain)
            best_idx = int(np.argmax(gains))
            if gains[best_idx] < 0.5:  # effectively zero gain
                break

            selected.append(best_idx)

            # Update: mark covered t-subsets, update gains
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = 0
                    remaining -= 1
                    # Decrease gain for all blocks covering this t_subset
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1.0

        return selected

    # 5. Fast deletion pass using cover counts
    def reduce_blocks(blocks_idx, time_budget=None):
        if not blocks_idx:
            return blocks_idx
        if time_budget is None:
            time_budget = deadline - 3

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed and time.time() < time_budget:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    # 5b. Aggressive multi-pass reduction
    def aggressive_reduce(blocks_idx, max_passes=5):
        if not blocks_idx:
            return blocks_idx

        best = list(blocks_idx)
        for _ in range(max_passes):
            if time.time() > deadline - 10:
                break
            candidate = reduce_blocks(best)
            if len(candidate) < len(best):
                best = candidate
            else:
                break
        return best

    # 6. Enhanced local search with multiple swap strategies
    def try_improve(sel_indices, max_rounds=5):
        """Try to reduce block count via targeted swaps with multiple strategies."""
        if not sel_indices:
            return sel_indices

        best_result = list(sel_indices)
        best_size = len(sel_indices)
        rng = random.Random(42)

        for _round in range(max_rounds):
            if time.time() > deadline - 8:
                break

            sel_indices = list(best_result)

            # Build cover count
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in sel_indices:
                for t_idx in k_sub_t_idxs[b_idx]:
                    cover_count[t_idx] += 1

            improved_this_round = False

            # Strategy 1: For each block, try removing it and finding replacements
            for i in range(len(sel_indices)):
                if time.time() > deadline - 8:
                    break
                if len(sel_indices) <= best_size:
                    break

                orig_idx = sel_indices[i]

                # Find t-subsets uniquely covered by orig_idx
                unique_t_idxs = []
                for t_idx in k_sub_t_idxs[orig_idx]:
                    if cover_count[t_idx] == 1:
                        unique_t_idxs.append(t_idx)

                if not unique_t_idxs:
                    # No unique coverage, can just remove
                    for t_idx in k_sub_t_idxs[orig_idx]:
                        cover_count[t_idx] -= 1
                    sel_indices.pop(i)
                    best_size = len(sel_indices)
                    best_result = list(sel_indices)
                    improved_this_round = True
                    i -= 1
                    continue

                # Find candidate blocks that cover ALL unique t-subsets
                if len(unique_t_idxs) == 1:
                    candidates = list(t_sub_to_k_idxs[unique_t_idxs[0]])
                else:
                    # Intersect candidate sets
                    candidates = set(t_sub_to_k_idxs[unique_t_idxs[0]])
                    for t_idx in unique_t_idxs[1:]:
                        candidates &= set(t_sub_to_k_idxs[t_idx])
                    candidates = list(candidates)

                candidates = [c for c in candidates if c != orig_idx]

                # Exclude already-selected blocks
                sel_set = set(sel_indices)
                candidates = [c for c in candidates if c not in sel_set]

                if not candidates:
                    continue

                # Sort candidates by how much they help (cover more of orig's t-subsets)
                candidates.sort(key=lambda c: len(k_sub_t_sets[c] & k_sub_t_sets[orig_idx]), reverse=True)

                # Try top candidates
                for new_idx in candidates[:20]:
                    if time.time() > deadline - 8:
                        break

                    # Perform swap
                    sel_indices[i] = new_idx

                    # Update cover counts
                    for t_idx in k_sub_t_idxs[orig_idx]:
                        cover_count[t_idx] -= 1
                    for t_idx in k_sub_t_idxs[new_idx]:
                        cover_count[t_idx] += 1

                    # Now try to delete redundant blocks
                    new_sel = list(sel_indices)
                    new_cover = cover_count.copy()

                    del_improved = True
                    while del_improved and time.time() < deadline - 8:
                        del_improved = False
                        # Process in reverse order for efficiency
                        for j in range(len(new_sel) - 1, -1, -1):
                            b = new_sel[j]
                            is_essential = False
                            for t_idx in k_sub_t_idxs[b]:
                                if new_cover[t_idx] == 1:
                                    is_essential = True
                                    break
                            if not is_essential:
                                del_improved = True
                                for t_idx in k_sub_t_idxs[b]:
                                    new_cover[t_idx] -= 1
                                new_sel.pop(j)

                    if len(new_sel) < best_size:
                        best_size = len(new_sel)
                        best_result = new_sel
                        sel_indices = new_sel
                        cover_count = new_cover
                        improved_this_round = True
                        i = -1  # restart loop
                    else:
                        # Undo swap
                        sel_indices[i] = orig_idx
                        for t_idx in k_sub_t_idxs[orig_idx]:
                            cover_count[t_idx] += 1
                        for t_idx in k_sub_t_idxs[new_idx]:
                            cover_count[t_idx] -= 1

            # Strategy 2: Try removing two blocks and adding one
            if not improved_this_round and len(sel_indices) > 2:
                sel_indices = list(best_result)
                cover_count = np.zeros(n_t, dtype=np.int32)
                for b_idx in sel_indices:
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] += 1

                # Try pairs of blocks to remove
                n_sel = len(sel_indices)
                # Limit pairs tried for speed
                max_pairs = min(n_sel * (n_sel - 1) // 2, 500)
                pairs_tried = 0

                for i in range(n_sel):
                    if time.time() > deadline - 8 or pairs_tried >= max_pairs:
                        break
                    for j in range(i + 1, n_sel):
                        if time.time() > deadline - 8 or pairs_tried >= max_pairs:
                            break
                        pairs_tried += 1

                        b1 = sel_indices[i]
                        b2 = sel_indices[j]

                        # Find t-subsets uniquely covered by b1 or b2
                        unique_t_idxs = []
                        for t_idx in k_sub_t_idxs[b1]:
                            if cover_count[t_idx] == 1:
                                unique_t_idxs.append(t_idx)
                        for t_idx in k_sub_t_idxs[b2]:
                            if cover_count[t_idx] == 1 and t_idx not in k_sub_t_sets[b1]:
                                unique_t_idxs.append(t_idx)

                        if not unique_t_idxs:
                            # Can remove both
                            for t_idx in k_sub_t_idxs[b1]:
                                cover_count[t_idx] -= 1
                            for t_idx in k_sub_t_idxs[b2]:
                                cover_count[t_idx] -= 1
                            sel_indices.pop(j)
                            sel_indices.pop(i)
                            best_size = len(sel_indices)
                            best_result = list(sel_indices)
                            improved_this_round = True
                            break

                        # Find single block covering all unique t-subsets
                        if len(unique_t_idxs) == 1:
                            candidates = list(t_sub_to_k_idxs[unique_t_idxs[0]])
                        else:
                            candidates = set(t_sub_to_k_idxs[unique_t_idxs[0]])
                            for t_idx in unique_t_idxs[1:]:
                                candidates &= set(t_sub_to_k_idxs[t_idx])
                            candidates = list(candidates)

                        candidates = [c for c in candidates if c != b1 and c != b2]
                        sel_set = set(sel_indices)
                        candidates = [c for c in candidates if c not in sel_set]

                        if not candidates:
                            continue

                        # Try best candidate
                        candidates.sort(key=lambda c: len(k_sub_t_sets[c] & (k_sub_t_sets[b1] | k_sub_t_sets[b2])), reverse=True)
                        for new_idx in candidates[:10]:
                            if time.time() > deadline - 8:
                                break

                            # Perform 2-for-1 swap
                            new_sel = [x for x in sel_indices if x != b1 and x != b2]
                            new_sel.append(new_idx)

                            new_cover = np.zeros(n_t, dtype=np.int32)
                            for b_idx in new_sel:
                                for t_idx in k_sub_t_idxs[b_idx]:
                                    new_cover[t_idx] += 1

                            # Delete redundant
                            del_improved = True
                            while del_improved and time.time() < deadline - 8:
                                del_improved = False
                                for jj in range(len(new_sel) - 1, -1, -1):
                                    b = new_sel[jj]
                                    is_essential = False
                                    for t_idx in k_sub_t_idxs[b]:
                                        if new_cover[t_idx] == 1:
                                            is_essential = True
                                            break
                                    if not is_essential:
                                        del_improved = True
                                        for t_idx in k_sub_t_idxs[b]:
                                            new_cover[t_idx] -= 1
                                        new_sel.pop(jj)

                            if len(new_sel) < best_size:
                                best_size = len(new_sel)
                                best_result = new_sel
                                sel_indices = new_sel
                                cover_count = new_cover
                                improved_this_round = True
                                break

                    if improved_this_round:
                        break

            if not improved_this_round:
                break

        return best_result

    # 7. Main: run multiple greedy restarts, keep best, then refine
    best_solution = None
    best_count = float('inf')

    # Run greedy variants with different seeds
    n_greedy = 8
    for seed in range(n_greedy):
        if time.time() > deadline - 60:  # Leave time for refinement
            break

        selected = greedy_cover(seed)
        reduced = reduce_blocks(selected, time_budget=deadline - 40)

        if len(reduced) < best_count:
            best_count = len(reduced)
            best_solution = reduced

    # Aggressive reduction on best
    if best_solution and time.time() < deadline - 30:
        agg = aggressive_reduce(best_solution, max_passes=5)
        if len(agg) < best_count:
            best_count = len(agg)
            best_solution = agg

    # Multi-round local search refinement - spend most time here
    if best_solution and time.time() < deadline - 10:
        refined = try_improve(best_solution, max_rounds=8)
        if len(refined) < best_count:
            best_count = len(refined)
            best_solution = refined

    # Final deletion pass
    if best_solution and time.time() < deadline - 3:
        final = reduce_blocks(best_solution)
        if len(final) < best_count:
            best_solution = final

    if best_solution is None:
        # Fallback: pure greedy
        best_solution = greedy_cover(0)
        best_solution = reduce_blocks(best_solution)

    return [list(all_k_subs[i]) for i in best_solution]
