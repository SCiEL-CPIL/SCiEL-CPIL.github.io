import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

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

    # 4. Greedy set cover using cover-count for fast updates
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered_count = np.ones(n_t, dtype=np.int32)  # 1 = uncovered, 0 = covered
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        selected = []

        for _ in range(n_k):
            if time.time() > deadline:
                break
            if uncovered_count.sum() == 0:
                break

            # Find best block (highest gain among uncovered)
            # Use argmax; blocks with gain 0 won't be picked if any have gain > 0
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break

            selected.append(best_idx)

            # Update: mark covered t-subsets, update gains
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered_count[t_idx]:
                    uncovered_count[t_idx] = 0
                    # Decrease gain for all blocks covering this t_subset
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1

        return selected

    # 5. Deletion pass using cover counts
    def reduce_blocks(blocks_idx):
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed and time.time() < deadline:
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

    # 6. Swap-based local search using reverse index for targeted candidates
    def try_improve(sel_indices):
        """Try to reduce block count via targeted swaps."""
        if not sel_indices:
            return sel_indices

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in sel_indices:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        best_result = list(sel_indices)
        best_size = len(sel_indices)

        # For each selected block, find its "unique" t-subsets (covered only by it)
        # Then find candidate blocks that cover those unique subsets
        for i in range(len(sel_indices)):
            if time.time() > deadline:
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
                i -= 1
                continue

            # Find candidate blocks that cover ALL unique t-subsets
            # A candidate must cover every t_idx in unique_t_idxs
            candidates = set(t_sub_to_k_idxs[unique_t_idxs[0]])
            for t_idx in unique_t_idxs[1:]:
                candidates &= set(t_sub_to_k_idxs[t_idx])
            candidates.discard(orig_idx)

            # Also exclude already-selected blocks
            sel_set = set(sel_indices)
            candidates -= sel_set

            if not candidates:
                continue

            # Try each candidate
            for new_idx in list(candidates)[:50]:  # Limit candidates for speed
                if time.time() > deadline:
                    break

                # Tentatively swap: remove orig_idx, add new_idx
                # Check if all t-subsets of orig_idx are still covered
                # (unique ones are covered by new_idx by construction)
                # Non-unique ones have cover_count >= 2, so removing orig_idx is fine

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

                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for j in range(len(new_sel) - 1, -1, -1):
                        b = new_sel[j]
                        is_essential = False
                        for t_idx in k_sub_t_idxs[b]:
                            if new_cover[t_idx] == 1:
                                is_essential = True
                                break
                        if not is_essential:
                            improved = True
                            for t_idx in k_sub_t_idxs[b]:
                                new_cover[t_idx] -= 1
                            new_sel.pop(j)

                if len(new_sel) < best_size:
                    best_size = len(new_sel)
                    best_result = new_sel
                    # Update state for next iteration
                    sel_indices = new_sel
                    cover_count = new_cover
                    # Restart the loop from beginning since indices shifted
                    i = -1
                else:
                    # Undo swap
                    sel_indices[i] = orig_idx
                    for t_idx in k_sub_t_idxs[orig_idx]:
                        cover_count[t_idx] += 1
                    for t_idx in k_sub_t_idxs[new_idx]:
                        cover_count[t_idx] -= 1

        return best_result

    # 7. Main: run multiple greedy restarts, keep best, then refine
    best_solution = None
    best_count = float('inf')

    # Run several greedy variants with different tie-breaking
    for seed in range(8):
        if time.time() > deadline - 20:  # Leave time for refinement
            break

        selected = greedy_cover(seed)
        reduced = reduce_blocks(selected)

        if len(reduced) < best_count:
            best_count = len(reduced)
            best_solution = reduced

    # Final refinement on the best solution
    if best_solution and time.time() < deadline - 5:
        refined = try_improve(best_solution)
        if len(refined) < best_count:
            best_count = len(refined)
            best_solution = refined

    # One more deletion pass
    if best_solution and time.time() < deadline - 2:
        final = reduce_blocks(best_solution)
        if len(final) < best_count:
            best_solution = final

    if best_solution is None:
        # Fallback: pure greedy
        best_solution = greedy_cover(0)
        best_solution = reduce_blocks(best_solution)

    return [list(all_k_subs[i]) for i in best_solution]
