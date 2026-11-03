import itertools
import time
import random
import numpy as np
import heapq
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Bitset representation
    tsub_mask = [0] * n_blocks
    tsub_blocks_mask = [0] * n_tsubsets
    block_tsubs = [[] for _ in range(n_blocks)]

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask
        
    full_tsub_mask = (1 << n_tsubsets) - 1
    
    # Precompute conflict masks for Packing LB
    # tsub_conflict_mask[i] has bits set for all tsubs j that share a block with i
    tsub_conflict_mask = [0] * n_tsubsets
    for b_idx in range(n_blocks):
        mask = tsub_mask[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tsub_conflict_mask[idx] |= mask
            m ^= lsb

    def packing_lb(uncovered_mask):
        """
        Calculate lower bound using maximal independent set of uncovered t-subsets.
        Two t-subsets are independent if no single block covers both.
        """
        count = 0
        remaining = uncovered_mask
        while remaining:
            # Pick the lowest set bit
            lsb = remaining & -remaining
            idx = lsb.bit_length() - 1
            count += 1
            # Remove this subset and all subsets that conflict with it
            remaining &= ~tsub_conflict_mask[idx]
        return count

    def compute_dual_weights(uncovered_mask, iterations=50):
        """
        Simplified fast weight computation.
        Weights are based on the number of blocks covering each t-subset.
        Rare t-subsets get higher weights.
        """
        # Count coverage for each t-subset in the current uncovered set?
        # No, just use static counts for speed.
        # tsub_freq[ts_idx] = number of blocks covering ts_idx
        # weight = 1 / freq
        
        # This is precomputed implicitly by tsub_blocks_mask bit counts
        weights = np.ones(n_tsubsets)
        # For speed, we just return uniform weights. The complexity of the
        # previous version was the bottleneck.
        return weights

    def weighted_greedy_cover(seed, use_weights=True):
        """
        Fast greedy cover.
        If time runs low, finishes with a fast unweighted greedy.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0
        weights = np.ones(n_tsubsets) # Static weights for speed

        while uncovered_mask:
            if time.time() > deadline - 15.0:
                # Fallback: Fast unweighted greedy to ensure valid covering
                # Just pick blocks with max coverage of remaining uncovered
                while uncovered_mask:
                    best_block = -1
                    best_gain = -1
                    # Sample a few random blocks to find a good one quickly?
                    # No, just do a full scan but without weights.
                    for b_idx in range(n_blocks):
                        if selected_mask & (1 << b_idx):
                            continue
                        mask = tsub_mask[b_idx] & uncovered_mask
                        if mask == 0:
                            continue
                        gain = mask.bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_block = b_idx
                    if best_block == -1:
                        break
                    selected.append(best_block)
                    selected_mask |= (1 << best_block)
                    uncovered_mask &= ~tsub_mask[best_block]
                break

            best_block = -1
            best_score = 0.0

            for b_idx in range(n_blocks):
                if selected_mask & (1 << b_idx):
                    continue
                mask = tsub_mask[b_idx] & uncovered_mask
                if mask == 0:
                    continue
                # Compute weighted gain (weights are 1, so this is just bit count)
                unweighted_gain = mask.bit_count()
                
                # Simple score: gain
                score = unweighted_gain
                
                if score > best_score:
                    best_score = score
                    best_block = b_idx

            if best_block == -1:
                break

            selected.append(best_block)
            selected_mask |= (1 << best_block)
            uncovered_mask &= ~tsub_mask[best_block]

        return selected

    def local_search(blocks, seed, max_swaps_per_iter=50):
        """
        Local search with improved swaps and deletions.
        Uses scoring to prefer swaps that maximize net coverage gain.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        iterations = 0
        while improved and time.time() < deadline - 5.0 and iterations < 200:
            improved = False
            iterations += 1

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: Try 1-for-1 swaps with scoring
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            swaps_made = 0
            for b in current_blocks:
                if time.time() > deadline - 5.0:
                    break
                if b not in block_set:
                    continue
                if swaps_made >= max_swaps_per_iter:
                    break

                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidates that cover all unique tsubs
                cand_mask = (1 << n_blocks) - 1
                for ts_idx in unique_tsubs:
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break

                if cand_mask == 0:
                    continue

                cand_mask &= ~(1 << b)

                # Score candidates by how many additional tsubs they cover
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    # Compute bonus: number of tsubs this block covers that are currently covered >= 2
                    bonus = 0
                    for ts_idx in block_tsubs[idx]:
                        if cover_count[ts_idx] >= 2:
                            bonus += 1
                    candidates.append((bonus, idx))
                    m ^= lsb
                    if len(candidates) > 50:
                        break

                if not candidates:
                    continue

                # Sort by bonus descending, then randomize ties
                candidates.sort(key=lambda x: (-x[0], rng.random()))
                
                # Try top candidates
                for bonus, r in candidates[:10]:
                    if r in block_set:
                        continue

                    # Simulate the swap to verify it's valid
                    # After removing b, check if all unique_tsubs are still covered by r
                    valid = True
                    for ts_idx in unique_tsubs:
                        if not ((tsub_blocks_mask[ts_idx] >> r) & 1):
                            valid = False
                            break
                    
                    if not valid:
                        continue

                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1

                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swaps_made += 1
                    break

                if improved:
                    break

        return list(block_set)

    def two_for_one_search(blocks, seed, n_checks=80):
        """
        Try to replace two blocks with one.
        Improved: score candidates by coverage bonus.
        """
        rng = random.Random(seed)
        if len(blocks) < 2:
            return blocks

        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for _ in range(n_checks):
                if time.time() > deadline - 5.0:
                    break
                if len(block_set) < 2:
                    break

                i, j = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[i]
                b2 = current_blocks[j]

                if b1 not in block_set or b2 not in block_set:
                    continue

                required_mask = 0
                for ts_idx in block_tsubs[b1]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)
                for ts_idx in block_tsubs[b2]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)

                if required_mask == 0:
                    continue

                cand_mask = (1 << n_blocks) - 1
                m = required_mask
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                    m ^= lsb

                if cand_mask == 0:
                    continue

                cand_mask &= ~(1 << b1)
                cand_mask &= ~(1 << b2)

                # Score candidates by how many extra tsubs they cover
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    # Bonus: tsubs covered by this block that are currently covered >= 2
                    bonus = 0
                    for ts_idx in block_tsubs[idx]:
                        if cover_count[ts_idx] >= 2:
                            bonus += 1
                    candidates.append((bonus, idx))
                    m ^= lsb
                    if len(candidates) > 20:
                        break

                if not candidates:
                    continue

                # Sort by bonus descending
                candidates.sort(key=lambda x: (-x[0], rng.random()))
                r = candidates[0][1]

                block_set.remove(b1)
                for ts_idx in block_tsubs[b1]:
                    cover_count[ts_idx] -= 1

                block_set.remove(b2)
                for ts_idx in block_tsubs[b2]:
                    cover_count[ts_idx] -= 1

                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1

                improved = True
                break

        return list(block_set)

    def branch_and_bound(uncovered_mask, current_blocks, current_size, best_size, selected_mask, depth=0):
        """
        Branch-and-bound with bitset-based lower bounds.
        """
        if time.time() > deadline - 5.0:
            return False, None

        if uncovered_mask == 0:
            return True, list(current_blocks)

        # Lower bound: Packing LB
        lb_pack = packing_lb(uncovered_mask)
        
        # Simple LB: ceil(n_uncovered / max_cover)
        n_uncovered = uncovered_mask.bit_count()
        if n_uncovered == 0:
            return True, list(current_blocks)
            
        # Estimate max cover quickly?
        # Just use packing LB as it's usually tighter for this problem structure.
        
        if current_size + lb_pack >= best_size:
            return False, None

        # Select the most constrained uncovered t-subset (fewest candidate blocks)
        best_ts = -1
        best_candidates = None
        best_cand_count = float('inf')
        
        m = uncovered_mask
        while m:
            lsb = m & (-m)
            ts_idx = lsb.bit_length() - 1
            candidates = tsub_blocks_mask[ts_idx] & ~selected_mask
            cand_count = candidates.bit_count()
            if cand_count < best_cand_count:
                best_cand_count = cand_count
                best_ts = ts_idx
                best_candidates = candidates
            if cand_count == 0:
                return False, None
            m ^= lsb

        if best_ts == -1:
            return True, list(current_blocks)

        # Branch on which block to use for best_ts
        candidate_list = []
        m = best_candidates
        while m:
            lsb = m & (-m)
            b_idx = lsb.bit_length() - 1
            cover = (tsub_mask[b_idx] & uncovered_mask).bit_count()
            candidate_list.append((cover, b_idx))
            m ^= lsb
        
        # Sort by coverage descending
        candidate_list.sort(reverse=True)
        
        # Limit branching factor
        max_branch = min(8, len(candidate_list))
        
        for cover, b_idx in candidate_list[:max_branch]:
            new_blocks = current_blocks + [b_idx]
            new_uncovered = uncovered_mask & ~tsub_mask[b_idx]
            new_selected_mask = selected_mask | (1 << b_idx)
            
            found, sol = branch_and_bound(
                new_uncovered, new_blocks, current_size + 1, best_size, new_selected_mask, depth + 1
            )
            
            if found and sol is not None:
                return True, sol
        
        return False, None

    def final_polish(blocks, seed, iterations=30):
        """
        High-precision polish: alternate between local search and 2-for-1 with 
        tighter convergence criteria.
        """
        rng = random.Random(seed)
        block_set = set(blocks)
        
        for it in range(iterations):
            if time.time() > deadline - 3.0:
                break
            
            # Local search with limited swaps
            ls_sol = local_search(list(block_set), seed + it, max_swaps_per_iter=20)
            block_set = set(ls_sol)
            
            # 2-for-1 with fewer checks
            t2o_sol = two_for_one_search(list(block_set), seed + 1000 + it, n_checks=30)
            new_block_set = set(t2o_sol)
            
            if len(new_block_set) < len(block_set):
                block_set = new_block_set
            else:
                # No improvement, try a few more local search iterations
                for _ in range(3):
                    if time.time() > deadline - 3.0:
                        break
                    ls_sol = local_search(list(block_set), seed + 2000 + it + _, max_swaps_per_iter=15)
                    new_block_set = set(ls_sol)
                    if len(new_block_set) < len(block_set):
                        block_set = new_block_set
                    else:
                        break
        
        return list(block_set)

    best_solution = None
    best_size = float('inf')

    # Run multiple restarts with both plain and weighted greedy
    num_restarts = 12
    for i in range(num_restarts):
        if time.time() > deadline - 25.0:
            break

        seed = 1000 + i
        # Alternate between plain and weighted greedy
        use_weights = (i % 2 == 0)
        greedy_sol = weighted_greedy_cover(seed, use_weights=use_weights)

        if not greedy_sol:
            continue

        # Local Search
        ls_sol = local_search(greedy_sol, seed)

        # 2-for-1 Search
        t2o_sol = two_for_one_search(ls_sol, seed)

        # B&B from scratch is too slow for this size. Skip it.
        # Instead, do more local search if time permits.
        if time.time() < deadline - 10.0:
            ls_sol2 = local_search(t2o_sol, seed + 5000)
            if len(ls_sol2) < len(t2o_sol):
                t2o_sol = ls_sol2

        if len(t2o_sol) < best_size:
            best_size = len(t2o_sol)
            best_solution = t2o_sol.copy()

    # Final polish pass on the best solution
    if best_solution is not None and time.time() < deadline - 5.0:
        polished = final_polish(best_solution, 9999, iterations=20)
        if len(polished) < best_size:
            best_size = len(polished)
            best_solution = polished

    # If no solution found (should not happen), fallback to basic greedy
    if best_solution is None:
        best_solution = weighted_greedy_cover(0, use_weights=False)

    return [list(all_blocks[idx]) for idx in best_solution]
