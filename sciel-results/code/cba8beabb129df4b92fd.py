import itertools
import numpy as np
import time
import random
import sys

def construct(v, k, t):
    # Deadline guard
    deadline = time.time() + 118.0  # 32s margin for safety (150s limit)
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Data structures
    block_tsubs = [[] for _ in range(n_blocks)]
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    # Bitsets for fast set operations
    # Note: Python integers are arbitrary precision, so we can use them as bitsets
    # However, for n_tsubsets = 3876, the integers will be ~500 bytes, which is fine.
    # For n_blocks = 50388, the integers will be ~6KB, which is also fine.
    
    tsub_mask = [0] * n_blocks       # tsub_mask[b] has bits for tsubs covered by block b
    tsub_blocks_mask = [0] * n_tsubsets # tsub_blocks_mask[ts] has bits for blocks covering ts
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask
        
    full_tsub_mask = (1 << n_tsubsets) - 1
    
    # Precompute conflict masks for Packing LB (independent sets)
    # Two t-subsets conflict if they are covered by the same block
    # We need this for the lower bound estimation
    tsub_conflict_mask = [0] * n_tsubsets
    for b_idx in range(n_blocks):
        mask = tsub_mask[b_idx]
        # For each t-subset in this block, all other t-subsets in this block conflict with it
        # Actually, the definition of packing lower bound is usually:
        # Max number of t-subsets such that no two are in the same block.
        # So two t-subsets conflict if they share a block.
        # We can iterate through bits of mask
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            # All subsets in this block conflict with idx
            tsub_conflict_mask[idx] |= mask
            m ^= lsb
            
    def packing_lb(uncovered_mask):
        """
        Calculate lower bound using maximal independent set of uncovered t-subsets.
        Two t-subsets are independent if no single block covers both.
        Greedy maximal independent set approximation.
        """
        count = 0
        remaining = uncovered_mask
        while remaining:
            # Pick an arbitrary t-subset (e.g., lowest bit)
            lsb = remaining & -remaining
            idx = lsb.bit_length() - 1
            count += 1
            # Remove all subsets that conflict with idx
            remaining &= ~tsub_conflict_mask[idx]
        return count

    def greedy_cover(seed):
        """
        Standard greedy using bitset operations for speed.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0
        
        # Initial gains: number of tsubs covered by each block
        # We can use bit_count for this
        gains = [tsub_mask[b].bit_count() for b in range(n_blocks)]
        
        while uncovered_mask:
            if time.time() > deadline - 10.0:
                break

            # Find best block (max gain)
            # To add randomness for tie-breaking, we can shuffle indices with same max gain
            max_gain = -1
            best_candidates = []
            for b_idx in range(n_blocks):
                if selected_mask & (1 << b_idx):
                    continue
                g = gains[b_idx]
                if g > max_gain:
                    max_gain = g
                    best_candidates = [b_idx]
                elif g == max_gain:
                    best_candidates.append(b_idx)
            
            if max_gain <= 0:
                break

            best_block = rng.choice(best_candidates)
            
            selected.append(best_block)
            selected_mask |= (1 << best_block)
            gains[best_block] = -1 # Mark as selected
            
            # Update gains for blocks covering newly covered tsubs
            new_covered = tsub_mask[best_block] & uncovered_mask
            uncovered_mask &= ~tsub_mask[best_block]
            
            # Iterate through newly covered t-subsets
            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                # For every block covering this ts_idx, decrease its gain
                cm = tsub_blocks_mask[ts_idx]
                while cm:
                    lsb2 = cm & -cm
                    b_idx2 = lsb2.bit_length() - 1
                    if not (selected_mask & (1 << b_idx2)):
                        gains[b_idx2] -= 1
                    cm ^= lsb2
                m ^= lsb

        return selected

    def local_search(blocks, seed):
        """
        Local search with swaps and deletions.
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
        while improved and time.time() < deadline - 5.0:
            improved = False

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

            # Phase 2: Try 1-for-1 swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 5.0:
                    break
                if b not in block_set:
                    continue

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
                # Start with the most constrained unique tsub
                min_ts = min(unique_tsubs, key=lambda x: tsub_blocks_mask[x].bit_count())
                candidates_mask = tsub_blocks_mask[min_ts]
                
                for ts_idx in unique_tsubs[1:]:
                    candidates_mask &= tsub_blocks_mask[ts_idx]
                    if not candidates_mask:
                        break

                if not candidates_mask:
                    continue

                # Exclude current block
                candidates_mask &= ~(1 << b)
                
                # Exclude already selected blocks? No, we are swapping, so we can pick any unselected block.
                # Actually, we remove b and add r. r should not be in block_set.
                candidates_mask &= ~selected_mask # Wait, selected_mask is not passed. 
                # Let's just check if r is in block_set
                cand_list = []
                m = candidates_mask
                while m:
                    lsb = m & -m
                    r_idx = lsb.bit_length() - 1
                    if r_idx not in block_set:
                        cand_list.append(r_idx)
                    m ^= lsb

                if not cand_list:
                    continue

                rng.shuffle(cand_list)
                
                # Limit candidates to avoid too much work per swap attempt
                if len(cand_list) > 20:
                    cand_list = cand_list[:20]

                swapped = False
                for r in cand_list:
                    # Perform swap
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1

                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1

                    # Check if valid (all cover counts >= 1)
                    valid = True
                    for ts_idx in block_tsubs[b]:
                        if cover_count[ts_idx] < 1:
                            valid = False
                            break
                    
                    if valid:
                        improved = True
                        swapped = True
                        break
                    else:
                        # Revert
                        block_set.remove(r)
                        for ts_idx in block_tsubs[r]:
                            cover_count[ts_idx] -= 1
                        
                        block_set.add(b)
                        for ts_idx in block_tsubs[b]:
                            cover_count[ts_idx] += 1

                if swapped:
                    break

        return list(block_set)

    best_solution = None
    best_size = float('inf')
    
    # Multi-start greedy + local search
    max_restarts = 500
    lower_bound = 131  # Known lower bound for C(19,7,4)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        # Early termination if we've hit the lower bound
        if best_size <= lower_bound:
            break
            
        # Run greedy
        solution = greedy_cover(restart)
        
        if not solution:
            continue
            
        # Run local search
        solution = local_search(solution, restart)
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution.copy()
            
            # If we reached the lower bound, stop immediately
            if best_size <= lower_bound:
                break

    # Fallback if no solution found (should not happen)
    if best_solution is None:
        # Simple greedy without local search
        uncovered_mask = full_tsub_mask
        selected = []
        while uncovered_mask:
            best_block = -1
            best_gain = -1
            for b_idx in range(n_blocks):
                if any(b == b_idx for b in selected):
                    continue
                g = (tsub_mask[b_idx] & uncovered_mask).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_block = b_idx
            if best_block == -1 or best_gain <= 0:
                break
            selected.append(best_block)
            uncovered_mask &= ~tsub_mask[best_block]
        best_solution = selected

    # Return list of lists of ints
    if best_solution is None:
        return []
    return [list(all_blocks[idx]) for idx in best_solution]
