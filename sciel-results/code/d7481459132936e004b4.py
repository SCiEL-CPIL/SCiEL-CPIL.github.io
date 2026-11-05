import itertools
import time
import random
import numpy as np
from collections import defaultdict
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Precompute all t-subsets and map to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubs = len(all_tsubsets)

    # 2. Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # 3. Precompute for each block, the list of t-subset indices it covers
    # Vectorized construction of block_tsubs_arr
    t_per_block = int(math.comb(k, t))
    
    # Initialize array
    block_tsubs_arr = np.empty((n_blocks, t_per_block), dtype=np.int32)
    
    # Fill it. 
    # itertools.combinations(range(v), k) generates tuples.
    # We need to map each 7-tuple to its 35 3-tuple indices.
    # Doing this in a pure Python loop is the bottleneck.
    # Let's try to speed it up slightly or just accept it.
    # For v=24, k=7, n_blocks ~ 346k. 
    # 346k * 35 = 12M lookups.
    # Python dict lookup is fast, but 12M is ~1-2s. Acceptable.
    
    for i, block in enumerate(all_blocks):
        # Generate combinations of the block
        # block is a tuple of ints
        tsubs = itertools.combinations(block, t)
        for j, ts in enumerate(tsubs):
            block_tsubs_arr[i, j] = t_sub_to_idx[ts]

    # 4. Precompute inverse mapping: for each t-subset, list of block indices covering it
    # We need this for "Add" moves in local search and potentially for smarter greedy.
    # Instead of a list of lists, let's keep it for random sampling of blocks covering a specific triple.
    tsub_to_blocks = defaultdict(list)
    # This loop is also 12M iterations.
    # We can vectorize the construction of the inverse index using numpy?
    # block_tsubs_arr is (n_blocks, t_per_block).
    # We want for each value x in block_tsubs_arr, to know which row index it came from.
    # This is essentially an inverse scatter.
    # For simplicity and speed, let's stick to the loop but maybe optimize it.
    # Or, we can build it on the fly if needed? No, it's needed for Add moves.
    
    # Let's use a more efficient way to build the inverse index if possible.
    # Or just accept the 2s cost.
    
    for b_idx in range(n_blocks):
        for ts_idx in block_tsubs_arr[b_idx]:
            tsub_to_blocks[ts_idx].append(b_idx)
            
    # Convert to numpy arrays for faster access in local search?
    # tsub_to_blocks_np = {k: np.array(v, dtype=np.int32) for k, v in tsub_to_blocks.items()}
    # This might use a lot of memory if not careful, but 2024 triples * ~6000 blocks * 4 bytes = ~48MB. OK.
    # Let's do it.
    tsub_to_blocks_np = {}
    for k, v in tsub_to_blocks.items():
        tsub_to_blocks_np[k] = np.array(v, dtype=np.int32)

    # Helper: Get blocks covering a specific triple index
    def get_blocks_for_triple(ts_idx):
        return tsub_to_blocks_np[ts_idx]

    # 5. Greedy Initialization with Vectorized Gains
    # We will implement a fast greedy that uses the "gain" array updated via the inverse index.
    # This avoids the O(n_blocks * t_per_block) scan for max gain every step.
    
    def greedy_set_cover_fast(seed, use_weighted=False):
        rng = random.Random(seed)
        # gain[b] is the number of currently uncovered triples covered by block b
        # Initially, all triples are uncovered, so gain[b] = t_per_block for all b.
        gain = np.full(n_blocks, t_per_block, dtype=np.int32)
        is_uncovered = np.ones(n_tsubs, dtype=bool)
        
        selected_blocks = []
        
        while np.any(is_uncovered):
            if time.time() > deadline - 5:
                # If we are running out of time, we need to return a valid cover.
                # The current selected_blocks might not cover everything.
                # We must finish the cover. 
                # A simple fallback: keep adding blocks greedily without time check until covered?
                # Risky. Better to check time at the start of the loop and break if too close,
                # then handle the incomplete cover outside.
                # But the contract says "Valid when every t-subset ... lies in at least one block".
                # So we MUST return a valid cover.
                # If we break here, we have to complete it.
                # Let's just continue without time check if we are close? No, that might timeout.
                # Let's assume the greedy finishes quickly. If not, we return what we have and hope it's valid?
                # No, we must ensure validity.
                # Strategy: If time is tight, switch to a "panic mode" where we pick blocks covering
                # the most uncovered triples, but we don't update the gain array efficiently, we just
                # compute it on the fly for a small subset? 
                # Actually, the most robust way is to ensure the greedy loop is fast enough.
                # With n_blocks ~ 346k, and ~80 iterations, and updates ~ 35*6000=210k per iter.
                # 80 * 210k = 16.8M operations. In Python, this is ~2-4 seconds.
                # So it should finish well within 115s.
                pass

            max_gain = np.max(gain)
            if max_gain == 0:
                # This shouldn't happen if is_uncovered has True values and there are blocks covering them.
                # Unless we have a bug.
                break
            
            # Find candidates with max gain
            candidates = np.where(gain == max_gain)[0]
            
            # Random tie-breaking
            if len(candidates) > 1:
                b_idx = candidates[rng.randrange(len(candidates))]
            else:
                b_idx = candidates[0]
                
            selected_blocks.append(b_idx)
            
            # Update is_uncovered and gain
            # For each triple in the selected block:
            for t_idx in block_tsubs_arr[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # Update gain for all blocks covering this triple
                    # gain[blocks_covering_t_idx] -= 1
                    gain[tsub_to_blocks_np[t_idx]] -= 1

        if np.any(is_uncovered):
            # Incomplete cover
            return None

        return selected_blocks

    # 6. Local Search (Simulated Annealing)
    # We will improve the local search from the current program.
    # Key improvements:
    # 1. Use the precomputed tsub_to_blocks_np for fast "Add" moves.
    # 2. Implement "Swap" moves correctly.
    # 3. Use a better cooling schedule or restart.
    
    def local_search(selected_indices, time_limit):
        current = list(selected_indices)
        if not current:
            return current

        # Precompute cover_count
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b_idx in current:
            cover_count[block_tsubs_arr[b_idx]] += 1
            
        n_uncovered = np.sum(cover_count == 0)
        
        best = list(current)
        best_cost = len(current) + 1000 * n_uncovered
        
        T = 10.0
        cooling = 0.99
        min_T = 0.01
        rng = random.Random(42)
        
        # Precompute some structures for speed?
        # Maybe a set of current block indices for O(1) lookup?
        current_set = set(current)
        
        while T > min_T and time.time() < time_limit:
            move_type = rng.random()
            
            if move_type < 0.3:
                # Remove
                if len(current) == 0: continue
                b_idx = current[rng.randrange(len(current))]
                tsubs = block_tsubs_arr[b_idx]
                
                # Check if safe: does it cover any unique triple?
                # unique_triples = tsubs where cover_count == 1
                delta_uncovered = np.sum(cover_count[tsubs] == 1)
                
                if delta_uncovered == 0:
                    # Safe remove
                    cover_count[tsubs] -= 1
                    current.remove(b_idx)
                    current_set.remove(b_idx)
                    cost = len(current) + 1000 * n_uncovered
                    if cost < best_cost:
                        best_cost = cost
                        best = list(current)
                else:
                    # Unsafe remove, accept with probability
                    delta_cost = -1 + 1000 * delta_uncovered
                    if rng.random() < np.exp(-delta_cost / T):
                        cover_count[tsubs] -= 1
                        current.remove(b_idx)
                        current_set.remove(b_idx)
                        n_uncovered += delta_uncovered
                        cost = len(current) + 1000 * n_uncovered
                        if cost < best_cost:
                            best_cost = cost
                            best = list(current)

            elif move_type < 0.6:
                # Add
                if n_uncovered == 0:
                    continue
                # Pick a random uncovered triple
                uncovered_indices = np.where(cover_count == 0)[0]
                if len(uncovered_indices) == 0:
                    continue
                target_ts = uncovered_indices[rng.randrange(len(uncovered_indices))]
                
                # Get blocks covering this triple
                candidates = tsub_to_blocks_np[target_ts]
                if len(candidates) == 0:
                    continue
                new_b_idx = candidates[rng.randrange(len(candidates))]
                
                if new_b_idx in current_set:
                    continue
                
                tsubs = block_tsubs_arr[new_b_idx]
                # Gain: number of uncovered triples this block covers
                gain = np.sum(cover_count[tsubs] == 0)
                delta_cost = 1 - 1000 * gain
                if rng.random() < np.exp(-delta_cost / T):
                    cover_count[tsubs] += 1
                    current.append(new_b_idx)
                    current_set.add(new_b_idx)
                    n_uncovered -= gain
                    cost = len(current) + 1000 * n_uncovered
                    if cost < best_cost:
                        best_cost = cost
                        best = list(current)

            else:
                # Swap
                if len(current) == 0: continue
                old_b_idx = current[rng.randrange(len(current))]
                
                # Pick a new block
                if n_uncovered > 0:
                    uncovered_indices = np.where(cover_count == 0)[0]
                    if len(uncovered_indices) > 0:
                        target_ts = uncovered_indices[rng.randrange(len(uncovered_indices))]
                        candidates = tsub_to_blocks_np[target_ts]
                        if len(candidates) > 0:
                            new_b_idx = candidates[rng.randrange(len(candidates))]
                        else:
                            new_b_idx = rng.randrange(n_blocks)
                    else:
                        new_b_idx = rng.randrange(n_blocks)
                else:
                    new_b_idx = rng.randrange(n_blocks)
                    
                if new_b_idx == old_b_idx or new_b_idx in current_set:
                    continue
                    
                old_tsubs = block_tsubs_arr[old_b_idx]
                new_tsubs = block_tsubs_arr[new_b_idx]
                
                # Calculate delta in n_uncovered
                # We need to know how many triples go from 0->1 or 1->0.
                # Let's compute the change in cover_count for the union of old and new tsubs.
                
                # Temporarily remove old
                cover_count[old_tsubs] -= 1
                # Temporarily add new
                cover_count[new_tsubs] += 1
                
                # Calculate new n_uncovered for the affected triples
                # affected = unique(old_tsubs, new_tsubs)
                # This is expensive to compute unique every time.
                # Alternative: 
                # old_uncovered_in_old = sum(cover_count[old_tsubs] == 0) AFTER removal?
                # No, we need the change in global n_uncovered.
                
                # Let's just compute the cost directly?
                # Cost = len(current) + 1000 * n_uncovered
                # len(current) is unchanged.
                # So we just need the change in n_uncovered.
                
                # Let's count the number of triples that are now uncovered (cover_count == 0)
                # in the set of triples affected by this swap.
                # affected_tsubs = np.unique(np.concatenate([old_tsubs, new_tsubs]))
                # new_uncovered_in_affected = np.sum(cover_count[affected_tsubs] == 0)
                # old_uncovered_in_affected = ... we need to know the previous state.
                
                # This is getting complicated. Let's stick to Remove and Add for now, 
                # or implement Swap carefully.
                
                # Let's try a simpler Swap evaluation:
                # 1. Save old cover_count state for affected triples? No, too much memory.
                # 2. Just compute the new n_uncovered by scanning all n_tsubs? No, too slow.
                
                # Let's revert and skip Swap for now, as it's complex to implement efficiently.
                cover_count[new_tsubs] -= 1
                cover_count[old_tsubs] += 1
                pass

            T *= cooling

        # Final cleanup: remove redundant blocks from best
        # We need to be careful to maintain validity.
        # The best solution might have uncovered triples if it was accepted via SA.
        # We should only return best if it is valid (n_uncovered == 0).
        # But wait, best_cost includes the penalty. If best has uncovered triples, it's invalid.
        # We must ensure we return a valid cover.
        
        # Let's check if best is valid.
        # We can reconstruct cover_count for best and check.
        # But we lost the cover_count for best.
        
        # Better approach: Keep track of the best VALID solution.
        # If a move results in an invalid solution, we don't update best.
        # In the code above, we update best_cost and best regardless of validity,
        # but the cost includes a large penalty for uncovered triples.
        # So best should ideally be valid if a valid solution exists.
        # However, to be safe, let's verify best at the end.
        
        # Re-verify best
        cover_count_best = np.zeros(n_tsubs, dtype=np.int32)
        for b_idx in best:
            cover_count_best[block_tsubs_arr[b_idx]] += 1
        if np.any(cover_count_best == 0):
            # Best is invalid. Return the initial selected_indices if it was valid?
            # Or just return best and hope the penalty made it prefer valid ones?
            # No, we must return a valid cover.
            # Let's fall back to the initial selected_indices if it was valid.
            # But we don't know if it was valid.
            # Actually, greedy_set_cover_fast returns a valid cover.
            # So selected_indices is valid.
            # If local search breaks it, we should return selected_indices.
            # But local search is supposed to improve it.
            # Let's just return best if it's valid, otherwise return selected_indices.
            # Wait, the function signature returns a list of indices.
            # Let's check validity of selected_indices (the input).
            # It came from greedy, so it should be valid.
            # So if best is invalid, return selected_indices.
            return selected_indices
        else:
            return best

    # 7. Main Loop
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy strategies
    seeds = range(20)
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        
        # Alternate between weighted and unweighted?
        # The current program uses use_weighted based on seed.
        # Let's stick to unweighted for speed, as the gain calculation is simpler.
        # Or try both.
        use_weighted = (seed % 2 == 0)
        
        selected = greedy_set_cover_fast(seed, use_weighted)
        if selected is None:
            continue
            
        # Remove redundant
        # We can use the remove_redundant logic from the current program.
        # It's fast.
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_tsubs_arr[b_idx]] += 1
            
        new_selected = []
        for b_idx in selected:
            tsubs_in_block = block_tsubs_arr[b_idx]
            if np.any(cover_count[tsubs_in_block] == 1):
                new_selected.append(b_idx)
            else:
                cover_count[tsubs_in_block] -= 1
                
        selected = new_selected
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = list(selected)
            
        # Local search on this solution
        if time.time() < deadline - 5:
            improved = local_search(selected, deadline - 5)
            if len(improved) < best_len:
                best_len = len(improved)
                best_blocks = improved

    if best_blocks is None:
        # Fallback: simple greedy
        selected = greedy_set_cover_fast(0, False)
        if selected:
            best_blocks = selected
        else:
            best_blocks = []

    return [list(all_blocks[b_idx]) for b_idx in best_blocks]

# Need to import math for comb
import math
