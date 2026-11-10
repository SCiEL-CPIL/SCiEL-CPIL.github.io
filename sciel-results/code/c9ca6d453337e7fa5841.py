# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Enumerate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage masks for all k-subsets
    # mask[i] is a bitmask where bit j is 1 if t-subset j is covered by k-subset i
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Optimization: Only consider k-subsets that cover at least some t-subsets.
    # Actually, all k-subsets cover C(k,t) t-subsets.
    # We store masks as Python integers (arbitrary precision).
    # n_t is 2024, so we need ~2024 bits. Python ints handle this fine.
    
    # Precomputing all masks might be memory heavy if done naively, 
    # but 346k * 2024 bits is ~85MB. This is acceptable.
    # However, generating them takes time. 
    # Let's generate on the fly or precompute efficiently.
    
    # Efficient precomputation:
    # For each k-subset, find all t-subsets within it.
    # A k-subset covers C(k,t) t-subsets.
    # C(7,3) = 35.
    # So each mask has 35 bits set.
    
    masks = [0] * n_k
    # To speed up, we can just iterate all k-subsets and set bits.
    for i, k_sub in enumerate(all_k_subsets):
        m = 0
        # Get all t-subsets of this k-subset
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            m |= (1 << idx)
        masks[i] = m

    # 3. Greedy Set Cover
    # uncovered_mask starts with all bits set (all t-subsets uncovered)
    uncovered = (1 << n_t) - 1
    selected_blocks = []
    
    # We need to pick blocks that maximize popcount(uncovered & mask)
    # Naive greedy: O(n_k) per step.
    # n_steps is roughly n_t / C(k,t) ~ 2024/35 ~ 58 blocks.
    # 58 * 346k operations = 20M bit ops. Very fast in Python if optimized?
    # Python int bitwise ops are C-speed. popcount is C-speed.
    # 20M iterations of Python loop might be slow (~10-20s).
    # We can improve by using a priority queue or just scanning.
    # Let's try simple scanning first.
    
    # To allow randomization and multiple restarts, we can shuffle the order of candidates
    # or use a randomized tie-breaking.
    
    best_solution = None
    best_len = float('inf')
    
    # We will run a few randomized greedy attempts
    # Randomness in: order of checking candidates, or tie-breaking.
    
    for attempt in range(50):
        if time.time() > deadline - 20:
            break
            
        rng = random.Random(attempt * 123 + 456)
        
        # Shuffle indices to randomize tie-breaking / exploration
        # But we need to find the max gain.
        # A common trick: add small random noise to gains? No, greedy needs max.
        # Instead, we can randomize the order in which we *evaluate* candidates?
        # No, we must find the global max gain for greedy to be standard.
        # However, we can randomize the *initial* set of blocks or use a "lazy" greedy.
        
        # Let's stick to standard greedy but with a randomized "order" of blocks in the list
        # so that in case of ties, different blocks might be picked if we just pick the first one found?
        # No, we should pick the true max.
        # To get diversity, we can perturb the gains slightly?
        # Or just run the same greedy? It's deterministic.
        # Let's try: Randomly remove some "bad" blocks from consideration?
        # Or: Start with a random valid partial cover?
        
        # Simpler: Just run the greedy once. It's very fast.
        # Then apply local search.
        
        current_uncovered = uncovered
        selected = []
        
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            
            # Scan all k-subsets
            # Optimization: If a block is already selected, skip? 
            # No, a block can only be selected once. We should mark used blocks.
            # But we are just picking indices.
            
            # To make it faster, we can maintain a list of available blocks?
            # For now, just scan all.
            
            for i in range(n_k):
                # If we don't track used, we might pick the same block twice.
                # But if we pick it, its contribution to *new* coverage becomes 0.
                # So `current_uncovered & masks[i]` will be 0 for used blocks 
                # only if we update `current_uncovered` correctly?
                # Yes, if we pick block i, we do `current_uncovered &= ~masks[i]`.
                # Then in the next iteration, `current_uncovered & masks[i]` is 0.
                # So we don't need to track used blocks explicitly if we update the mask.
                
                gain_mask = current_uncovered & masks[i]
                # popcount
                gain = gain_mask.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_mask = gain_mask
                elif gain == best_gain and gain > 0:
                    # Random tie-break
                    if rng.random() < 0.1:
                        best_gain = gain
                        best_idx = i
                        best_mask = gain_mask
            
            if best_idx == -1:
                # Should not happen if problem is solvable
                break
                
            selected.append(best_idx)
            current_uncovered &= ~best_mask
            
            if time.time() > deadline - 20:
                break
        
        # Check if valid
        if current_uncovered == 0:
            # Convert indices to actual blocks
            blocks = [list(all_k_subsets[i]) for i in selected]
            
            # Local Search: Try to remove redundant blocks
            # Sort by index or something to stabilize?
            # We can try to remove a block if all its t-subsets are covered by others.
            
            # Coverage count per t-subset
            # This is expensive to compute from scratch every time.
            # Let's do a simple pass:
            # For each block, check if it's essential.
            
            # To speed up, we can just return the greedy result if it's good enough.
            # But local search can reduce count significantly.
            
            if len(selected) < best_len:
                best_len = len(selected)
                best_solution = blocks
            
            # Try local search with limited time
            if time.time() < deadline - 5:
                # Simple local search: iterate through blocks, try to remove
                # If removing block b leaves all its t-subsets covered by other blocks, remove it.
                
                # Precompute coverage masks for the selected blocks
                # cov_mask[t_idx] is a bitmask of which selected blocks cover it?
                # Or simpler: for each t-subset, count how many blocks cover it.
                
                # Let's use a "coverage count" array.
                # n_t = 2024.
                # cov_counts = [0] * n_t
                # for b in selected_blocks:
                #    for ts in combinations(b, t):
                #       cov_counts[t_to_idx[ts]] += 1
                #
                # To check if block b is removable:
                #   for ts in combinations(b, t):
                #      if cov_counts[t_to_idx[ts]] == 1:
                #          return False # b is essential
                #   return True
                #
                # This is O(C(k,t)) per check.
                # Total O(n_blocks * C(k,t)).
                # n_blocks ~ 60. C(7,3)=35. 2100 ops. Very fast.
                
                # We can do multiple passes.
                
                # Let's implement a quick local search
                # We need the actual blocks to compute t-subsets.
                # We have `blocks` list.
                
                # To avoid recomputing t-subsets for every block every time,
                # we can precompute the list of t-subset indices for each selected block.
                
                block_t_indices = []
                for b in blocks:
                    indices = []
                    for ts in itertools.combinations(b, t):
                        indices.append(t_to_idx[ts])
                    block_t_indices.append(indices)
                
                # Initialize cov_counts
                cov_counts = [0] * n_t
                for indices in block_t_indices:
                    for idx in indices:
                        cov_counts[idx] += 1
                
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    # Try to remove blocks in reverse order (newer ones might be less essential?)
                    # Or random order.
                    order = list(range(len(blocks)))
                    rng.shuffle(order)
                    
                    for bi in order:
                        if bi >= len(blocks):
                            continue
                        
                        # Check if block bi is redundant
                        redundant = True
                        for idx in block_t_indices[bi]:
                            if cov_counts[idx] == 1:
                                redundant = False
                                break
                        
                        if redundant:
                            # Remove block bi
                            for idx in block_t_indices[bi]:
                                cov_counts[idx] -= 1
                            
                            blocks.pop(bi)
                            block_t_indices.pop(bi)
                            
                            # Adjust order? We are iterating over a snapshot of indices.
                            # If we pop, indices shift.
                            # It's safer to mark for removal and clean up later,
                            # or iterate backwards if we don't shuffle.
                            # Since we shuffled, indices are mixed.
                            # Let's just break and restart the pass? Or handle carefully.
                            # Simpler: collect indices to remove.
                            
                            # Actually, popping changes indices.
                            # Let's just restart the loop if we remove one.
                            improved = True
                            break # Restart the while loop with new list
                
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_solution = blocks

    if best_solution is None:
        # Fallback: Return a valid greedy cover (should have been found in first attempt)
        # If not, try a very slow but sure method?
        # For now, return empty or raise?
        # The task says "Valid when every t-subset...".
        # If we failed to find a valid one, we return the best we have (invalid?)
        # No, we must return a valid one.
        # The greedy should always find one.
        # If it didn't, it's a bug or timeout.
        
        # Let's force a valid cover using a simpler, slower method if needed
        # But given the speed, it should be fine.
        
        # Re-run greedy without local search, strictly
        current_uncovered = uncovered
        selected = []
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            for i in range(n_k):
                gain = (current_uncovered & masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            current_uncovered &= ~masks[best_idx]
        
        best_solution = [list(all_k_subsets[i]) for i in selected]
        
    return best_solution
# EVOLVE-BLOCK-END
