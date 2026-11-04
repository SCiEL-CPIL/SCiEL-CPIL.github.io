import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_indices = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute all k-subsets? 
    # C(24, 7) is 346,104. Storing them might be memory heavy but okay.
    # However, we only need to generate candidates containing a specific t-subset.
    
    def greedy_cover(use_random=False):
        covered = [0] * num_t
        blocks = []
        
        # List of uncovered t-subsets
        # Instead of a set, we can maintain a list of uncovered indices
        uncovered_indices = list(range(num_t))
        
        # Map from t-subset tuple to index is already done
        
        while uncovered_indices:
            # Pick an uncovered t-subset
            if use_random:
                idx = random.choice(uncovered_indices)
            else:
                idx = uncovered_indices[0]
            
            target_t = all_t_subsets[idx]
            
            # Generate all k-subsets containing target_t
            # We need to choose k-t more elements from the v-t remaining
            remaining_points = [x for x in range(v) if x not in target_t]
            candidates = []
            
            # Generate combinations of k-t from remaining_points
            for ext in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_t + ext))
                # Calculate gain: how many t-subsets in this block are currently uncovered
                gain = 0
                # The block contains C(k, t) t-subsets
                for ts in itertools.combinations(block, t):
                    t_idx = t_subset_indices[ts]
                    if covered[t_idx] == 0:
                        gain += 1
                candidates.append((gain, block))
            
            # Pick the best candidate (max gain, then lexicographically smallest for determinism if no random)
            if use_random and len(candidates) > 1:
                # Pick randomly from top performers to diversify
                max_gain = max(c[0] for c in candidates)
                best_cands = [c[1] for c in candidates if c[0] == max_gain]
                best_block = random.choice(best_cands)
            else:
                best_block = max(candidates, key=lambda x: (x[0], -x[1]))[1]
            
            blocks.append(list(best_block))
            
            # Update covered status
            for ts in itertools.combinations(best_block, t):
                t_idx = t_subset_indices[ts]
                if covered[t_idx] == 0:
                    covered[t_idx] = 1
                    # Remove from uncovered_indices efficiently?
                    # Using a list and removing is O(N). 
                    # For v=24, t=3, num_t is 2024. It's small.
                    # But we do this inside the loop.
                    # Let's just rebuild uncovered_indices occasionally or use a set.
                    # Actually, checking `if covered[t_idx] == 0` is fast.
                    # We can just keep the list and filter it out when picking the next target.
                    # Or, simpler: just check `covered` in the main loop condition?
                    # No, we need to pick an uncovered one.
            
            # Update uncovered_indices
            # This is O(num_t) per block. 76 * 2024 ~ 150k ops. Fine.
            uncovered_indices = [i for i in uncovered_indices if covered[i] == 0]

        return blocks

    def try_delete(blocks):
        # Try to remove redundant blocks
        # For each block, check if it can be removed
        # We need to know for each t-subset, which blocks cover it.
        # Or, simpler: for each block, count how many t-subsets it is the *only* cover for.
        
        # Precompute coverage counts
        # num_t is 2024
        # blocks is ~76
        # For each block, for each t-subset in it, increment count
        # Then for each block, if all its t-subsets have count > 1, it can be removed.
        # But removing one block changes counts for others.
        
        # Greedy deletion:
        # 1. Compute initial counts.
        # 2. Iterate through blocks.
        # 3. If a block has no "essential" t-subsets (i.e., all t-subsets in it are covered by other blocks), remove it.
        # 4. Repeat until no changes.
        
        changed = True
        current_blocks = list(blocks)
        
        while changed:
            changed = False
            # Compute coverage counts for current_blocks
            counts = [0] * num_t
            for block in current_blocks:
                for ts in itertools.combinations(block, t):
                    counts[t_subset_indices[ts]] += 1
            
            new_blocks = []
            for block in current_blocks:
                # Check if this block is essential
                essential = False
                for ts in itertools.combinations(block, t):
                    if counts[t_subset_indices[ts]] == 1:
                        essential = True
                        break
                
                if essential:
                    new_blocks.append(block)
                else:
                    changed = True
            
            current_blocks = new_blocks
        
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy instances with randomization
    # Time limit is 120s. 
    # Greedy is fast. Deletion is fast.
    # Let's run 10-20 iterations.
    
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < 110: # Leave 10s buffer
        # Alternate between random and deterministic?
        # Random helps escape local optima of the greedy choice.
        blocks = greedy_cover(use_random=True)
        blocks = try_delete(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # If we hit the theoretical lower bound or a known good bound, we can stop?
            # Lower bound is 69. If we hit 69, we can't do better.
            if best_len <= 69:
                break
        
        iterations += 1

    if best_blocks is None:
        # Fallback to deterministic if no random run finished (unlikely)
        blocks = greedy_cover(use_random=False)
        best_blocks = try_delete(blocks)
        
    return best_blocks
