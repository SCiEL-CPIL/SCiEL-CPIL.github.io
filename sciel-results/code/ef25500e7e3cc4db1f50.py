import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Deadline guard: 115s to be safe within 120s limit
    deadline = start_time + 115.0
    
    # 1. Precomputation
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubsets)
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubsets)}
    
    # Generate all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage mask for each block
    # mask[b] is an integer where bit i is 1 if block b covers t-subset i
    block_masks = [0] * num_blocks
    bit_vals = [1 << i for i in range(num_tsubs)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for tsub in itertools.combinations(block, t):
            idx = tsub_to_idx[tsub]
            mask |= bit_vals[idx]
        block_masks[b_idx] = mask
        
    # Full mask representing all t-subsets
    full_mask = (1 << num_tsubs) - 1
    
    # 2. Construction Phase: Weighted Greedy
    # Instead of pure greedy, we use a stochastic selection among top candidates
    # to avoid deterministic local optima.
    
    covered_mask = 0
    used_blocks = []
    
    # We will run a few restarts if time permits, but first let's do one strong run.
    # Actually, given the time limit, a single run with good local search is better.
    
    while covered_mask != full_mask:
        # Calculate gains for all blocks
        # Optimization: We only need to consider blocks that add *some* new coverage.
        # We can filter out blocks with gain 0.
        
        # To speed this up, we maintain a list of "active" blocks that might have gain > 0.
        # However, since we start with all blocks, and gain changes as covered_mask grows,
        # we just scan all blocks. 34610 blocks is small.
        
        gains = [0] * num_blocks
        candidates = []
        
        # Vectorized calculation using numpy might be faster for the AND and popcount?
        # Python big ints are quite fast, but let's stick to Python for simplicity unless slow.
        # Let's try a hybrid: use Python loops but optimize the inner check.
        
        max_gain = 0
        for b_idx in range(num_blocks):
            new_bits = block_masks[b_idx] & ~covered_mask
            gain = new_bits.bit_count()
            if gain > 0:
                gains[b_idx] = gain
                if gain > max_gain:
                    max_gain = gain
                candidates.append(b_idx)
        
        if not candidates:
            break # Should not happen if solution exists
            
        # Stochastic Selection:
        # Pick a block from the top 'top_k' candidates with probability proportional to gain?
        # Or just pick from the top 10 with some randomness.
        # Let's sort candidates by gain descending and pick from the top 20.
        # This allows escaping the single best greedy choice if it leads to a dead end later.
        
        candidates.sort(key=lambda x: gains[x], reverse=True)
        
        # Pick from top 10 or 20
        top_n = min(20, len(candidates))
        top_candidates = candidates[:top_n]
        
        # Weighted random choice based on gain?
        # Simpler: Uniform random from top candidates.
        # This introduces diversity.
        selected_idx = random.choice(top_candidates)
        
        used_blocks.append(selected_idx)
        covered_mask |= block_masks[selected_idx]
        
        if time.time() > deadline - 30.0:
            # If we are running low on time during construction, stop and go to search
            # But we must ensure validity. If we stop early, we have an invalid cover.
            # So we must finish construction.
            # 34610 * 76 iterations is fast. This check is mostly for safety.
            pass

    # 3. Local Search Phase: Swap/Exchange
    # Goal: Minimize len(used_blocks).
    # Move: Remove block i, Add block j.
    # Condition: The new set (used_blocks - {i} + {j}) is still a valid cover.
    # This reduces count by 1 if j was not in used_blocks? No, count stays same.
    # Wait. If we remove i and add j, count is same.
    # We need to find a sequence of moves that leads to a state where a block is redundant.
    # Or, simpler: 
    # 1. Try to remove any redundant block (as before).
    # 2. If no redundant block, try to "improve" the set by swapping a block with another 
    #    that allows a *different* block to become redundant?
    
    # Actually, the standard "Local Search for Set Cover" works like this:
    # State: S (set of blocks).
    # Move: Remove block b from S. If S \ {b} is still a cover, accept (count decreases).
    # If not, try to add a block b' to S \ {b} such that (S \ {b} + {b'}) is a cover 
    #    AND b' is "better" than b in some sense? 
    #    Actually, if we just swap, count is same. But it might open up a redundancy for a 3rd block.
    
    # Let's use a 2-step move:
    # 1. Pick a block b to remove.
    # 2. If S \ {b} is valid, remove it.
    # 3. If not, find a block b' not in S such that (S \ {b} + {b'}) is valid.
    #    If found, swap b for b'. This doesn't reduce count immediately, but changes the configuration.
    #    We can then retry step 1 with the new configuration.
    #    To avoid infinite loops, we can limit swaps or use a "tabu" list.
    
    # However, a simpler and very effective heuristic for small instances is:
    # "Randomized Restarted Greedy with Pruning"
    # Since the instance is small (C(24,7,3)), we can afford multiple runs.
    
    best_count = len(used_blocks)
    best_blocks = list(used_blocks)
    
    # Time remaining for local search / restarts
    time_left = deadline - time.time()
    
    # We will perform local search on the current solution first.
    
    def is_valid_cover(block_indices):
        union = 0
        for idx in block_indices:
            union |= block_masks[idx]
        return union == full_mask

    # Local Search Loop
    current_blocks = list(used_blocks)
    no_improvement_count = 0
    max_no_improvement = 50 # Limit search time
    
    while time.time() < deadline - 10.0 and no_improvement_count < max_no_improvement:
        improved = False
        
        # 1. Try to remove redundant blocks
        # Iterate through blocks in random order
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if i >= len(current_blocks):
                continue
            
            block_to_remove = current_blocks[i]
            
            # Check if removing it breaks coverage
            # Calculate union of others
            union_others = 0
            for j in range(len(current_blocks)):
                if j != i:
                    union_others |= block_masks[current_blocks[j]]
            
            if (block_masks[block_to_remove] & ~union_others) == 0:
                # Redundant
                del current_blocks[i]
                improved = True
                no_improvement_count = 0
                break # Restart loop as indices are stale
        
        if improved:
            continue
            
        # 2. If no redundant block found, try a SWAP
        # Pick a random block to remove.
        # Find a block to add that restores coverage and potentially enables future removals.
        
        if len(current_blocks) > 1:
            i = random.randint(0, len(current_blocks) - 1)
            block_to_remove = current_blocks[i]
            
            # What is the "hole" created by removing block_to_remove?
            # hole = mask[block_to_remove] & ~union_others
            union_others = 0
            for j in range(len(current_blocks)):
                if j != i:
                    union_others |= block_masks[current_blocks[j]]
            
            hole = block_masks[block_to_remove] & ~union_others
            
            if hole != 0:
                # We need to find a block b' not in current_blocks such that 
                # mask[b'] covers 'hole'.
                # To be a "good" swap, b' should ideally cover parts of other blocks too, 
                # making them potentially redundant.
                
                # Score candidates by how much of the hole they cover + how much new overlap they create?
                # Let's just find any block that covers the hole.
                # To speed up, we can precompute which blocks cover which t-subsets?
                # Or just scan.
                
                # Optimization: 
                # We only care about blocks that intersect the hole.
                # hole is a bitmask.
                
                best_swap = -1
                best_score = -1
                
                # We can limit the search to blocks that are "similar" or just random sample?
                # Scanning all 34610 blocks is fast enough.
                
                # To avoid adding a block that is already in the set (which is a no-op for coverage 
                # but might be a different block object? No, indices are unique).
                
                # We want to swap block_to_remove with a block b'.
                # Condition: mask[b'] covers 'hole'.
                # Preference: b' should cover many t-subsets that are *only* covered by block_to_remove 
                # in the current set? No, that's the hole.
                # Preference: b' should help make *other* blocks redundant.
                # Metric: Count of t-subsets in mask[b'] that are covered by exactly 1 other block in 
                # (current_blocks - {i}). If we add b', those subsets will be covered by 2 blocks.
                # This doesn't directly help.
                
                # Simpler heuristic: Pick a block that covers the hole and has high total overlap 
                # with the rest of the set, hoping to create redundancy.
                
                # Let's just pick a random valid swap to break symmetry.
                # Find all valid swaps.
                valid_swaps = []
                
                # To speed this up, we can precompute a list of blocks for each t-subset?
                # No, let's just scan.
                
                # Optimization: If hole has few bits, we can check blocks containing those t-subsets.
                # But hole is a bitmask.
                
                # Let's just scan all blocks not in current_blocks.
                # To avoid O(N^2) overhead, we can use a set of current indices for O(1) lookup.
                current_set = set(current_blocks)
                
                # We will sample a subset of blocks to speed up if N is large.
                # But N=34610 is small.
                
                for b_idx in range(num_blocks):
                    if b_idx in current_set:
                        continue
                    
                    # Does b_idx cover the hole?
                    if (block_masks[b_idx] & hole) == hole:
                        # Valid swap candidate.
                        # Score: How many new redundancies does this create?
                        # This is expensive to calculate exactly.
                        # Approximation: Prefer blocks with high overlap with current set.
                        overlap = (block_masks[b_idx] & union_others).bit_count()
                        # We want high overlap to make other blocks redundant?
                        # Actually, if b' covers things covered by others, it might allow removal of others.
                        # Let's just add to valid_swaps.
                        valid_swaps.append((b_idx, overlap))
                        
                        # Early exit if we find a very good one?
                        # For now, collect a few.
                        if len(valid_swaps) > 50:
                            break
                
                if valid_swaps:
                    # Pick one with highest overlap (most likely to enable redundancy)
                    valid_swaps.sort(key=lambda x: x[1], reverse=True)
                    new_block_idx = valid_swaps[0][0]
                    
                    # Perform swap
                    current_blocks[i] = new_block_idx
                    improved = True # Count as an improvement in search progress
                    no_improvement_count = 0
                    continue
        
        no_improvement_count += 1
        
        # If we've tried many swaps without reduction, break and maybe restart?
        # For now, just break the local search and keep the best.
        
    # Update best if current is better
    if len(current_blocks) < best_count:
        best_count = len(current_blocks)
        best_blocks = list(current_blocks)
    
    # If time remains, we could do another restart of the whole construction with different random seed.
    # Given the complexity, one strong run with local search is likely sufficient for this improvement.
    # But let's add a quick restart loop if time > 20s remains.
    
    while time.time() < deadline - 15.0:
        # Restart construction
        covered_mask = 0
        used_blocks = []
        
        # Simple greedy for speed in restarts, or slightly randomized
        while covered_mask != full_mask:
            max_gain = 0
            best_idx = -1
            # Randomized greedy: pick from top 5
            candidates = []
            for b_idx in range(num_blocks):
                new_bits = block_masks[b_idx] & ~covered_mask
                gain = new_bits.bit_count()
                if gain > 0:
                    candidates.append((gain, b_idx))
                    if gain > max_gain:
                        max_gain = gain
            
            if not candidates:
                break
                
            candidates.sort(reverse=True)
            top_n = min(5, len(candidates))
            # Pick from top 5
            _, selected_idx = random.choice(candidates[:top_n])
            
            used_blocks.append(selected_idx)
            covered_mask |= block_masks[selected_idx]
            
            if time.time() > deadline - 15.0:
                break
        
        # Quick local search on this new solution
        if len(used_blocks) < best_count:
            # Potential improvement, do a quick prune
            current_blocks = list(used_blocks)
            # Simple redundancy removal
            changed = True
            while changed and time.time() < deadline - 15.0:
                changed = False
                indices = list(range(len(current_blocks)))
                random.shuffle(indices)
                for i in indices:
                    if i >= len(current_blocks): continue
                    block_to_remove = current_blocks[i]
                    union_others = 0
                    for j in range(len(current_blocks)):
                        if j != i:
                            union_others |= block_masks[current_blocks[j]]
                    if (block_masks[block_to_remove] & ~union_others) == 0:
                        del current_blocks[i]
                        changed = True
                        break
            if len(current_blocks) < best_count:
                best_count = len(current_blocks)
                best_blocks = list(current_blocks)

    # Final Validation
    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    
    # Verify validity
    union = 0
    for i in best_blocks:
        union |= block_masks[i]
        
    if union != full_mask:
        # Fallback to the first solution if it was valid
        union1 = 0
        for i in used_blocks:
            union1 |= block_masks[i]
        if union1 == full_mask:
            final_blocks = [list(all_blocks[i]) for i in used_blocks]
        else:
            # Critical error: return a trivial valid cover (all blocks? No, too many)
            # Return the first 76 blocks? 
            # We must return a valid cover.
            # Since greedy usually works, this is a safety net.
            # We'll just return the used_blocks from the first run and hope it's valid.
            # If not, we fail. But the code logic ensures used_blocks is a cover 
            # if the loop finished.
            pass

    return final_blocks

if __name__ == "__main__":
    blocks = construct(24, 7, 3)
    print(f"Number of blocks: {len(blocks)}")
    all_tsubs = set(itertools.combinations(range(24), 3))
    covered = set()
    for b in blocks:
        for t in itertools.combinations(b, 3):
            covered.add(t)
    if all_tsubs == covered:
        print("Valid Cover")
    else:
        print(f"Invalid! Missing {len(all_tsubs - covered)} subsets")
