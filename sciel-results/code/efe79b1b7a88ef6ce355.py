import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Map each block to its t-subsets
    block_tsubs = []
    for b in all_blocks:
        tsubs = [tsub_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_tsubs.append(tsubs)
        
    # Map each t-subset to the blocks that contain it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
            
    best_solution = None
    best_len = float('inf')
    
    # Run multiple randomized greedy attempts
    # The randomness comes from tie-breaking in the max selection
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        freq = [0] * num_tsubs
        score = [len(block_tsubs[0])] * num_blocks # Initially all cover C(k,t)
        used = [False] * num_blocks
        solution = []
        
        # Greedy loop
        # Optimization: Instead of finding max every time from scratch (O(N)), 
        # we can use a heap, but scores decrease. 
        # Given N=50388, O(N) scan is ~0.005s. 150 steps -> 0.75s. 
        # This is acceptable for 10-20 runs.
        
        covered_count = 0
        while covered_count < num_tsubs:
            # Find block with max score among unused
            max_score = -1
            best_block = -1
            candidates = []
            
            # To allow randomization, we can collect top candidates
            # Or just find max and if ties, pick random
            # For speed, we just find max. Ties are rare if scores are distinct enough, 
            # but initially many are 35.
            
            for i in range(num_blocks):
                if not used[i] and score[i] > max_score:
                    max_score = score[i]
                    best_block = i
                    candidates = [i]
                elif not used[i] and score[i] == max_score:
                    candidates.append(i)
            
            if best_block == -1:
                break # Should not happen if valid input
                
            # Randomly pick from candidates to break ties
            b = random.choice(candidates)
            
            used[b] = True
            solution.append(b)
            
            # Update scores
            for s in block_tsubs[b]:
                if freq[s] == 0:
                    freq[s] = 1
                    covered_count += 1
                    # Decrease score of all other blocks covering s
                    for b_prime in tsub_to_blocks[s]:
                        if not used[b_prime]:
                            score[b_prime] -= 1
                            
        if len(solution) < best_len:
            best_len = len(solution)
            best_solution = solution[:]
            
        # If we are getting good results, we might want to stop early or continue
        # If best_len is close to known lower bound (131), we might stop.
        # But let's keep trying to beat the current best.
        
    # Post-processing: Remove redundant blocks
    if best_solution:
        # Convert solution indices to actual blocks
        final_blocks = [all_blocks[i] for i in best_solution]
        
        # Try to remove blocks
        # We need to check if a block is essential.
        # A block is redundant if all its t-subsets are covered by other blocks.
        
        # Build coverage count for the final solution
        # This is a separate pass to be safe and allow removal
        
        # Re-initialize freq for final solution
        freq_final = [0] * num_tsubs
        for b_idx in best_solution:
            for s in block_tsubs[b_idx]:
                freq_final[s] += 1
                
        new_solution = []
        # Sort by "rarity" or just iterate? 
        # Removing a block might uncover subsets. 
        # We can try to remove blocks one by one in a random order.
        
        indices_to_check = list(range(len(best_solution)))
        random.shuffle(indices_to_check)
        
        for idx in indices_to_check:
            b_idx = best_solution[idx]
            # Check if removing this block leaves all its t-subsets covered
            can_remove = True
            for s in block_tsubs[b_idx]:
                if freq_final[s] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove it
                for s in block_tsubs[b_idx]:
                    freq_final[s] -= 1
                # We don't remove from best_solution list immediately to keep indices stable, 
                # but we mark it as removed. 
                # Easier: build a set of removed indices
                # Let's just build a new list at the end.
                # We need to track which ones are kept.
                # Let's use a boolean array `keep`
                if not hasattr(construct, '_keep'): # This is messy in a function
                    pass
                
        # Let's do a cleaner removal pass
        keep = [True] * len(best_solution)
        # We need to be careful: removing one block might make another redundant.
        # So we iterate and check.
        
        # Actually, a single pass is not enough. 
        # Let's do a loop until no more can be removed.
        changed = True
        while changed:
            changed = False
            for i in range(len(best_solution)):
                if not keep[i]:
                    continue
                b_idx = best_solution[i]
                can_remove = True
                for s in block_tsubs[b_idx]:
                    if freq_final[s] == 1:
                        can_remove = False
                        break
                if can_remove:
                    keep[i] = False
                    changed = True
                    for s in block_tsubs[b_idx]:
                        freq_final[s] -= 1
                    
        final_blocks = [all_blocks[best_solution[i]] for i in range(len(best_solution)) if keep[i]]
        
        return final_blocks
        
    return []
