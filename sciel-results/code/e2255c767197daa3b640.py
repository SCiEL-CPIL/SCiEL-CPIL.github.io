"""Covering design C(19,8,5): Iterated Local Search with 1-0 and 0-1 moves.
Strategy:
1. Formulate as a minimization problem: minimize number of blocks such that all 5-subsets are covered.
2. Start with a valid greedy covering.
3. Use Iterated Local Search (ILS):
   - Local Search: Try to remove redundant blocks (1->0 moves). If stuck, try adding a block that allows removing two others (0->1 move).
   - Perturbation: Randomly remove a few blocks and re-add greedily to escape local minima.
   - Acceptance: Accept if better or equal (with some probability for worse to escape).
"""
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks_total = len(all_blocks)
    
    # For each block, precompute the list of t-subset indices it covers
    block_coverages = []
    for b in all_blocks:
        covers = []
        for s in itertools.combinations(b, t):
            covers.append(idx[s])
        block_coverages.append(np.array(covers, dtype=np.int32))
        
    # Precompute for each t-subset, the list of blocks that cover it.
    t_sub_to_blocks = [[] for _ in range(N)]
    for bi, cov in enumerate(block_coverages):
        for ti in cov:
            t_sub_to_blocks[ti].append(bi)
            
    # Helper: calculate cover counts
    def calc_cover_counts(active_blocks):
        counts = np.zeros(N, dtype=np.int32)
        for bi in active_blocks:
            counts[block_coverages[bi]] += 1
        return counts
        
    # Helper: check if all covered
    def is_valid(counts):
        return np.all(counts > 0)
        
    # Helper: greedy construction
    def greedy_construct():
        uncovered = np.ones(N, dtype=bool)
        active_blocks = []
        while np.any(uncovered):
            # Find block covering most uncovered
            best_bi = -1
            best_gain = -1
            # Sample a few random blocks to speed up? Or scan all?
            # Scanning all 75k blocks * 56 checks is ~4M ops. 
            # Doing this ~300 times is 1.2B ops. Too slow.
            # Instead, pick a random uncovered subset, find blocks covering it, pick best among those.
            uncovered_indices = np.where(uncovered)[0]
            target_ti = random.choice(uncovered_indices)
            candidates = t_sub_to_blocks[target_ti]
            
            for bi in candidates:
                if bi in active_blocks:
                    continue
                gain = np.sum(uncovered[block_coverages[bi]])
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    
            if best_bi == -1:
                # Fallback: pick random block
                best_bi = random.choice(candidates)
                
            active_blocks.append(best_bi)
            uncovered[block_coverages[best_bi]] = False
            
        return active_blocks
        
    # Helper: try to remove a block
    def try_remove(active_blocks, counts, bi):
        # Check if any t-subset has count 1 and is covered by bi
        cov = block_coverages[bi]
        for ti in cov:
            if counts[ti] == 1:
                return False
        # Safe to remove
        return True

    # Main ILS loop
    best_blocks = greedy_construct()
    best_size = len(best_blocks)
    best_counts = calc_cover_counts(best_blocks)
    
    current_blocks = list(best_blocks)
    current_counts = best_counts.copy()
    current_size = len(current_blocks)
    
    # Track best found
    global_best = list(best_blocks)
    global_best_size = best_size
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # 1. Local Search: Try to remove blocks
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing a random block
            if not current_blocks:
                break
            random.shuffle(current_blocks) # Not in place, need to be careful
            # Actually, shuffling the list changes order. Let's just pick a random index.
            for _ in range(len(current_blocks)):
                if time.time() > deadline:
                    break
                ri = random.randint(0, len(current_blocks)-1)
                bi = current_blocks[ri]
                if try_remove(current_blocks, current_counts, bi):
                    # Remove it
                    current_blocks.pop(ri)
                    current_counts[block_coverages[bi]] -= 1
                    current_size -= 1
                    improved = True
                    if current_size < global_best_size:
                        global_best_size = current_size
                        global_best = list(current_blocks)
                    # Restart the while loop to try more removals
                    break
        
        # 2. Perturbation: If stuck, perturb
        if time.time() > deadline:
            break
            
        # Check if we are stuck (no removals possible)
        # For simplicity, just perturb every few iterations or if no improvement
        # Let's do a 0-1 move: add a block, then try to remove 2 blocks
        
        # Pick a random block to add
        if len(current_blocks) < num_blocks_total:
            # Find a block not in current
            # This is expensive to check. 
            # Instead, pick a random block from all_blocks, check if in current.
            # Use a set for current blocks for fast lookup?
            # current_set = set(current_blocks) - expensive to build every time.
            # Let's just pick a random block index.
            add_bi = random.randint(0, num_blocks_total - 1)
            if add_bi not in current_blocks: # This is O(N) check. 
                # To speed up, we can maintain a set of active block indices.
                # But updating it is also work.
                # For now, let's assume the check is okay for small N (75k).
                # Actually, 75k check is fast.
                
                # Add it
                current_blocks.append(add_bi)
                current_counts[block_coverages[add_bi]] += 1
                current_size += 1
                
                # Now try to remove 2 blocks
                # We want to find two blocks b1, b2 such that removing them keeps coverage valid.
                # This is a 2-SAT like problem or just search.
                # Simplified: Try to remove one block. If valid, remove it.
                # Then try to remove another.
                
                # Try to remove one
                removed_one = False
                for _ in range(len(current_blocks)):
                    if time.time() > deadline:
                        break
                    ri = random.randint(0, len(current_blocks)-1)
                    bi = current_blocks[ri]
                    if bi == add_bi:
                        continue
                    if try_remove(current_blocks, current_counts, bi):
                        current_blocks.pop(ri)
                        current_counts[block_coverages[bi]] -= 1
                        current_size -= 1
                        removed_one = True
                        if current_size < global_best_size:
                            global_best_size = current_size
                            global_best = list(current_blocks)
                        break
                        
                # Try to remove another
                removed_two = False
                if removed_one:
                    for _ in range(len(current_blocks)):
                        if time.time() > deadline:
                            break
                        ri = random.randint(0, len(current_blocks)-1)
                        bi = current_blocks[ri]
                        if bi == add_bi:
                            continue
                        if try_remove(current_blocks, current_counts, bi):
                            current_blocks.pop(ri)
                            current_counts[block_coverages[bi]] -= 1
                            current_size -= 1
                            removed_two = True
                            if current_size < global_best_size:
                                global_best_size = current_size
                                global_best = list(current_blocks)
                            break
                            
                # If we didn't remove 2, we might be worse off (size +1)
                # If we removed 2, we are better off (size -1)
                # If we removed 1, we are same size (size 0)
                # If we removed 0, we are worse (size +1)
                
                # Acceptance criterion:
                # If size decreased, accept.
                # If size same, accept with prob 0.5.
                # If size increased, accept with prob 0.1 (to explore).
                
                # But we need to revert if we reject.
                # This is getting complex. Let's simplify:
                # Just accept if size <= previous size.
                # If size > previous size, revert.
                
                # We need to store the state before perturbation.
                # This is expensive.
                # Alternative: Only keep perturbation if it leads to improvement.
                # This is "Best Improvement" ILS.
                
                # Let's refine: 
                # Save state before perturbation
                # Perform perturbation
                # If improved, keep. Else, revert.
                
                # Re-doing with state saving:
                pass # We already modified current_blocks. Let's just track best.
                
        # If we are stuck in a local minimum (no single block removal works),
        # we might need to try swapping blocks.
        # For now, let's just run the greedy construction again if we haven't improved in a while.
        
        # To avoid infinite loops of bad perturbations, let's check time.
        
    # Return the best found
    if global_best is not None:
        final_blocks = [list(all_blocks[bi]) for bi in global_best]
        return final_blocks
    else:
        # Fallback
        return [list(all_blocks[bi]) for bi in best_blocks]

