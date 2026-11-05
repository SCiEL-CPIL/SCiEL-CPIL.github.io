import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precomputation
    # 1. All t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_subset_index = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # 2. All k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    
    # 3. Map each block to the indices of t-subsets it covers
    block_to_t_indices = []
    for block in all_k_blocks:
        indices = []
        for sub in itertools.combinations(block, t):
            indices.append(t_subset_index[sub])
        block_to_t_indices.append(indices)
        
    # 4. Map each t-subset to the list of block indices covering it
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, indices in enumerate(block_to_t_indices):
        for t_idx in indices:
            t_to_blocks[t_idx].append(b_idx)

    def greedy_cover(seed):
        random.seed(seed)
        uncovered = set(range(num_t))
        cover = []
        while uncovered:
            best_block = -1
            max_covered = 0
            # To speed up, we can sample or use a heuristic. 
            # With num_k=75k and num_t=11k, iterating all blocks each step might be slow if many steps.
            # But greedy usually takes ~200-300 steps. 300 * 75k * 56 ops ~ 1.2 billion ops? Too slow in Python.
            
            # Optimization: Instead of checking all blocks, we can pick a random uncovered t-subset 
            # and look at the blocks covering it.
            # This is a randomized greedy.
            if not uncovered: break
            target_t = random.choice(list(uncovered))
            candidates = t_to_blocks[target_t]
            if not candidates:
                # Should not happen if we start with empty uncovered and valid universe
                break
                
            best_block = -1
            max_covered = 0
            for b_idx in candidates:
                covered = 0
                for t_idx in block_to_t_indices[b_idx]:
                    if t_idx in uncovered:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            cover.append(best_block)
            # Update uncovered
            for t_idx in block_to_t_indices[best_block]:
                uncovered.discard(t_idx)
                
        return cover

    def local_search(cover, seed):
        random.seed(seed)
        # cover is a list of block indices
        # Initialize counts
        counts = [0] * num_t
        for b_idx in cover:
            for t_idx in block_to_t_indices[b_idx]:
                counts[t_idx] += 1
        
        # Check validity (should be valid from greedy)
        # for c in counts:
        #    if c == 0: return None
            
        improved = True
        no_improvement_streak = 0
        max_swaps = 5000 # Limit swaps to avoid infinite loops
        
        while improved and (time.time() < deadline - 5.0):
            improved = False
            # Shuffle order to avoid bias
            indices = list(range(len(cover)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(cover): continue # Cover might have shrunk
                b_idx = cover[i]
                
                # Find essential t-subsets for b_idx
                essential = []
                for t_idx in block_to_t_indices[b_idx]:
                    if counts[t_idx] == 1:
                        essential.append(t_idx)
                        
                if not essential:
                    # Block is redundant
                    # Remove it
                    cover.pop(i)
                    for t_idx in block_to_t_indices[b_idx]:
                        counts[t_idx] -= 1
                    improved = True
                    break # Restart loop
                    
                else:
                    # Try to find a replacement
                    # Candidates must cover all essential t-subsets
                    # Intersection of t_to_blocks[s] for s in essential
                    # Start with the smallest list
                    min_list_idx = min(essential, key=lambda s: len(t_to_blocks[s]))
                    candidates = set(t_to_blocks[min_list_idx])
                    
                    for s in essential:
                        if s != min_list_idx:
                            candidates &= set(t_to_blocks[s])
                        if not candidates:
                            break
                    
                    candidates.discard(b_idx)
                    
                    if candidates:
                        # Pick best candidate
                        # Score: number of subsets in block that have count 1 (after removing b_idx)
                        # Since we are removing b_idx, counts for essential subsets will become 0, 
                        # but we are adding b', so they become 1.
                        # We want b' to cover other subsets that are currently count 1 (essential to other blocks)
                        # to potentially make those blocks redundant.
                        
                        best_cand = None
                        best_score = -1
                        
                        # To speed up, don't check all candidates if many. 
                        # But candidates intersection should be small.
                        for c_idx in candidates:
                            score = 0
                            for t_idx in block_to_t_indices[c_idx]:
                                # If this subset is in essential, it was count 1 (due to b_idx), 
                                # after removing b_idx it's 0, adding c_idx makes it 1.
                                # It's not "redundancy creating" in the same way.
                                # We care about subsets NOT in essential that are count 1.
                                if counts[t_idx] == 1 and t_idx not in essential:
                                    score += 1
                            if score > best_score:
                                best_score = score
                                best_cand = c_idx
                                
                        if best_cand is not None:
                            # Perform swap
                            # Remove b_idx
                            for t_idx in block_to_t_indices[b_idx]:
                                counts[t_idx] -= 1
                            # Add best_cand
                            for t_idx in block_to_t_indices[best_cand]:
                                counts[t_idx] += 1
                            cover[i] = best_cand
                            improved = True
                            no_improvement_streak = 0
                            break # Restart loop
                            
            if not improved:
                # Try a few random swaps even if no essential sets? No, that's just wandering.
                # If no essential sets and no swaps found, we are stuck.
                pass

        return cover

    best_cover = None
    best_len = float('inf')
    
    # Run multiple instances
    seed_counter = 0
    while time.time() < deadline:
        seed = seed_counter
        seed_counter += 1
        
        # Initial greedy
        cover = greedy_cover(seed)
        if not cover:
            continue
            
        # Local search
        cover = local_search(cover, seed + 1000)
        
        if len(cover) < best_len:
            best_len = len(cover)
            best_cover = cover.copy()
            
        # If we hit a very good score, we can tighten the time or break early?
        # Theoretical lower bound is 208. Best known 335.
        # If we get close to 208, we should spend more time.
        
    # Convert block indices back to point lists
    if best_cover is None:
        # Fallback: return empty or a trivial invalid cover? 
        # The task says "Valid when every t-subset...". 
        # If we can't find a valid one, we should probably return a valid one.
        # Greedy always returns a valid one.
        # So best_cover should not be None.
        return []
        
    result = []
    for b_idx in best_cover:
        result.append(list(all_k_blocks[b_idx]))
        
    return result

# Test locally if needed
# if __name__ == "__main__":
#     import time
#     start = time.time()
#     blocks = construct(19, 8, 5)
#     end = time.time()
#     print(f"Time: {end - start}, Blocks: {len(blocks)}")
