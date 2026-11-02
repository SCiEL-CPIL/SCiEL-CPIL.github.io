import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 138.0  # Safety margin before 150s

    # Precompute all t-subsets and their indices
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute block_coverages: for each block, the indices of t-subsets it covers
    # Also build inverse index: for each t-subset, the list of blocks that cover it
    block_coverages = []
    t_sub_to_blocks = [[] for _ in range(N)]
    
    for bi, b in enumerate(all_blocks):
        covers = []
        for s in itertools.combinations(b, t):
            ti = t_idx[s]
            covers.append(ti)
            t_sub_to_blocks[ti].append(bi)
        block_coverages.append(np.array(covers, dtype=np.int32))
    
    def greedy_cover(uncovered_mask, used_blocks_set=None, max_blocks=None):
        """
        Greedy construction: repeatedly pick the block covering the most uncovered subsets.
        Returns list of block indices.
        """
        if used_blocks_set is None:
            used_blocks_set = set()
        
        blocks = []
        mask = uncovered_mask.copy()
        
        # For efficiency, we'll maintain a count of uncovered per block
        # But recomputing is simpler and faster in practice for this problem size
        
        while np.any(mask):
            if time.time() > deadline:
                return blocks
            
            # Pick a random uncovered t-subset to focus on
            unc_idx = np.where(mask)[0]
            if len(unc_idx) == 0:
                break
            
            target = random.choice(unc_idx)
            candidates = t_sub_to_blocks[target]
            
            best_bi = -1
            best_gain = 0
            
            for bi in candidates:
                if bi in used_blocks_set:
                    continue
                gain = np.sum(mask[block_coverages[bi]])
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    # Early exit if we found a block covering many
                    if gain >= 50:
                        break
            
            if best_bi == -1:
                break
            
            blocks.append(best_bi)
            used_blocks_set.add(best_bi)
            mask[block_coverages[best_bi]] = False
            
            if max_blocks and len(blocks) >= max_blocks:
                break
        
        return blocks
    
    def remove_redundant(blocks_list):
        """
        Try to remove redundant blocks from a valid covering.
        Returns a new list with redundant blocks removed.
        """
        if not blocks_list:
            return []
        
        # Build coverage map: for each t-subset, which blocks cover it
        # We'll use the precomputed t_sub_to_blocks but filter to our blocks
        block_set = set(blocks_list)
        
        # For each t-subset, count how many blocks from our solution cover it
        cover_count = np.zeros(N, dtype=np.int32)
        for bi in blocks_list:
            cover_count[block_coverages[bi]] += 1
        
        # If any t-subset has cover_count == 0, we can't remove anything that would break coverage
        # But we need to check: can we remove block bi without leaving any t-subset with count 0?
        
        new_blocks = []
        for bi in blocks_list:
            # Check if removing bi leaves all t-subsets covered
            cov = block_coverages[bi]
            can_remove = True
            for ti in cov:
                if cover_count[ti] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove bi: decrement cover_count for its t-subsets
                cover_count[cov] -= 1
                # Don't add to new_blocks
            else:
                new_blocks.append(bi)
        
        return new_blocks
    
    def local_search(blocks_list):
        """
        Local search: try to swap blocks to reduce total count.
        """
        if not blocks_list:
            return blocks_list
        
        current = list(blocks_list)
        current_size = len(current)
        
        # Compute cover_count
        cover_count = np.zeros(N, dtype=np.int32)
        for bi in current:
            cover_count[block_coverages[bi]] += 1
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_size = len(current)
            
            # Recompute cover_count if needed (it's maintained)
            
            # Try removing each block and see if we can repair
            for i in range(len(current)):
                if time.time() > deadline - 2:
                    break
                
                bi = current[i]
                cov = block_coverages[bi]
                
                # Check which t-subsets would become uncovered
                uncovered_by_removal = []
                for ti in cov:
                    if cover_count[ti] == 1:
                        uncovered_by_removal.append(ti)
                
                if not uncovered_by_removal:
                    # Safe to remove
                    cover_count[cov] -= 1
                    current.pop(i)
                    improved = True
                    break  # Restart from beginning
            
            if not improved and time.time() < deadline - 5:
                # Try swap: replace a block with another that covers more
                # Pick a random block to consider removing
                if len(current) > 0:
                    i = random.randint(0, len(current) - 1)
                    bi = current[i]
                    cov = block_coverages[bi]
                    
                    uncovered_by_removal = [ti for ti in cov if cover_count[ti] == 1]
                    
                    if uncovered_by_removal:
                        # Need to find a replacement block that covers all uncovered_by_removal
                        # and ideally covers additional uncovered subsets
                        # But we don't have "uncovered" in the global sense, only within this solution
                        
                        # Try to find a block that covers all of uncovered_by_removal
                        # and has high total coverage
                        candidates = set(t_sub_to_blocks[uncovered_by_removal[0]])
                        for ti in uncovered_by_removal[1:]:
                            candidates &= set(t_sub_to_blocks[ti])
                        
                        # Remove blocks already in current
                        current_set = set(current)
                        candidates -= current_set
                        candidates.add(bi)  # Allow keeping bi? No, we're replacing
                        candidates.discard(bi)
                        
                        best_new_bi = -1
                        best_score = -1
                        
                        for new_bi in candidates:
                            if new_bi in current_set:
                                continue
                            # Score: how many of uncovered_by_removal does it cover (should be all)
                            # Plus how many other subsets it covers that are covered by exactly 1 other block
                            new_cov = block_coverages[new_bi]
                            # All uncovered_by_removal must be in new_cov
                            if not all(ti in set(new_cov) for ti in uncovered_by_removal):
                                continue
                            
                            # Score based on additional coverage benefit
                            score = 0
                            for ti in new_cov:
                                if cover_count[ti] == 1 and ti not in set(cov):
                                    score += 2  # High value: was critically covered, now doubly
                                elif cover_count[ti] == 1 and ti in set(cov):
                                    score += 1  # Replacing critical coverage
                            
                            if score > best_score:
                                best_score = score
                                best_new_bi = new_bi
                        
                        if best_new_bi != -1:
                            # Perform swap
                            cover_count[cov] -= 1
                            cover_count[block_coverages[best_new_bi]] += 1
                            current[i] = best_new_bi
                            improved = True
                            # After swap, try to remove redundant again
                            # The loop will restart
        
        return current
    
    # Phase 1: Generate initial solution with greedy
    best_blocks = None
    best_size = float('inf')
    
    # Try a few greedy instances
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        
        random.seed(seed)
        uncovered = np.ones(N, dtype=bool)
        blocks = greedy_cover(uncovered, used_blocks_set=set())
        
        if blocks and np.all(cover_count >= 1) if False else True:
            # Verify
            test_cover = np.zeros(N, dtype=bool)
            for bi in blocks:
                test_cover[block_coverages[bi]] = True
            if np.all(test_cover):
                # Remove redundant
                blocks = remove_redundant(blocks)
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: simple greedy
        random.seed(42)
        uncovered = np.ones(N, dtype=bool)
        best_blocks = greedy_cover(uncovered)
        best_size = len(best_blocks)
    
    # Phase 2: Local search
    if time.time() < deadline - 5:
        improved_blocks = local_search(best_blocks)
        if len(improved_blocks) < best_size:
            best_blocks = improved_blocks
            best_size = best_size
    
    # Phase 3: More local search passes
    for _ in range(3):
        if time.time() > deadline - 3:
            break
        improved_blocks = local_search(best_blocks)
        if len(improved_blocks) < best_size:
            best_blocks = improved_blocks
            best_size = len(best_blocks)
    
    # Convert back to point lists
    final_blocks = [list(all_blocks[bi]) for bi in best_blocks]
    
    return final_blocks
