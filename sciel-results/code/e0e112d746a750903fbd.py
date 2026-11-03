import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks_total = len(all_blocks)
    
    # Precompute block_coverages: list of np.array of t-subset indices
    # This is the core data structure for the GS heuristic
    block_coverages = []
    for b in all_blocks:
        covers = []
        for s in itertools.combinations(b, t):
            covers.append(idx[s])
        block_coverages.append(np.array(covers, dtype=np.int32))
        
    # Precompute inverse index: t_sub_to_blocks
    # t_sub_to_blocks[ti] = list of block indices that cover t-subset ti
    t_sub_to_blocks = [[] for _ in range(N)]
    for bi, cov in enumerate(block_coverages):
        for ti in cov:
            t_sub_to_blocks[ti].append(bi)
            
    # Convert to numpy arrays for faster access if needed, but lists are fine for small counts
    # However, for the GS heuristic, we need to know which blocks cover which t-subsets efficiently.
    
    def get_initial_cover():
        """
        Generate an initial valid cover using a greedy algorithm.
        Returns a list of block indices.
        """
        uncovered = np.ones(N, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            # Find the block that covers the most uncovered t-subsets
            # To speed this up, we can pick a random uncovered t-subset and only consider blocks covering it?
            # No, standard greedy picks the global best.
            # With N=11628 and 75k blocks, a full scan is 75k * 56 ops ~ 4M.
            # Doing this ~300 times is 1.2B ops. Too slow in Python.
            
            # Optimized Greedy:
            # 1. Pick a random uncovered t-subset s.
            # 2. Get all blocks covering s. (Size ~ 364)
            # 3. Among these, pick the one that covers the most *other* uncovered subsets.
            # 4. Add it, update mask.
            
            # This is O(364 * 56) per step ~ 20k ops. 300 steps -> 6M ops. Fast.
            
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
                
            # Pick a random uncovered subset to anchor the search
            # Or pick the one with fewest covering blocks? 
            # Let's stick to random for diversity in initial solution.
            target_ti = random.choice(uncovered_indices)
            
            candidate_blocks = t_sub_to_blocks[target_ti]
            if not candidate_blocks:
                # Should not happen
                break
                
            best_block = -1
            best_score = -1
            
            # Evaluate candidates
            for bi in candidate_blocks:
                if bi in blocks:
                    continue
                cov = block_coverages[bi]
                # Count uncovered covered by this block
                score = np.sum(uncovered[cov])
                if score > best_score:
                    best_score = score
                    best_block = bi
                    
            if best_block == -1:
                # No valid block found (should not happen if candidate_blocks is not empty and we don't reuse)
                # If we are forcing distinct blocks, and all candidates are used, we might fail.
                # But in initial cover, we start with empty, so this is fine.
                # If we get stuck, break.
                break
                
            blocks.append(best_block)
            uncovered[block_coverages[best_block]] = False
            
        return blocks

    def is_valid_cover(blocks):
        """Check if the list of blocks covers all t-subsets."""
        if not blocks:
            return False
        covered = np.zeros(N, dtype=bool)
        for bi in blocks:
            covered[block_coverages[bi]] = True
        return np.all(covered)

    def calculate_redundancy(blocks):
        """
        Identify redundant blocks in the current cover.
        A block is redundant if all t-subsets it covers are also covered by other blocks.
        Returns a set of indices of redundant blocks.
        """
        if not blocks:
            return set()
        
        # Count how many times each t-subset is covered
        cover_counts = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            cover_counts[block_coverages[bi]] += 1
            
        redundant = set()
        for bi in blocks:
            cov = block_coverages[bi]
            # If all covered t-subsets have count > 1, then this block is redundant
            # i.e., min(cover_counts[cov]) > 1
            if np.min(cover_counts[cov]) > 1:
                redundant.add(bi)
                
        return redundant

    def grassberger_scott_optimize(initial_blocks, time_limit):
        """
        Optimize the cover using GS heuristic.
        1. Start with a valid cover.
        2. Iteratively remove redundant blocks.
        3. If stuck (no redundant blocks), perform a "kick":
           - Remove a random block (making the cover invalid).
           - Add a block that covers the most newly uncovered t-subsets to restore validity.
           - Repeat removal phase.
        """
        current_blocks = list(initial_blocks)
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        # Precompute a list of block indices for random selection
        all_block_indices = list(range(num_blocks_total))
        
        while time.time() < time_limit:
            # Phase 1: Remove redundant blocks
            redundant = calculate_redundancy(current_blocks)
            
            # Remove all redundant blocks at once? Or one by one?
            # Removing all at once is faster and safe.
            if redundant:
                current_blocks = [bi for bi in current_blocks if bi not in redundant]
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                continue
            
            # Phase 2: No redundant blocks found. Perform a Kick.
            # Strategy:
            # 1. Pick a random block from current_blocks.
            # 2. Remove it.
            # 3. Identify the t-subsets that are now uncovered (count dropped to 0).
            # 4. Find the best block to add back that covers the most of these.
            
            if not current_blocks:
                break
                
            # Pick a random block to remove
            idx_to_remove = random.randint(0, len(current_blocks) - 1)
            bi_removed = current_blocks[idx_to_remove]
            current_blocks.pop(idx_to_remove)
            
            # Identify newly uncovered t-subsets
            # We need to know the cover counts before removal?
            # Easier: Just calculate which t-subsets are now uncovered.
            # Re-calculate cover counts for the current set (without the removed block).
            # This is O(M * 56). M ~ 300. 16k ops. Fast.
            
            cover_counts = np.zeros(N, dtype=np.int32)
            for bi in current_blocks:
                cover_counts[block_coverages[bi]] += 1
                
            uncovered_mask = (cover_counts == 0)
            
            if not np.any(uncovered_mask):
                # The removed block was actually redundant? 
                # This shouldn't happen if Phase 1 was thorough, but due to concurrency or logic, maybe.
                # Or maybe it was part of a "cycle" where removing it didn't expose new gaps immediately?
                # No, if it was redundant, it would have been caught in Phase 1.
                # So this implies the block was essential, but we removed it anyway.
                # Wait, if it wasn't redundant, removing it MUST leave some t-subsets uncovered.
                # Unless... the t-subsets it covered were covered by others? That's the definition of redundant.
                # So if not redundant, removing it leaves gaps.
                # So uncovered_mask should be True for some.
                pass
            
            if np.any(uncovered_mask):
                # Find the best block to add to cover the most uncovered
                # We can restrict candidates to blocks that cover at least one uncovered t-subset.
                # Or just pick a random uncovered t-subset and look at its covering blocks.
                
                uncovered_indices = np.where(uncovered_mask)[0]
                target_ti = random.choice(uncovered_indices)
                
                candidate_blocks = t_sub_to_blocks[target_ti]
                best_block = -1
                best_score = -1
                
                for bi in candidate_blocks:
                    if bi in current_blocks:
                        continue
                    cov = block_coverages[bi]
                    # Score: number of currently uncovered t-subsets it covers
                    score = np.sum(uncovered_mask[cov])
                    if score > best_score:
                        best_score = score
                        best_block = bi
                        
                if best_block != -1:
                    current_blocks.append(best_block)
                else:
                    # Couldn't fix it? Restore removed block?
                    current_blocks.append(bi_removed)
            else:
                # If no uncovered, we improved?
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                    
        return best_blocks

    # Main Logic
    best_blocks_global = None
    best_size_global = float('inf')
    
    # Run multiple iterations of GS with different random initial covers
    # Each GS run has a time slice.
    # Total time ~ 145s.
    # Let's do 10 iterations of 14s each? Or dynamic.
    
    num_iterations = 10
    time_per_iter = (deadline - time.time()) / num_iterations
    
    for i in range(num_iterations):
        if time.time() > deadline - 2:
            break
            
        # Generate a new random initial cover
        # To make it different, we can vary the random seed or the heuristic in get_initial_cover
        random.seed(100 + i)
        
        init_blocks = get_initial_cover()
        
        if not is_valid_cover(init_blocks):
            continue
            
        current_size = len(init_blocks)
        if current_size < best_size_global:
            best_size_global = current_size
            best_blocks_global = list(init_blocks)
            
        # Optimize with GS
        # Leave some time for the next iteration
        gs_deadline = time.time() + time_per_iter * 0.8
        optimized_blocks = grassberger_scott_optimize(init_blocks, gs_deadline)
        
        if optimized_blocks and is_valid_cover(optimized_blocks):
            if len(optimized_blocks) < best_size_global:
                best_size_global = len(optimized_blocks)
                best_blocks_global = list(optimized_blocks)
                
    if best_blocks_global is None:
        # Fallback
        init_blocks = get_initial_cover()
        best_blocks_global = init_blocks
        
    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[bi]) for bi in best_blocks_global]
    
    return final_blocks
