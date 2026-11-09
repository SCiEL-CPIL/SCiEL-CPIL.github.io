import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Solve the covering design problem C(v, k, t) using a randomized greedy 
    set cover approach followed by local search for block removal.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Precompute all t-subsets (the universe to be covered)
    # For v=24, k=7, t=3, there are C(24,3) = 2024 t-subsets.
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each potential block (k-subset), which t-subsets it covers.
    # There are C(24,7) = 346104 blocks. 
    # Each block covers C(7,3) = 35 t-subsets.
    # Storing this as a list of lists of indices is memory efficient enough.
    
    # Optimization: We don't need to precompute ALL blocks if we are greedy.
    # But for local search and faster greedy, precomputing the "coverage mask" 
    # or list of covered t-subset indices for every block is beneficial.
    
    # Let's precompute the coverage for all blocks.
    # block_to_covered_t_indices: dict mapping frozenset(block) -> list of t-subset indices
    # This might be heavy: 346k blocks * 35 ints. ~12 million ints. 
    # In Python, this could be slow to build or memory heavy.
    
    # Alternative: Generate blocks on the fly for the greedy step, but use a 
    # smarter gain calculation.
    
    # Let's try a different approach for the greedy step to be faster:
    # 1. Start with empty covered set.
    # 2. At each step, find the block that covers the most uncovered t-subsets.
    #    To do this efficiently without checking all 346k blocks every time:
    #    - We can maintain a count of how many blocks cover each t-subset? No, that's for removal.
    #    - Standard greedy set cover: for each candidate set, calculate gain.
    #    - With 346k candidates and 2024 universe elements, calculating gain for all 
    #      candidates is 346k * 35 operations per step. 
    #      If we need ~70 blocks, that's 70 * 12M = 840M operations. Too slow in Python.
    
    # Better Greedy Strategy:
    # Instead of iterating all blocks, iterate uncovered t-subsets.
    # Pick an uncovered t-subset. Find a block that covers it and maximizes additional coverage.
    # But finding "best block covering a specific t-subset" still requires scanning blocks.
    
    # Let's use a different heuristic:
    # Randomized Greedy with Precomputed Block Properties.
    # Since v=24 is small, let's precompute the coverage lists.
    
    # Memory check:
    # 346,104 blocks.
    # Each block covers 35 t-subsets.
    # Total entries: ~12 million.
    # List of lists of ints. 
    # Building this takes time. Let's see if we can do it fast enough.
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_coverage = []
    for b in all_blocks:
        covered = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_coverage.append(covered)
    
    # Now we have block_coverage[i] = list of t-subset indices covered by block i.
    
    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered_mask = [1] * num_t_subsets
        num_uncovered = num_t_subsets
        selected_blocks = []
        
        while num_uncovered > 0:
            if time.time() > deadline:
                break
            
            best_block_idx = -1
            best_gain = 0
            
            # Optimization: Only consider blocks that cover at least one uncovered subset?
            # Or just scan all?
            # Scanning all 346k blocks every step is expensive.
            # Let's try a heuristic: Pick a random uncovered t-subset, 
            # then only consider blocks that cover that specific t-subset.
            
            # Find an uncovered t-subset index
            # Instead of scanning the whole list, we can maintain a list of uncovered indices
            # but removing from a list is O(N). 
            # Let's use a set of uncovered indices?
            
            # Let's stick to a simpler, robust method first:
            # Randomized Greedy:
            # 1. Pick a random uncovered t-subset.
            # 2. Find all blocks that cover it.
            # 3. Among those, pick the one with the highest gain (covering most other uncovered subsets).
            # 4. Add to solution.
            
            # To make this fast, we need a reverse index: t_subset -> list of blocks covering it.
            # Let's build that.
            
            # Re-do precomputation with reverse index.
            # This is O(12M) memory and time.
            
            # Wait, I can build the reverse index in the same loop.
            
            # Let's restart the precomputation to include reverse index.
            pass # Placeholder for logic restructure
            
        return selected_blocks

    # RESTRUCTURE:
    # 1. Precompute all blocks and their covered t-subsets.
    # 2. Build reverse index: t_subset_idx -> list of block_indices covering it.
    # 3. Run multiple randomized greedy instances.
    # 4. Post-process with local search (remove redundant blocks).
    
    # Step 1 & 2: Precomputation
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_covered_t = [None] * num_blocks
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, b in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_to_blocks[idx].append(i)
        block_covered_t[i] = covered_indices
    
    # Step 3: Randomized Greedy
    def run_greedy(seed):
        random.seed(seed)
        uncovered_set = set(range(num_t_subsets))
        selected = []
        
        while uncovered_set:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset to focus on
            # This reduces the search space from all blocks to blocks covering this t-subset
            target_t_idx = random.choice(list(uncovered_set))
            candidates = t_to_blocks[target_t_idx]
            
            best_block = -1
            best_gain = -1
            
            for b_idx in candidates:
                # Calculate gain: number of uncovered t-subsets covered by this block
                gain = 0
                for t_idx in block_covered_t[b_idx]:
                    if t_idx in uncovered_set:
                        gain += 1
                
                # Randomized tie-breaking: add small random noise or just pick max
                # To diversify, we can pick randomly among top performers?
                # Or just strict max. Let's do strict max for now, 
                # but the randomness comes from the choice of target_t_idx.
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break # Should not happen if uncovered_set is non-empty and valid input
                
            selected.append(best_block)
            
            # Update uncovered set
            for t_idx in block_covered_t[best_block]:
                if t_idx in uncovered_set:
                    uncovered_set.remove(t_idx)
        
        return selected
    
    # Step 4: Local Search - Remove Redundant Blocks
    def try_remove_blocks(blocks):
        if not blocks:
            return blocks
        
        # Convert to set of frozensets for fast lookup?
        # Or just work with indices.
        
        # A block is redundant if all t-subsets it covers are also covered by other blocks.
        # To check this, we need to know for each t-subset, how many blocks cover it.
        
        # Build coverage counts
        cover_count = [0] * num_t_subsets
        for b_idx in blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        # A block is redundant if for every t_idx it covers, cover_count[t_idx] > 1.
        # If we remove it, we must decrement counts.
        
        # Greedy removal:
        # Iterate through blocks. If a block is redundant, remove it.
        # Repeat until no more can be removed?
        # Or just one pass? One pass is faster but less effective.
        
        # Let's do a multi-pass removal.
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks
                
                b_idx = blocks[i]
                is_redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block i
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart loop since indices changed
        
        return blocks

    # Step 5: Main Search Loop
    best_blocks = None
    best_len = float('inf')
    
    # Try a few deterministic seeds first
    seeds = [42, 123, 999, 0, 1, 2, 3, 4, 5]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        greedy_res = run_greedy(seed)
        if not greedy_res:
            continue
            
        # Post-process
        processed_res = try_remove_blocks(greedy_res)
        
        if len(processed_res) < best_len:
            best_len = len(processed_res)
            best_blocks = processed_res
            
        # Check if we reached the theoretical lower bound or a known good bound?
        # Lower bound for C(24,7,3) is 69.
        # If we hit 69, we can stop.
        if best_len <= 69:
            break

    # If we have time, try more random seeds
    if time.time() < deadline:
        while time.time() < deadline - 5.0: # Keep 5s buffer
            seed = random.randint(0, 1000000)
            greedy_res = run_greedy(seed)
            if not greedy_res:
                continue
                
            processed_res = try_remove_blocks(greedy_res)
            
            if len(processed_res) < best_len:
                best_len = len(processed_res)
                best_blocks = processed_res
                
            if best_len <= 69:
                break

    if best_blocks is None:
        # Fallback: return an empty list or a single block? 
        # The problem requires a valid covering. 
        # If greedy failed (shouldn't happen for these params), we need a fallback.
        # But for v=24, k=7, t=3, greedy will always find a solution.
        return []
        
    # Convert block indices back to the actual blocks (lists of points)
    # The evaluator expects a list of blocks, where each block is a list of k integers.
    # all_blocks is a list of tuples.
    
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    # Ensure distinctness (should be guaranteed by indices) and sorting
    for b in final_blocks:
        b.sort()
        
    return final_blocks
