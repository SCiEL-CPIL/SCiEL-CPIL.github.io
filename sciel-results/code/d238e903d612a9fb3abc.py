"""New program using Randomized Local Search with NumPy acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import itertools
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110
    
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(all_points, t))
    n_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-subsets and the t-subsets they cover
    # C(19, 7) = 50388
    all_ksubsets = list(itertools.combinations(all_points, k))
    n_ksubsets = len(all_ksubsets)
    
    # coverage_matrix: list of lists of indices
    # coverage_matrix[block_idx] = list of tsubset_idx covered by block_idx
    coverage_matrix = []
    for block in all_ksubsets:
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(tsubset_to_idx[ts])
        coverage_matrix.append(covered)
    
    # Convert to numpy arrays for faster slicing if needed, 
    # but lists are often faster for small lengths in Python loops if not vectorizing the whole thing.
    # However, we will use a cover_count array.
    
    def get_block_coverage(block_idx):
        return coverage_matrix[block_idx]

    def greedy_init():
        """Fast greedy initialization."""
        uncovered = np.ones(n_tsubsets, dtype=np.bool_)
        blocks = []
        
        # Precompute for each block, the list of t-subset indices it covers
        # We already have this in coverage_matrix
        
        # To speed up greedy, we can just iterate. 
        # C(19,7) is small enough that even a simple loop might work, 
        # but let's be smart: 
        # We want to pick the block covering the most uncovered t-subsets.
        
        # Since we need to do this repeatedly, and n_ksubsets is ~50k, 
        # checking all 50k blocks each step is 50k * 35 ops.
        # Number of blocks in solution ~150. 
        # 150 * 50k * 35 = 262M ops. Might be too slow in Python.
        
        # Better: Randomized Greedy or just random init + local search?
        # Random init with 200 blocks is very fast.
        # Then local search removes/replaces.
        
        # Let's try a hybrid: 
        # 1. Pick random blocks until all covered.
        # 2. Local search to minimize.
        
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        remaining_uncovered = n_tsubsets
        rng = random.Random(42)
        
        while remaining_uncovered > 0:
            # Pick a random block
            b_idx = rng.randint(0, n_ksubsets - 1)
            covs = coverage_matrix[b_idx]
            
            # Check if it adds anything
            # If we just add it, we update cover_count.
            # We need to know if it covers any *uncovered* t-subset.
            
            # Optimization: 
            # Instead of checking all, just add it. 
            # If it covers nothing new, we wasted a block. 
            # To avoid this, we can pick blocks more intelligently.
            
            # Let's do a simple greedy:
            # Find the block that covers the most currently uncovered t-subsets.
            # This is expensive. 
            # Alternative: Just pick random, if it doesn't cover new ones, try again?
            # Or: Pick a random uncovered t-subset, pick a random block covering it.
            
            # Strategy: 
            # 1. Find an uncovered t-subset.
            # 2. Pick a random block that covers it.
            
            # To find uncovered t-subset:
            # We can maintain a list of uncovered indices, or just scan.
            # Scanning 11628 t-subsets is fast.
            
            # Find first uncovered
            unc_idx = -1
            for i in range(n_tsubsets):
                if cover_count[i] == 0:
                    unc_idx = i
                    break
            
            if unc_idx == -1:
                break
            
            # Pick a random block that covers unc_idx
            # We need a reverse map: tsubset_idx -> list of block_idx covering it
            # Precomputing this is O(n_ksubsets * 35) which is fine.
            
            # Let's precompute reverse map outside this function? 
            # No, let's do it once.
            
            # But I'm inside greedy_init. I should precompute before calling.
            # Let's refactor.
            
            pass # Placeholder

    # Precompute reverse map: tsubset -> list of blocks covering it
    # This is memory intensive? 11628 tsubsets * avg blocks covering them.
    # Each tsubset is in C(19-4, 7-4) = C(15,3) = 455 blocks.
    # 11628 * 455 = 5.3M entries. Fine.
    
    tsub_to_blocks = [[] for _ in range(n_tsubsets)]
    for b_idx, covs in enumerate(coverage_matrix):
        for ts_idx in covs:
            tsub_to_blocks[ts_idx].append(b_idx)
    
    def fast_greedy_init(seed):
        rng = random.Random(seed)
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        blocks = []
        
        while True:
            # Find an uncovered t-subset
            # Efficient way: maintain a set of uncovered? 
            # Or just scan. 11k scan is ~10us.
            unc_idx = -1
            for i in range(n_tsubsets):
                if cover_count[i] == 0:
                    unc_idx = i
                    break
            
            if unc_idx == -1:
                break
            
            # Pick a random block from tsub_to_blocks[unc_idx]
            candidates = tsub_to_blocks[unc_idx]
            b_idx = rng.choice(candidates)
            
            # Add block
            blocks.append(b_idx)
            for ts_idx in coverage_matrix[b_idx]:
                cover_count[ts_idx] += 1
        
        return blocks, cover_count

    def local_search(blocks, cover_count, time_limit):
        """
        Randomized Local Search:
        1. Try to remove a random block. If valid, accept.
        2. Try to replace a random block with a random other block. If valid, accept.
        """
        n_blocks = len(blocks)
        rng = random.Random(123)
        
        # We will track the current number of blocks
        current_len = n_blocks
        
        # To speed up validity check, we rely on cover_count.
        # A state is valid if min(cover_count) >= 1.
        
        # We can keep track of the number of t-subsets with count == 1 (critical)
        # and count == 0 (invalid).
        
        # Let's just use min(cover_count) > 0 for simplicity, 
        # but that's O(N). 
        # Better: maintain `n_critical` (count==1) and `n_invalid` (count==0).
        
        n_critical = np.sum(cover_count == 1)
        n_invalid = np.sum(cover_count == 0)
        
        # Helper to update counts
        def update_counts(block_idx, sign):
            nonlocal n_critical, n_invalid
            for ts_idx in coverage_matrix[block_idx]:
                old_val = cover_count[ts_idx]
                new_val = old_val + sign
                cover_count[ts_idx] = new_val
                
                # Update counters
                if old_val == 0:
                    n_invalid -= 1
                elif old_val == 1:
                    n_critical -= 1
                    
                if new_val == 0:
                    n_invalid += 1
                elif new_val == 1:
                    n_critical += 1

        best_len = current_len
        best_blocks = blocks[:]
        
        while time.time() < time_limit:
            # Try removal
            if current_len > 0:
                # Pick a random block index
                pos = rng.randint(0, current_len - 1)
                b_idx = blocks[pos]
                
                # Check if removal is safe
                # Safe if no t-subset in b_idx has count == 1
                # i.e., min(cover_count[coverage_matrix[b_idx]]) > 1
                
                # This is O(35) check.
                safe = True
                covs = coverage_matrix[b_idx]
                for ts_idx in covs:
                    if cover_count[ts_idx] == 1:
                        safe = False
                        break
                
                if safe:
                    # Remove
                    update_counts(b_idx, -1)
                    blocks.pop(pos)
                    current_len -= 1
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = blocks[:]
                
            # Try replacement
            # Only if we didn't just remove, or randomly
            if current_len > 0:
                pos = rng.randint(0, current_len - 1)
                old_b_idx = blocks[pos]
                
                # Pick a random new block
                new_b_idx = rng.randint(0, n_ksubsets - 1)
                
                if new_b_idx == old_b_idx:
                    continue
                
                # Check if replacement keeps validity
                # We remove old, add new.
                # Valid if after removal, any t-subset that becomes 0 is covered by new.
                
                # Let's simulate:
                # 1. Remove old
                # 2. Check if any 0s
                # 3. Add new
                # 4. Check if all 0s are gone
                
                # Optimization:
                # Find t-subsets in old_b that have count == 1.
                # These are the "critical" ones that would become 0.
                critical_ts = []
                covs_old = coverage_matrix[old_b_idx]
                for ts_idx in covs_old:
                    if cover_count[ts_idx] == 1:
                        critical_ts.append(ts_idx)
                
                if not critical_ts:
                    # Removal is safe, so replacement is safe (adding more doesn't hurt)
                    # Actually, if removal is safe, we should have removed it in the removal step.
                    # But maybe we didn't pick this one.
                    # So replacement is definitely safe.
                    update_counts(old_b_idx, -1)
                    update_counts(new_b_idx, 1)
                    blocks[pos] = new_b_idx
                    # Length unchanged
                else:
                    # Check if new_b covers all critical_ts
                    covs_new = set(coverage_matrix[new_b_idx])
                    # Check intersection
                    all_covered = True
                    for ts_idx in critical_ts:
                        if ts_idx not in covs_new:
                            all_covered = False
                            break
                    
                    if all_covered:
                        update_counts(old_b_idx, -1)
                        update_counts(new_b_idx, 1)
                        blocks[pos] = new_b_idx
                    else:
                        # Invalid replacement
                        pass
        
        return best_blocks, best_len

    # Run multiple restarts? 
    # Greedy init is deterministic-ish if seed fixed, but random choice makes it vary.
    # Let's run 1 greedy init, then local search.
    
    blocks, cover_count = fast_greedy_init(0)
    
    # Local search
    best_blocks, best_len = local_search(blocks, cover_count, time_limit)
    
    # Convert indices back to point lists
    result = []
    for b_idx in best_blocks:
        result.append(list(all_ksubsets[b_idx]))
        
    return result

# EVOLVE-BLOCK-END
