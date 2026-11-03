import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start_time = time.time()
    # Evaluator limit is 120s, but we should leave a safety margin for overhead/return
    limit = 115.0 
    deadline = start_time + limit

    # 1. Precomputation
    # Generate all t-subsets
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # Generate all k-blocks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Map each block to the indices of the t-subsets it covers
    # block_covers[i] is a list of subset indices covered by block i
    block_covers = []
    for block in blocks:
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        block_covers.append(np.array(covers, dtype=np.int32))
        
    if not _deadline_ok(deadline):
        return []

    # Build reverse mapping: for each t-subset, which blocks cover it?
    # sub_to_blocks[j] is a list of block indices covering subset j
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
    
    # Pre-compute a numpy array for subset coverage? 
    # It's a sparse matrix of size n_t x n_blocks. 
    # Given n_t=11628 and n_blocks=75582, a dense matrix is 880M entries (too big).
    # We stick to list of lists for sub_to_blocks and lists of arrays for block_covers.

    # 2. Fast Greedy Initialization
    # We need a valid solution to start local search.
    # Greedy: repeatedly pick the block that covers the most uncovered subsets.
    
    uncovered_mask = np.ones(n_t, dtype=bool)
    selected = set()
    
    # To speed up greedy, we can't easily vectorize the "gain" calculation without a matrix.
    # However, n_t is small enough that we can iterate.
    # Optimization: Pre-compute the size of block_covers? They are all C(k,t).
    # Actually, the gain depends on how many of its subsets are currently uncovered.
    
    # Simple Greedy Loop
    while np.any(uncovered_mask):
        if not _deadline_ok(deadline):
            break
            
        best_block = -1
        max_gain = 0
        
        # Iterate through blocks. To speed this up, we can skip blocks that don't cover any uncovered subset.
        # We can check this by looking at the intersection of block_covers[i] and uncovered indices.
        # But iterating all n_blocks every step is O(n_blocks * n_t) which is ~ 75k * 11k = 800M ops per step.
        # Too slow.
        
        # Better Greedy Strategy:
        # 1. Find an uncovered subset.
        # 2. Pick a random block that covers it, or the one that covers the most other uncovered subsets.
        
        # Let's pick a random uncovered subset to guide the search, or the "most constrained" one.
        # "Most constrained" = subset with fewest covering blocks.
        # We can pre-compute the count of blocks for each subset.
        subset_counts = [len(lst) for lst in sub_to_blocks]
        
        # Find uncovered subset with minimum count
        # We need to track which subsets are uncovered.
        # Let's maintain a list of uncovered subsets.
        
        # Actually, a simpler greedy for initialization:
        # Randomly pick an uncovered subset, pick a random block covering it.
        # This is fast but might yield a large solution.
        # Then local search fixes it.
        
        # Let's do a slightly smarter greedy:
        # Pick the uncovered subset with the fewest options (min count).
        # Among blocks covering it, pick the one with highest "potential gain".
        
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Find the index in uncovered_indices with min subset_count
        # subset_counts is a list, so we can map it to numpy
        # To avoid O(n_t) scan every time, we could maintain a priority queue, but for initialization 
        # a simple scan is fine if we limit steps or use randomization.
        
        # Let's just pick a random uncovered subset to keep it O(1) per step roughly.
        # This produces a valid cover quickly.
        target_idx = random.choice(uncovered_indices)
        
        candidates = sub_to_blocks[target_idx]
        if not candidates:
            break # Should not happen for valid covering designs parameters
            
        # Pick the candidate that covers the most other uncovered subsets
        # We can approximate this or just pick random for speed in init.
        # Let's pick the one that covers the most uncovered subsets.
        best_cand = -1
        max_gain = 0
        # Limit candidates checked if too many?
        # Usually sub_to_blocks sizes are manageable.
        for b in candidates:
            # Count how many of block_covers[b] are in uncovered_mask
            # block_covers[b] is a numpy array of indices.
            # We can use np.isin or boolean indexing.
            # uncovered_mask is a boolean array of length n_t.
            gain = np.sum(uncovered_mask[block_covers[b]])
            if gain > max_gain:
                max_gain = gain
                best_cand = b
                
        if best_cand != -1:
            selected.add(best_cand)
            uncovered_mask[block_covers[best_cand]] = False

    if not selected:
        return []

    # 3. Local Search Optimization
    # We now have a valid cover. We want to minimize |selected|.
    
    # Helper: Get coverage counts for a set of blocks
    # Instead of rebuilding counts from scratch every time, we update them incrementally.
    
    # Initialize counts
    counts = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        counts[block_covers[b]] += 1
    
    # Helper: Check if a block is redundant given current counts
    # A block is redundant if all subsets it covers have count > 1.
    def is_redundant(b, counts):
        return np.all(counts[block_covers[b]] > 1)
        
    # Helper: Try to remove a block. If it causes gaps, try to fix.
    def try_remove_and_fix(b, selected, counts, deadline):
        """
        Attempts to remove block b.
        Returns True if successful (and updates selected/counts), False otherwise.
        """
        if not _deadline_ok(deadline):
            return False
            
        # Check if b is redundant
        if is_redundant(b, counts):
            selected.remove(b)
            counts[block_covers[b]] -= 1
            return True
            
        # If not redundant, it covers some subsets with count == 1.
        # These are the "critical" subsets for b.
        covers_b = block_covers[b]
        critical_subsets = covers_b[counts[covers_b] == 1]
        
        if len(critical_subsets) == 0:
            # Should be redundant, but just in case
            selected.remove(b)
            counts[covers_b] -= 1
            return True
            
        # We need to find a replacement block b_new such that:
        # 1. b_new covers all critical_subsets.
        # 2. b_new is not in selected.
        # 3. Ideally, b_new covers other subsets that are already covered (so we don't increase "waste" too much), 
        #    or simply that the swap is valid.
        
        # Find candidate blocks that cover ALL critical_subsets.
        # Intersection of sub_to_blocks[s] for s in critical_subsets.
        
        if len(critical_subsets) == 1:
            candidates = sub_to_blocks[critical_subsets[0]]
        else:
            # Find intersection
            # Start with the subset that has the fewest candidates
            min_count = float('inf')
            target = -1
            for s in critical_subsets:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target = s
            
            candidates = set(sub_to_blocks[target])
            for s in critical_subsets:
                if s == target: continue
                candidates.intersection_update(sub_to_blocks[s])
                if not candidates:
                    return False
            
            candidates = list(candidates)
            
        # Filter candidates: must not be in selected
        valid_candidates = [c for c in candidates if c not in selected]
        
        if not valid_candidates:
            return False
            
        # Evaluate candidates.
        # Metric: We want to minimize the "cost" of the new block.
        # A good proxy: Number of subsets covered by b_new that are currently covered exactly once by other blocks?
        # Or simply: Does b_new cover any subsets that are currently uncovered? (No, because we only removed b, 
        # and b's critical subsets are the only ones at risk. If b_new covers them, they become covered.
        # The other subsets covered by b_new were already covered by other blocks in selected, 
        # so their count increases. This is "safe" but might make them harder to remove later.
        
        # Heuristic: Pick the candidate that covers the most subsets that are currently "over-covered" (count > 1)?
        # Or the one with the smallest "impact"?
        # Actually, the simplest valid swap is any valid candidate.
        # To improve, we can look for a candidate that also helps cover other "weakly covered" subsets?
        # Let's just pick a random valid candidate to allow exploration.
        # Or pick the one that covers the most subsets with count == 2 (making them count 3, less critical)?
        
        # Let's try to pick the candidate that minimizes the number of subsets it covers 
        # that are currently covered exactly once by blocks OTHER than b.
        # Wait, if a subset s is covered by b and b_new, and count[s] was 1 (only b), 
        # after removing b, count[s] becomes 0. Adding b_new makes it 1.
        # If count[s] was 2 (b and x), removing b makes it 1. Adding b_new makes it 2.
        
        # The "critical" subsets are those with count 1. They MUST be covered by b_new.
        # For non-critical subsets covered by b_new, the count goes from C to C+1.
        
        # Let's just pick a random one for speed and diversity.
        b_new = random.choice(valid_candidates)
        
        # Perform swap
        selected.remove(b)
        counts[covers_b] -= 1
        selected.add(b_new)
        counts[block_covers[b_new]] += 1
        
        return True

    # Main Local Search Loop
    # We iterate, trying to remove random blocks.
    # If removal fails (block is critical), we try to fix it with a swap.
    # If swap succeeds, we continue. If it fails, we leave the block.
    
    # To avoid getting stuck, we can use a "temperature" or just random restarts of the local search phase.
    
    best_size = len(selected)
    best_selected = set(selected)
    best_counts = counts.copy()
    
    # We run local search for a fixed amount of time or iterations.
    # Since we have a deadline, we loop until deadline.
    
    # Optimization: Keep a list of blocks in selected for fast random access
    selected_list = list(selected)
    
    # We'll use a counter to track consecutive failures to detect stagnation
    # and potentially restart or change strategy.
    
    while _deadline_ok(deadline):
        if not selected_list:
            break
            
        # Pick a random block to try to remove
        idx = random.randint(0, len(selected_list) - 1)
        b = selected_list[idx]
        
        success = try_remove_and_fix(b, selected, counts, deadline)
        
        if success:
            # Update the list representation
            # Since we modified 'selected' (set), we need to sync 'selected_list'
            # try_remove_and_fix might have removed b and added b_new, or just removed b.
            # Rebuilding the list is O(N), which is okay if N is small (~300-400).
            selected_list = list(selected)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = set(selected)
                best_counts = counts.copy()
                
    # Final Result
    # Ensure we return the best found
    if len(best_selected) < len(selected):
        selected = best_selected
        
    # Convert to list of lists
    result = [list(blocks[i]) for i in selected]
    return result
