import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # Map each k-subset to the list of t-subset indices it covers
    k_to_t_indices = []
    for k_sub in all_k_subsets:
        t_indices = []
        for t_sub in itertools.combinations(k_sub, t):
            t_indices.append(t_subset_to_idx[t_sub])
        k_to_t_indices.append(t_indices)
        
    # Helper to run a local search starting from an initial set of blocks
    def local_search(initial_blocks):
        # initial_blocks is a list of indices into all_k_subsets
        selected = set(initial_blocks)
        coverage = [0] * num_t
        
        # Initialize coverage
        for ki in selected:
            for ti in k_to_t_indices[ki]:
                coverage[ti] += 1
                
        current_size = len(selected)
        
        # Function to check if a block is redundant
        def is_redundant(ki):
            for ti in k_to_t_indices[ki]:
                if coverage[ti] == 1:
                    return False
            return True
            
        # Function to remove a block
        def remove_block(ki):
            selected.remove(ki)
            for ti in k_to_t_indices[ki]:
                coverage[ti] -= 1
                
        # Function to add a block
        def add_block(ki):
            selected.add(ki)
            for ti in k_to_t_indices[ki]:
                coverage[ti] += 1
                
        # Phase 1: Remove all redundant blocks
        changed = True
        while changed:
            changed = False
            to_remove = []
            for ki in list(selected):
                if is_redundant(ki):
                    to_remove.append(ki)
            for ki in to_remove:
                remove_block(ki)
                changed = True
                current_size -= 1
                
        # Phase 2: Local Search Moves
        # We try to swap blocks or add/remove to reduce size
        # Since removing a block might break coverage, we only remove if redundant.
        # To improve, we can try:
        # 1. Add a random block, then remove redundancies. If size < current_size, accept.
        # 2. Swap a selected block with an unselected one if it allows removing another block?
        
        # Let's try a simple hill climbing:
        # Try to replace a selected block with a different one that covers the same "critical" sets?
        # Actually, a better move for set cover is:
        # Pick a selected block A. Remove it.
        # If coverage is broken, find the best block B to restore coverage.
        # Add B.
        # Remove any new redundancies.
        # If total size < previous size, accept. Else revert.
        
        improved = True
        max_iterations = 1000 # Limit to prevent infinite loops
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            # Try random swaps
            # Pick a random selected block
            if not selected:
                break
            a = random.choice(list(selected))
            
            # Save state
            old_selected = set(selected)
            old_coverage = coverage[:]
            old_size = current_size
            
            # Remove A
            remove_block(a)
            
            # Check if coverage is broken
            broken = [ti for ti in range(num_t) if coverage[ti] == 0]
            
            if not broken:
                # A was redundant? We should have removed it in Phase 1.
                # But maybe it became redundant after other changes?
                # If we removed A and didn't break coverage, we have a smaller set.
                # This is a pure improvement.
                current_size -= 1
                improved = True
                # Continue to Phase 1 to clean up
                continue
            
            # Find best block B to cover the broken sets
            # Score B by how many broken sets it covers
            best_b = -1
            best_score = 0
            # To speed up, we can just check a subset of candidates or all
            # 75k candidates * 56 t-indices is ~4M ops. Might be slow in Python.
            # We can pre-compute for each t-subset, the list of blocks covering it.
            
            # Let's use a heuristic: Pick a random broken set, and pick a random block covering it?
            # Or pick the block covering the most broken sets.
            
            # Optimization: Precompute t_to_k_indices
            # This is expensive to build but worth it.
            # However, we are inside a function called multiple times? No, construct is called once.
            
            # Let's build t_to_k_indices once outside local_search?
            # Yes, let's move it to the outer scope.
            
            # For now, let's assume we have a way to find B quickly.
            # We'll pass t_to_k_indices as an argument or closure.
            
            # Let's refactor to pass t_to_k_indices
            pass # Placeholder
            
        return list(selected)

    # We need t_to_k_indices for the local search to be fast.
    t_to_k_indices = [[] for _ in range(num_t)]
    for ki, t_indices in enumerate(k_to_t_indices):
        for ti in t_indices:
            t_to_k_indices[ti].append(ki)
            
    # Redefine local_search with access to t_to_k_indices
    def local_search_v2(initial_blocks):
        selected = set(initial_blocks)
        coverage = [0] * num_t
        
        for ki in selected:
            for ti in k_to_t_indices[ki]:
                coverage[ti] += 1
                
        current_size = len(selected)
        
        def is_redundant(ki):
            for ti in k_to_t_indices[ki]:
                if coverage[ti] == 1:
                    return False
            return True
            
        def remove_block(ki):
            selected.remove(ki)
            for ti in k_to_t_indices[ki]:
                coverage[ti] -= 1
                
        def add_block(ki):
            selected.add(ki)
            for ti in k_to_t_indices[ki]:
                coverage[ti] += 1
                
        # Phase 1: Remove redundant
        changed = True
        while changed:
            changed = False
            to_remove = [ki for ki in selected if is_redundant(ki)]
            for ki in to_remove:
                remove_block(ki)
                changed = True
                current_size -= 1
                
        # Phase 2: Local Search
        improved = True
        max_iterations = 500
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try to find a swap that reduces size
            # Pick a random selected block A
            if not selected:
                break
            a = random.choice(list(selected))
            
            # Snapshot
            old_selected = set(selected)
            old_coverage = coverage[:]
            old_size = current_size
            
            # Remove A
            remove_block(a)
            
            # Identify broken sets
            broken = [ti for ti in range(num_t) if coverage[ti] == 0]
            
            if not broken:
                # Pure removal
                current_size -= 1
                improved = True
                continue
            
            # Find best B
            # Count how many broken sets each candidate covers
            # Candidates are blocks covering at least one broken set
            # We can iterate through broken sets and collect candidates
            candidate_counts = {}
            for ti in broken:
                for ki in t_to_k_indices[ti]:
                    if ki not in selected:
                        candidate_counts[ki] = candidate_counts.get(ki, 0) + 1
                        
            if not candidate_counts:
                # Can't fix? Revert
                selected = old_selected
                coverage = old_coverage
                current_size = old_size
                continue
                
            # Pick best candidate
            best_b = max(candidate_counts, key=candidate_counts.get)
            
            # Add B
            add_block(best_b)
            
            # Remove new redundancies
            to_remove = [ki for ki in selected if is_redundant(ki)]
            for ki in to_remove:
                remove_block(ki)
                current_size -= 1
                
            if current_size < old_size:
                improved = True
            else:
                # Revert
                selected = old_selected
                coverage = old_coverage
                current_size = old_size
                
        return list(selected)

    # Generate initial solutions using Greedy
    def run_greedy(order):
        coverage = [0] * num_t
        uncovered = [True] * num_t
        selected = set()
        uncovered_count = num_t
        
        # We need to efficiently find the block with max coverage of uncovered sets.
        # This is the hard part.
        # Simple approach: Iterate through all blocks, calculate coverage. O(N_k * 56).
        # 75k * 56 ~ 4M ops.
        # We do this until uncovered_count == 0.
        # If we pick ~300 blocks, total ops ~ 300 * 4M = 1.2B. Too slow in Python.
        
        # Optimization: Use a priority queue or bucketing?
        # Or just use the previous method but optimized?
        # The previous method used buckets. Let's reuse that logic but ensure it's fast.
        
        # Actually, for v=19, N_k=75582.
        # 75582 * 56 = 4,232,592 operations per greedy step.
        # If we need 335 steps, that's 1.4 billion operations.
        # Python does ~10M-100M ops/sec. This will take 10-100 seconds.
        # We have 120s limit.
        # We need a faster greedy or fewer restarts.
        
        # Let's use a "Randomized Greedy" that picks random blocks but biases towards those covering more uncovered sets?
        # Or just use the bucket method from the original code, which is faster because it updates incrementally.
        
        # The original code's bucket method is:
        # 1. Init buckets with all blocks in bucket[max_cov].
        # 2. Pop from highest non-empty bucket.
        # 3. If stale, re-bucket.
        # 4. Select.
        # 5. Update coverage of neighbors.
        
        # This is O(N_k * 56) for the update step per selection.
        # Total O(N_selected * N_k * 56)? No, the update is only for neighbors of the selected block.
        # Number of neighbors is number of blocks sharing a t-subset with the selected block.
        # Each t-subset is in C(19-5, 8-5) = C(14,3) = 364 blocks.
        # So each selected block affects ~56 * 364 / (overlap factor) blocks?
        # Roughly, the update step is fast.
        
        # Let's stick to the bucket greedy for initialization, but add the Local Search.
        
        max_cov = len(k_to_t_indices[0])
        buckets = [[] for _ in range(max_cov + 1)]
        for ki in order:
            buckets[max_cov].append(ki)
            
        current_max = max_cov
        blocks = []
        selected_set = set()
        
        # We need to track coverage for the "stale" check in buckets?
        # The original code did:
        # if coverage[ki] != current_max: re-bucket.
        # This requires maintaining `coverage` array for all blocks?
        # Yes, `coverage[ki]` is the number of *uncovered* t-subsets covered by block ki.
        
        coverage = [max_cov] * num_k
        uncovered = [True] * num_t
        uncovered_count = num_t
        
        while uncovered_count > 0:
            while current_max >= 0 and not buckets[current_max]:
                current_max -= 1
            if current_max < 0:
                break
                
            ki = buckets[current_max].pop()
            
            if ki in selected_set:
                continue
                
            if coverage[ki] != current_max:
                buckets[coverage[ki]].append(ki)
                continue
                
            if coverage[ki] == 0:
                continue
                
            # Select ki
            selected_set.add(ki)
            blocks.append(ki)
            
            # Update
            for ti in k_to_t_indices[ki]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    uncovered_count -= 1
                    for other_ki in t_to_k_indices[ti]:
                        if other_ki not in selected_set:
                            coverage[other_ki] -= 1
                            
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run a few greedy instances
    for i in range(5):
        order = list(range(num_k))
        if i > 0:
            random.shuffle(order)
        init_blocks = run_greedy(order)
        
        # Improve with Local Search
        improved_blocks = local_search_v2(init_blocks)
        
        if len(improved_blocks) < best_len:
            best_len = len(improved_blocks)
            best_blocks = improved_blocks
            
    # Convert indices to lists
    return [list(all_k_subsets[ki]) for ki in best_blocks]
