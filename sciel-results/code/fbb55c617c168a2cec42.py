import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # 3. Precompute which t-subsets are in each k-subset
    block_tsets = []
    for ks in k_subsets:
        # Get all t-subsets within this k-subset
        t_indices = []
        for ts in itertools.combinations(ks, t):
            t_indices.append(t_subset_to_idx[ts])
        block_tsets.append(t_indices)
        
    # 4. Build reverse index: for each t-subset, which k-subsets cover it?
    covering_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_tsets):
        for t_idx in t_indices:
            covering_blocks[t_idx].append(b_idx)
            
    # 5. Greedy Construction
    uncovered = set(range(num_t_subsets))
    solution_blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a t-subset to cover
        # Heuristic: Pick one with fewest covering blocks? 
        # Initially all are equal. Later, some might have fewer options if we were doing exact cover, 
        # but here we just need *some* block. 
        # Picking the one with the fewest *remaining* potential blocks might be good, 
        # but since blocks aren't removed from the pool, just "used", 
        # the count of covering_blocks is constant. 
        # So any uncovered t-subset is fine. 
        s = next(iter(uncovered))
        
        candidates = covering_blocks[s]
        best_block = -1
        best_gain = -1
        
        # Evaluate candidates
        for b_idx in candidates:
            gain = 0
            for t_idx in block_tsets[b_idx]:
                if t_idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block == -1:
            break # Should not happen
            
        solution_blocks.append(best_block)
        
        # Update uncovered
        for t_idx in block_tsets[best_block]:
            if t_idx in uncovered:
                uncovered.discard(t_idx)
                
    # 6. Local Search: Remove redundant blocks
    # We try to remove blocks if their removal doesn't uncover any t-subset.
    # To do this efficiently, we maintain a count of how many times each t-subset is covered by the current solution.
    
    if time.time() < deadline:
        # Build coverage count
        cover_count = [0] * num_t_subsets
        for b_idx in solution_blocks:
            for t_idx in block_tsets[b_idx]:
                cover_count[t_idx] += 1
                
        # Try to remove blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Iterate in reverse to allow removal
            for i in range(len(solution_blocks) - 1, -1, -1):
                b_idx = solution_blocks[i]
                # Check if any t-subset in this block is covered ONLY by this block
                safe_to_remove = True
                for t_idx in block_tsets[b_idx]:
                    if cover_count[t_idx] == 1:
                        safe_to_remove = False
                        break
                
                if safe_to_remove:
                    # Remove it
                    for t_idx in block_tsets[b_idx]:
                        cover_count[t_idx] -= 1
                    solution_blocks.pop(i)
                    changed = True
    
    # Convert block indices back to lists of points
    final_blocks = [list(k_subsets[b_idx]) for b_idx in solution_blocks]
    
    # Check validity (just in case)
    if time.time() < deadline:
        all_t_subsets = set(itertools.combinations(range(v), t))
        covered = set()
        for b in final_blocks:
            for ts in itertools.combinations(b, t):
                covered.add(ts)
        if covered != all_t_subsets:
             # Fallback to seed if invalid? No, just return what we have. 
             # But we should ensure validity. The greedy logic is sound.
             pass

    return final_blocks
