import itertools
import numpy as np
from scipy.special import comb
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a Covering Density heuristic.
    
    Strategy:
    1. Initialize with a dense, highly symmetric pool of blocks derived from cyclic structures.
    2. Use vectorized greedy set cover to get an initial covering.
    3. Apply fast local search:
       a. Delete redundant blocks (all t-subsets covered >1 time).
       b. Swap blocks: If a block is 'expensive' (covers many unique subsets), try to replace it 
          with a cheaper block that covers the same unique subsets, or simply delete if possible.
          Actually, standard heuristic: 
          - Try to remove a block. If it breaks coverage, find a replacement block that covers 
            the broken subsets and has low marginal cost (covers many other subsets that are 
            already well-covered, or covers the broken ones efficiently).
    """
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Precompute t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Generate candidate blocks
    random.seed(42)
    
    # Generate diverse base blocks
    base_blocks = []
    
    # 1. Structured bases: Cyclic shifts of small sets to ensure symmetry
    # We want blocks that are "spread out". 
    # For v=24, k=7. 
    # Let's generate bases by taking random samples but ensuring they are distinct.
    
    # Add random base blocks
    for _ in range(100):
        base = sorted(random.sample(range(v), k))
        base_blocks.append(base)
    
    # Add some "structured" bases: e.g., every 3rd element, etc.
    for step in [2, 3, 4, 5, 6, 7, 8, 11, 13]:
        if step < v:
            base = [i * step % v for i in range(k)]
            base = sorted(base)
            if len(base) == k: # Ensure distinct
                base_blocks.append(base)
    
    # Generate all cyclic shifts
    candidate_blocks = []
    seen_blocks = set()
    for base in base_blocks:
        for shift in range(v):
            block = tuple(sorted((x + shift) % v for x in base))
            if block not in seen_blocks:
                seen_blocks.add(block)
                candidate_blocks.append(block)
    
    # For each candidate block, compute the t-subset indices it covers
    # We will store these as a list of arrays
    block_t_indices = []
    for block in candidate_blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[t_sub])
        block_t_indices.append(np.array(indices, dtype=np.int32))
    
    # Create a matrix M where M[b, s] = 1 if block b covers subset s
    # This is sparse, but for v=24, k=7, t=3, C(7,3)=35 triples per block.
    # Number of blocks ~ 100*24 = 2400. Number of subsets C(24,3)=2024.
    # Matrix size 2400x2024 ~ 5M entries. This is manageable in memory.
    
    # However, building the full matrix might be slow. 
    # Let's stick to the list of arrays but use a smarter greedy.
    
    # Greedy Set Cover
    uncovered = np.ones(n_t_subsets, dtype=np.bool_)
    selected_block_indices = []
    
    # To speed up greedy, we can pre-compute the count of uncovered subsets for each block?
    # No, uncovered changes.
    
    # Optimization: 
    # Instead of looping through all candidates to find max coverage, 
    # we can maintain a count of how many uncovered subsets each block covers.
    # When we add a block, we decrement the count for all blocks that share those subsets.
    # This is O(N_blocks * N_subsets_per_block) per step, which is similar to the original.
    
    # Let's try a different approach for Greedy:
    # 1. Calculate initial coverage of uncovered sets for all blocks.
    # 2. Pick max.
    # 3. Update.
    
    # This is still O(N) per step. With ~2000 blocks and ~2000 steps (worst case), it's 4M operations.
    # Python loops are slow.
    
    # Let's use the matrix M for vectorized greedy.
    # M: (num_candidates, n_t_subsets) boolean matrix
    M = np.zeros((len(candidate_blocks), n_t_subsets), dtype=np.bool_)
    for i, indices in enumerate(block_t_indices):
        M[i, indices] = True
    
    # Greedy with Matrix
    # uncovered is a vector of length n_t_subsets
    # We want argmax(M @ uncovered)
    
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
        
        # Calculate coverage of uncovered sets for each candidate
        # M is boolean, uncovered is boolean. 
        # Sum of (M[i, :] & uncovered)
        # Using matrix multiplication: M @ uncovered (where uncovered is 0/1)
        # Since M is bool, we can cast to int or use bitwise.
        # Numpy dot product on bools might be slow or unsupported for sum.
        # Let's use: np.sum(M & uncovered[:, None], axis=1)
        
        # This creates a large temporary array (N_blocks x N_subsets).
        # 2400 x 2024 ~ 5M bools. 5MB. This is fine.
        
        covers = np.sum(M & uncovered[:, None], axis=1)
        
        best_block_idx = np.argmax(covers)
        max_cover = covers[best_block_idx]
        
        if max_cover == 0:
            break
        
        selected_block_indices.append(best_block_idx)
        
        # Update uncovered
        uncovered[M[best_block_idx, :]] = False
    
    # Convert to list of blocks
    blocks = [list(candidate_blocks[i]) for i in selected_block_indices]
    
    # If we didn't get a complete covering, add more blocks
    if uncovered.any():
        # Add random blocks until covered
        while uncovered.any():
            if time.time() - start_time > time_limit:
                break
            block = tuple(sorted(random.sample(range(v), k)))
            if block in seen_blocks:
                continue
            seen_blocks.add(block)
            indices = []
            for t_sub in itertools.combinations(block, t):
                indices.append(t_subset_to_idx[t_sub])
            indices_arr = np.array(indices, dtype=np.int32)
            if np.sum(uncovered[indices_arr]) > 0:
                blocks.append(list(block))
                uncovered[indices_arr] = False
    
    # Local Search: Minimize block count
    # 1. Build coverage counts
    coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
    block_t_indices_current = []
    for block in blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[t_sub])
        indices_arr = np.array(indices, dtype=np.int32)
        block_t_indices_current.append(indices_arr)
        coverage_counts[indices_arr] += 1
    
    # Helper: Check if a block is redundant (all its triples have count > 1)
    def is_redundant(block_idx):
        return np.all(coverage_counts[block_t_indices_current[block_idx]] > 1)
    
    # Helper: Remove a block
    def remove_block(block_idx):
        coverage_counts[block_t_indices_current[block_idx]] -= 1
        blocks.pop(block_idx)
        block_t_indices_current.pop(block_idx)
    
    # Phase 1: Delete redundant blocks
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(blocks)):
            if is_redundant(i):
                remove_block(i)
                changed = True
                break  # Restart loop as indices changed
    
    # Phase 2: Swap to reduce count
    # Idea: For each block, try to find a replacement block from the candidate pool 
    # that covers all the "unique" triples of the current block (triples with count 1 
    # that would become 0 if we removed the current block) AND has a lower "cost".
    # Cost can be defined as: number of triples it covers that are already well-covered (count > 1).
    # Or simply: Does the replacement block exist that covers the critical triples?
    # If we remove block i, the critical triples are those with count == 1 in block i.
    # We need a block j such that it covers ALL critical triples of i.
    # If such a j exists, and j is not already in the set (or if it is, we just remove i), 
    # we can replace i with j. 
    # Wait, if j is already in the set, we can just remove i if j covers the criticals? 
    # No, if j is in the set, the criticals of i are already covered by j? 
    # If count[i, triple] == 1, and j covers triple, then count[j, triple] >= 1.
    # So if we remove i, the triple is still covered by j. 
    # So if there is ANY block in the current set (or candidate pool) that covers all critical triples of i, 
    # we can remove i.
    
    # Let's refine:
    # 1. Identify critical triples for block i: C_i = {s | s in i and count[s] == 1}
    # 2. If C_i is empty, remove i (redundant).
    # 3. Else, check if any other block in the current set covers ALL triples in C_i.
    #    If yes, remove i.
    # 4. Else, check if any candidate block covers ALL triples in C_i.
    #    If yes, replace i with that candidate (if candidate is not already in set).
    #    If candidate is in set, then it covers C_i, so we could have removed i in step 3? 
    #    Wait, if candidate j is in set, and j covers C_i, then for any s in C_i, count[s] >= 1 (from j).
    #    But we defined C_i as triples with count == 1. 
    #    So if j covers s, and count[s] == 1, then j is the ONLY block covering s.
    #    So if j is in the set, and j covers s, then i does NOT cover s? 
    #    Contradiction: s is in i. So i covers s.
    #    So count[s] would be at least 2 (from i and j).
    #    Therefore, if s is in C_i (count 1), NO other block in the set covers s.
    #    So Step 3 is never true if C_i is not empty? 
    #    Wait. If s is in C_i, count[s] == 1. This means ONLY block i covers s.
    #    So no other block in the set covers s.
    #    So we MUST find a new block from the candidate pool that covers s.
    
    # So the algorithm is:
    # For each block i:
    #   C_i = {s in i | count[s] == 1}
    #   If C_i is empty: remove i.
    #   Else:
    #       Find a candidate block j (not in current set) that covers ALL s in C_i.
    #       If found:
    #           Remove i.
    #           Add j.
    #           Update counts.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(len(blocks)):
            # Get critical triples
            indices_i = block_t_indices_current[i]
            critical_mask = (coverage_counts[indices_i] == 1)
            critical_indices = indices_i[critical_mask]
            
            if len(critical_indices) == 0:
                # Redundant
                remove_block(i)
                improved = True
                break
            
            # Find a replacement
            # We need a block j such that M[j, critical_indices] are all True.
            # Let's check candidates.
            # To speed up, we can pre-filter candidates that cover the first critical triple.
            # Then check if they cover the rest.
            
            if len(critical_indices) > 0:
                first_crit = critical_indices[0]
                # Find candidates covering first_crit
                # M is (N_cand, N_sub)
                # Candidates covering first_crit: np.where(M[:, first_crit])[0]
                potential_cands = np.where(M[:, first_crit])[0]
                
                # Filter out candidates that are already in the set?
                # We don't strictly need to, but adding a duplicate is bad.
                # We can check if the candidate is in `blocks`.
                # Since blocks are lists, checking membership is slow.
                # Let's store block tuples in a set `current_block_set`.
                
                # For now, let's just pick the first candidate that covers all criticals.
                # To avoid adding duplicates, we can check if the candidate tuple is in a set of current block tuples.
                
                found_replacement = False
                for cand_idx in potential_cands:
                    # Check if it covers all criticals
                    if np.all(M[cand_idx, critical_indices]):
                        # Check if this candidate is already in the set
                        cand_block = candidate_blocks[cand_idx]
                        # Simple check: is it in blocks?
                        # This is O(N_blocks) per check. 
                        # Let's maintain a set of current block tuples.
                        
                        # To optimize, let's just add it and handle duplicates in a cleanup phase?
                        # Or better, maintain a set.
                        
                        # Let's assume for now we just add it. 
                        # If we add a duplicate, we have two identical blocks.
                        # One of them will be redundant in the next pass.
                        
                        # Remove i
                        remove_block(i)
                        
                        # Add cand_idx
                        blocks.append(list(cand_block))
                        new_indices = block_t_indices[cand_idx].copy()
                        block_t_indices_current.append(new_indices)
                        coverage_counts[new_indices] += 1
                        
                        improved = True
                        found_replacement = True
                        break
                
                if found_replacement:
                    break # Restart loop
    
    # Phase 3: Final cleanup of duplicates and redundancies
    # Remove duplicates
    unique_blocks = []
    seen = set()
    for b in blocks:
        if tuple(b) not in seen:
            seen.add(tuple(b))
            unique_blocks.append(b)
    blocks = unique_blocks
    
    # Recalculate counts
    coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
    block_t_indices_current = []
    for block in blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[t_sub])
        indices_arr = np.array(indices, dtype=np.int32)
        block_t_indices_current.append(indices_arr)
        coverage_counts[indices_arr] += 1
        
    # Remove redundant blocks again
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(blocks)):
            if np.all(coverage_counts[block_t_indices_current[i]] > 1):
                remove_block(i)
                changed = True
                break

    # Verify coverage
    final_counts = np.zeros(n_t_subsets, dtype=np.int32)
    for block in blocks:
        for t_sub in itertools.combinations(block, t):
            final_counts[t_subset_to_idx[t_sub]] += 1
    
    # If there are uncovered subsets, add blocks to cover them
    if np.any(final_counts == 0):
        uncovered_mask = (final_counts == 0)
        while uncovered_mask.any():
            if time.time() - start_time > time_limit:
                break
            # Find a block that covers the most uncovered subsets
            best_block = None
            best_count = 0
            for _ in range(100):
                block = tuple(sorted(random.sample(range(v), k)))
                count = 0
                for t_sub in itertools.combinations(block, t):
                    if uncovered_mask[t_subset_to_idx[t_sub]]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = block
            if best_block is not None:
                blocks.append(list(best_block))
                for t_sub in itertools.combinations(best_block, t):
                    final_counts[t_subset_to_idx[t_sub]] += 1
                    uncovered_mask[t_subset_to_idx[t_sub]] = False
    
    return blocks
