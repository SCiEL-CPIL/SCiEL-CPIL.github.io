import itertools
import time
import random
import numpy as np

# Precomputation for the specific task v=19, k=8, t=5
# However, the function signature is construct(v, k, t).
# The prompt says "The evaluator passes this task's parameters v, k and t."
# And "Keep the function names and signatures".
# But precomputation for general v, k, t is expensive if done inside construct.
# Given the constraint "Each evaluation must finish within 120 s", and v=19 is small,
# we can check if v, k, t match the known task and use precomputed structures,
# or just compute them on the fly if they are small enough.
# C(19, 5) = 11628 subsets. C(19, 8) = 75582 blocks.
# This is small enough to precompute in < 1s in Python if done carefully, 
# or we can hardcode the precomputation logic to run once.

# Since the prompt implies a single task (C(19,8,5)), but the interface is generic,
# I will implement a generic solver that optimizes for small v.
# If v is large, this might fail, but the task is specifically C(19,8,5).

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute all k-blocks and the t-subsets they cover
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # block_subsets[b] = list of indices of t-subsets covered by block b
    block_subsets = []
    for block in all_k_blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(subset_to_idx[tuple(sub)])
        block_subsets.append(subs)
        
    # subset_blocks[s] = list of block indices that cover subset s
    # This is useful for finding replacement candidates
    subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, subs in enumerate(block_subsets):
        for s_idx in subs:
            subset_blocks[s_idx].append(b_idx)

    # Helper to convert block index to list of points
    def get_block_points(b_idx):
        return list(all_k_blocks[b_idx])

    # 3. Greedy Initialization
    def greedy_init():
        uncovered = set(range(num_t_subsets))
        blocks = []
        # For speed, we can pre-compute for each block how many subsets it covers
        # But the "uncovered" set changes.
        
        # Optimization: 
        # Instead of iterating all blocks for each step, 
        # we can iterate points? No, blocks are the unit of selection.
        # With 75k blocks and ~300 steps, 22.5M checks. 
        # Each check: intersection of block_subsets[b] with uncovered.
        # Using sets: len(set(block_subsets[b]) & uncovered)
        
        # To speed up, we can maintain a count of uncovered subsets per block?
        # When we add a block, we update counts for all blocks that share subsets.
        # This is complex. Let's try the simple set intersection first.
        
        while uncovered:
            best_block = -1
            best_count = -1
            # Randomize start to break symmetry?
            # Iterate all blocks
            for b_idx in range(num_k_blocks):
                # Count how many subsets in this block are in uncovered
                # Optimization: if best_count is already high, skip?
                count = 0
                for s_idx in block_subsets[b_idx]:
                    if s_idx in uncovered:
                        count += 1
                        if count <= best_count:
                            break
                if count > best_count:
                    best_count = count
                    best_block = b_idx
                    if best_count == len(block_subsets[best_block]):
                        break # Perfect block found
            
            if best_block == -1:
                break # Should not happen if valid cover exists
            
            blocks.append(best_block)
            # Update uncovered
            for s_idx in block_subsets[best_block]:
                uncovered.discard(s_idx)
                
            # Time check
            if time.time() > deadline - 10:
                break

        return blocks

    # 4. Local Search
    def local_search(initial_blocks):
        # Convert to list of indices
        blocks = list(initial_blocks)
        if not blocks:
            return blocks
            
        # Calculate coverage
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in blocks:
            coverage[block_subsets[b_idx]] += 1
        
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Removal
            new_blocks = []
            for b_idx in blocks:
                is_critical = False
                for s_idx in block_subsets[b_idx]:
                    if coverage[s_idx] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    # Remove b_idx
                    coverage[block_subsets[b_idx]] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks
            
            # Phase 2: Replacement
            # For each block, see if we can replace it with another block
            # that covers all its critical subsets
            for i in range(len(blocks)):
                b_idx = blocks[i]
                criticals = []
                for s_idx in block_subsets[b_idx]:
                    if coverage[s_idx] == 1:
                        criticals.append(s_idx)
                
                if not criticals:
                    # Should have been removed in Phase 1, but maybe coverage changed?
                    # If no criticals, it's redundant.
                    coverage[block_subsets[b_idx]] -= 1
                    blocks.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                
                # Find candidates that cover all criticals
                # Intersection of subset_blocks[s] for s in criticals
                candidates = set(subset_blocks[criticals[0]])
                for s_idx in criticals[1:]:
                    candidates.intersection_update(subset_blocks[s_idx])
                
                if not candidates:
                    continue
                    
                # Pick a candidate
                # Heuristic: Pick one that minimizes the increase in "fragility"
                # Or just random
                # Let's pick the one that covers the most subsets with coverage == 2
                # This increases the buffer for those subsets.
                best_cand = -1
                best_score = -1
                # Limit candidates to check for speed if list is huge
                cand_list = list(candidates)
                if len(cand_list) > 50:
                    random.shuffle(cand_list)
                    cand_list = cand_list[:50]
                
                for c_idx in cand_list:
                    if c_idx == b_idx:
                        continue
                    # Calculate score
                    score = 0
                    for s_idx in block_subsets[c_idx]:
                        if coverage[s_idx] == 2:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cand = c_idx
                        
                if best_cand != -1:
                    # Replace
                    coverage[block_subsets[b_idx]] -= 1
                    coverage[block_subsets[best_cand]] += 1
                    blocks[i] = best_cand
                    improved = True
                    break # Restart loop to allow removals
                    
            # Time check
            if time.time() > deadline - 10:
                break
                
        return blocks

    # Main Loop
    best_blocks = greedy_init()
    best_len = len(best_blocks)
    
    # ILS: Perturb and Local Search
    while time.time() < deadline - 5:
        # Perturb: Remove 1-2 blocks and repair
        curr = list(best_blocks)
        if len(curr) < 2:
            break
            
        # Remove random blocks
        num_remove = min(2, len(curr))
        for _ in range(num_remove):
            if not curr: break
            idx = random.randint(0, len(curr) - 1)
            curr.pop(idx)
            
        # Repair: Greedy add until valid
        # Calculate coverage of curr
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in curr:
            coverage[block_subsets[b_idx]] += 1
            
        uncovered = [i for i in range(num_t_subsets) if coverage[i] == 0]
        
        while uncovered:
            best_block = -1
            best_count = -1
            for b_idx in range(num_k_blocks):
                count = 0
                for s_idx in block_subsets[b_idx]:
                    if coverage[s_idx] == 0:
                        count += 1
                        if count <= best_count:
                            break
                if count > best_count:
                    best_count = count
                    best_block = b_idx
            if best_block == -1:
                break
            curr.append(best_block)
            coverage[block_subsets[best_block]] += 1
            
        # Local Search
        curr = local_search(curr)
        
        if len(curr) < best_len:
            best_blocks = curr
            best_len = len(curr)
            
    # Convert indices back to point lists
    return [get_block_points(b) for b in best_blocks]

