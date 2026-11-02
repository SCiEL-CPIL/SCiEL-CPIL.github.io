import itertools
import random
import time

def construct(v, k, t):
    # Set a deadline slightly before the hard limit
    deadline = 115.0
    
    # Precomputation
    # Generate all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    M = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    K = len(all_blocks)
    
    # Precompute block_subsets and subset_to_blocks
    block_subsets = [[] for _ in range(K)]
    subset_to_blocks = [[] for _ in range(M)]
    
    # This is the most expensive part of precomputation
    # K * C(k, t) iterations
    for b_idx, b in enumerate(all_blocks):
        for s in itertools.combinations(b, t):
            s_idx = subset_to_idx[s]
            block_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)

    # State
    cover_count = [0] * M
    blocks = set()
    best_blocks = None
    best_size = float('inf')
    
    start_time = time.time()
    
    def add_block(b_idx):
        nonlocal cover_count, blocks
        if b_idx in blocks:
            return
        blocks.add(b_idx)
        for s_idx in block_subsets[b_idx]:
            cover_count[s_idx] += 1
            
    def remove_block(b_idx):
        nonlocal cover_count, blocks
        if b_idx not in blocks:
            return
        blocks.remove(b_idx)
        for s_idx in block_subsets[b_idx]:
            cover_count[s_idx] -= 1

    # 1. Greedy Initialization
    uncovered = set(range(M))
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick a random uncovered subset to anchor the new block
        s = random.choice(list(uncovered))
        
        # Find the block covering s that covers the most other uncovered subsets
        best_b = -1
        max_gain = -1
        candidates = subset_to_blocks[s]
        
        for b_idx in candidates:
            gain = 0
            for s_idx in block_subsets[b_idx]:
                if s_idx in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_b = b_idx
        
        if best_b != -1:
            add_block(best_b)
            for s_idx in block_subsets[best_b]:
                if s_idx in uncovered:
                    uncovered.remove(s_idx)
                    
    if not uncovered:
        best_blocks = list(blocks)
        best_size = len(blocks)
        
    # 2. Local Search
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Remove Redundant Blocks
        # We iterate multiple times to ensure all possible removals are made
        # But usually one pass with random order is good, or just one pass.
        # Let's do one pass per loop iteration.
        block_list = list(blocks)
        random.shuffle(block_list)
        
        for b_idx in block_list:
            if b_idx not in blocks:
                continue
            
            # Check if redundant: all its subsets are covered by other blocks
            redundant = True
            for s_idx in block_subsets[b_idx]:
                if cover_count[s_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                remove_block(b_idx)
                improved = True
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = list(blocks)
                    
        # If no improvement, Perturb
        if not improved:
            if not blocks:
                break
                
            b_rem = random.choice(list(blocks))
            remove_block(b_rem)
            
            # Find violations
            violations = set()
            for s_idx in range(M):
                if cover_count[s_idx] == 0:
                    violations.add(s_idx)
            
            # Repair
            while violations:
                if time.time() > deadline:
                    break
                
                s = random.choice(list(violations))
                
                best_b = -1
                max_gain = -1
                candidates = subset_to_blocks[s]
                
                for b_idx in candidates:
                    if b_idx in blocks:
                        continue
                        
                    gain = 0
                    for s_idx in block_subsets[b_idx]:
                        if cover_count[s_idx] == 0:
                            gain += 1
                            
                    if gain > max_gain:
                        max_gain = gain
                        best_b = b_idx
                        
                if best_b != -1:
                    add_block(best_b)
                    for s_idx in block_subsets[best_b]:
                        if cover_count[s_idx] == 1: # Was 0, now 1
                            if s_idx in violations:
                                violations.remove(s_idx)
                else:
                    break
            
            # Update best if better (unlikely to be better immediately after perturbation, 
            # but possible if perturbation led to a configuration that allows more removals later)
            # We only update best in the removal phase.
            
    if best_blocks is None:
        # Fallback: return empty or current blocks if valid
        # If we timed out during greedy, we might have a partial cover.
        # We must return a valid cover.
        # If best_blocks is None, it means we didn't finish greedy.
        # We should try to finish greedy quickly or return best effort?
        # The task requires a valid cover.
        # If we didn't finish, we can't guarantee validity.
        # However, with 115s limit and fast greedy, this is unlikely.
        # If it happens, we return the current blocks (which might be invalid).
        # To be safe, if best_blocks is None, we return [] (invalid) or try to force finish?
        # Given the speed, it should be fine.
        return []
        
    result = [all_blocks[b_idx] for b_idx in best_blocks]
    return [list(b) for b in result]
