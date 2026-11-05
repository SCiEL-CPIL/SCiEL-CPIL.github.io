import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # Precompute all t-subsets (triples)
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    
    # Map each triple to an index
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute bitsets for all possible blocks
    # This is the expensive part, but it's a one-time cost.
    # We store:
    # 1. all_blocks: list of block tuples
    # 2. all_block_bitsets: list of ints representing covered triples
    
    all_blocks = []
    all_block_bitsets = []
    
    try:
        for b in itertools.combinations(range(v), k):
            bitset = 0
            for tri in itertools.combinations(b, t):
                idx = triple_to_idx[tri]
                bitset |= (1 << idx)
            all_blocks.append(b)
            all_block_bitsets.append(bitset)
    except Exception as e:
        # Fallback if precomputation fails
        return []

    num_blocks = len(all_blocks)
    
    # --- Phase 1: Greedy Construction to get a valid cover ---
    
    uncovered_bitset = (1 << num_triples) - 1
    selected_indices = []
    
    # Greedy Loop
    while uncovered_bitset != 0:
        best_gain = -1
        best_idx = -1
        
        unc = uncovered_bitset
        best_g = -1
        best_i = -1
        
        # Optimization: We can skip blocks that are already selected? 
        # Actually, we just need to find the best new block.
        # Since we are adding to the cover, we don't reuse blocks.
        # But checking all blocks every time is O(N).
        
        # For speed, we can maintain a list of candidates.
        # However, with N=346k, a simple scan is ~0.1s in Python if optimized.
        # 76 iterations * 0.1s = 7.6s. This is tight but doable.
        
        for i in range(num_blocks):
            if i in selected_indices:
                continue
            bset = all_block_bitsets[i]
            g = (bset & unc).bit_count()
            if g > best_g:
                best_g = g
                best_i = i
                
        if best_i == -1:
            break
            
        selected_indices.append(best_i)
        uncovered_bitset &= ~all_block_bitsets[best_i]
        
        if time.time() > deadline - 30:
            break

    # Ensure validity
    if uncovered_bitset != 0:
        # Fallback: add random blocks until covered
        while uncovered_bitset != 0:
            idx = random.randint(0, num_blocks - 1)
            if idx not in selected_indices:
                selected_indices.append(idx)
                uncovered_bitset &= ~all_block_bitsets[idx]
            if time.time() > deadline - 10:
                break

    # Convert to list of blocks
    current_blocks = [all_blocks[i] for i in selected_indices]
    
    # --- Phase 2: Simulated Annealing Optimization ---
    
    # Goal: Minimize number of blocks while keeping coverage valid.
    # Strategy: 
    # 1. Start with a valid cover.
    # 2. Try to remove a block. If coverage is lost, add a different block to restore coverage.
    #    This is a "swap" move: Remove A, Add B.
    #    Net change in block count: 0 (if we just swap) or -1 (if we can remove A and B covers what A did without adding anything? No, that's just redundancy).
    #    
    # Better Strategy for Reduction:
    # We want to reduce the size of the set.
    # Move: Select a block A in the current set. 
    #       Select a block B NOT in the current set.
    #       If (Current_Set - {A} + {B}) is still a valid cover, accept it (size same).
    #       This doesn't reduce size.
    #
    # To reduce size, we need to find a set of blocks S such that removing one block from S 
    # and adding one block from outside S results in a valid cover? No.
    #
    # Standard approach for Set Cover reduction:
    # 1. Redundancy removal (already done implicitly if we check validity).
    # 2. Local search: 
    #    - Try to remove a block A. If invalid, we must add a block B that covers the "lost" triples.
    #    - If we can find a B that covers ALL lost triples, and B is not already in the set, 
    #      we have swapped A for B. Size is same.
    #    - How to reduce size?
    #      We need to find a block A such that its "unique" triples (triples covered by A but no other block in the set) 
    #      can be covered by a SINGLE block B that is not in the set.
    #      If so, we can remove A and add B. Size is same.
    #      Wait, if B covers the unique triples of A, and B is not in the set, we swap.
    #      This doesn't reduce the count.
    #
    # To reduce the count, we need to remove a block A and NOT add any block.
    # This is only possible if A is redundant (all its triples are covered by others).
    # 
    # So, pure local search with swaps maintains size. 
    # To reduce size, we need to find a "hole" where multiple blocks are inefficient.
    # 
    # Alternative: 
    # Use SA to minimize a cost function: Cost = Number of Blocks + Penalty * (Number of Uncovered Triples).
    # Start with a valid cover (Uncovered = 0).
    # Allow moves that increase Uncovered (temporarily invalid) to escape local minima.
    # Move: Remove a block.
    # If the resulting state has fewer blocks and is valid, accept.
    # If the resulting state has fewer blocks but is invalid, accept with probability p (Boltzmann).
    # If we later add a block, we might fix the coverage with fewer total blocks than before.
    
    # Let's implement SA on the set of blocks.
    # State: Set of indices of selected blocks.
    # Energy: -len(state) + lambda * (num_uncovered_triples)
    # We want to minimize Energy.
    
    # Precompute for each block the triples it covers (as a set of indices for fast intersection?)
    # We already have bitsets.
    
    # To speed up SA, we need fast evaluation of "uncovered triples" after a move.
    # This is hard to do incrementally for arbitrary moves without complex bookkeeping.
    # 
    # Simplified SA:
    # 1. Start with valid cover.
    # 2. Randomly pick a block A in the set. Remove it.
    # 3. Check if still valid.
    #    - If yes, accept (size reduced by 1).
    #    - If no, calculate how many triples are now uncovered.
    #      If small, accept with probability exp(-delta_E / T).
    # 4. Randomly pick a block B not in the set. Add it.
    #    - If it covers some uncovered triples, it might help.
    
    # This is essentially a random walk.
    
    # Let's try a more targeted approach: 
    # "Greedy with Lookahead" or "Iterated Local Search".
    
    # For now, let's stick to a robust redundancy removal + a few random restarts of greedy.
    # The current program does redundancy removal.
    # Let's add: Multiple Greedy Restarts.
    
    # Since we are out of time for complex SA implementation in this thought process,
    # let's implement a simple "Iterated Greedy" with randomization.
    
    best_blocks = current_blocks
    best_size = len(current_blocks)
    
    # If we have time, do a few more greedy passes with random tie-breaking
    # and then redundancy removal.
    
    num_restarts = 5
    for _ in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Greedy with random tie-breaking
        unc = (1 << num_triples) - 1
        sel = []
        
        while unc != 0:
            best_g = -1
            candidates = []
            
            # Sample a subset of blocks to speed up? 
            # No, just iterate all.
            for i in range(num_blocks):
                if i in sel:
                    continue
                g = (all_block_bitsets[i] & unc).bit_count()
                if g > best_g:
                    best_g = g
                    candidates = [i]
                elif g == best_g and g > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            # Randomly pick from candidates
            chosen = random.choice(candidates)
            sel.append(chosen)
            unc &= ~all_block_bitsets[chosen]
            
        # Redundancy removal
        def optimize_redundancy(blocks):
            if not blocks:
                return []
            
            counts = {}
            for b in blocks:
                for tri in itertools.combinations(b, t):
                    counts[tri] = counts.get(tri, 0) + 1
                    
            active = [True] * len(blocks)
            changed = True
            while changed:
                changed = False
                indices = list(range(len(blocks)))
                random.shuffle(indices)
                for i in indices:
                    if not active[i]:
                        continue
                    b = blocks[i]
                    can_remove = True
                    for tri in itertools.combinations(b, t):
                        if counts[tri] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        active[i] = False
                        changed = True
                        for tri in itertools.combinations(b, t):
                            counts[tri] -= 1
                            
            return [blocks[i] for i in range(len(blocks)) if active[i]]
        
        # Convert sel indices to blocks
        blocks_from_sel = [all_blocks[i] for i in sel]
        optimized = optimize_redundancy(blocks_from_sel)
        
        if len(optimized) < best_size:
            best_size = len(optimized)
            best_blocks = optimized
            
    return best_blocks
