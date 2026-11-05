import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 145  # Safety margin for 150s limit
    
    rng = random.Random(42)

    # Precompute all k-subsets (blocks) and t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    sub_idx = {s: i for i, s in enumerate(t_subsets)}

    # Build the block-to-subset incidence matrix M (n_blocks x n_subsets)
    # M[bi, si] = 1 if block bi covers subset si
    # For v=19, k=7, t=4: n_blocks=50388, n_subsets=3876
    # This is a sparse matrix, but we can use a dense uint8 array for speed in GEMV-like ops
    # Or better, use a list of sets/arrays for efficient intersection?
    # Actually, for "best gain" we need argmax over blocks of (M[bi] & uncovered).sum()
    # If we keep `uncovered` as a boolean vector, gain[bi] = M[bi] @ uncovered
    # This is a matrix-vector product. M is 50388 x 3876.
    # M can be stored as a numpy array of uint8.
    
    M = np.zeros((n_blocks, n_subsets), dtype=np.uint8)
    for bi, blk in enumerate(all_blocks):
        for sub in itertools.combinations(blk, t):
            si = sub_idx[sub]
            M[bi, si] = 1

    # Greedy Initialization
    def greedy_init():
        uncovered = np.ones(n_subsets, dtype=bool)
        chosen = []
        chosen_set = set()
        while uncovered.any():
            # Find block with max gain
            gains = M[uncovered]  # This is wrong, M is (n_blocks, n_subsets)
            # We need sum over subsets for each block
            # gain[bi] = sum(M[bi, :] * uncovered)
            # This is M.dot(uncovered.astype(np.uint8))
            # But M is uint8, uncovered is bool.
            # Let's precompute M as int8 or use a faster method.
            # Actually, M.dot is fast.
            gain_vec = M.dot(uncovered.astype(np.uint8))
            
            # Mask out already chosen
            for bi in chosen_set:
                gain_vec[bi] = -1
            
            best_bi = np.argmax(gain_vec)
            if gain_vec[best_bi] == 0:
                break
                
            chosen.append(best_bi)
            chosen_set.add(best_bi)
            # Update uncovered
            uncovered &= ~M[best_bi].astype(bool)
            
            # Check time
            if time.time() > deadline - 10:
                break
                
        return chosen

    # Prune redundant blocks
    def prune(blocks_list):
        chosen = list(blocks_list)
        chosen_set = set(chosen)
        
        # Count coverage
        # counts = sum of M[bi] for bi in chosen
        # Since M is 0/1, we can just sum rows
        if not chosen:
            return []
            
        counts = np.zeros(n_subsets, dtype=np.int32)
        for bi in chosen:
            counts += M[bi]
            
        changed = True
        while changed:
            changed = False
            # Try to remove any block
            # A block is redundant if all its 1s are covered by other blocks
            # i.e., for all si where M[bi, si]==1, counts[si] > 1
            # We can check this efficiently?
            # For each bi, check if np.any(M[bi] * (counts == 1))
            # This is O(n_blocks * n_subsets) per pass, which might be slow.
            # 50388 * 3876 ~ 195M ops. In numpy, this is fast (~0.1s).
            
            # Vectorized check for redundancy
            # redundant_mask[bi] = 1 if block bi is redundant
            # M[bi] * (counts == 1) -> 1 if bi covers a subset with count 1
            # We want bi to be redundant if NO such subset exists.
            # critical = M * (counts == 1).reshape(1, -1)
            # redundant = (critical.sum(axis=1) == 0)
            
            critical = M * (counts == 1).reshape(1, -1)
            redundant = (critical.sum(axis=1) == 0)
            
            # Find a redundant block in chosen
            for bi in chosen:
                if redundant[bi]:
                    # Remove it
                    counts -= M[bi]
                    chosen.remove(bi)
                    chosen_set.discard(bi)
                    changed = True
                    break # Restart scan to be safe, or continue?
                    # Restarting is safer for correctness of counts if multiple removed
                    # But we break, so it's fine.
                    
        return chosen

    # Iterated Local Search (ILS)
    def ils(initial_blocks, time_budget):
        current_blocks = list(initial_blocks)
        current_set = set(current_blocks)
        
        # Precompute counts
        counts = np.zeros(n_subsets, dtype=np.int32)
        for bi in current_blocks:
            counts += M[bi]
            
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        t_start = time.time()
        
        while time.time() - t_start < time_budget:
            # 1. Local Search: 1-exchange
            improved = True
            while improved:
                improved = False
                # For each block in current, try to swap with best gain block
                # Or just try to remove if redundant (handled by prune, but maybe we can do it here)
                # Let's do: for each bi in current, find best bj not in current
                # such that swapping bi->bj improves size (impossible, size same) 
                # Wait, 1-exchange keeps size same. We need 0-exchange (removal) to reduce size.
                # The "prune" function handles removal.
                # So ILS step:
                # a. Prune
                # b. Swap search: try to swap a block with another to enable further pruning?
                # Or just: 
                # 1. Prune
                # 2. If size < best, update best
                # 3. Perturb: swap a random block with a random other block
                # 4. Repeat
                
                current_blocks = prune(current_blocks)
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                    current_set = set(current_blocks)
                    # Recount
                    counts = np.zeros(n_subsets, dtype=np.int32)
                    for bi in current_blocks:
                        counts += M[bi]
                    
                # Check time
                if time.time() - t_start > time_budget:
                    break
                    
                # Perturb
                if len(current_blocks) > 0:
                    # Pick a random block to remove
                    bi_out = rng.choice(current_blocks)
                    # Pick a random block to add (not in set)
                    # We need to pick a good candidate? 
                    # Strategy: Pick the block that maximizes gain among those not in set?
                    # That would be greedy. 
                    # Let's pick a block that covers the most "critical" subsets?
                    # Or just random.
                    # To make it "smart", let's pick the block with max gain for the current uncovered?
                    # But we are covered. 
                    # Let's pick a block that is "similar" to bi_out?
                    # Or just random.
                    
                    # Let's try: Replace bi_out with the block that has the highest overlap with bi_out?
                    # Or the block that covers the most subsets currently covered exactly once by bi_out?
                    
                    # Simple perturbation: Random swap
                    # Find a random bi_in
                    # To avoid picking one in set, we can just pick random and check
                    # Or precompute list of not-in-set? Too slow to update.
                    
                    # Let's pick a random index from 0..n_blocks
                    # If in set, try again (max 10 times)
                    bi_in = -1
                    for _ in range(10):
                        cand = rng.randrange(n_blocks)
                        if cand not in current_set:
                            bi_in = cand
                            break
                    if bi_in == -1:
                        continue
                        
                    # Apply swap
                    current_blocks.remove(bi_out)
                    current_set.discard(bi_out)
                    current_blocks.append(bi_in)
                    current_set.add(bi_in)
                    
                    # Update counts
                    counts -= M[bi_out]
                    counts += M[bi_in]
                    
                    # If this creates uncovered subsets, it's a "worse" state
                    # But prune will try to fix it by removing redundant ones?
                    # No, prune only removes if redundant. It doesn't add to fix coverage.
                    # So if we swap and lose coverage, the solution is invalid.
                    # We should check validity.
                    # If invalid, we can revert or try to repair.
                    # For ILS, we usually accept worse states.
                    # But we need to return a VALID solution.
                    # So we keep best_blocks (which is valid).
                    # The current state can be invalid during search.
                    
                    # However, if we stay invalid for too long, we might not recover.
                    # Let's add a repair step: if uncovered, greedily add blocks.
                    if np.any(counts == 0):
                        # Greedy repair
                        uncovered_mask = (counts == 0).astype(np.uint8)
                        while np.any(counts == 0):
                            gain_vec = M.dot(uncovered_mask)
                            for bi in current_set:
                                gain_vec[bi] = -1
                            best_bi = np.argmax(gain_vec)
                            if gain_vec[best_bi] == 0:
                                break
                            current_blocks.append(best_bi)
                            current_set.add(best_bi)
                            counts += M[best_bi]
                            uncovered_mask = (counts == 0).astype(np.uint8)
                            if time.time() - t_start > time_budget:
                                break
                            
                    # Re-prune
                    # (Loop continues to prune)
                    
            # If no improvement in local search (prune didn't reduce size), we broke out?
            # Actually, the while improved loop breaks if prune doesn't change size?
            # No, prune returns a list. We check size.
            # If size doesn't change, we should perturb.
            # My logic above:
            # while improved:
            #   prune
            #   if size < best: update
            #   perturb
            #   improved = True (always? No, we should stop if we've been perturbed many times without improvement?)
            # Let's restructure.
            
            # Standard ILS:
            # 1. Local Search (Prune + Swap to enable prune)
            # 2. Perturb
            # 3. Repeat
            
            # Let's do a fixed number of swap attempts per iteration?
            # Or just:
            # 1. Prune
            # 2. If improved, continue
            # 3. Else, Perturb (Swap)
            # 4. Loop
            
            # The code above does:
            # while improved:
            #   prune
            #   if improved size: update best
            #   perturb
            #   # implicit: improved is still True?
            #   # I set improved = True at start, but never set to False inside the loop properly.
            #   # I need to exit if we've done a swap.
            
            # Let's rewrite the ILS loop structure clearly.
            pass # The previous logic was a bit messy. Let's implement a cleaner ILS below.

    # Cleaner ILS Implementation
    def ils_clean(initial_blocks, time_budget):
        current_blocks = list(initial_blocks)
        current_set = set(current_blocks)
        
        counts = np.zeros(n_subsets, dtype=np.int32)
        for bi in current_blocks:
            counts += M[bi]
            
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        t_start = time.time()
        
        while time.time() - t_start < time_budget:
            # 1. Local Search: Prune
            current_blocks = prune(current_blocks)
            current_set = set(current_blocks)
            
            # Recount if pruned
            if len(current_blocks) < best_size or len(current_blocks) != len(best_blocks):
                 # Just recount to be safe if we pruned
                 counts = np.zeros(n_subsets, dtype=np.int32)
                 for bi in current_blocks:
                     counts += M[bi]
                 
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks)
                
            # 2. Perturb: Swap
            if len(current_blocks) > 0:
                bi_out = rng.choice(current_blocks)
                
                # Choose bi_in
                # Heuristic: Choose block with max gain on subsets covered ONLY by bi_out?
                # Or just random.
                # Let's try "Smart Random": Pick from top 10% gain blocks?
                # Or just random for diversity.
                
                bi_in = -1
                for _ in range(5):
                    cand = rng.randrange(n_blocks)
                    if cand not in current_set:
                        bi_in = cand
                        break
                if bi_in == -1:
                    continue
                    
                # Apply Swap
                current_blocks.remove(bi_out)
                current_set.discard(bi_out)
                current_blocks.append(bi_in)
                current_set.add(bi_in)
                
                counts -= M[bi_out]
                counts += M[bi_in]
                
                # 3. Repair if invalid
                if np.any(counts == 0):
                    # Greedy Repair
                    uncovered_mask = (counts == 0).astype(np.uint8)
                    while np.any(counts == 0):
                        gain_vec = M.dot(uncovered_mask)
                        for bi in current_set:
                            gain_vec[bi] = -1
                        best_bi = np.argmax(gain_vec)
                        if gain_vec[best_bi] == 0:
                            break
                        current_blocks.append(best_bi)
                        current_set.add(best_bi)
                        counts += M[best_bi]
                        uncovered_mask = (counts == 0).astype(np.uint8)
                        if time.time() - t_start > time_budget:
                            break
                            
        return best_blocks

    # Main
    init_blocks = greedy_init()
    pruned_init = prune(init_blocks)
    
    best_solution = pruned_init
    best_size = len(pruned_init)
    
    if time.time() < deadline - 5:
        sa_result = ils_clean(pruned_init, time_budget=deadline - time.time() - 5)
        pruned_sa = prune(sa_result)
        if len(pruned_sa) < best_size:
            best_solution = pruned_sa
            best_size = len(pruned_sa)
            
    final_solution = prune(best_solution)
    
    return [list(all_blocks[bi]) for bi in final_solution]
