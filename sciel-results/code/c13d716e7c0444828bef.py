"""Simulated Annealing for covering design C(24,7,3)."""
import itertools
import random
import time
import sys
import numpy as np

# EVOLVE-BLOCK-START

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # Precomputation
    # 1. All t-sets
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # 2. For each block (k-subset), precompute the bitset of t-sets it covers
    # We represent the coverage state as a numpy array of uint8 (0=uncovered, 1=covered)
    # Actually, for SA we need to know the "gain" or "loss" of a move.
    # Let's maintain:
    #   covered_count: array of length num_tsets, counting how many blocks cover each t-set
    #   num_blocks: current number of blocks
    
    # Precompute t-sets for all possible k-blocks? 
    # C(24,7) is ~346k. Storing bitsets for all is 346k * (C(24,3)/8 bytes) ~ 346k * 2300 bytes ~ 800MB. Too much.
    # Instead, compute on the fly or cache recent ones.
    # But for SA, we need fast evaluation.
    # Let's store the list of t-set indices for each block in the current solution.
    
    def get_tset_indices(block):
        """Return list of indices of t-sets contained in block."""
        return [tset_index[ts] for ts in itertools.combinations(block, t)]
    
    # Initial solution: Greedy covering
    def greedy_initial(v, k, t, rng):
        uncovered = set(all_tsets)
        blocks = []
        # Precompute t-sets containing each pair for speed? 
        # Just simple greedy
        while uncovered:
            # Pick a random uncovered t-set
            ts = next(iter(uncovered))
            block = list(ts)
            while len(block) < k:
                # Find element that covers most uncovered t-sets
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    # Check all (t-1) subsets of current block
                    for sub in itertools.combinations(block, t-1):
                        # sub is a tuple of t-1 elements
                        # Check if x + sub is in uncovered
                        # We need to form the t-set
                        # Since block is sorted? No, block is a list.
                        # Let's just check if the set {x} U sub is in uncovered
                        # But uncovered contains tuples.
                        # Construct tuple: sorted(sub + (x,))
                        candidate = tuple(sorted(sub + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(tuple(block))
            # Remove covered
            for ts_in_block in itertools.combinations(block, t):
                if ts_in_block in uncovered:
                    uncovered.remove(ts_in_block)
        return blocks

    # Simulated Annealing
    def simulated_annealing(initial_blocks, v, k, t, rng, time_limit):
        blocks = [list(b) for b in initial_blocks]
        n = len(blocks)
        
        # State variables
        # covered_counts: array of size num_tsets, int
        covered_counts = np.zeros(num_tsets, dtype=np.int32)
        
        # Block t-set indices
        block_tsets = []
        for b in blocks:
            tsets = get_tset_indices(b)
            block_tsets.append(tsets)
            for idx in tsets:
                covered_counts[idx] += 1
        
        # Check validity (should be valid from greedy)
        # min_cover = np.min(covered_counts)
        
        current_blocks = n
        best_blocks = blocks.copy()
        best_len = n
        best_cover_counts = covered_counts.copy()
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.01
        cooling_rate = 0.9995 # Adjust based on expected steps
        
        # Precompute all possible k-blocks? No, too many.
        # Generate neighbors by:
        # 1. Add a random block
        # 2. Remove a random block (if valid)
        # 3. Swap a random block with a random new block
        
        # Helper to check if removing a block keeps validity
        def can_remove(block_idx):
            tsets = block_tsets[block_idx]
            for idx in tsets:
                if covered_counts[idx] <= 1:
                    return False
            return True
        
        # Helper to calculate delta in objective function
        # Objective: Minimize number of blocks.
        # If we add a block: +1 block. Validity might improve (but we start valid).
        # If we remove a block: -1 block. Must remain valid.
        # If we swap: 0 change in count. But might allow further removals later?
        # Actually, standard SA for covering:
        # State: Set of blocks.
        # Move: Add, Remove, Swap.
        # Cost: Number of blocks.
        # Constraint: All t-sets covered.
        # If a move violates constraint, it's rejected (or penalized heavily).
        
        # Let's use a penalty method:
        # Cost = num_blocks + lambda * (num_uncovered_tsets)
        # lambda is large enough to ensure validity is prioritized.
        
        lambda_penalty = 100.0
        
        def get_cost(num_blocks, num_uncovered):
            return num_blocks + lambda_penalty * num_uncovered
        
        current_num_uncovered = np.sum(covered_counts == 0)
        current_cost = get_cost(n, current_num_uncovered)
        
        T = T_start
        steps = 0
        max_steps = 1000000
        
        while time.time() < time_limit and steps < max_steps:
            steps += 1
            
            # Choose a move
            move_type = rng.random()
            
            if move_type < 0.3:
                # Add a block
                new_block = sorted(rng.sample(range(v), k))
                new_block_tsets = get_tset_indices(new_block)
                
                # Calculate new cost
                # num_blocks increases by 1
                # num_uncovered decreases by number of t-sets in new_block that were uncovered
                newly_covered = 0
                for idx in new_block_tsets:
                    if covered_counts[idx] == 0:
                        newly_covered += 1
                
                new_num_uncovered = current_num_uncovered - newly_covered
                new_num_blocks = n + 1
                new_cost = get_cost(new_num_blocks, new_num_uncovered)
                
                delta = new_cost - current_cost
                
                # Acceptance
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    # Perform move
                    blocks.append(new_block)
                    block_tsets.append(new_block_tsets)
                    for idx in new_block_tsets:
                        covered_counts[idx] += 1
                    n += 1
                    current_num_uncovered = new_num_uncovered
                    current_cost = new_cost
                    current_num_blocks = n
                    
                    if n < best_len:
                        best_len = n
                        best_blocks = [b[:] for b in blocks]
                        best_cover_counts = covered_counts.copy()
                        
            elif move_type < 0.6:
                # Remove a block
                if n <= best_len: # Don't remove if we are at best? No, explore
                    idx_to_remove = rng.randint(0, n - 1)
                    if can_remove(idx_to_remove):
                        # Calculate new cost
                        # num_blocks decreases by 1
                        # num_uncovered increases by 0 (since it was valid)
                        # Wait, if we remove a block that is the ONLY cover for some t-set, it becomes invalid.
                        # can_remove checks if any t-set has count <= 1. If yes, removing makes it 0.
                        # So if can_remove is True, no t-set becomes uncovered.
                        
                        new_num_blocks = n - 1
                        new_num_uncovered = current_num_uncovered
                        new_cost = get_cost(new_num_blocks, new_num_uncovered)
                        
                        delta = new_cost - current_cost
                        
                        if delta < 0 or rng.random() < np.exp(-delta / T):
                            # Perform move
                            removed_block_tsets = block_tsets.pop(idx_to_remove)
                            blocks.pop(idx_to_remove)
                            for idx in removed_block_tsets:
                                covered_counts[idx] -= 1
                            n -= 1
                            current_num_uncovered = new_num_uncovered
                            current_cost = new_cost
                            
                            if n < best_len:
                                best_len = n
                                best_blocks = [b[:] for b in blocks]
                                best_cover_counts = covered_counts.copy()
                                
            else:
                # Swap a block
                if n > 0:
                    idx_to_swap = rng.randint(0, n - 1)
                    new_block = sorted(rng.sample(range(v), k))
                    old_block = blocks[idx_to_swap]
                    if new_block == old_block:
                        continue
                    
                    new_block_tsets = get_tset_indices(new_block)
                    old_block_tsets = block_tsets[idx_to_swap]
                    
                    # Calculate change in uncovered
                    # t-sets covered only by old_block (and not new_block) become uncovered
                    # t-sets covered only by new_block (and not old_block) become covered
                    
                    # This is tricky with counts.
                    # Let's just simulate the counts change.
                    
                    # Temporarily apply
                    # 1. Remove old
                    for idx in old_block_tsets:
                        covered_counts[idx] -= 1
                    # 2. Add new
                    for idx in new_block_tsets:
                        covered_counts[idx] += 1
                    
                    new_num_uncovered = np.sum(covered_counts == 0)
                    new_num_blocks = n
                    new_cost = get_cost(new_num_blocks, new_num_uncovered)
                    
                    delta = new_cost - current_cost
                    
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        # Keep change
                        blocks[idx_to_swap] = new_block
                        block_tsets[idx_to_swap] = new_block_tsets
                        current_num_uncovered = new_num_uncovered
                        current_cost = new_cost
                        
                        if n < best_len:
                            best_len = n
                            best_blocks = [b[:] for b in blocks]
                            best_cover_counts = covered_counts.copy()
                    else:
                        # Revert
                        # 1. Revert Add new
                        for idx in new_block_tsets:
                            covered_counts[idx] -= 1
                        # 2. Revert Remove old
                        for idx in old_block_tsets:
                            covered_counts[idx] += 1
                        
            T *= cooling_rate
            if T < T_end:
                T = T_end
                
        return best_blocks

    # Run SA with a few different initializations
    best_overall = None
    best_len_overall = float('inf')
    
    # 1. Greedy init
    rng = random.Random(42)
    init_blocks = greedy_initial(v, k, t, rng)
    
    # Run SA for remaining time
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 5.0:
        sa_blocks = simulated_annealing(init_blocks, v, k, t, rng, remaining_time)
        if len(sa_blocks) < best_len_overall:
            best_len_overall = len(sa_blocks)
            best_overall = sa_blocks
            
    # If SA didn't improve or failed, return the greedy init
    if best_overall is None:
        best_overall = init_blocks
        
    # Final check
    # The evaluator checks validity. Our SA should maintain validity or revert.
    # However, if lambda is not high enough, we might return an invalid solution.
    # Let's verify and fix if needed.
    
    def verify_and_fix(blocks, v, k, t):
        # Check validity
        covered = set()
        for b in blocks:
            for ts in itertools.combinations(b, t):
                covered.add(ts)
        
        # Find uncovered
        all_tsets_set = set(all_tsets)
        uncovered = all_tsets_set - covered
        
        if not uncovered:
            return blocks
        
        # Greedily add blocks to cover uncovered
        while uncovered:
            ts = next(iter(uncovered))
            block = list(ts)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for sub in itertools.combinations(block, t-1):
                        candidate = tuple(sorted(sub + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(tuple(block))
            for ts_in_block in itertools.combinations(block, t):
                if ts_in_block in uncovered:
                    uncovered.remove(ts_in_block)
        return blocks

    final_blocks = verify_and_fix(best_overall, v, k, t)
    
    return final_blocks

# EVOLVE-BLOCK-END
