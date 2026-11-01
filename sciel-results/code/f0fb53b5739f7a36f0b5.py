import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Precomputation
    # Generate all t-subsets (4-subsets) and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Generate all possible k-blocks (7-subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    # block_coverage[i] is a list of indices into all_t_subs
    block_coverage = []
    for block in all_blocks:
        covered = []
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            covered.append(idx)
        block_coverage.append(covered)
        
    # Track uncovered t-subsets
    # Initially all are uncovered
    uncovered = [True] * n_t_subs
    n_uncovered = n_t_subs
    
    # Weights for subgradient descent
    weights = [1.0] * n_t_subs
    # Lagrangian multiplier (penalty for using a block)
    # We want to minimize number of blocks while maximizing weighted coverage.
    # Score(block) = sum(weights[j] for j in block_coverage[block] if uncovered[j]) - lambda
    # If Score > 0, we consider adding the block.
    
    best_solution = None
    best_score = float('inf')
    
    # Subgradient descent parameters
    step_size = 1.0
    lambda_val = 0.1 # Initial penalty
    
    # To speed up, we might not want to check all blocks if n_blocks is large, 
    # but 50k is manageable.
    
    while time.time() < deadline:
        # Reset for a new run with slightly perturbed initial conditions or restart
        # Actually, subgradient descent is a continuous process, but for covering,
        # we often restart or use a heuristic that fixes the solution and tweaks weights.
        # A common approach: 
        # 1. Start with empty solution.
        # 2. Repeatedly pick best block.
        # 3. Update weights.
        # 4. If stuck, restart with different weights.
        
        solution = []
        uncovered = [True] * n_t_subs
        n_uncovered = n_t_subs
        weights = [1.0] * n_t_subs
        
        # We will run a greedy guided by weights for a certain number of steps
        # or until no more gain.
        
        # To allow for different "perspectives", we can randomize initial weights slightly
        # or just run the process.
        
        # Optimization: Instead of recalculating score for ALL blocks every step,
        # we can maintain a heap or just scan. With 50k blocks, scanning is fast (~0.01s).
        
        while n_uncovered > 0 and time.time() < deadline - 10:
            best_block_idx = -1
            best_gain = 0
            best_block_score = 0
            
            # Find block with max (sum of weights of uncovered subsets) - lambda
            # We only care if it covers at least one new subset.
            
            for b_idx in range(n_blocks):
                cov = block_coverage[b_idx]
                # Calculate gain: sum of weights of uncovered items in cov
                gain = 0
                # Optimization: if n_uncovered is small, we could check differently,
                # but summing a list of ~35 items (C(7,4)=35) is very fast.
                for idx in cov:
                    if uncovered[idx]:
                        gain += weights[idx]
                
                score = gain - lambda_val
                if score > best_block_score:
                    best_block_score = score
                    best_block_idx = b_idx
                    best_gain = gain
            
            if best_block_idx == -1 or best_gain == 0:
                # No block provides net positive gain.
                # This means lambda is too high or weights are too low.
                # For covering, we must cover everything. 
                # If we can't find a block with score > 0, we should pick the one with max gain
                # regardless of penalty, or reduce lambda.
                # Let's pick the one with max gain if gain > 0.
                max_gain_block = -1
                max_gain = 0
                for b_idx in range(n_blocks):
                    cov = block_coverage[b_idx]
                    gain = 0
                    for idx in cov:
                        if uncovered[idx]:
                            gain += 1 # Count uncovered, not weighted, for fallback
                    if gain > max_gain:
                        max_gain = gain
                        max_gain_block = b_idx
                
                if max_gain_block == -1 or max_gain == 0:
                    break # Stuck? Should not happen if n_uncovered > 0 and v,k,t valid.
                
                best_block_idx = max_gain_block
                best_gain = max_gain
            
            # Add the block
            solution.append(all_blocks[best_block_idx])
            
            # Update uncovered
            cov = block_coverage[best_block_idx]
            for idx in cov:
                if uncovered[idx]:
                    uncovered[idx] = False
                    n_uncovered -= 1
            
            # Update weights (Subgradient step)
            # For covered subsets, decrease weight (or keep low).
            # For uncovered subsets, increase weight.
            # Standard subgradient for set cover: 
            # w_j <- max(0, w_j + step * (1 - sum_{i: j in S_i} x_i))
            # Here x_i is 1 if block i is in solution.
            # Since we are doing greedy, we update weights to reflect "need".
            
            # Simplified update:
            # Increase weight of uncovered items to encourage covering them.
            # Decrease weight of covered items to avoid over-covering.
            for j in range(n_t_subs):
                if uncovered[j]:
                    weights[j] += step_size
                else:
                    weights[j] = max(0.1, weights[j] - step_size * 0.1)
            
            # Adaptive step size
            # step_size = step_size * 0.99 or similar
            step_size = max(0.01, step_size * 0.98)
            
            # Occasionally reset lambda or adjust it
            # If we are adding many blocks, maybe increase lambda to be more selective?
            # No, we want to minimize blocks. High lambda means we only add if gain is high.
            # If we are stuck, low lambda helps.
            
        # Post-processing: Remove redundant blocks
        # Check if any block in solution is redundant (all its subsets are covered by other blocks)
        # This is O(N^2 * C(k,t)). N is ~150. 150^2 * 35 ~ 800k ops. Fast.
        
        # Create a set of covered subsets for each block in solution
        # Actually, simpler: 
        # For each block in solution, check if all its t-subsets are covered by OTHER blocks.
        
        # Efficient check:
        # 1. Count how many times each t-subset is covered by the current solution.
        # 2. If a block covers a t-subset that is covered exactly once, that block is essential.
        # 3. If all t-subsets of a block are covered >= 2 times, the block is redundant.
        
        subset_count = [0] * n_t_subs
        for b_idx in solution:
            # b_idx is a tuple from all_blocks
            # We need the index in all_blocks to use block_coverage
            # Wait, solution stores the tuple. Let's store indices in solution instead.
            pass
        
        # Let's redo the solution storage to store indices for faster lookup
        # Actually, let's just do the redundancy check on the list of tuples.
        
        # Re-implement redundancy check properly
        if solution:
            # Map block tuple to index in all_blocks for fast coverage lookup
            block_to_idx = {b: i for i, b in enumerate(all_blocks)}
            
            # Count coverage
            cov_count = [0] * n_t_subs
            sol_indices = []
            for b in solution:
                idx = block_to_idx[b]
                sol_indices.append(idx)
                for sub_idx in block_coverage[idx]:
                    cov_count[sub_idx] += 1
            
            # Remove redundant blocks
            new_sol_indices = []
            for idx in sol_indices:
                redundant = True
                for sub_idx in block_coverage[idx]:
                    if cov_count[sub_idx] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_sol_indices.append(idx)
            
            # Update solution
            solution = [all_blocks[i] for i in new_sol_indices]
        
        if len(solution) < best_score:
            best_score = len(solution)
            best_solution = [list(b) for b in solution]
            
        # Check if we should restart with different parameters
        # If we are close to deadline, break
        if time.time() > deadline - 20:
            break
            
    if best_solution is None:
        # Fallback: Greedy without weights
        # Just pick random blocks? No, standard greedy.
        uncovered_set = set(range(n_t_subs))
        blocks = []
        while uncovered_set and time.time() < deadline - 5:
            best_b = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = 0
                for idx in block_coverage[b_idx]:
                    if idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1: break
            blocks.append(all_blocks[best_b])
            for idx in block_coverage[best_b]:
                uncovered_set.discard(idx)
        
        if blocks:
            best_solution = [list(b) for b in blocks]
            
    return best_solution if best_solution else []
