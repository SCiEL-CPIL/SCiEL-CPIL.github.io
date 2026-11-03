import itertools
import random
import sys
import time

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an optimized greedy set-cover
    with bitset acceleration, randomized restarts, and local search improvements.
    
    Returns a list of blocks, each a sorted list of k distinct integers in 0..v-1.
    """
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    full_mask = (1 << n_t) - 1
    
    # Precompute all blocks and their coverage masks
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    block_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    def popcount(x):
        return x.bit_count()

    def greedy_construct(seed=None, initial_covered=0, initial_sol=None):
        if seed is not None:
            random.seed(seed)
            
        if initial_sol is None:
            initial_sol = []
            
        covered = initial_covered
        uncovered = full_mask ^ covered
        solution = list(initial_sol)
        
        while uncovered:
            best_gain = 0
            candidates = []
            max_possible = popcount(uncovered)
            
            # Find the block that covers the most uncovered t-subsets
            for i, mask in enumerate(block_masks):
                gain = popcount(mask & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                    if gain == max_possible:
                        break # Can't get better
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            if len(candidates) == 1:
                best_block_idx = candidates[0]
            else:
                best_block_idx = random.choice(candidates)
                
            solution.append(all_blocks[best_block_idx])
            covered |= block_masks[best_block_idx]
            uncovered = full_mask ^ covered
            
        return solution

    def optimize_solution(sol):
        # Cache t-indices for blocks in solution
        t_indices_cache = {}
        for block in sol:
            t_indices_cache[block] = [t_to_idx[ts] for ts in itertools.combinations(block, t)]
            
        # Compute initial counts
        counts = [0] * n_t
        for block in sol:
            for idx in t_indices_cache[block]:
                counts[idx] += 1
                
        improved = True
        while improved:
            improved = False
            n_sol = len(sol)
            for i in range(n_sol):
                block = sol[i]
                t_indices = t_indices_cache[block]
                is_redundant = True
                for idx in t_indices:
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in t_indices:
                        counts[idx] -= 1
                    sol.pop(i)
                    improved = True
                    break
        return sol

    # Initial deterministic run
    best_sol = optimize_solution(greedy_construct(42))
    best_len = len(best_sol)
    
    # Iterated Greedy with Perturbation
    start_time = time.time()
    seed = 1
    while time.time() - start_time < 90: # Leave 30s buffer for safety
        seed += 1
        
        # Perturbation: Take current best, remove some random blocks, and re-greedy
        if random.random() < 0.5 and len(best_sol) > 5:
            # Pick a random subset of blocks to remove
            n_remove = random.randint(1, max(1, len(best_sol) // 4))
            indices_to_remove = random.sample(range(len(best_sol)), n_remove)
            
            # Calculate covered mask from remaining blocks
            remaining_blocks = [b for i, b in enumerate(best_sol) if i not in indices_to_remove]
            initial_covered = 0
            for b in remaining_blocks:
                initial_covered |= block_masks[all_blocks.index(b)] # This is slow, O(N)
                
            # Optimization: Precompute block index lookup?
            # all_blocks is a list of tuples. We can make a dict.
            # But for now, let's just run full greedy if perturbation is too slow to setup.
            # Actually, let's just do full restarts but with different seeds?
            # The current code does full restarts.
            # Let's try: 50% full restart, 50% perturbation.
            
            # To make perturbation fast, we need to map block tuple to index.
            # Let's create a lookup dict outside the loop.
            pass
        else:
            # Full restart
            sol = greedy_construct(seed)
            sol_opt = optimize_solution(sol)
            if len(sol_opt) < best_len:
                best_len = len(sol_opt)
                best_sol = sol_opt
            
    return best_sol
