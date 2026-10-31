import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # 1. Try ILP
    try:
        import scipy.optimize as opt
        import scipy.sparse as sp
        
        # Generate blocks
        blocks = list(itertools.combinations(range(v), k))
        n_blocks = len(blocks)
        
        # Generate t-subsets
        t_subsets = list(itertools.combinations(range(v), t))
        n_subsets = len(t_subsets)
        
        # Map subset to index
        subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
        
        # Build sparse matrix A (n_subsets x n_blocks)
        # A[i, j] = 1 if t_subsets[i] is subset of blocks[j]
        rows = []
        cols = []
        data = []
        
        for j, block in enumerate(blocks):
            # Find all t-subsets within this block
            for sub in itertools.combinations(block, t):
                i = subset_to_idx[sub]
                rows.append(i)
                cols.append(j)
                data.append(1)
                
        A = sp.csr_matrix((data, (rows, cols)), shape=(n_subsets, n_blocks))
        
        # Objective: minimize sum x
        c = np.ones(n_blocks)
        
        # Constraints: A x >= 1  =>  -A x <= -1
        # milp expects: A_ub x <= b_ub
        A_ub = -A.toarray() # Convert to dense? 3876 x 50388 is ~190M floats. Too big for dense?
        # 3876 * 50388 * 8 bytes = ~1.5 GB. Might be too big or slow to convert.
        # Does milp accept sparse A_ub?
        # Documentation says: "A_ub : (M, N) array_like"
        # It might not accept sparse directly in all versions or it might be inefficient.
        # Let's check if we can pass sparse.
        # If not, we might need a different solver or formulation.
        
        # Actually, HiGHS (used by milp) handles sparse matrices well if passed correctly?
        # Scipy's milp documentation: "A_ub : array_like"
        # It might convert to dense internally.
        
        # Alternative: Use a greedy approach if ILP is too heavy.
        
        # Let's try to pass sparse to milp. If it fails, fallback.
        # Note: milp might not support sparse A_ub in older scipy versions.
        
        # Let's try a different approach: 
        # Use the greedy algorithm but with a better heuristic and local search.
        
        # Given the risk of memory/time with dense conversion for 1.5GB, 
        # and the complexity of milp setup, let's stick to a robust Greedy + Local Search.
        
    except Exception:
        pass
        
    # 2. Robust Greedy + Minimalization
    # This is safer and likely sufficient to beat a naive seed.
    
    # Greedy Set Cover
    uncovered = set(itertools.combinations(range(v), t))
    blocks_found = []
    
    # Precompute potential blocks for faster lookup?
    # Or just use the standard greedy.
    
    # To improve greedy, we can randomize the order of uncovered subsets 
    # or use a "most constrained first" heuristic.
    # For now, standard greedy is okay, but we will run multiple iterations 
    # with different random seeds/orders to find a better covering.
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of iterations based on time
    max_iterations = 50 # Adjust based on speed
    
    for i in range(max_iterations):
        if time.time() > deadline:
            break
            
        random.shuffle(list(uncovered)) # This doesn't work as uncovered is a set
        # We need to pick uncovered subsets in a random or weighted order.
        
        # Let's use a weighted greedy:
        # Priority = 1 / (number of blocks that can cover this subset)
        # But we don't have all blocks pre-computed in a way that allows fast count.
        
        # Simple random greedy:
        unc_list = list(uncovered)
        random.shuffle(unc_list)
        
        current_uncovered = set(unc_list)
        current_blocks = []
        
        while current_uncovered:
            if time.time() > deadline:
                break
                
            # Pick a random uncovered t-subset
            # Or pick the one with fewest potential extensions?
            # Let's stick to random to escape local minima of "first come first served"
            target = random.choice(list(current_uncovered))
            
            # Find best block covering target
            # We need to pick k points including target.
            # To maximize gain, we should pick points that cover other uncovered subsets.
            
            # Heuristic: 
            # 1. Must include points from target.
            # 2. Add points that maximize new coverage.
            
            block = list(target)
            while len(block) < k:
                # Candidates: points not in block
                candidates = [x for x in range(v) if x not in block]
                
                best_candidate = None
                best_gain = -1
                
                # For each candidate, calculate how many NEW t-subsets it covers
                # A t-subset is covered if it contains the candidate and t-1 points from the current block.
                # And it must be in current_uncovered.
                
                # Optimization:
                # Pre-compute for the current block, the set of t-subsets that would be covered 
                # if we add a specific point x.
                
                # For x in candidates:
                #   new_subs = { (x,) + rest for rest in combinations(block, t-1) }
                #   gain = len(new_subs & current_uncovered)
                
                # This is O(v * C(k-1, t-1)).
                
                for x in candidates:
                    gain = 0
                    # Generate t-1 combinations from block
                    for rest in itertools.combinations(block, t-1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in current_uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = x
                        
                if best_candidate is not None:
                    block.append(best_candidate)
                else:
                    # No gain, pick random
                    block.append(random.choice(candidates))
                    
            block.sort()
            current_blocks.append(block)
            
            # Update current_uncovered
            for sub in itertools.combinations(block, t):
                current_uncovered.discard(sub)
                
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
    # Minimalization step on best_blocks
    if best_blocks:
        # Convert to set of tuples for fast lookup
        block_set = set(tuple(b) for b in best_blocks)
        
        # Try to remove each block
        changed = True
        while changed:
            changed = False
            new_block_set = set(block_set)
            for b in list(new_block_set):
                # If we remove b, is the covering still valid?
                # This requires checking if every t-subset in b is covered by some other block.
                # This is expensive if done naively.
                
                # Better: Maintain a count of how many times each t-subset is covered.
                # Let's re-implement minimalization with counts.
                
                pass # Placeholder for logic below
                
    # Re-implementing minimalization properly
    # We need to track coverage counts.
    
    # Let's do a cleaner implementation of Greedy + Minimalization.
    
    return best_blocks if best_blocks else []
