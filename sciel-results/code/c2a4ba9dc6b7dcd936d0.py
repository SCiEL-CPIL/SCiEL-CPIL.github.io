"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import Counter

# We will use a vectorized greedy algorithm with a deadline guard.
# The core idea is to maintain a count of how many times each t-subset is covered.
# A t-subset is "uncovered" if its count is 0.
# We select blocks to maximize the number of newly covered t-subsets.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 115s deadline to be safe within 120s limit
    
    # Map each t-subset to an index for fast array access
    # Total number of t-subsets is C(v, t). For v=24, t=3, this is 2024.
    # We can enumerate all t-subsets.
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map from subset (tuple) to index
    subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Coverage count array
    coverage = [0] * num_t_subsets
    
    # Set of uncovered subset indices
    # Initially all are uncovered
    uncovered_count = num_t_subsets
    
    blocks = []
    
    # Precompute for each point x, the list of (t-1)-subsets of the other points?
    # No, that's not quite right.
    # When we add a point x to a partial block B, the new t-subsets covered are
    # {S ∪ {x} : S ⊆ B, |S| = t-1}.
    # So we need to iterate over (t-1)-subsets of B.
    
    # To speed up the greedy step:
    # For a given partial block B, we want to find x not in B that maximizes:
    #   sum_{S ⊆ B, |S|=t-1} [1 if S ∪ {x} is uncovered else 0]
    
    # We can precompute for each x, a list of t-subsets containing x?
    # That would be C(v-1, t-1) per x.
    # Alternatively, for each (t-1)-subset S, the set of x such that S ∪ {x} is a t-subset
    # is just {x : x ∉ S}. But we need to check if S ∪ {x} is uncovered.
    
    # Let's try a direct approach first and see if it's fast enough.
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        # Pick the lexicographically first uncovered t-subset to start a block
        # This is a heuristic; we could also pick randomly or based on some other criterion.
        # Let's stick with lexicographic first for determinism and simplicity.
        # Actually, we can find the first uncovered index.
        first_uncovered_idx = -1
        for i in range(num_t_subsets):
            if coverage[i] == 0:
                first_uncovered_idx = i
                break
        
        if first_uncovered_idx == -1:
            break
            
        # Start the block with the elements of this t-subset
        initial_subset = all_t_subsets[first_uncovered_idx]
        block = list(initial_subset)
        
        # Mark these t-subsets? No, the t-subset itself is covered, but we need to cover
        # all t-subsets that are subsets of the final block.
        # Wait, the block will have k elements. The t-subsets covered are all C(k, t) subsets.
        # We start with a t-subset, so we need to add k-t more elements.
        
        while len(block) < k:
            if time.time() > deadline:
                break
                
            best_x = -1
            best_gain = -1
            
            # For each candidate x not in block
            for x in range(v):
                if x in block:
                    continue
                    
                # Calculate gain: number of new t-subsets covered by adding x
                # New t-subsets are {S ∪ {x} : S ⊆ block, |S| = t-1}
                gain = 0
                for S in itertools.combinations(block, t-1):
                    new_subset = tuple(sorted(S + (x,)))
                    idx = subset_to_idx[new_subset]
                    if coverage[idx] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
            if best_x == -1:
                # This shouldn't happen if uncovered_count > 0 and k > t
                break
                
            block.append(best_x)
            
        # Finalize the block
        block.sort()
        
        # Update coverage
        newly_covered = 0
        for S in itertools.combinations(block, t):
            idx = subset_to_idx[S]
            if coverage[idx] == 0:
                coverage[idx] = 1
                newly_covered += 1
            else:
                coverage[idx] += 1
                
        uncovered_count -= newly_covered
        blocks.append(block)
        
    return blocks
# EVOLVE-BLOCK-END
