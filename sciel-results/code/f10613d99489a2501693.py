import time
import math
from math import gcd
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Candidates 2..256
    n_candidates = 255
    candidates = list(range(2, 257))
    
    # Sort by value descending (1/x), which means x ascending.
    # We process candidates in order of decreasing weight.
    # This is crucial for B&B: we decide on the most valuable items first.
    # Since weight = 1/x, smaller x is more valuable.
    # So we iterate x from 2 to 256.
    
    # Precompute compatibility bitmask for each candidate.
    # mask[i] is a bitmask of candidates j > i that are INCOMPATIBLE with i.
    # We only need to look ahead, so we store conflicts with higher indices.
    # Total candidates: 255. Indices 0..254 correspond to 2..256.
    # 255 bits fit in a Python int (arbitrary precision, but fast bitwise ops).
    
    masks = [0] * n_candidates
    for i in range(n_candidates):
        val_i = candidates[i]
        for j in range(i + 1, n_candidates):
            val_j = candidates[j]
            lcm_val = (val_i * val_j) // gcd(val_i, val_j)
            if lcm_val <= 256:
                masks[i] |= (1 << (j - i - 1)) # Offset to keep small? 
                # Actually, Python ints handle large shifts fine.
                # But let's keep indices relative to i for easier manipulation?
                # No, absolute indices are easier for the "available" mask.
                # Let's use absolute indices in the mask.
                # mask[i] has bit j set if i and j conflict.
                masks[i] = 0 # Reset
                # We will rebuild properly below for clarity
    
    # Rebuild masks with absolute indices for clarity and correctness
    # mask[i] contains bits for all j != i where conflict exists.
    # But in B&B, we only care about j > i (future candidates).
    # So mask[i] will have bit j set if j > i and conflict(i, j).
    
    for i in range(n_candidates):
        val_i = candidates[i]
        m = 0
        for j in range(i + 1, n_candidates):
            val_j = candidates[j]
            lcm_val = (val_i * val_j) // gcd(val_i, val_j)
            if lcm_val <= 256:
                m |= (1 << j)
        masks[i] = m

    # Precompute weights
    weights = [1.0 / candidates[i] for i in range(n_candidates)]
    
    # Global best
    best_sum = 0.0
    best_set = []
    
    # B&B State:
    # idx: current candidate index to decide on (0..254)
    # current_mask: bitmask of selected candidates so far.
    # We need to know which candidates are "blocked" by current selections.
    # Let's maintain a `blocked_mask` which is the union of masks of all selected candidates.
    # A candidate j is available if bit j is NOT in blocked_mask AND j >= idx.
    
    def get_upper_bound(idx, blocked_mask):
        """
        Calculate a fast upper bound for the remaining candidates (idx..254).
        We can simply sum the weights of all candidates j >= idx that are not blocked.
        This is an upper bound because we ignore conflicts among the remaining candidates.
        For this problem, the graph is sparse enough that this bound might be loose,
        but it's O(N) per node.
        
        To make it tighter and faster, we can precompute suffix sums of weights.
        But we must subtract blocked ones.
        
        Alternative: Greedy Relaxation.
        Greedily pick available candidates in order of weight (which is index order here).
        This gives a valid independent set, so its sum is a LOWER bound.
        We need an UPPER bound for pruning.
        
        The sum of all available weights is a valid upper bound.
        Let's optimize this sum calculation.
        """
        # Naive sum is O(N). 255 iterations.
        # With 2^k nodes, this could be slow.
        # Let's try to make it faster or use a tighter bound.
        
        # Actually, since we process in order of decreasing weight,
        # the "sum of all available" is a very loose bound initially.
        
        # Better Bound: Fractional Knapsack style?
        # No, this is Max Weight Independent Set.
        
        # Let's stick to "Sum of all available weights" but optimize the loop.
        # We can precompute total_suffix_sum[i] = sum(weights[i:])
        # Then UB = total_suffix_sum[idx] - sum(weights[j] for j in available_if_not_blocked_but_in_suffix)
        # This is still O(N).
        
        # Given N=255, O(N) per node is acceptable if the tree is pruned well.
        # Let's implement the simple sum first.
        
        s = 0.0
        # Iterate j from idx to 254
        # Check if bit j is in blocked_mask
        for j in range(idx, n_candidates):
            if not (blocked_mask >> j) & 1:
                s += weights[j]
        return s

    def solve(idx, blocked_mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        if idx == n_candidates:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(current_set)
            return
        
        # Pruning:
        # Upper bound = current_sum + max_possible_remaining
        # We calculate max_possible_remaining as sum of weights of all j >= idx not in blocked_mask.
        # Optimization: If current_sum + sum(weights[idx:]) <= best_sum, prune.
        # This is a very loose bound.
        
        # Tighter bound:
        # We can compute the sum of available items more efficiently if we maintain it?
        # No, blocked_mask changes.
        
        # Let's use a heuristic bound:
        # Greedy solve of the remaining subproblem? Too slow.
        
        # Let's use the "Sum of available" bound.
        # To speed up, we can precompute suffix sums.
        # suffix_sum[i] = sum(weights[i:])
        # But we need to subtract blocked ones.
        
        # Let's just do the loop. 255 iterations is fast in Python if optimized?
        # 255 * (bit shift + and) is ~microseconds.
        
        ub_rem = 0.0
        # Local variable for speed
        bm = blocked_mask
        w = weights
        for j in range(idx, n_candidates):
            if not (bm >> j) & 1:
                ub_rem += w[j]
                
        if current_sum + ub_rem <= best_sum:
            return
        
        val = candidates[idx]
        
        # Option 1: Include val
        # Check if val is available: bit idx not in blocked_mask
        if not (blocked_mask >> idx) & 1:
            new_blocked = blocked_mask | masks[idx]
            # Note: masks[idx] only has bits for j > idx.
            # So new_blocked correctly marks all future conflicts.
            new_set = current_set + [val]
            new_sum = current_sum + weights[idx]
            
            solve(idx + 1, new_blocked, new_sum, new_set)
            
        # Option 2: Exclude val
        # blocked_mask remains same for future? 
        # Wait, if we exclude val, it doesn't block anyone.
        # So blocked_mask stays the same.
        solve(idx + 1, blocked_mask, current_sum, current_set)

    # Initial call
    # blocked_mask = 0
    # current_sum = 0.0
    # current_set = []
    
    # Optimization: 
    # The "Sum of available" bound is very loose. 
    # Example: If we pick 2, it blocks 4, 6, 8... 
    # The sum of all remaining weights is still large.
    # We might explore too many branches.
    
    # Improvement: Use a tighter bound.
    # We can compute the "Greedy Independent Set Sum" of the remaining available nodes as a LOWER bound? No, we need Upper.
    
    # What if we use a fractional bound?
    # Sort remaining available by weight (already sorted by index).
    # Take them all? That's the sum.
    
    # Let's try to improve the bound calculation speed and tightness.
    # Precompute suffix sums.
    suffix_sum = [0.0] * (n_candidates + 1)
    for i in range(n_candidates - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + weights[i]
        
    # Now, UB = current_sum + (suffix_sum[idx] - sum(weights[j] for j in available_j_ge_idx_that_are_blocked))
    # We still need to sum the blocked ones.
    # But maybe fewer iterations? No.
    
    # Let's add a simple heuristic: 
    # If the number of available items is small, exact solve?
    # No, let's stick to the simple B&B but add a time check inside the loop?
    # The deadline check is at the start of the function.
    
    # One major issue: Python recursion depth.
    # Depth 255 is fine (default limit is 1000).
    
    # Let's run it.
    solve(0, 0, 0.0, [])
    
    # If best_set is empty (shouldn't happen, but just in case), pick 256
    if not best_set:
        best_set = [256]
        
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
    print(f"Sum: {sum(1/x for x in res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if (res[i]*res[j])//gcd(res[i], res[j]) <= 256:
                valid = False
                print(f"Invalid pair: {res[i]}, {res[j]}")
    print(f"Valid: {valid}")
