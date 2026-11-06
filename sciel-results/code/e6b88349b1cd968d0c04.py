"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm, gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s limit
    
    # Candidates are 2..256
    # We want to maximize sum(1/x) subject to lcm(a,b) > 256 for all distinct a,b in S.
    
    # Precompute pairwise compatibility
    # Two numbers a, b are compatible if lcm(a,b) > 256.
    # lcm(a,b) = a*b / gcd(a,b).
    # Condition: a*b / gcd(a,b) > 256.
    
    # To make the search efficient, we should order candidates by "value" (1/x).
    # Since we want to maximize sum(1/x), smaller x are more valuable.
    # However, smaller x are also more restrictive (easier to violate lcm constraint).
    # A common strategy for B&B on knapsack-like problems is to sort by value/weight ratio.
    # Here, "weight" is the constraint impact, which is hard to quantify simply.
    # But since the objective is sum(1/x), and x is the item, items with smaller x have larger value.
    # Let's sort candidates in descending order of 1/x (i.e., ascending order of x).
    # Wait, if we pick small x, we gain a lot of value.
    # If we pick large x, we gain little value but are less restrictive.
    
    # Let's try a B&B where we decide for each candidate (sorted by x ascending) whether to include it.
    # State: index i, current set S, current sum.
    # Upper bound: current sum + sum(1/x_j for j > i).
    # If upper bound <= best_sum, prune.
    
    # Candidates: 2 to 256
    candidates = list(range(2, 257))
    # Sort by x ascending (so 1/x descending). This is a standard ordering for B&B on subset sum/maximization.
    # Actually, for maximization of sum(1/x), picking small x is high reward.
    # If we include a small x, it restricts future picks more.
    
    # Let's precompute compatibility matrix or just check on the fly.
    # n=255 candidates.
    # B&B with n=255 might be too slow if the tree is deep.
    # However, the constraint is quite strong.
    # If we pick 2, no even number can be picked (lcm(2, 2k) = 2k <= 256 if 2k <= 256, wait.
    # lcm(2, y) = 2*y/gcd(2,y). If y is even, gcd=2, lcm=y. So if y<=256, lcm<=256. Invalid.
    # If y is odd, gcd=1, lcm=2y. If 2y > 256 => y > 128.
    # So if 2 is in S, all other elements must be odd and > 128.
    
    # This suggests the set size is small.
    # Let's run B&B.
    
    best_set = []
    best_sum = Fraction(0)
    
    # To speed up, we can pre-filter candidates?
    # No, let's let B&B handle it.
    
    # Optimization: Precompute lcm > 256 check.
    # is_comp[a][b] = (lcm(a,b) > 256)
    
    # Since n=255, a 255x255 matrix is fine.
    # But we can compute on the fly with gcd which is fast.
    
    def bnb(idx, current_set, current_sum):
        nonlocal best_set, best_sum
        
        if time.time() > deadline:
            return
        
        # Update best if current sum is better
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(current_set)
            
        # If we've considered all candidates
        if idx == len(candidates):
            return
            
        # Upper bound calculation
        # The maximum possible additional sum is the sum of 1/x for all remaining candidates.
        # We can precompute suffix sums of 1/x.
        # suffix_sum[i] = sum(1/x_j for j from i to end)
        
        # Check if current_sum + suffix_sum[idx] <= best_sum
        # If so, prune.
        
        # We need suffix_sum as Fractions for exact comparison.
        
        # Branch 1: Include candidates[idx]
        x = candidates[idx]
        
        # Check if x is compatible with all in current_set
        can_add = True
        for y in current_set:
            if lcm(x, y) <= 256:
                can_add = False
                break
                
        if can_add:
            new_set = current_set + [x]
            bnb(idx + 1, new_set, current_sum + Fraction(1, x))
            
        # Branch 2: Exclude candidates[idx]
        bnb(idx + 1, current_set, current_sum)
        
    # Precompute suffix sums
    suffix_sums = [Fraction(0)] * (len(candidates) + 1)
    for i in range(len(candidates) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i+1] + Fraction(1, candidates[i])
        
    # Modify bnb to use suffix_sums for pruning
    def bnb_opt(idx, current_set, current_sum):
        nonlocal best_set, best_sum
        
        if time.time() > deadline:
            return
            
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(current_set)
            
        if idx == len(candidates):
            return
            
        # Pruning
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        x = candidates[idx]
        
        # Include x
        can_add = True
        for y in current_set:
            if lcm(x, y) <= 256:
                can_add = False
                break
                
        if can_add:
            bnb_opt(idx + 1, current_set + [x], current_sum + Fraction(1, x))
            
        # Exclude x
        bnb_opt(idx + 1, current_set, current_sum)
        
    bnb_opt(0, [], Fraction(0))
    
    return best_set
# EVOLVE-BLOCK-END
