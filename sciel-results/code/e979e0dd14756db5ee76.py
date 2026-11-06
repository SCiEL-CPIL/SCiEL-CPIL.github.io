"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute all 3-AP conflicts
    n = 256
    # conflict[i] = set of j such that {i, j} forms part of a 3-AP with some k
    # Actually, we need: if we include i and j, then we cannot include k where i,j,k form 3-AP
    # Let's precompute for each pair (a,b) with a<b, the forbidden third element
    # 3-AP: x, x+d, x+2d. So if we have a and b:
    #   If b-a = d, then a-d and a+d are forbidden (if in range)
    #   Actually: a, b, c form 3-AP if 2b = a+c (b is middle) or a, c, b or c, a, b
    #   Three cases: b is middle: c = 2b-a; a is middle: c = 2a-b; b is "end": c = 2b-a... 
    #   Let me think: 3 elements {x,y,z} form a 3-AP if one is the average of the other two.
    #   So for a pair (a,b), the forbidden third element c satisfies:
    #     2a = b + c => c = 2a - b
    #     2b = a + c => c = 2b - a
    #   (The third case 2c = a+b gives c = (a+b)/2, which is only integer if a+b even)
    
    # For B&B, we process elements in order of decreasing weight (1/x), so start with small x
    # weights[i] = 1/i for i in 1..256
    
    # Precompute conflicts: for each i, conflicts[i] = set of j that cannot coexist with i
    # Two elements i,j conflict if there exists k in [1,256] such that {i,j,k} is a 3-AP
    # i.e., 2i=j+k or 2j=i+k or 2k=i+j for some k in [1,256]
    # Equivalent: for pair (i,j), they conflict if:
    #   2i - j in [1,256] and 2i-j != i and 2i-j != j  (k = 2i-j, i is middle)
    #   2j - i in [1,256] and 2j-i != i and 2j-i != j  (k = 2j-i, j is middle)
    #   (i+j) even and (i+j)//2 in [1,256] and (i+j)//2 != i and (i+j)//2 != j  (k=(i+j)/2, k is middle)
    
    # Actually simpler: i and j are in a 3-AP together iff there exists k != i,j in [1,256]
    # such that {i,j,k} forms a 3-AP. This happens iff:
    #   |i-j| = d, and i-d in [1,256] or j+d in [1,256] (for the AP i-d, i, i+d or j-d, j, j+d)
    #   OR (i+j) is even and (i+j)/2 is in [1,256] and (i+j)/2 != i and (i+j)/2 != j
    
    # Let me just precompute: for each i in 1..256, find all j > i that conflict with i
    # i,j conflict iff:
    #   2*i - j in [1,256] and 2*i-j not in {i,j}
    #   OR 2*j - i in [1,256] and 2*j-i not in {i,j}  
    #   OR (i+j) % 2 == 0 and (i+j)//2 in [1,256] and (i+j)//2 not in {i,j}
    
    conflict = [set() for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            c1 = 2*i - j
            c2 = 2*j - i
            c3 = (i+j)//2 if (i+j) % 2 == 0 else None
            has_conflict = False
            if 1 <= c1 <= n and c1 != i and c1 != j:
                has_conflict = True
            if 1 <= c2 <= n and c2 != i and c2 != j:
                has_conflict = True
            if c3 is not None and 1 <= c3 <= n and c3 != i and c3 != j:
                has_conflict = True
            if has_conflict:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Weights: weight[i] = Fraction(1, i)
    # Process in order of decreasing weight: 1, 2, 3, ..., 256
    order = list(range(1, n+1))  # already in decreasing weight order since weight[i]=1/i
    
    best_set = []
    best_weight = Fraction(0)
    
    # Greedy initial solution
    greedy = []
    for x in order:
        can_add = True
        for y in greedy:
            if x in conflict[y]:
                can_add = False
                break
        if can_add:
            greedy.append(x)
    best_set = list(greedy)
    best_weight = sum(Fraction(1, x) for x in greedy)
    
    # Branch and bound
    # State: current set, current weight, index in order (next candidate)
    # Upper bound: current weight + sum of weights of all remaining candidates that don't conflict with current set
    
    # For efficiency, precompute suffix sums of weights
    suffix_weight = [Fraction(0)] * (n + 2)
    for idx in range(n-1, -1, -1):
        suffix_weight[idx] = suffix_weight[idx+1] + Fraction(1, order[idx])
    
    def branch_and_bound(current_set, current_weight, idx, deadline):
        nonlocal best_set, best_weight
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_weight + sum of weights of remaining elements that can be added
        # Quick upper bound: current_weight + suffix_weight[idx]
        if current_weight + suffix_weight[idx] <= best_weight:
            return
        
        if idx == n:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = list(current_set)
            return
        
        x = order[idx]
        
        # Option 1: skip x
        branch_and_bound(current_set, current_weight, idx + 1, deadline)
        
        # Option 2: include x (if it doesn't conflict)
        can_include = True
        for y in current_set:
            if x in conflict[y]:
                can_include = False
                break
        
        if can_include:
            current_set.append(x)
            branch_and_bound(current_set, current_weight + Fraction(1, x), idx + 1, deadline)
            current_set.pop()
    
    branch_and_bound([], Fraction(0), 0, deadline)
    
    return best_set
# EVOLVE-BLOCK-END
