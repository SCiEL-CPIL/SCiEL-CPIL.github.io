import time
from fractions import Fraction
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s limit)
    
    # Weights as Fractions for exact comparison
    weights = [Fraction(0)] + [Fraction(1, i) for i in range(1, N + 1)]
    
    # Precompute suffix sums of weights for upper bounding
    # suffix_sum[i] = sum of weights[i..N]
    suffix_sum = [Fraction(0)] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    # Precompute for each x, the list of pairs (a, b) such that {a, b, x} forms a 3-AP
    # This helps with fast conflict checking
    # For element x, if we have a in S, then 2a-x must not be in S, and 2x-a must not be in S
    
    # Greedy initial solution: add elements in order of increasing value (decreasing weight)
    in_set = [False] * (N + 1)
    S = set()
    
    def can_add(x):
        """Check if x can be added without creating a 3-AP"""
        for a in S:
            # x is middle: a, x, z where z = 2x - a
            z = 2 * x - a
            if z != a and z != x and 1 <= z <= N and in_set[z]:
                return False
            # x is end: a, b, x where b = (a+x)/2, or x, b, a where b = (x+a)/2
            # If x is the larger end: a, (a+x)/2, x -> need (a+x) even and (a+x)//2 in S
            if (a + x) % 2 == 0:
                m = (a + x) // 2
                if m != a and m != x and 1 <= m <= N and in_set[m]:
                    return False
            # If x is the smaller end: x, (x+a)/2, a -> same check as above
            # Already covered by (a+x)//2
        return True
    
    for x in range(1, N + 1):
        if can_add(x):
            S.add(x)
            in_set[x] = True
    
    best_set = S.copy()
    best_weight = sum(weights[x] for x in S)
    
    # Local search: try single additions and swaps
    # First, try all possible single additions
    for x in range(1, N + 1):
        if not in_set[x]:
            if can_add(x):
                S.add(x)
                in_set[x] = True
                new_weight = best_weight + weights[x]
                if new_weight > best_weight:
                    best_weight = new_weight
                    best_set = S.copy()
                S.remove(x)
                in_set[x] = False
    
    # Try swaps: remove x, add y
    for x in list(S):
        if time.time() > deadline - 30:
            break
        # Remove x
        S.remove(x)
        in_set[x] = False
        current_weight = best_weight - weights[x]
        
        # Try adding each y not in S
        for y in range(1, N + 1):
            if y == x or in_set[y]:
                continue
            if can_add(y):
                S.add(y)
                in_set[y] = True
                new_weight = current_weight + weights[y]
                if new_weight > best_weight:
                    best_weight = new_weight
                    best_set = S.copy()
                S.remove(y)
                in_set[y] = False
        
        # Restore x
        S.add(x)
        in_set[x] = True
    
    # Branch and bound search for further improvement
    # Process elements in order 1..N, at each step decide include or exclude
    # Use exact Fraction-based upper bounds
    
    # For B&B, we need a more efficient representation
    # Use a list of booleans for in_set, and maintain the current weight
    
    # We'll do a limited B&B focusing on the most promising branches
    # Key insight: small elements have high weight, so we should prioritize them
    # But excluding a small element might allow multiple slightly larger ones
    
    # Precompute for each pair (i, j) with i < j, the third element that would complete a 3-AP
    # If i and j are both selected, then certain elements are forbidden
    
    # For B&B, we process elements in order 1, 2, ..., N
    # At each element x, we decide to include or exclude it
    # Upper bound: current_weight + suffix_sum[x+1] (optimistic: assume all remaining can be added)
    # Tighter: for each remaining element y, check if it conflicts with current S
    #   If it conflicts, it contributes 0 to the bound
    
    # To keep B&B fast, we'll limit the depth and use a time check
    # We'll focus on the first ~50 elements where the weight differences are most significant
    
    # Actually, let's do a more targeted search:
    # Identify "critical" elements where the decision matters most
    # These are elements that, when excluded, allow multiple other elements to be included
    
    # For now, let's do a simple B&B on the first K elements
    # and use the greedy solution for the rest
    
    # Let's try a different approach: iterative improvement with random restarts
    # within the local search neighborhood
    
    # Multiple local search passes with different starting points
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        # Create a perturbed version of the best set
        if restart == 0:
            S = best_set.copy()
            in_set = [False] * (N + 1)
            for x in S:
                in_set[x] = True
        else:
            # Remove a random element and re-greedy
            S = best_set.copy()
            if len(S) > 1:
                # Remove a random element
                to_remove = random.choice(list(S))
                S.remove(to_remove)
            
            in_set = [False] * (N + 1)
            for x in S:
                in_set[x] = True
            
            # Re-greedy: try to add elements in order
            for x in range(1, N + 1):
                if not in_set[x] and can_add(x):
                    S.add(x)
                    in_set[x] = True
        
        # Local search on this starting point
        current_weight = sum(weights[x] for x in S)
        
        # Single additions
        for x in range(1, N + 1):
            if not in_set[x]:
                if can_add(x):
                    S.add(x)
                    in_set[x] = True
                    new_weight = current_weight + weights[x]
                    if new_weight > current_weight:
                        current_weight = new_weight
                    else:
                        S.remove(x)
                        in_set[x] = False
        
        # Swaps
        improved = True
        while improved:
            improved = False
            for x in list(S):
                if time.time() > deadline - 5:
                    break
                S.remove(x)
                in_set[x] = False
                temp_weight = current_weight - weights[x]
                
                for y in range(1, N + 1):
                    if y == x or in_set[y]:
                        continue
                    if can_add(y):
                        S.add(y)
                        in_set[y] = True
                        new_weight = temp_weight + weights[y]
                        if new_weight > current_weight:
                            current_weight = new_weight
                            improved = True
                            break
                        S.remove(y)
                        in_set[y] = False
                        in_set[x] = False  # x is already removed
                
                # Restore x if we didn't find a better swap
                if not in_set[x]:
                    S.add(x)
                    in_set[x] = True
                    current_weight += weights[x]
                
                if time.time() > deadline - 5:
                    break
            
            if time.time() > deadline - 5:
                break
        
        # Update best
        if current_weight > best_weight:
            best_weight = current_weight
            best_set = S.copy()
    
    return list(best_set)
