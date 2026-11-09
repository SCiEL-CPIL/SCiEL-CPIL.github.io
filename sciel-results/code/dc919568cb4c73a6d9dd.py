"""
Strategy: Systematic Multi-Start Greedy with Efficient O(K) Incremental Validation 
and Aggressive Local Search (Add/Remove/Swap).

Key Improvements over Seed:
1. O(K) Incremental 3-AP Check: Instead of checking all pairs in the set (O(K^2)) 
   for every insertion attempt, we maintain a set of 'blocked' numbers. 
   When we add x, we only need to check if x is already blocked.
   When we remove x, we re-calculate blocked status for its neighbors.
   This reduces the cost of a single add/remove operation from O(K^2) to O(K) 
   (actually O(1) for check, O(K) for update if we track dependencies, 
   but here we use a simpler 'recompute blocked set' approach for removal 
   or a 'check against existing' approach for addition).
   
   Actually, the most efficient way for 3-AP free sets is:
   - Maintain a boolean array `in_set` of size N+1.
   - To check if x can be added: 
     For each d in 1..N:
       if x-d > 0 and x+d <= N and in_set[x-d] and in_set[x+d]: return False
       if x-2d > 0 and in_set[x-2d] and in_set[x-d]: return False (x is last)
       if x+2d <= N and in_set[x+2d] and in_set[x+d]: return False (x is first)
   This is O(N) per check. With N=256, this is very fast.
   We can optimize by only checking d up to N/3.

2. Aggressive Local Search:
   - Try adding any number from 1..N that isn't in the set.
   - Try removing any number in the set and then adding a different one.
   - Try swapping two numbers (remove one, add another).
   - Use a priority queue or sorted list to try highest value (1/x) additions first.

3. Multi-Start:
   - Run multiple greedy initializations with different orderings 
     (ascending, descending, random, value-based).
   - Apply local search to each.
   - Keep the best result.
"""

import time
import random
from fractions import Fraction

N = 256

def can_add(x, in_set, N):
    """
    Check if x can be added to the set represented by in_set 
    without creating a 3-term AP.
    Returns True if valid.
    """
    # x is in the middle: a, x, b => x-a = b-x => a = 2x-b
    # We need to check if there exist a, b in set such that a+b = 2x.
    # Or x is an endpoint:
    # 1. x, x+d, x+2d => x+d, x+2d in set.
    # 2. x-2d, x-d, x => x-2d, x-d in set.
    
    # Iterate d
    # Max d is such that 3 terms fit in [1, N]
    # If x is middle, d can be up to min(x-1, N-x)
    # If x is first, d can be up to (N-x)//2
    # If x is last, d can be up to (x-1)//2
    
    # Efficient check:
    # For each d:
    #   Case 1: x is middle. a = x-d, b = x+d.
    #   Case 2: x is first. b = x+d, c = x+2d.
    #   Case 3: x is last. a = x-2d, b = x-d.
    
    # We can just iterate d from 1 to N//2 + 1
    for d in range(1, N // 2 + 2):
        # x is middle
        a = x - d
        b = x + d
        if a >= 1 and b <= N:
            if in_set[a] and in_set[b]:
                return False
        
        # x is first
        b = x + d
        c = x + 2 * d
        if b <= N and c <= N:
            if in_set[b] and in_set[c]:
                return False
        
        # x is last
        a = x - 2 * d
        b = x - d
        if a >= 1 and b >= 1:
            if in_set[a] and in_set[b]:
                return False
                
    return True

def greedy_construct(order, in_set=None, S=None):
    """
    Construct a 3-free set using the given order.
    """
    if in_set is None:
        in_set = [False] * (N + 1)
    if S is None:
        S = []
        
    for x in order:
        if not in_set[x]:
            if can_add(x, in_set, N):
                in_set[x] = True
                S.append(x)
    return S, in_set

def sum_reciprocals(S):
    """Calculate exact sum of reciprocals."""
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total

def local_search(S, in_set, time_limit=1.0):
    """
    Perform local search to improve the set.
    1. Try adding any available number (prioritize small x).
    2. Try removing a number and adding a better one.
    3. Try swapping.
    """
    start_time = time.time()
    S_set = set(S)
    current_weight = sum_reciprocals(S)
    
    # List of candidates to add, sorted by 1/x (descending) => x ascending
    candidates = list(range(1, N + 1))
    candidates.sort(key=lambda x: 1.0 / x, reverse=True)
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
            
        # 1. Try adding
        added_any = False
        for x in candidates:
            if in_set[x]:
                continue
            if can_add(x, in_set, N):
                # Add it
                in_set[x] = True
                S.append(x)
                S_set.add(x)
                added_any = True
                # Restart loop as set changed
                improved = True
                break
                
        if added_any:
            continue
            
        # 2. Try swap: remove y, add x
        # To be efficient, we can try removing the "worst" elements (largest x) 
        # and see if we can add a "better" element (smaller x).
        # However, removing a large x might allow adding multiple small x's?
        # Let's try a simple 1-for-1 swap.
        
        # Sort current set by value (descending 1/x)
        current_sorted = sorted(S, key=lambda x: 1.0 / x, reverse=True)
        
        swap_made = False
        # Try removing from the end (largest x, smallest value)
        # and adding from the front (smallest x, largest value)
        for y in current_sorted:
            if time.time() - start_time > time_limit:
                break
                
            # Temporarily remove y
            in_set[y] = False
            S_set.discard(y)
            
            # Try to add any x < y (since we want higher weight)
            # Actually, we can add any x not in set.
            # But adding x > y would likely decrease weight unless it allows further adds.
            # Let's prioritize x < y.
            best_x = None
            best_gain = Fraction(0, 1)
            
            # Optimization: Only check x that are "attractive"
            # We can just iterate through all candidates x < y
            for x in range(1, y):
                if in_set[x]:
                    continue
                if can_add(x, in_set, N):
                    gain = Fraction(1, x) - Fraction(1, y)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            
            if best_x is not None:
                # Perform swap
                in_set[best_x] = True
                S_set.add(best_x)
                S.remove(y)
                S.append(best_x)
                swap_made = True
                improved = True
                break
            else:
                # Restore y
                in_set[y] = True
                S_set.add(y)
                
        if swap_made:
            continue
            
    return S, in_set, sum_reciprocals(S)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    best_in_set = None
    best_weight = Fraction(0, 1)
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy Ascending (Prioritize small numbers)
    order1 = list(range(1, N + 1))
    s1, is1 = greedy_construct(order1)
    s1, is1, w1 = local_search(s1, is1, time_limit=0.5)
    if w1 > best_weight:
        best_weight = w1
        best_set = s1
        best_in_set = is1
        
    # Strategy 2: Greedy Descending (Prioritize large numbers - usually worse for sum 1/x, 
    # but might lead to different local optima)
    order2 = list(range(N, 0, -1))
    s2, is2 = greedy_construct(order2)
    s2, is2, w2 = local_search(s2, is2, time_limit=0.5)
    if w2 > best_weight:
        best_weight = w2
        best_set = s2
        best_in_set = is2
        
    # Strategy 3: Randomized Multi-Start
    # We have ~80s left. Let's do many random starts with quick local search.
    # Each random start + local search should be fast.
    
    trial = 0
    while time.time() < deadline:
        # Create random order
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        # Greedy
        s, iset = greedy_construct(order)
        
        # Local Search with tight time limit
        # Adjust time limit based on remaining time
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        ls_time = min(0.5, remaining / 10) # Allow for 10 more trials
        
        s, iset, w = local_search(s, iset, time_limit=ls_time)
        
        if w > best_weight:
            best_weight = w
            best_set = s
            best_in_set = iset
            
        trial += 1
        
        # Occasionally try a "biased" random order favoring small numbers
        if trial % 5 == 0:
            # Weighted random selection
            weights = [1.0 / x for x in range(1, N + 1)]
            # Normalize
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            
            # Sample without replacement
            order = list(range(1, N + 1))
            # Simple weighted shuffle approximation: 
            # Sort by random() / weight? No, that's not standard.
            # Let's just use a few random shuffles but sort by 1/x with some noise
            noisy_order = sorted(range(1, N + 1), key=lambda x: (1.0/x) + rng.gauss(0, 0.01))
            s, iset = greedy_construct(noisy_order)
            s, iset, w = local_search(s, iset, time_limit=ls_time)
            if w > best_weight:
                best_weight = w
                best_set = s
                best_in_set = iset
            trial += 1

    if not best_set:
        return [1]
        
    return best_set

# Keep the main entry point as construct()
# The task specifies: construct() -> a nonempty subset...
# The current program has a main() but construct() is the interface.
# I will ensure construct() is the one called.

if __name__ == "__main__":
    # For local testing
    res = construct()
    print(f"Result: {res}")
    print(f"Size: {len(res)}")
    print(f"Weight: {sum_reciprocals(res)}")
