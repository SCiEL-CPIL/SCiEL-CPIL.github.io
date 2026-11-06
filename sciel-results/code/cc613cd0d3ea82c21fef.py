import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill
    n = 256
    
    # Precompute conflicts: for each pair (a,b), check if they form a 3AP with some c
    # A 3AP is (a, b, c) where a+c = 2b, so for any two elements, we can check
    # if adding a third would create a 3AP
    
    # For branch-and-bound, we need to check if a candidate element conflicts with existing set
    def can_add(S, x):
        """Check if x can be added to S without creating a 3AP."""
        for a in S:
            # Check all three positions for a 3AP involving x and a
            # Case 1: x is the middle term: a, x, c where c = 2x - a
            c = 2 * x - a
            if c in S and c != x and c != a:
                return False
            # Case 2: x is the first term: x, a, c where c = 2a - x
            c = 2 * a - x
            if c in S and c != x and c != a:
                return False
            # Case 3: x is the last term: a, c, x where c = (a+x)/2 if integer
            if (a + x) % 2 == 0:
                c = (a + x) // 2
                if c in S and c != x and c != a:
                    return False
        return True
    
    # Greedy initialization to get a good starting point
    S_greedy = set()
    for x in range(1, n + 1):
        if can_add(S_greedy, x):
            S_greedy.add(x)
    
    best_set = set(S_greedy)
    best_weight = sum(Fraction(1, x) for x in S_greedy)
    
    # Branch and bound
    # Order elements by weight (descending) - smaller numbers have larger 1/x
    elements = list(range(1, n + 1))
    elements.sort(key=lambda x: -1.0/x)  # Descending by weight
    
    # Precompute suffix sums of weights for upper bound
    # suffix_sum[i] = sum of weights of elements[i:]
    suffix_sum = [Fraction(0)] * (len(elements) + 1)
    for i in range(len(elements) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + Fraction(1, elements[i])
    
    def branch_and_bound(current_set, current_weight, idx):
        """Branch and bound to find optimal subset."""
        nonlocal best_set, best_weight
        
        if time.time() > deadline - 1.0:
            return
        
        # Upper bound: current weight + sum of all remaining elements
        upper_bound = current_weight + suffix_sum[idx]
        
        # If upper bound can't beat best, prune
        if upper_bound <= best_weight:
            return
        
        # If we've considered all elements, update best if needed
        if idx == len(elements):
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
        
        x = elements[idx]
        
        # Option 1: Include x if valid
        if can_add(current_set, x):
            current_set.add(x)
            branch_and_bound(current_set, current_weight + Fraction(1, x), idx + 1)
            current_set.remove(x)
        
        # Option 2: Exclude x
        branch_and_bound(current_set, current_weight, idx + 1)
    
    # Run branch and bound with initial greedy solution as lower bound
    branch_and_bound(set(), Fraction(0), 0)
    
    return sorted(best_set)
