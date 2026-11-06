from itertools import combinations

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors
    divisors.sort()
    
    # Try to find a solution of size 4
    # Since we want to minimize the number of terms, we check size 4 first.
    # If found, return it.
    # The number of combinations is ~487k, which is fast.
    for combo in combinations(divisors, 4):
        if sum(combo) == target:
            return list(combo)
            
    # If no size 4 solution found, use the greedy approach for size 5
    # The greedy approach is known to work for this specific instance
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    
    # Verify greedy solution
    if sum(chosen) == target and len(set(chosen)) == len(chosen):
        return chosen
        
    # Fallback (should not happen)
    return []
