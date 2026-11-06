import itertools

def construct():
    n = 5040
    target = 4979
    
    # Generate all distinct positive divisors of n
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    
    # Filter out divisors larger than the target, as they cannot be part of a sum equal to target
    # (since all divisors are positive)
    divisors = [d for d in divisors if d <= target]
    
    # We want to minimize the number of divisors.
    # Iterate k from 1 upwards.
    # The maximum number of divisors is len(divisors), but we expect a small k.
    # We know k=5 works from the seed.
    # Checking up to 10 is safe and fast given the small size of the divisor set (~59).
    
    for k in range(1, 11):
        # Generate all combinations of k distinct divisors
        for combo in itertools.combinations(divisors, k):
            if sum(combo) == target:
                return list(combo)
                
    # Fallback: If no small-k solution is found (which is theoretically impossible 
    # given the seed works with k=5, but for safety), use the greedy approach 
    # from the seed program to guarantee a valid output.
    remainder = target
    chosen = []
    for d in range(n, 0, -1):
        if n % d == 0 and d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen
