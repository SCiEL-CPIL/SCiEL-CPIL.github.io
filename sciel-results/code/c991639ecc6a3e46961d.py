import itertools
import random

def is_valid(A):
    n = len(A)
    # S[k] is a set of sums for subsets of size k
    S = [set() for _ in range(n + 1)]
    S[0].add(0)
    
    for i, a in enumerate(A):
        # Update S in reverse to avoid using 'a' multiple times in the same step
        # Actually, we are adding 'a' to existing subsets.
        # Let's just compute all subset sums directly for simplicity if n is small (< 20)
        pass
        
    # Better: compute all subset sums
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    for mask in range(1, 1 << n):
        # This is too slow for n=20. 
        # Use DP:
        pass

    # Efficient DP for subset sums by cardinality
    # dp[k] is a set of sums for cardinality k
    dp = [set() for _ in range(n + 1)]
    dp[0].add(0)
    
    for a in A:
        # Update from high k to low k
        for k in range(n, 0, -1):
            if dp[k-1]:
                for s in dp[k-1]:
                    dp[k].add(s + a)
                    
    # Check disjointness
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if dp[i] & dp[j]:
                return False
    return True

def construct():
    # Try to find a set of size 10
    # Start with powers of 2
    base = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Check if base is valid (it is)
    # Try to see if we can replace 256 with something else to allow more elements?
    # Or just try to add to a smaller base.
    
    # Let's try a greedy construction
    A = []
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    
    # Sort candidates? Greedy usually works better with specific order.
    # Let's try to build up to size 10.
    
    # Heuristic: Try to fit 10 numbers.
    # We know 9 works.
    
    # Try random restarts
    best_set = base
    for _ in range(1000):
        # Start with a random subset of size 8 or 9?
        # Or start with powers of 2 and perturb.
        
        # Let's try a simple hill climbing from powers of 2
        current = [1, 2, 4, 8, 16, 32, 64, 128]
        # Try to add 2 more numbers
        # We have 256 left.
        
        # This is getting complicated.
        pass

    return best_set
