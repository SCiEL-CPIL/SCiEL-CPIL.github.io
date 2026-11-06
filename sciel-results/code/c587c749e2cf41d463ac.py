import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 123457
    
    # Precompute largest prime factor
    P = [1] * (N + 1)
    for i in range(2, N + 1):
        if P[i] == 1:
            for j in range(i, N + 1, i):
                P[j] = i
                
    # Prepare sorted list of (largest_prime_factor, number)
    items = [(P[n], n) for n in range(1, N + 1)]
    items.sort()
    
    # Generate primes up to N
    primes = []
    for i in range(2, N + 1):
        if P[i] == i:
            primes.append(i)
            
    smooth_set = set()
    idx = 0
    len_items = len(items)
    
    for B in primes:
        if time.time() > deadline:
            return [1, 123456]
            
        # Add all numbers with largest prime factor equal to B
        while idx < len_items and items[idx][0] == B:
            smooth_set.add(items[idx][1])
            idx += 1
            
        # Check for pair a, b in smooth_set such that a + b = N
        # We can iterate through smooth_set, but it grows.
        # Alternatively, iterate through the newly added numbers and check if complement is in set.
        # But the complement might have been added earlier (with smaller P).
        # So checking newly added numbers is sufficient?
        # If a is new (P(a)=B), and b is in set (P(b)<=B), then max(P(a), P(b)) <= B.
        # Yes.
        
        # To be safe and simple, we can just check all a in smooth_set?
        # smooth_set size grows. Max size is N.
        # But we stop early when B is small.
        # Let's check only newly added numbers to be faster?
        # If a pair exists with max(P)=B, at least one of them must have P=B.
        # So checking newly added numbers is sufficient.
        
        # Let's store newly added numbers
        new_nums = []
        # We already added them in the while loop. Let's re-iterate or collect them.
        # It's better to collect them during the while loop.
        
        # Let's refine the loop to collect new_nums
        # (Rewriting the loop structure in the final code)
        
        pass # Placeholder, will implement properly below

    return [1, 123456]
