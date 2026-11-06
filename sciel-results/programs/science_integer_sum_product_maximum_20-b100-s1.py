import time
import random
from itertools import combinations

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    """
    sum_set = set()
    prod_set = set()
    # Using explicit loops for clarity and slight speed optimization over list comprehension in some cases, 
    # though for n=20, list comprehension is very fast. 
    # We include diagonal pairs (a+a, a*a) as per task description "equal summands/factors allowed".
    
    for a in A:
        for b in A:
            sum_set.add(a + b)
            prod_set.add(a * b)
            
    return max(len(sum_set), len(prod_set))

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin: 45s out of 60s limit
    
    # Strategy:
    # 1. Initialize with a set that naturally has small product sets.
    #    Powers of 2 are ideal because the product of two powers of 2 is a power of 2.
    #    For A = {2^0, 2^1, ..., 2^19}, A*A is {2^0, ..., 2^38}, size 39.
    #    A+A is much larger, but this is a strong starting point for the product side.
    #    However, 2^19 = 524288 > 10000. So we can't use pure powers of 2 up to 20 elements.
    #    Max power of 2 <= 10000 is 2^13 = 8192. So we can have at most 14 powers of 2 (2^0 to 2^13).
    #    
    #    Alternative Seed: Mixed structure.
    #    Let's try a seed that balances sums and products.
    #    Small integers like 1..20 have:
    #    Sumset: 2..40 (size 39)
    #    Prodset: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 25, 30, 32, 36, 40, 45, 48, 50, 60, 64, 75, 80, 90, 100, 120, 144, 160, 180, 200, 225, 256, 300, 320, 360, 400, 450, 500, 576, 640, 720, 800, 900, 1000, 1152, 1296, 1440, 1600, 1800, 2000, 2304, 2500, 2880, 3200, 3600, 4000, 4410? No 21*21=441. 
    #    Let's just compute it.
    
    # Seed 1: 1..20
    A1 = list(range(1, 21))
    score1 = compute_metrics(A1)
    
    # Seed 2: Powers of 2 (as many as possible) + small integers to fill up to 20
    # 2^0..2^13 are 14 numbers.
    # We need 6 more. Let's add small odd numbers that don't create too many new products.
    # Actually, mixing powers of 2 with other numbers creates a "grid" of products which can be large.
    # A set of form {2^i * 3^j} might be better?
    # Let's try a set of "smooth" numbers.
    
    # Let's try a random search around a good seed.
    # But first, let's look at the structure of the problem.
    # We want |A+A| and |A*A| to be small.
    # |A+A| >= 2n-1 = 39.
    # |A*A| >= 2n-1 = 39.
    # So the theoretical minimum for max(|A+A|, |A*A|) is 39.
    # Can we achieve 39?
    # If A is an arithmetic progression, |A+A| = 2n-1 = 39.
    # A = {a, a+d, ..., a+19d}.
    # A*A = { (a+id)(a+jd) }.
    # The number of distinct products in an AP is generally larger than 2n-1.
    # For A={1..20}, |A+A|=39. |A*A|=?
    # Let's calculate score1.
    
    # If score1 is close to 39, that's great.
    # Let's run a local search to improve upon the best seed found.
    
    best_A = A1[:]
    best_score = score1
    
    # Try a few other seeds
    # Seed: Geometric progression (limited by max value)
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14 nums)
    # Add 3, 5, 7, 9, 11, 13?
    A2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # Need 6 more distinct positive integers <= 10000 not in A2.
    # To minimize products, we want numbers that share factors with existing ones.
    # But we want to minimize the *set* size.
    # Adding 3: Products with 3 will be 3*2^k. These are new.
    # Adding 9: 9*2^k.
    # This creates a 2D grid.
    
    # Let's stick to the AP seed 1..20 as it minimizes the sumset to the theoretical minimum.
    # The product set is the bottleneck.
    
    # Local Search:
    # We will perform a hill-climbing search.
    # Move: Replace one element x in A with a new element y.
    # Constraint: y is not in A, y <= 10000, y > 0.
    # Objective: Minimize max(|A+A|, |A*A|).
    
    # To make this efficient, we need to calculate the change in score quickly?
    # For n=20, calculating full metrics takes ~400 ops.
    # 60s / 400 ops = 150k evaluations. That's plenty for simple hill climbing.
    
    # Let's implement a simple local search with random restarts or just one long run.
    # Given the landscape might be rugged, a simple 1-swap hill climb might get stuck.
    # Let's use a "best improvement" or "first improvement" strategy.
    # To avoid local minima, we can occasionally accept worse moves (simulated annealing) or do random restarts.
    # But 45s is short for many restarts. Let's try a robust local search.
    
    # We'll maintain a pool of candidates.
    # Initially, candidates can be numbers 1..10000.
    # But random picking from 1..10000 is inefficient.
    # We can bias towards numbers that are "similar" to current set or small numbers.
    
    # Let's try a different approach:
    # The set 1..20 has sumset size 39.
    # Product set size for 1..20:
    # Let's compute it to see the baseline.
    
    # I will write the code to compute score1 and then run a local search.
    
    # Optimization: Pre-compute pair sums and products?
    # When we swap x for y:
    # New sums: (y+a) for a in A\{x}. Old sums: (x+a) for a in A\{x}. Plus x+x and y+y.
    # We can update the set size incrementally?
    # It's complex because removing an element might not reduce the set size if another pair produces the same sum.
    # Given n=20, full recomputation is ~20*20 = 400 additions and 400 multiplications.
    # In Python, 400 operations is very fast (< 0.001s).
    # So we can do ~10,000 - 100,000 evaluations in 45s.
    
    # Let's do a random local search.
    
    A = A1[:]
    current_score = best_score
    
    # Candidate generation strategy:
    # 1. Small numbers (1-100)
    # 2. Numbers close to existing elements
    # 3. Random numbers 1-10000
    
    # Let's start with a simple loop.
    
    while time.time() < deadline:
        # Choose a random element to replace
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Generate a few candidate replacements
        found_improvement = False
        
        # Try 10 random candidates per step? Or just 1?
        # Let's try a batch of candidates to increase chance of finding a good move.
        # But we need to be fast.
        
        # Candidate 1: Random small number
        # Candidate 2: Random number 1-10000
        # Candidate 3: old_val +/- 1
        
        candidates = []
        
        # Small numbers are often good for keeping products small
        for _ in range(5):
            candidates.append(random.randint(1, 50))
            
        # Numbers near current value
        candidates.append(max(1, old_val - 1))
        candidates.append(old_val + 1)
        
        # Random large number
        candidates.append(random.randint(1, 10000))
        
        # Remove invalid candidates (already in A)
        A_set = set(A)
        A_set.remove(old_val) # We are replacing this, so it's allowed to pick something else
        # Actually, if we pick a value already in A (other than old_val), it's invalid.
        
        best_candidate = None
        best_candidate_score = current_score
        
        for c in candidates:
            if c in A_set:
                continue
            if c == old_val:
                continue
            if c < 1 or c > 10000:
                continue
            
            # Test candidate
            A[idx] = c
            new_score = compute_metrics(A)
            
            if new_score < best_candidate_score:
                best_candidate_score = new_score
                best_candidate = c
                
        if best_candidate is not None:
            A[idx] = best_candidate
            current_score = best_candidate_score
            if current_score < best_score:
                best_score = current_score
                best_A = A[:]
        else:
            # No improvement found in this batch, revert if we changed A during testing
            # We did change A[idx] in the loop, so we need to revert to old_val if we didn't accept a move
            # Actually, the loop sets A[idx] = c. If no c was accepted, A[idx] is the last tested c.
            # We must restore old_val.
            A[idx] = old_val

    return best_A

# Let's add a quick check to ensure the returned set is valid.
# The task requires distinct positive integers <= 10000.
# My local search ensures distinctness and bounds.

# One refinement: The initial seed 1..20 might not be the best starting point.
# What if we start with a set that has a smaller product set initially?
# For example, powers of 2.
# Let's initialize best_A with the best of a few seeds.

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    # Seed 1: 1..20
    seeds = [
        list(range(1, 21)),
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 3, 5, 7, 9, 11, 13], # Mixed
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71], # Primes (bad for products, good for sums?)
    ]
    
    best_A = None
    best_score = float('inf')
    
    for seed in seeds:
        # Ensure seed is valid (distinct, positive, <=10000)
        if len(set(seed)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in seed):
            continue
        
        score = compute_metrics(seed)
        if score < best_score:
            best_score = score
            best_A = seed[:]
            
    # Local Search
    A = best_A[:]
    current_score = best_score
    
    # To speed up, we can pre-compute the set of sums and products?
    # No, full recompute is fast enough for n=20.
    
    while time.time() < deadline:
        # Randomly select an index to perturb
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Generate candidates
        # Bias towards small numbers and numbers that create collisions
        candidates = set()
        
        # 1. Small numbers
        for _ in range(3):
            candidates.add(random.randint(1, 30))
            
        # 2. Neighbors
        candidates.add(max(1, old_val - 1))
        candidates.add(old_val + 1)
        candidates.add(max(1, old_val - 2))
        candidates.add(old_val + 2)
        
        # 3. Random
        for _ in range(2):
            candidates.add(random.randint(1, 10000))
            
        A_set = set(A)
        A_set.discard(old_val)
        
        best_cand = None
        best_cand_score = current_score
        
        for c in candidates:
            if c in A_set:
                continue
            if c == old_val:
                continue
            if c < 1 or c > 10000:
                continue
            
            A[idx] = c
            s = compute_metrics(A)
            if s < best_cand_score:
                best_cand_score = s
                best_cand = c
                
        if best_cand is not None:
            A[idx] = best_cand
            current_score = best_cand_score
            if current_score < best_score:
                best_score = current_score
                best_A = A[:]
        else:
            A[idx] = old_val
            
    return best_A

# Final check: The function must return a list of 20 distinct positive integers <= 10000.
# The local search maintains this invariant.
