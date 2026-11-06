"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import sys
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _covered_length(classes, cap=10**5):
    """Compute the largest L such that every integer 1..L is covered by some class."""
    # For speed, use a boolean array up to cap
    # But we can stop early when we find a gap
    # Use a set-based approach for small L, array for large
    # Actually, let's use a bytearray for efficiency
    covered = bytearray(cap + 1)  # 0-indexed, we care about 1..cap
    for p, r in classes:
        # Mark all n in 1..cap where n % p == r
        # Start from the smallest n >= 1 with n % p == r
        # n = r if r >= 1, else p + r (but r is in [0, p-1], so if r==0, start at p)
        # Actually n % p == r means n = r, r+p, r+2p, ... but n must be >= 1
        # If r == 0: n = p, 2p, 3p, ...
        # If r > 0: n = r, r+p, r+2p, ...
        start = r if r > 0 else p
        for n in range(start, cap + 1, p):
            covered[n] = 1
    
    # Find the first gap
    L = 0
    for n in range(1, cap + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _budget_ok(classes):
    """Check if sum of 1/p <= 1 using exact rationals."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total <= 1

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    # Precompute reciprocals as Fractions
    reciprocals = [Fraction(1, p) for p in primes]
    
    # Initial solution: the seed approach
    # Use primes from 13 upward, covering integer i with prime i (0-indexed from 13)
    # Actually the seed uses primes starting at 13, and the i-th such prime covers i+1
    # Let's replicate and improve
    
    # Strategy: Greedy with local search
    # 1. Start with a good initial covering
    # 2. Try to extend the covered range by adding/replacing primes
    # 3. Use local search to optimize residues
    
    # Initial construction: use small primes to cover early integers densely
    # Then use larger primes to extend coverage
    
    # Key insight: To maximize L, we want to cover 1, 2, 3, ... consecutively
    # The budget is sum(1/p) <= 1
    
    # Let's try a greedy approach:
    # - Maintain a set of selected (p, r) pairs
    # - At each step, find the prime/residue that extends coverage the most
    # - But this is NP-hard, so we use heuristics
    
    # Better approach: 
    # 1. Select a subset of primes with total reciprocal sum <= 1
    # 2. For each selected prime, choose the residue that maximizes coverage
    # 3. Optimize both the subset and the residues
    
    # Phase 1: Select primes
    # We want to maximize coverage. Smaller primes cover more integers per unit budget.
    # But we need to cover consecutively from 1.
    
    # Let's try: select primes greedily by "value" = how many new integers they cover
    # given current coverage, divided by their cost (1/p)
    
    # But first, let's establish a baseline
    # Baseline: use primes 2,3,5,7,11,... and choose residues to cover 1..L
    
    # Actually, let's think differently. 
    # The covering problem: we have residue classes mod p. We want to cover 1..L.
    # For a fixed set of primes, the optimal residues can be found by... it's still hard.
    
    # Practical approach:
    # 1. Start with a set of primes
    # 2. For each prime, try all residues and pick the one that best extends coverage
    # 3. Iterate
    
    # Let's define a function to compute coverage for given (primes, residues)
    
    def compute_coverage(prime_indices, residues, cap=10**5):
        """Compute covered length for given prime indices and residues."""
        # Use a bytearray for efficiency
        covered = bytearray(cap + 1)
        for idx, r in zip(prime_indices, residues):
            p = primes[idx]
            start = r if r > 0 else p
            for n in range(start, cap + 1, p):
                covered[n] = 1
        
        L = 0
        for n in range(1, cap + 1):
            if covered[n]:
                L = n
            else:
                break
        return L
    
    # Greedy prime selection with residue optimization
    # Start with no primes, add one at a time
    
    # But this is slow. Let's use a smarter initialization.
    
    # Initialize with primes that are "good" for covering small integers
    # Prime 2 covers every other number - very efficient
    # Prime 3 covers every third - also good
    
    # Let's try: start with primes 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    # And see what budget that uses
    
    # Sum of reciprocals of primes up to 100:
    # Let's compute it
    small_primes = [p for p in primes if p <= 100]
    small_budget = sum(reciprocals[primes.index(p)] for p in small_primes)
    
    # Actually, let's just do a greedy construction
    # Keep track of selected primes and their best residues
    
    selected_primes = []  # list of prime indices
    selected_residues = []  # corresponding residues
    current_coverage = 0
    current_budget = Fraction(0, 1)
    
    # We'll try to build up coverage
    # For each candidate prime, try all residues and see which extends coverage most
    
    # This is O(n_primes * p * L) which is too slow. Let's optimize.
    
    # Better: for each prime, the best residue is the one that covers the most uncovered numbers in 1..(current_coverage+something)
    
    # Let's use a different strategy:
    # 1. Start with a reasonable set of primes
    # 2. Optimize residues via local search
    # 3. Try to add more primes or swap primes
    
    # Phase 1: Initialize with small primes
    # Use primes up to some limit, choosing residues greedily
    
    # Let's try primes up to 100 first
    init_primes = [p for p in primes if p <= 100]
    init_indices = [primes.index(p) for p in init_primes]
    
    # Check budget
    init_budget = sum(reciprocals[i] for i in init_indices)
    if init_budget > 1:
        # Too much, remove largest primes until budget ok
        while init_budget > 1 and init_indices:
            # Remove the prime with largest p (smallest 1/p) - wait, we want to keep small p
            # Actually, remove the one that contributes least to coverage per unit budget
            # For simplicity, just remove the largest prime
            init_indices.sort(key=lambda i: primes[i], reverse=True)
            init_indices.pop()
            init_budget = sum(reciprocals[i] for i in init_indices)
    
    # Now optimize residues for these primes
    # For each prime, try all residues and keep the best
    
    def optimize_residues(prime_indices, max_iter=100):
        """Optimize residues for given prime indices."""
        n = len(prime_indices)
        if n == 0:
            return []
        
        # Initialize residues to 0
        residues = [0] * n
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                p = primes[prime_indices[i]]
                best_r = residues[i]
                best_cov = compute_coverage(prime_indices, residues, cap=min(10**5, current_coverage + 10000))
                
                # Try a few random residues and the current one
                # For efficiency, try all residues if p is small
                candidates = list(range(p)) if p <= 50 else [residues[i]] + [random.randint(0, p-1) for _ in range(20)]
                
                for r in candidates:
                    if r == best_r:
                        continue
                    old_r = residues[i]
                    residues[i] = r
                    cov = compute_coverage(prime_indices, residues, cap=min(10**5, current_coverage + 10000))
                    if cov > best_cov:
                        best_cov = cov
                        best_r = r
                    residues[i] = old_r
                
                if best_r != residues[i]:
                    residues[i] = best_r
                    improved = True
            
            if not improved:
                break
        
        return residues
    
    import random
    
    # Let's use a simpler, more efficient approach
    # We'll represent coverage as a bit array and update incrementally
    
    def greedy_construct():
        """Greedy construction of covering classes."""
        nonlocal deadline
        
        # Start with no primes
        selected = []  # list of (prime_index, residue)
        budget = Fraction(0, 1)
        
        # We'll maintain coverage as a set of covered numbers
        # But for efficiency, let's use a different approach
        
        # Try adding primes one by one
        # For each prime, find the residue that maximizes new coverage
        
        # Precompute which primes are affordable
        affordable = [i for i in range(n_primes) if reciprocals[i] <= 1 - budget]
        
        # This greedy approach is too slow. Let's use a fixed set and optimize.
        pass
    
    # Let's use a practical approach:
    # 1. Use a fixed set of primes (those that fit in budget)
    # 2. Optimize residues via hill climbing
    # 3. Try to swap primes to improve coverage
    
    # Select primes: start with all primes up to some limit, then trim
    # The sum of reciprocals of all primes up to 1000 is about log(log(1000)) + M ≈ 2.9 + 0.261 ≈ 3.16
    # So we can only use a fraction of them.
    
    # Let's try: use primes in increasing order of 1/p (i.e., decreasing p) until budget is full
    # No wait, we want small p for better coverage.
    
    # Use primes in increasing order of p (smallest first) until budget is full
    selected_indices = []
    budget = Fraction(0, 1)
    for i in range(n_primes):
        if budget + reciprocals[i] <= 1:
            selected_indices.append(i)
            budget += reciprocals[i]
        else:
            break
    
    # Now we have a set of primes. Optimize residues.
    # Use hill climbing: for each prime, try changing its residue
    
    residues = [0] * len(selected_indices)
    
    def get_coverage(indices, res, cap=10**5):
        """Fast coverage computation."""
        if not indices:
            return 0
        covered = bytearray(cap + 1)
        for idx, r in zip(indices, res):
            p = primes[idx]
            start = r if r > 0 else p
            for n in range(start, cap + 1, p):
                covered[n] = 1
        L = 0
        for n in range(1, cap + 1):
            if covered[n]:
                L = n
            else:
                break
        return L
    
    # Optimize residues
    best_coverage = get_coverage(selected_indices, residues)
    
    for outer_iter in range(50):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(len(selected_indices)):
            p = primes[selected_indices[i]]
            old_r = residues[i]
            best_r = old_r
            best_cov = best_coverage
            
            # Try all residues for small p, sample for large p
            if p <= 30:
                candidates = list(range(p))
            else:
                candidates = [old_r] + [random.randint(0, p-1) for _ in range(30)]
            
            for r in candidates:
                if r == best_r:
                    continue
                residues[i] = r
                cov = get_coverage(selected_indices, residues)
                if cov > best_cov:
                    best_cov = cov
                    best_r = r
                residues[i] = old_r
            
            if best_r != old_r:
                residues[i] = best_r
                best_coverage = best_cov
                improved = True
        
        if not improved:
            break
    
    # Try to add more primes if budget allows
    # Or swap primes
    
    # Build the result
    result = [[primes[i], residues[j]] for j, i in enumerate(selected_indices)]
    
    # Verify
    if not _budget_ok(result):
        # Fallback to seed
        primes13 = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        return [[p, (i + 1) % p] for i, p in enumerate(primes13)]
    
    return result

# EVOLVE-BLOCK-END
