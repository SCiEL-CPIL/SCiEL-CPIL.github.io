"""Narrow admissible tuple: fixed-diameter maximization in [0, N]."""
import time
import math

N = 5000
DEADLINE = 320.0  # safety margin before 330s kill

def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

def _is_admissible(points, primes_to_check):
    """Check if the set of points is admissible for the given primes."""
    for p in primes_to_check:
        residues = set()
        for v in points:
            residues.add(v % p)
        if len(residues) == p:
            return False
    return True

def construct():
    start_time = time.time()
    
    # We need primes up to N (since tuple size could be up to N+1)
    # But realistically, we'll have fewer points. Let's get primes up to N.
    all_primes = _primes(N)
    
    # Start with all integers 0..N
    # Using a boolean array for speed
    active = bytearray(b'\x01') * (N + 1)
    # points are 0..N
    # We need to remove residue classes for each prime p <= len(tuple)
    # Initially len(tuple) = N+1, so we need primes up to N+1.
    # Let's just use primes up to N for simplicity, as len will decrease.
    
    # Optimization: Process primes in increasing order.
    # For a prime p, we need to ensure not all p residues are present.
    # If all p residues are present, we must remove at least one class.
    # To maximize the final size, we should remove the class with the fewest elements.
    
    # However, the order of processing matters.
    # Smaller primes have more classes to cover (p classes) and thus constrain more.
    # But removing elements for a small prime might help with larger primes?
    # Actually, the condition is: for EVERY prime p <= len(S), some residue class mod p is missed.
    
    # Strategy:
    # 1. Start with S = {0, ..., N}.
    # 2. Let M = len(S).
    # 3. Find all primes p <= M.
    # 4. For each such p, check if all residues mod p are covered.
    # 5. If yes, remove the least populated residue class mod p.
    # 6. Repeat until stable.
    
    # To make this efficient, we can iterate primes in order.
    # When we remove elements, M decreases, so the set of relevant primes shrinks.
    
    # Let's maintain a list of active points.
    # Since N=5000, a list of ints is fine.
    points = list(range(N + 1))
    
    # We'll repeatedly check and repair.
    # To avoid O(P * N) per check where P is num primes, we can be smarter.
    # But N=5000 is small. Let's just do it directly first and see if it's fast enough.
    
    # Efficient approach:
    # Keep a count of points per residue for each prime?
    # That's expensive to update.
    
    # Let's try a greedy loop:
    # 1. Determine current max prime to check: p_max = largest prime <= len(points).
    # 2. Iterate primes p <= p_max.
    # 3. If p covers all residues, remove least frequent class.
    # 4. Update points.
    # 5. If points changed, restart loop (or continue, but len changed).
    
    # Actually, removing points can only make it "easier" to miss residues for other primes?
    # No, removing points reduces the set, so it might *uncover* a residue class?
    # No, if we remove points, we are removing elements.
    # If we had all residues for p, and we remove some elements, we might still have all residues.
    # If we remove enough elements from one residue class, we might miss it.
    # So removing elements for prime p1 might help satisfy the condition for prime p2?
    # Or it might not affect p2 at all.
    
    # The safest and simplest greedy:
    # While True:
    #   changed = False
    #   for p in primes up to len(points):
    #     if all residues mod p are present:
    #       remove least frequent class mod p
    #       changed = True
    #       break  # restart loop because len changed and other primes might now be okay or not
    #   if not changed:
    #     break
    
    # This is O(N^2 * log N) in worst case? N=5000.
    # 5000 * 5000 * ~10 = 250 million ops. Might be too slow in Python.
    
    # Let's optimize the "remove least frequent class" step.
    # And maybe process primes in a specific order.
    
    # Alternative:
    # Pre-compute residues for all points for all primes?
    # Memory: 5000 points * ~600 primes = 3 million entries. Feasible.
    
    # Let's try a different approach:
    # We want to keep as many points as possible.
    # This is equivalent to finding the largest subset S of [0,N] such that for all p <= |S|, S is not a complete set of residues mod p.
    
    # Let's use the "repair" strategy but optimize the inner loop.
    
    points = list(range(N + 1))
    # We'll keep a list of active points.
    # To speed up, we can use a set for O(1) removal? No, we need to iterate.
    # List is fine for 5000 elements.
    
    while True:
        if time.time() - start_time > DEADLINE:
            break
        
        m = len(points)
        # Get primes up to m
        # We can reuse the prime list, just filter
        primes_to_check = [p for p in all_primes if p <= m]
        
        changed = False
        for p in primes_to_check:
            # Count residues
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            
            # Check if all residues are covered
            if min(counts) > 0:
                # All residues covered. Need to remove one class.
                # Remove the class with minimum count.
                min_count = min(counts)
                # There might be ties. Which class to remove?
                # Removing a class that is "shared" with other primes' constraints might be better?
                # For now, just pick the first min or any min.
                # Let's pick the one that removes the fewest points (which is min_count).
                # If ties, maybe pick the one with smallest index? Or largest?
                # Let's try smallest index for determinism.
                target_residue = counts.index(min_count)
                
                # Remove all points v where v % p == target_residue
                points = [v for v in points if v % p != target_residue]
                changed = True
                break  # Restart the while loop
        
        if not changed:
            break
    
    return points
