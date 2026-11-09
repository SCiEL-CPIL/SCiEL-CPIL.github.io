"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

_ALL_PRIMES = _primes(N)
# Only primes up to N are relevant for the condition p <= len(tuple) <= N+1
# Actually len can be N+1 = 5001, so primes up to 5001
# But max possible size is N+1=5001. Primes > 5001 don't matter.
# _ALL_PRIMES contains primes up to N=5000. We might need prime 5003? No, 5003 > 5001.
# So _ALL_PRIMES is sufficient.

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    def is_admissible(points):
        # Check if tuple is admissible
        # For every prime p <= len(tuple), some residue class mod p is missed
        n = len(points)
        if n == 0:
            return True
        # We only need to check primes p <= n
        for p in _ALL_PRIMES:
            if p > n:
                break
            # Check if all residues 0..p-1 are present
            residues = np.zeros(p, dtype=bool)
            for v in points:
                residues[v % p] = True
            if residues.all():
                return False
        return True

    def repair(points):
        points = list(points)
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            current_len = len(points)
            # Only check primes p <= current_len
            for p in _ALL_PRIMES:
                if p > current_len:
                    break
                # Count residues
                counts = np.zeros(p, dtype=int)
                for v in points:
                    counts[v % p] += 1
                
                # If all residues are covered, we must remove one class
                if counts.min() > 0:
                    # Find least populated class
                    min_size = counts.min()
                    # Pick smallest residue with min size for determinism
                    target_res = np.where(counts == min_size)[0][0]
                    points = [v for v in points if v % p != target_res]
                    changed = True
                    break
        return points

    def add_points(points):
        points = list(points)
        point_set = set(points)
        
        # M[p] is the set of missing residues mod p
        # If M[p] has size 1 and the only missing residue is r, 
        # then adding x where x%p == r will violate condition for p.
        
        M = {}
        for p in _ALL_PRIMES:
            if p > len(points) + 1: # We only care if p <= len(points)+1
                # But as we add points, len increases.
                # Let's just compute for all p in _ALL_PRIMES
                pass
            
            counts = np.zeros(p, dtype=int)
            for v in points:
                counts[v % p] += 1
            M[p] = set(r for r in range(p) if counts[r] == 0)
            
        candidates = [i for i in range(N + 1) if i not in point_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            new_len = len(points) + 1
            safe = True
            for p in _ALL_PRIMES:
                if p > new_len:
                    break
                if M[p] == {x % p}:
                    safe = False
                    break
            
            if safe:
                points.append(x)
                point_set.add(x)
                # Update M
                for p in _ALL_PRIMES:
                    if p > new_len:
                        break
                    r = x % p
                    if r in M[p]:
                        M[p].remove(r)
        
        return points

    def local_search(points):
        # Try to remove points that are "blocking" many candidates?
        # Or simply try to swap a point with a non-point to increase size?
        # If we can swap x in points with y not in points, and size stays same, 
        # it might allow future additions.
        # But we want to maximize size.
        # Let's stick to repair and add.
        pass

    best_points = []
    
    # Strategy 1: Full set repair
    full_set = list(range(N + 1))
    repaired_full = repair(full_set)
    added_full = add_points(repaired_full)
    
    if len(added_full) > len(best_points):
        best_points = added_full
    
    # Strategy 2: Random restarts
    # Use remaining time for multiple random restarts
    while time.time() < deadline - 10:
        # Random subset
        # Size heuristic: start large, repair will shrink it
        size = random.randint(2000, 4000)
        initial_points = random.sample(range(N + 1), size)
        
        repaired = repair(initial_points)
        added = add_points(repaired)
        
        if len(added) > len(best_points):
            best_points = added
            
    return best_points
# EVOLVE-BLOCK-END
