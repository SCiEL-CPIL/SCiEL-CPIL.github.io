"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

N = 5000
DEADLINE = 325  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START

def _primes_upto(limit):
    """Sieve of Eratosthenes for primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]


def _is_admissible(points, primes):
    """Check if points is admissible: for every prime p <= len(points), some residue mod p is missed."""
    n = len(points)
    if n == 0:
        return True
    max_p = n
    # Only need to check primes <= max_p
    for p in primes:
        if p > max_p:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def _greedy_repair(points, primes, deadline):
    """
    Greedy repair: while not admissible, remove the least-populated residue class
    for the first (smallest) prime that covers all residues.
    Use 1-step lookahead: when choosing which class to remove, prefer the removal
    that causes the fewest new violations for other primes.
    """
    points = list(points)
    n = len(points)
    # Precompute primes up to N (max possible tuple length)
    all_primes = _primes_upto(N)
    
    # We only need primes up to len(points) for the admissibility check
    # But as we remove points, len decreases, so we need primes up to initial len
    
    while True:
        if time.time() > deadline:
            break
        n = len(points)
        if n == 0:
            break
        
        # Find the first prime p <= n where all p residue classes are covered
        violating_p = None
        violating_classes = None
        for p in all_primes:
            if p > n:
                break
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            if len(classes) == p:
                violating_p = p
                violating_classes = classes
                break
        
        if violating_p is None:
            break  # Admissible!
        
        # Choose which residue class to remove
        # Strategy: remove the smallest class, but with lookahead
        # Lookahead: for each candidate class, count how many other primes would
        # become violated (or stay violated) after removal
        candidates = sorted(violating_classes.items(), key=lambda x: len(x[1]))
        
        best_class = None
        best_score = None
        
        # Limit lookahead to top few candidates for speed
        for r, vals in candidates[:3]:
            # Simulate removal
            removed_set = set(vals)
            # New points
            new_points = [v for v in points if v not in removed_set]
            new_n = len(new_points)
            
            # Score: how many primes <= new_n are now fully covered (violations)
            violations = 0
            for p in all_primes:
                if p > new_n:
                    break
                residues = set(v % p for v in new_points)
                if len(residues) == p:
                    violations += 1
            
            score = (violations, len(vals))  # Minimize violations, then class size
            if best_score is None or score < best_score:
                best_score = score
                best_class = r
        
        if best_class is None:
            best_class = candidates[0][0]
        
        # Remove the chosen class
        to_remove = set(violating_classes[best_class])
        points = [v for v in points if v not in to_remove]
    
    return points


def _greedy_repair_simple(points, primes, deadline):
    """Simpler greedy: always remove the smallest violating class for the smallest prime."""
    points = list(points)
    all_primes = _primes_upto(N)
    
    while True:
        if time.time() > deadline:
            break
        n = len(points)
        if n == 0:
            break
        
        violating_p = None
        violating_classes = None
        for p in all_primes:
            if p > n:
                break
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            if len(classes) == p:
                violating_p = p
                violating_classes = classes
                break
        
        if violating_p is None:
            break
        
        # Remove the smallest class
        smallest_r, smallest_vals = min(violating_classes.items(), key=lambda x: len(x[1]))
        to_remove = set(smallest_vals)
        points = [v for v in points if v not in to_remove]
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start from all integers in [0, N]
    points = list(range(N + 1))
    
    all_primes = _primes_upto(N)
    
    # Phase 1: Greedy repair with lookahead
    points = _greedy_repair(points, all_primes, deadline)
    
    if time.time() > deadline - 30:
        # Not enough time for more, return what we have
        # Verify admissibility
        n = len(points)
        for p in all_primes:
            if p > n:
                break
            residues = set(v % p for v in points)
            if len(residues) == p:
                # Need to fix - use simple repair
                points = _greedy_repair_simple(points, all_primes, deadline)
                break
        return points
    
    # Phase 2: Try to improve by local search / different removal orders
    # The greedy approach is deterministic. Let's try a few variations.
    
    best_points = points
    best_len = len(points)
    
    # Try: start from all integers but with different initial removal strategies
    # Strategy: remove from the largest prime violation first
    points2 = list(range(N + 1))
    
    while time.time() < deadline - 10:
        n = len(points2)
        if n == 0:
            break
        
        # Find ALL violating primes and pick the largest one
        violating = []
        for p in all_primes:
            if p > n:
                break
            classes = {}
            for v in points2:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            if len(classes) == p:
                violating.append((p, classes))
        
        if not violating:
            break
        
        # Pick the largest violating prime
        p, classes = max(violating, key=lambda x: x[0])
        # Remove the smallest class
        smallest_r, smallest_vals = min(classes.items(), key=lambda x: len(x[1]))
        to_remove = set(smallest_vals)
        points2 = [v for v in points2 if v not in to_remove]
        
        if len(points2) > best_len:
            best_len = len(points2)
            best_points = list(points2)
    
    if len(points2) > len(best_points):
        best_points = points2
    
    # Final verification and repair if needed
    n = len(best_points)
    for p in all_primes:
        if p > n:
            break
        residues = set(v % p for v in best_points)
        if len(residues) == p:
            best_points = _greedy_repair_simple(best_points, all_primes, deadline)
            break
    
    return best_points

# EVOLVE-BLOCK-END
