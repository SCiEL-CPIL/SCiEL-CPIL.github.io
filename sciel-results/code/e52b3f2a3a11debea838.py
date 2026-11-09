"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import log, exp

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

_ALL_PRIMES = _primes(N)

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    def is_admissible(points):
        n = len(points)
        if n == 0:
            return True
        for p in _ALL_PRIMES:
            if p > n:
                break
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
            for p in _ALL_PRIMES:
                if p > current_len:
                    break
                counts = np.zeros(p, dtype=int)
                for v in points:
                    counts[v % p] += 1
                if counts.min() > 0:
                    min_size = counts.min()
                    target_res = np.where(counts == min_size)[0][0]
                    points = [v for v in points if v % p != target_res]
                    changed = True
                    break
        return points

    def add_points(points):
        points = list(points)
        point_set = set(points)
        
        M = {}
        for p in _ALL_PRIMES:
            if p > len(points) + 1:
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
                for p in _ALL_PRIMES:
                    if p > new_len:
                        break
                    r = x % p
                    if r in M[p]:
                        M[p].remove(r)
        return points

    def simulated_annealing(points):
        """Simulated annealing to improve tuple size"""
        points = list(points)
        point_set = set(points)
        current_size = len(points)
        best_points = list(points)
        best_size = current_size
        
        T = 100.0
        cooling_rate = 0.995
        max_steps = 50000
        
        # Precompute residue counts for efficiency
        def compute_residue_counts(pts):
            counts = {}
            for p in _ALL_PRIMES:
                if p > len(pts):
                    break
                c = np.zeros(p, dtype=int)
                for v in pts:
                    c[v % p] += 1
                counts[p] = c
            return counts
        
        current_counts = compute_residue_counts(points)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Choose a random move: add or remove a point
            if random.random() < 0.6 and len(points) < N + 1:
                # Try to add a point
                x = random.randint(0, N)
                if x in point_set:
                    continue
                
                # Check if adding x would violate admissibility
                new_len = len(points) + 1
                safe = True
                for p in _ALL_PRIMES:
                    if p > new_len:
                        break
                    r = x % p
                    if current_counts[p][r] == 0:
                        # This would be the only point in this residue class
                        # Check if all other residues are already covered
                        other_covered = all(current_counts[p][rr] > 0 for rr in range(p) if rr != r)
                        if other_covered:
                            safe = False
                            break
                
                if safe:
                    points.append(x)
                    point_set.add(x)
                    current_size += 1
                    # Update counts
                    for p in _ALL_PRIMES:
                        if p <= new_len:
                            current_counts[p][x % p] += 1
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_points = list(points)
                else:
                    # Accept with probability based on temperature
                    # Use a smooth penalty for constraint violation
                    penalty = 0.0
                    for p in _ALL_PRIMES:
                        if p <= new_len:
                            r = x % p
                            if current_counts[p][r] == 0:
                                # Count how many other residues are covered
                                other_covered = sum(1 for rr in range(p) if rr != r and current_counts[p][rr] > 0)
                                if other_covered == p - 1:
                                    penalty += 1.0
                    
                    if penalty > 0 and random.random() < exp(-penalty / T):
                        # Accept the violating move temporarily
                        points.append(x)
                        point_set.add(x)
                        current_size += 1
                        for p in _ALL_PRIMES:
                            if p <= new_len:
                                current_counts[p][x % p] += 1
            else:
                # Try to remove a point
                if len(points) == 0:
                    continue
                idx = random.randint(0, len(points) - 1)
                x = points.pop(idx)
                point_set.discard(x)
                current_size -= 1
                
                # Update counts
                for p in _ALL_PRIMES:
                    if p <= len(points):
                        current_counts[p][x % p] -= 1
            
            # Cool down
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
        
        # Final repair to ensure admissibility
        best_points = repair(best_points)
        best_points = add_points(best_points)
        
        return best_points

    def polish(points):
        """Final high-precision polish pass"""
        points = list(points)
        point_set = set(points)
        
        # Try to add any remaining safe points
        M = {}
        for p in _ALL_PRIMES:
            if p > len(points) + 1:
                pass
            counts = np.zeros(p, dtype=int)
            for v in points:
                counts[v % p] += 1
            M[p] = set(r for r in range(p) if counts[r] == 0)
        
        candidates = [i for i in range(N + 1) if i not in point_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 2:
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
                for p in _ALL_PRIMES:
                    if p > new_len:
                        break
                    r = x % p
                    if r in M[p]:
                        M[p].remove(r)
        
        return points

    best_points = []
    
    # Strategy 1: Full set repair
    full_set = list(range(N + 1))
    repaired_full = repair(full_set)
    added_full = add_points(repaired_full)
    
    if len(added_full) > len(best_points):
        best_points = added_full
    
    # Strategy 2: Simulated annealing from different starting points
    while time.time() < deadline - 30:
        # Random subset
        size = random.randint(2000, 4000)
        initial_points = random.sample(range(N + 1), size)
        
        repaired = repair(initial_points)
        sa_result = simulated_annealing(repaired)
        polished = polish(sa_result)
        
        if len(polished) > len(best_points):
            best_points = polished
    
    # Final polish on best result
    best_points = polish(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
