"""
Narrow admissible tuple: maximize |S| for S ⊂ [0, N] such that for every prime p ≤ |S|,
S does not cover all residue classes mod p.

Strategy:
1. Branch over residue classes for the first few primes (2,3,5,7,11) to generate diverse
   initial configurations that avoid the local optima of single-path greedy.
2. For each branch, run a fast greedy repair.
3. Use simulated annealing with swap moves to improve the best solution found.
"""

import time
import random
from math import isqrt

N = 5000
DEADLINE = 320.0  # seconds, safety margin
START_TIME = time.time()


def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]


_PRIMES = _primes(N)
# Primes that are "small" enough to be constraints for sets up to size 5000
_SMALL_PRIMES = [p for p in _PRIMES if p <= 100]  # We'll use these for checking


def _is_admissible(points, primes_list):
    """Check if points is admissible: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    if n == 0:
        return True
    for p in primes_list:
        if p > n:
            break
        # Check if all residue classes 0..p-1 are covered
        seen = [False] * p
        for v in points:
            seen[v % p] = True
            if all(seen):
                return False
    return True


def _get_smallest_violated_prime(points, primes_list):
    """Return the smallest prime p <= len(points) such that points covers all residues mod p, or None."""
    n = len(points)
    for p in primes_list:
        if p > n:
            break
        seen = [False] * p
        for v in points:
            seen[v % p] = True
        if all(seen):
            return p
    return None


def _greedy_remove(points, primes_list):
    """
    Greedily repair points by removing the least-populated residue class for the smallest violated prime.
    Returns the repaired list.
    """
    points = list(points)
    changed = True
    max_iter = 100
    while changed and max_iter > 0:
        max_iter -= 1
        changed = False
        p = _get_smallest_violated_prime(points, primes_list)
        if p is None:
            break
        # Count residues
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        # Find min count residue
        min_count = min(counts)
        min_residues = [r for r in range(p) if counts[r] == min_count]
        # Remove one of the min residue classes (deterministic: smallest)
        target_r = min_residues[0]
        points = [v for v in points if v % p != target_r]
        changed = True
    return points


def _greedy_add(points, primes_list, candidates):
    """
    Try to add candidates to points while maintaining admissibility.
    Returns the improved list.
    """
    points = list(points)
    point_set = set(points)
    for v in candidates:
        if v in point_set:
            continue
        # Check if adding v maintains admissibility
        test = points + [v]
        if _is_admissible(test, primes_list):
            points = test
            point_set.add(v)
    return points


def _branch_and_bound_initial(branch_primes, primes_list):
    """
    Branch over residue classes for the given primes to create diverse initial sets.
    For each combination of forbidden residue classes, start with all numbers not in any forbidden class,
    then repair.
    Returns a list of candidate initial sets.
    """
    # For each prime in branch_primes, we forbid one residue class.
    # We'll try all combinations of the first few primes.
    # To keep it manageable, branch over at most 5 primes.
    bp = branch_primes[:5]
    if not bp:
        return [list(range(N + 1))]
    
    candidates = []
    # Generate all combinations of forbidden residues
    def _gen(idx, forbidden):
        if idx == len(bp):
            # Create initial set: all numbers not in any forbidden class
            init = []
            for v in range(N + 1):
                ok = True
                for (p, r) in forbidden:
                    if v % p == r:
                        ok = False
                        break
                if ok:
                    init.append(v)
            candidates.append(init)
            return
        
        p = bp[idx]
        for r in range(p):
            forbidden.append((p, r))
            _gen(idx + 1, forbidden)
            forbidden.pop()
    
    _gen(0, [])
    return candidates


def _simulated_annealing(points, primes_list, time_limit=60.0):
    """
    Simulated annealing to improve the admissible set.
    Moves: add a random excluded element, or swap an included with an excluded.
    """
    start = time.time()
    best = list(points)
    best_len = len(best)
    current = list(points)
    current_set = set(current)
    
    # Precompute excluded elements
    excluded = [v for v in range(N + 1) if v not in current_set]
    
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start < time_limit:
        if T < T_min:
            break
        
        move_type = random.random()
        if move_type < 0.5:
            # Add a random excluded element
            if not excluded:
                break
            v = random.choice(excluded)
            new_set = list(current) + [v]
            new_set_set = current_set | {v}
            # Check admissibility
            if _is_admissible(new_set, primes_list):
                current = new_set
                current_set = new_set_set
                excluded.remove(v)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                # Reject (could accept with probability, but for simplicity reject)
                pass
        elif move_type < 0.8:
            # Swap: remove a random included, add a random excluded
            if not excluded or not current:
                break
            v_remove = random.choice(current)
            v_add = random.choice(excluded)
            if v_remove == v_add:
                continue
            new_set = [v for v in current if v != v_remove] + [v_add]
            if _is_admissible(new_set, primes_list):
                current = new_set
                current_set = set(current)
                excluded.remove(v_add)
                excluded.append(v_remove)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
        else:
            # Remove a random included element (to explore smaller sets temporarily)
            if not current:
                break
            v_remove = random.choice(current)
            new_set = [v for v in current if v != v_remove]
            # Always accept removal (it makes the set more admissible)
            current = new_set
            current_set = set(current)
            excluded.append(v_remove)
        
        T *= cooling
    
    return best


def construct():
    start = time.time()
    best = []
    best_len = 0
    
    # Phase 1: Branch and bound over first few primes
    # Use primes 2, 3, 5, 7 for branching (2*3*5*7 = 210 combinations)
    branch_primes = [2, 3, 5, 7]
    candidates = _branch_and_bound_initial(branch_primes, _PRIMES)
    
    # For each candidate, run greedy repair and keep the best
    for i, cand in enumerate(candidates):
        if time.time() - start > DEADLINE - 60:
            break
        repaired = _greedy_remove(cand, _PRIMES)
        if len(repaired) > best_len:
            best = list(repaired)
            best_len = len(repaired)
    
    # Phase 2: Try some random restarts with greedy
    random.seed(42)
    for trial in range(50):
        if time.time() - start > DEADLINE - 30:
            break
        # Random initial set: start with all, random repair
        points = list(range(N + 1))
        # Randomly remove some elements first
        for _ in range(random.randint(0, 100)):
            if points:
                idx = random.randint(0, len(points) - 1)
                points.pop(idx)
        repaired = _greedy_remove(points, _PRIMES)
        if len(repaired) > best_len:
            best = list(repaired)
            best_len = len(repaired)
    
    # Phase 3: Simulated annealing on the best
    if time.time() - start < DEADLINE - 5:
        sa_time = min(60.0, DEADLINE - 5 - (time.time() - start))
        if sa_time > 5:
            best = _simulated_annealing(best, _PRIMES, time_limit=sa_time)
    
    return best
