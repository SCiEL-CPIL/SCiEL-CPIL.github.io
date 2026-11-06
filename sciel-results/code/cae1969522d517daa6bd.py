"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes_upto(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def _check_admissible(points, all_primes):
    """Check if points is admissible for all primes p <= len(points)."""
    m = len(points)
    for p in all_primes:
        if p > m:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True


def _greedy_repair(points, all_primes, deadline):
    """
    Greedy repair: repeatedly find a prime p <= len(points) whose residue classes
    are all covered, and remove the least-populated class. Continue until admissible
    or deadline.
    """
    points = list(points)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        m = len(points)
        for p in all_primes:
            if p > m:
                break
            class_counts = [0] * p
            for v in points:
                class_counts[v % p] += 1
            if all(c > 0 for c in class_counts):
                min_count = min(class_counts)
                to_remove = set()
                for v in points:
                    if class_counts[v % p] == min_count:
                        to_remove.add(v)
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points


def _build_residue_state(points, all_primes, max_p):
    """Build residue count and nonempty count structures for given points."""
    residue_counts = {}
    nonempty_counts = {}
    relevant_primes = [p for p in all_primes if p <= max_p]
    for p in relevant_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        residue_counts[p] = counts
        nonempty_counts[p] = sum(1 for c in counts if c > 0)
    return residue_counts, nonempty_counts, relevant_primes


def _local_improvement_fast(points, all_primes, deadline):
    """
    Fast local improvement: try to add elements back using incremental residue tracking.
    Uses candidate ordering based on scarcity of residue classes.
    """
    points_set = set(points)
    points = list(points)
    m = len(points)
    
    residue_counts, nonempty_counts, relevant_primes = _build_residue_state(points, all_primes, m + 1)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        
        # Update relevant primes
        new_relevant = [p for p in all_primes if p <= m + 1]
        for p in new_relevant:
            if p not in residue_counts:
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                residue_counts[p] = counts
                nonempty_counts[p] = sum(1 for c in counts if c > 0)
        relevant_primes = new_relevant
        
        # Score candidates: prefer those that fill scarce residue classes
        # and don't risk violating constraints
        candidates = []
        for candidate in range(N + 1):
            if candidate in points_set:
                continue
            if time.time() >= deadline:
                break
            
            # Check admissibility and compute a score
            admissible = True
            score = 0
            for p in relevant_primes:
                counts = residue_counts[p]
                ne = nonempty_counts[p]
                r = candidate % p
                if counts[r] == 0:
                    if ne + 1 == p:
                        admissible = False
                        break
                    # This candidate fills a scarce class - good
                    score += 1 / (counts[r] + 1)
                else:
                    # Already in this class - neutral
                    score += 0.1 / (counts[r] + 1)
            
            if admissible:
                candidates.append((score, candidate))
        
        # Sort by score descending, try best candidates first
        candidates.sort(key=lambda x: -x[0])
        
        for score, candidate in candidates:
            if time.time() >= deadline:
                return points
            
            # Re-check admissibility (state may have changed)
            admissible = True
            for p in relevant_primes:
                counts = residue_counts[p]
                ne = nonempty_counts[p]
                r = candidate % p
                if counts[r] == 0:
                    if ne + 1 == p:
                        admissible = False
                        break
            
            if admissible:
                points.append(candidate)
                points_set.add(candidate)
                m += 1
                for p in relevant_primes:
                    r = candidate % p
                    if residue_counts[p][r] == 0:
                        nonempty_counts[p] += 1
                    residue_counts[p][r] += 1
                # Add new primes if m+1 is a prime
                if m + 1 <= N + 1:
                    for p in all_primes:
                        if p == m + 1:
                            counts = [0] * p
                            for v in points:
                                counts[v % p] += 1
                            residue_counts[p] = counts
                            nonempty_counts[p] = sum(1 for c in counts if c > 0)
                            relevant_primes.append(p)
                            break
                improved = True
                break  # Only add one per outer iteration
    
    return points


def _swap_improvement(points, all_primes, deadline):
    """
    Try swap operations: remove one element, add another.
    This can unlock solutions where direct addition fails.
    """
    points = list(points)
    points_set = set(points)
    m = len(points)
    
    residue_counts, nonempty_counts, relevant_primes = _build_residue_state(points, all_primes, m + 1)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        
        # Update relevant primes
        new_relevant = [p for p in all_primes if p <= m + 1]
        for p in new_relevant:
            if p not in residue_counts:
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                residue_counts[p] = counts
                nonempty_counts[p] = sum(1 for c in counts if c > 0)
        relevant_primes = new_relevant
        
        # Try all (remove, add) pairs
        best_swap = None
        best_score = 0
        
        for idx, v_remove in enumerate(points):
            if time.time() >= deadline:
                break
            
            # Tentatively remove v_remove
            old_counts = {}
            for p in relevant_primes:
                r = v_remove % p
                old_counts[p] = (residue_counts[p][r], nonempty_counts[p])
                residue_counts[p][r] -= 1
                if residue_counts[p][r] == 0:
                    nonempty_counts[p] -= 1
            
            # Try adding each candidate
            for v_add in range(N + 1):
                if v_add in points_set:
                    continue
                if v_add == v_remove:
                    continue
                
                admissible = True
                for p in relevant_primes:
                    r = v_add % p
                    if residue_counts[p][r] == 0:
                        if nonempty_counts[p] + 1 == p:
                            admissible = False
                            break
                
                if admissible:
                    # Compute score: prefer swaps that add scarce classes
                    score = 0
                    for p in relevant_primes:
                        r = v_add % p
                        if residue_counts[p][r] == 0:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_swap = (idx, v_remove, v_add)
            
            # Restore state
            for p in relevant_primes:
                r = v_remove % p
                old_count, old_ne = old_counts[p]
                residue_counts[p][r] = old_count
                nonempty_counts[p] = old_ne
        
        if best_swap is not None:
            idx, v_remove, v_add = best_swap
            # Perform the swap
            points[idx] = v_add
            points_set.remove(v_remove)
            points_set.add(v_add)
            
            # Update residue counts
            for p in relevant_primes:
                r_remove = v_remove % p
                r_add = v_add % p
                residue_counts[p][r_remove] -= 1
                if residue_counts[p][r_remove] == 0:
                    nonempty_counts[p] -= 1
                residue_counts[p][r_add] += 1
                if residue_counts[p][r_add] == 1:
                    nonempty_counts[p] += 1
            
            # Add new primes if m+1 is a prime (size didn't change, but check)
            improved = True
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill

    # Precompute primes up to N+1
    all_primes = _primes_upto(N + 1)

    best_points = []
    
    # Strategy 1: Start from all integers and repair
    points_dense = list(range(N + 1))
    points_dense = _greedy_repair(points_dense, all_primes, deadline)
    if len(points_dense) > len(best_points):
        best_points = points_dense
    
    # Strategy 2: Start from odd numbers
    if time.time() < deadline - 40:
        points_odd = list(range(1, N + 1, 2))
        points_odd = _greedy_repair(points_odd, all_primes, deadline)
        if len(points_odd) > len(best_points):
            best_points = points_odd
    
    # Strategy 3: Start from multiples of 3
    if time.time() < deadline - 40:
        points_mult3 = list(range(0, N + 1, 3))
        points_mult3 = _greedy_repair(points_mult3, all_primes, deadline)
        if len(points_mult3) > len(best_points):
            best_points = points_mult3

    # Local improvement: try to add back elements using fast method
    if time.time() < deadline - 20 and len(best_points) > 0:
        best_points = _local_improvement_fast(best_points, all_primes, deadline)

    # Swap improvement
    if time.time() < deadline - 10 and len(best_points) > 0:
        best_points = _swap_improvement(best_points, all_primes, deadline)

    # Final local improvement after swaps
    if time.time() < deadline - 5 and len(best_points) > 0:
        best_points = _local_improvement_fast(best_points, all_primes, deadline)

    # Final verification and repair if needed
    if not _check_admissible(best_points, all_primes):
        best_points = _greedy_repair(best_points, all_primes, deadline)

    return best_points
# EVOLVE-BLOCK-END
