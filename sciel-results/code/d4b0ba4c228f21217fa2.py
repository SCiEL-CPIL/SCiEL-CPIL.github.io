"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

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
            # Count residue classes using a list for speed
            class_counts = [0] * p
            for v in points:
                class_counts[v % p] += 1
            if all(c > 0 for c in class_counts):
                # Violation: all p classes covered
                min_count = min(class_counts)
                to_remove = set()
                for v in points:
                    if class_counts[v % p] == min_count:
                        to_remove.add(v)
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points


def _local_improvement_fast(points, all_primes, deadline):
    """
    Fast local improvement: try to add elements back using incremental residue tracking.
    For each relevant prime, maintain:
      - residue_counts[p][r]: count of points in residue class r mod p
      - nonempty_count[p]: number of residue classes with count > 0
    
    Adding candidate c is safe iff for every relevant prime p, 
    nonempty_count[p] + (1 if residue_counts[p][c%p]==0 else 0) < p
    """
    points_set = set(points)
    points = list(points)
    m = len(points)
    
    # Precompute relevant primes
    relevant_primes = [p for p in all_primes if p <= m + 1]
    
    # Build residue count structures for current points
    residue_counts = {}
    nonempty_count = {}
    for p in relevant_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        residue_counts[p] = counts
        nonempty_count[p] = sum(1 for c in counts if c > 0)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        
        # Update relevant primes if m changed
        new_relevant = [p for p in all_primes if p <= m + 1]
        for p in new_relevant:
            if p not in residue_counts:
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                residue_counts[p] = counts
                nonempty_count[p] = sum(1 for c in counts if c > 0)
        relevant_primes = new_relevant
        
        # Try candidates in order
        for candidate in range(N + 1):
            if candidate in points_set:
                continue
            if time.time() >= deadline:
                return points
            
            # Check if adding candidate would cause any prime to cover all residues
            admissible = True
            for p in relevant_primes:
                r = candidate % p
                if residue_counts[p][r] == 0:
                    # Adding would make this class non-empty
                    if nonempty_count[p] + 1 == p:
                        admissible = False
                        break
                # else: class already non-empty, no change to nonempty_count
            
            if admissible:
                # Add the candidate
                points.append(candidate)
                points_set.add(candidate)
                m += 1
                # Update residue counts and nonempty counts
                for p in relevant_primes:
                    r = candidate % p
                    if residue_counts[p][r] == 0:
                        nonempty_count[p] += 1
                    residue_counts[p][r] += 1
                
                # Check if a new prime p == m+1 is now relevant
                if m + 1 in residue_counts:
                    pass  # Already tracked
                else:
                    # Check if m+1 is prime
                    for p in all_primes:
                        if p == m + 1:
                            counts = [0] * p
                            for v in points:
                                counts[v % p] += 1
                            residue_counts[p] = counts
                            nonempty_count[p] = sum(1 for c in counts if c > 0)
                            relevant_primes.append(p)
                            break
                
                improved = True
    
    return points


def _swap_improvement(points, all_primes, deadline):
    """
    Try to swap one element out and two elements in to increase size by 1.
    For each element x in points, try removing it and then adding two new elements.
    This is more expensive but can find improvements that simple addition misses.
    """
    points_set = set(points)
    points = list(points)
    m = len(points)
    
    # Build residue tracking
    relevant_primes = [p for p in all_primes if p <= m + 2]
    residue_counts = {}
    nonempty_count = {}
    for p in relevant_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        residue_counts[p] = counts
        nonempty_count[p] = sum(1 for c in counts if c > 0)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        relevant_primes = [p for p in all_primes if p <= m + 2]
        for p in relevant_primes:
            if p not in residue_counts:
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                residue_counts[p] = counts
                nonempty_count[p] = sum(1 for c in counts if c > 0)
        
        # Try each element as a removal candidate
        for i, x in enumerate(points):
            if time.time() >= deadline:
                return points
            
            # Temporarily remove x
            for p in relevant_primes:
                r = x % p
                residue_counts[p][r] -= 1
                if residue_counts[p][r] == 0:
                    nonempty_count[p] -= 1
            
            # Now try to add two elements
            added = []
            for c1 in range(N + 1):
                if c1 in points_set or c1 == x:
                    continue
                if time.time() >= deadline:
                    break
                
                # Check if c1 can be added
                ok1 = True
                for p in relevant_primes:
                    r = c1 % p
                    if residue_counts[p][r] == 0 and nonempty_count[p] + 1 == p:
                        ok1 = False
                        break
                
                if not ok1:
                    continue
                
                # Temporarily add c1
                for p in relevant_primes:
                    r = c1 % p
                    if residue_counts[p][r] == 0:
                        nonempty_count[p] += 1
                    residue_counts[p][r] += 1
                
                # Try to add c2
                found_c2 = False
                for c2 in range(N + 1):
                    if c2 in points_set or c2 == x or c2 == c1:
                        continue
                    if time.time() >= deadline:
                        break
                    
                    ok2 = True
                    for p in relevant_primes:
                        r = c2 % p
                        if residue_counts[p][r] == 0 and nonempty_count[p] + 1 == p:
                            ok2 = False
                            break
                    
                    if ok2:
                        # Add c2
                        for p in relevant_primes:
                            r = c2 % p
                            if residue_counts[p][r] == 0:
                                nonempty_count[p] += 1
                            residue_counts[p][r] += 1
                        added = [c1, c2]
                        found_c2 = True
                        break
                
                # Undo c1
                for p in relevant_primes:
                    r = c1 % p
                    residue_counts[p][r] -= 1
                    if residue_counts[p][r] == 0:
                        nonempty_count[p] -= 1
                
                if found_c2:
                    break
            
            if len(added) == 2:
                # Commit the swap: remove x, add c1 and c2
                points.pop(i)
                points_set.discard(x)
                for c in added:
                    points.append(c)
                    points_set.add(c)
                improved = True
                break
            else:
                # Undo removal of x
                for p in relevant_primes:
                    r = x % p
                    if residue_counts[p][r] == 0:
                        nonempty_count[p] += 1
                    residue_counts[p][r] += 1
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill

    # Precompute primes up to N+1
    all_primes = _primes_upto(N + 1)

    best_points = []
    
    # Strategy 1: Start from a sparse set (1 and primes > sqrt(N)) and repair
    points_sparse = [1] + [p for p in all_primes if p * p > N]
    points_sparse = _greedy_repair(points_sparse, all_primes, deadline)
    if len(points_sparse) > len(best_points):
        best_points = points_sparse
    
    # Strategy 2: Start from all integers and repair
    if time.time() < deadline - 60:
        points_dense = list(range(N + 1))
        points_dense = _greedy_repair(points_dense, all_primes, deadline)
        if len(points_dense) > len(best_points):
            best_points = points_dense
    
    # Strategy 3: Start from evens + 1 (odd numbers)
    if time.time() < deadline - 60:
        points_odd = list(range(1, N + 1, 2))
        points_odd = _greedy_repair(points_odd, all_primes, deadline)
        if len(points_odd) > len(best_points):
            best_points = points_odd

    # Local improvement: try to add back elements using fast method
    if time.time() < deadline and len(best_points) > 0:
        best_points = _local_improvement_fast(best_points, all_primes, deadline)
    
    # Swap improvement: try to increase size by 1 via swaps
    if time.time() < deadline - 30 and len(best_points) > 0:
        best_points = _swap_improvement(best_points, all_primes, deadline)
    
    # Another round of local improvement after swaps
    if time.time() < deadline and len(best_points) > 0:
        best_points = _local_improvement_fast(best_points, all_primes, deadline)

    # Final verification and repair if needed
    if not _check_admissible(best_points, all_primes):
        best_points = _greedy_repair(best_points, all_primes, deadline)

    return best_points
# EVOLVE-BLOCK-END
