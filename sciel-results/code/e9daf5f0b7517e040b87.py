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


def _local_improvement_fast(points, all_primes, deadline):
    """
    Fast local improvement: try to add elements back using incremental residue tracking.
    Uses dynamic candidate limiting and early termination for speed.
    """
    points_set = set(points)
    points = list(points)
    m = len(points)
    
    if m == 0:
        return points
    
    relevant_primes = [p for p in all_primes if p <= m + 1]
    
    residue_counts = {}
    for p in relevant_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        residue_counts[p] = counts
    
    nonempty_counts = {}
    for p in relevant_primes:
        nonempty_counts[p] = sum(1 for c in residue_counts[p] if c > 0)
    
    # Precompute which residue classes are "critical" (adding would complete the set)
    critical_mask = {}
    for p in relevant_primes:
        ne = nonempty_counts[p]
        if ne + 1 == p:
            empty_classes = [r for r in range(p) if residue_counts[p][r] == 0]
            critical_mask[p] = set(empty_classes)
        else:
            critical_mask[p] = set()
    
    # Build a conflict score array for all candidates
    # conflict_score[i] = number of primes p where i % p is in critical_mask[p]
    conflict_scores = np.zeros(N + 1, dtype=np.int32)
    for p in relevant_primes:
        if critical_mask[p]:
            for r in critical_mask[p]:
                # Mark all numbers congruent to r mod p
                for i in range(r, N + 1, p):
                    conflict_scores[i] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        elapsed = time.time() - start_time if 'start_time' in globals() else 0
        
        new_relevant = [p for p in all_primes if p <= m + 1]
        for p in new_relevant:
            if p not in residue_counts:
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                residue_counts[p] = counts
                nonempty_counts[p] = sum(1 for c in counts if c > 0)
                ne = nonempty_counts[p]
                if ne + 1 == p:
                    empty_classes = [r for r in range(p) if residue_counts[p][r] == 0]
                    critical_mask[p] = set(empty_classes)
                else:
                    critical_mask[p] = set()
        relevant_primes = new_relevant
        
        # Recompute conflict scores efficiently
        for p in relevant_primes:
            if critical_mask[p]:
                for r in critical_mask[p]:
                    for i in range(r, N + 1, p):
                        if i not in points_set:
                            conflict_scores[i] += 1
            else:
                pass
        # Actually, let's just recompute from scratch each iteration for correctness
        conflict_scores = np.zeros(N + 1, dtype=np.int32)
        for p in relevant_primes:
            if critical_mask[p]:
                for r in critical_mask[p]:
                    for i in range(r, N + 1, p):
                        conflict_scores[i] += 1
        
        # Get candidates sorted by conflict score, limit to top K for speed
        available = [c for c in range(N + 1) if c not in points_set]
        if not available:
            break
        
        # Sort by conflict score (fewer conflicts first)
        available.sort(key=lambda c: conflict_scores[c])
        
        # Limit candidates to improve speed - use more candidates if we have time
        remaining_time = deadline - time.time()
        if remaining_time > 60:
            max_candidates = min(len(available), 500)
        elif remaining_time > 30:
            max_candidates = min(len(available), 300)
        else:
            max_candidates = min(len(available), 150)
        
        candidates = available[:max_candidates]
        
        for candidate in candidates:
            if candidate in points_set:
                continue
            if time.time() >= deadline:
                return points
            
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
                    was_empty = residue_counts[p][r] == 0
                    residue_counts[p][r] += 1
                    if was_empty:
                        nonempty_counts[p] += 1
                        if nonempty_counts[p] + 1 == p:
                            empty_classes = [rr for rr in range(p) if residue_counts[p][rr] == 0]
                            critical_mask[p] = set(empty_classes)
                        else:
                            critical_mask[p] = set()
                if m + 1 <= N + 1:
                    for p in all_primes:
                        if p == m + 1:
                            counts = [0] * p
                            for v in points:
                                counts[v % p] += 1
                            residue_counts[p] = counts
                            nonempty_counts[p] = sum(1 for c in counts if c > 0)
                            ne = nonempty_counts[p]
                            if ne + 1 == p:
                                empty_classes = [r for r in range(p) if residue_counts[p][r] == 0]
                                critical_mask[p] = set(empty_classes)
                            else:
                                critical_mask[p] = set()
                            relevant_primes.append(p)
                            break
                improved = True
                # Break after adding one element to recompute scores
                break
    
    return points


def _swap_improvement(points, all_primes, deadline):
    """
    Swap improvement: try to replace one element with two others to increase tuple size.
    Optimized with limited search and early termination.
    """
    points_set = set(points)
    points = list(points)
    m = len(points)
    
    if m == 0:
        return points
    
    relevant_primes = [p for p in all_primes if p <= m + 2]
    
    residue_counts = {}
    for p in relevant_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        residue_counts[p] = counts
    
    nonempty_counts = {}
    for p in relevant_primes:
        nonempty_counts[p] = sum(1 for c in residue_counts[p] if c > 0)
    
    critical_mask = {}
    for p in relevant_primes:
        ne = nonempty_counts[p]
        if ne + 1 == p:
            empty_classes = [r for r in range(p) if residue_counts[p][r] == 0]
            critical_mask[p] = set(empty_classes)
        else:
            critical_mask[p] = set()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        
        new_relevant = [p for p in all_primes if p <= m + 2]
        for p in new_relevant:
            if p not in residue_counts:
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                residue_counts[p] = counts
                nonempty_counts[p] = sum(1 for c in counts if c > 0)
                ne = nonempty_counts[p]
                if ne + 1 == p:
                    empty_classes = [r for r in range(p) if residue_counts[p][r] == 0]
                    critical_mask[p] = set(empty_classes)
                else:
                    critical_mask[p] = set()
        relevant_primes = new_relevant
        
        # Try swap: for each x in points, try to remove x and add two candidates
        # Limit to a small number of elements to keep runtime bounded
        remaining_time = deadline - time.time()
        if remaining_time > 120:
            max_swaps_to_try = min(len(points), 20)
        elif remaining_time > 60:
            max_swaps_to_try = min(len(points), 10)
        else:
            max_swaps_to_try = min(len(points), 5)
        
        for x_idx in range(max_swaps_to_try):
            if time.time() >= deadline:
                return points
            x = points[x_idx]
            
            temp_set = points_set - {x}
            
            # Update residue counts for temp state
            temp_residue_counts = {}
            for p in relevant_primes:
                counts = list(residue_counts[p])
                r = x % p
                counts[r] -= 1
                if counts[r] < 0:
                    counts[r] = 0
                temp_residue_counts[p] = counts
            
            temp_nonempty = {}
            for p in relevant_primes:
                temp_nonempty[p] = sum(1 for c in temp_residue_counts[p] if c > 0)
            
            temp_critical = {}
            for p in relevant_primes:
                ne = temp_nonempty[p]
                if ne + 1 == p:
                    empty_classes = [r for r in range(p) if temp_residue_counts[p][r] == 0]
                    temp_critical[p] = set(empty_classes)
                else:
                    temp_critical[p] = set()
            
            # Get candidate list sorted by conflict score
            # Limit candidates significantly for speed
            remaining_time = deadline - time.time()
            if remaining_time > 90:
                max_cand_a = 80
                max_cand_b = 50
            elif remaining_time > 45:
                max_cand_a = 50
                max_cand_b = 30
            else:
                max_cand_a = 30
                max_cand_b = 20
            
            candidates = [c for c in range(N + 1) if c not in temp_set]
            
            def conflict_score(c):
                score = 0
                for p in relevant_primes:
                    if p in temp_critical and c % p in temp_critical[p]:
                        score += 1
                return score
            
            candidates.sort(key=conflict_score)
            candidates = candidates[:max_cand_a]
            
            found_swap = False
            for a in candidates:
                if a in temp_set:
                    continue
                if time.time() >= deadline:
                    break
                
                admissible_a = True
                for p in relevant_primes:
                    counts = temp_residue_counts[p]
                    ne = temp_nonempty[p]
                    r = a % p
                    if counts[r] == 0:
                        if ne + 1 == p:
                            admissible_a = False
                            break
                
                if not admissible_a:
                    continue
                
                # Tentatively add a
                for p in relevant_primes:
                    r = a % p
                    was_empty = temp_residue_counts[p][r] == 0
                    temp_residue_counts[p][r] += 1
                    if was_empty:
                        temp_nonempty[p] += 1
                        if temp_nonempty[p] + 1 == p:
                            empty_classes = [rr for rr in range(p) if temp_residue_counts[p][rr] == 0]
                            temp_critical[p] = set(empty_classes)
                        else:
                            temp_critical[p] = set()
                temp_set.add(a)
                
                # Now try to add a second candidate b
                candidates_b = [c for c in candidates if c not in temp_set][:max_cand_b]
                for b in candidates_b:
                    if b in temp_set:
                        continue
                    if time.time() >= deadline:
                        break
                    
                    admissible_b = True
                    for p in relevant_primes:
                        counts = temp_residue_counts[p]
                        ne = temp_nonempty[p]
                        r = b % p
                        if counts[r] == 0:
                            if ne + 1 == p:
                                admissible_b = False
                                break
                    
                    if not admissible_b:
                        continue
                    
                    # Found a valid swap! Apply it
                    for p in relevant_primes:
                        r = b % p
                        was_empty = temp_residue_counts[p][r] == 0
                        temp_residue_counts[p][r] += 1
                        if was_empty:
                            temp_nonempty[p] += 1
                            if temp_nonempty[p] + 1 == p:
                                empty_classes = [rr for rr in range(p) if temp_residue_counts[p][rr] == 0]
                                temp_critical[p] = set(empty_classes)
                            else:
                                temp_critical[p] = set()
                    temp_set.add(b)
                    
                    # Update global state
                    points = [v for v in points if v != x] + [a, b]
                    points_set = temp_set
                    m += 1
                    residue_counts = temp_residue_counts
                    nonempty_counts = temp_nonempty
                    for p in relevant_primes:
                        ne = nonempty_counts[p]
                        if ne + 1 == p:
                            empty_classes = [r for r in range(p) if residue_counts[p][r] == 0]
                            critical_mask[p] = set(empty_classes)
                        else:
                            critical_mask[p] = set()
                    improved = True
                    found_swap = True
                    break
                
                # Undo adding a if we didn't find b
                if not found_swap:
                    for p in relevant_primes:
                        r = a % p
                        temp_residue_counts[p][r] -= 1
                        if temp_residue_counts[p][r] == 0:
                            temp_nonempty[p] -= 1
                            if temp_nonempty[p] + 1 == p:
                                empty_classes = [rr for rr in range(p) if temp_residue_counts[p][rr] == 0]
                                temp_critical[p] = set(empty_classes)
                            else:
                                temp_critical[p] = set()
                    temp_set.discard(a)
                
                if found_swap:
                    break
            
            if found_swap:
                break
    
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
    
    # Strategy 2: Start from numbers coprime to 6 (more selective, often better)
    if time.time() < deadline - 40:
        points_coprime6 = [x for x in range(N + 1) if x % 2 != 0 and x % 3 != 0]
        points_coprime6 = _greedy_repair(points_coprime6, all_primes, deadline)
        if len(points_coprime6) > len(best_points):
            best_points = points_coprime6
    
    # Strategy 3: Start from odd numbers
    if time.time() < deadline - 30:
        points_odd = list(range(1, N + 1, 2))
        points_odd = _greedy_repair(points_odd, all_primes, deadline)
        if len(points_odd) > len(best_points):
            best_points = points_odd

    # Local improvement: try to add back elements using fast method
    if time.time() < deadline and len(best_points) > 0:
        best_points = _local_improvement_fast(best_points, all_primes, deadline)

    # Swap improvement: try 1-out-2-in exchanges
    if time.time() < deadline - 10 and len(best_points) > 0:
        best_points = _swap_improvement(best_points, all_primes, deadline)

    # Final local improvement pass
    if time.time() < deadline - 5 and len(best_points) > 0:
        best_points = _local_improvement_fast(best_points, all_primes, deadline)

    # Second local improvement if time remains
    if time.time() < deadline - 5 and len(best_points) > 0:
        best_points = _local_improvement_fast(best_points, all_primes, deadline)

    # Final verification and repair if needed
    if not _check_admissible(best_points, all_primes):
        best_points = _greedy_repair(best_points, all_primes, deadline)

    return best_points
# EVOLVE-BLOCK-END
