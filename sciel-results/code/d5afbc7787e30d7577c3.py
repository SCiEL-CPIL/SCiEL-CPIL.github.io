"""Narrow admissible tuple: fixed-diameter maximization in [0, N]."""
import time
import math
import random
import numpy as np

N = 5000
DEADLINE = 315.0  # safety margin before 330s kill


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


def _is_admissible(points, all_primes):
    """Check if a set of points forms an admissible tuple."""
    m = len(points)
    if m == 0:
        return True
    point_set = set(points)
    for p in all_primes:
        if p > m:
            break
        # Check if all residues mod p are covered
        residues = set()
        for v in point_set:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True


def construct():
    start_time = time.time()
    
    # Get primes up to N+1 (max possible tuple size is N+1)
    all_primes = _primes(N + 1)
    
    # Strategy: Start with full set, greedily remove residue classes,
    # then try to add back elements.
    
    # Phase 1: Greedy removal from full set
    points = list(range(N + 1))
    
    while True:
        if time.time() - start_time > DEADLINE:
            break
        
        m = len(points)
        if m == 0:
            break
        
        # Get primes up to m
        primes_to_check = [p for p in all_primes if p <= m]
        
        changed = False
        for p in primes_to_check:
            # Count residues using numpy for speed
            arr = np.array(points, dtype=np.int64)
            counts = np.bincount(arr % p, minlength=p)
            
            if counts.min() > 0:
                # All residues covered. Remove the class with minimum count.
                min_count = counts.min()
                min_residues = np.where(counts == min_count)[0]
                
                # Pick the residue that, when removed, causes least future damage
                # Heuristic: pick the one with most "overlap" with other critical classes
                # For simplicity, pick randomly among min-count residues
                target_residue = int(random.choice(min_residues))
                
                # Remove all points v where v % p == target_residue
                mask = (arr % p) != target_residue
                points = arr[mask].tolist()
                changed = True
                break
        
        if not changed:
            break
    
    best_points = points[:]
    best_len = len(points)
    
    # Phase 2: Try to add back removed elements
    point_set = set(points)
    candidates = [v for v in range(N + 1) if v not in point_set]
    
    # Multiple random restarts for the addition phase
    num_trials = 30
    for trial in range(num_trials):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        random.shuffle(candidates)
        
        current = best_points[:]
        current_set = set(current)
        current_arr = np.array(current, dtype=np.int64)
        
        # Pre-compute residue coverage for current set
        # For each prime p <= len(current), track which residues are present
        residue_coverage = {}
        for p in all_primes:
            if p > len(current):
                break
            residues = set(current_arr % p)
            residue_coverage[p] = residues
        
        for v in candidates:
            if v in current_set:
                continue
            
            m_new = len(current) + 1
            
            # Check admissibility of current + [v]
            admissible = True
            
            for p in all_primes:
                if p > m_new:
                    break
                
                v_res = v % p
                if p in residue_coverage:
                    cur_res = residue_coverage[p]
                    if v_res in cur_res:
                        # Adding v doesn't change residue coverage
                        continue
                    else:
                        # v adds a new residue
                        if len(cur_res) == p - 1:
                            # This would complete all residues -> violation
                            admissible = False
                            break
                else:
                    # p > len(current), so p >= m_new (since p is prime and m_new = len(current)+1)
                    # Actually if p > len(current) and p <= m_new, then p == m_new
                    # In this case, we need all residues 0..p-1 to NOT be covered
                    # With just m_new = p elements, we can cover at most p residues
                    # We need to check if all p residues are covered
                    # This is unlikely but let's check
                    residues = set((current_arr % p))
                    residues.add(v_res)
                    if len(residues) == p:
                        admissible = False
                        break
            
            if admissible:
                current.append(v)
                current_set.add(v)
                current_arr = np.append(current_arr, v)
                
                # Update residue coverage
                for p in all_primes:
                    if p > m_new:
                        break
                    if p not in residue_coverage:
                        residue_coverage[p] = set(current_arr % p)
                    else:
                        residue_coverage[p].add(v % p)
        
        if len(current) > best_len:
            best_points = current
            best_len = len(current)
    
    # Phase 3: Local search - try swapping elements
    # Try to replace some elements with others to increase size
    for iteration in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        improved = False
        point_set = set(best_points)
        
        # Try removing one element and adding two others
        # This is expensive, so only try a few random pairs
        
        for _ in range(100):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            if len(best_points) < 2:
                break
            
            # Pick a random element to remove
            idx = random.randint(0, len(best_points) - 1)
            removed = best_points[idx]
            
            # Pick two random candidates to add
            candidates_not_in = [v for v in range(N + 1) if v not in point_set]
            if len(candidates_not_in) < 2:
                break
            
            i1 = random.randint(0, len(candidates_not_in) - 1)
            i2 = random.randint(0, len(candidates_not_in) - 1)
            if i1 == i2:
                continue
            add1 = candidates_not_in[i1]
            add2 = candidates_not_in[i2]
            
            # Try the swap
            new_points = [v for i, v in enumerate(best_points) if i != idx]
            new_points.append(add1)
            new_points.append(add2)
            
            if _is_admissible(new_points, all_primes):
                best_points = new_points
                best_len = len(best_points)
                improved = True
                break
        
        if not improved:
            break
    
    return best_points
