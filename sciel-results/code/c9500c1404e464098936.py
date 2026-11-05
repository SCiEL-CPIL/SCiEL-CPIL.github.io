import time
import random

N = 5000
DEADLINE = 325  # seconds, safety margin before 330s kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def _greedy_construct():
    points = list(range(N + 1))
    primes = _primes(N + 1)
    
    # We iterate primes in increasing order.
    # If p > len(points), break.
    for p in primes:
        if p > len(points):
            break
        
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        
        if min(counts) > 0:
            min_idx = counts.index(min(counts))
            points = [v for v in points if v % p != min_idx]
            
    return points

def construct():
    start_time = time.time()
    
    # 1. Greedy Initial Solution
    points = _greedy_construct()
    in_set = set(points)
    
    # 2. Preparation for Local Search
    # We need to track missed residues for each prime p <= len(points) + 1
    # Let's precompute primes up to N+1
    primes = _primes(N + 1)
    
    # Initialize counts and missed sets
    # We only care about primes p where p <= len(points) + 1
    # But as we add points, len(points) increases.
    # Max len is N+1.
    
    # To be efficient, we can just re-calculate or update.
    # Let's maintain `counts[p]` as a list of length p.
    # And `missed_count[p]` as the number of residues with count 0.
    # Actually, we need to know *which* residues are missed to check candidates.
    # So `missed[p]` as a set of residues.
    
    # Initialize for current points
    counts = {}
    missed = {}
    
    # We will check primes up to N+1 potentially.
    # But initially, only p <= len(points) are constraints.
    # However, for the insertion check, we need to know if adding a point violates
    # the condition for p = len(points)+1 (if it's prime).
    # So we need stats for p up to N+1.
    
    # Let's compute stats for all primes in `primes` list.
    # This is O(N * pi(N)) ~ 3M ops. Fast.
    
    for p in primes:
        counts[p] = [0] * p
        for v in points:
            counts[p][v % p] += 1
        missed[p] = set()
        for r in range(p):
            if counts[p][r] == 0:
                missed[p].add(r)
                
    # 3. Local Search
    candidates = [i for i in range(N + 1) if i not in in_set]
    random.shuffle(candidates)
    
    current_len = len(points)
    
    # We will iterate through candidates multiple times until no improvement or time limit.
    # But a single pass over a shuffled list is often good.
    # Let's do a loop that keeps trying to insert until a full pass yields no inserts,
    # or time runs out.
    
    improved = True
    while improved and (time.time() - start_time < DEADLINE - 5):
        improved = False
        # Re-shuffle candidates? Or just iterate.
        # If we iterate in fixed order, we might get stuck.
        # Shuffling each pass is better.
        random.shuffle(candidates)
        
        for cand in candidates:
            if cand in in_set:
                continue
            
            # Check if cand can be added
            # New length will be current_len + 1
            new_len = current_len + 1
            
            # We need to check all primes p <= new_len
            can_add = True
            
            # Optimization:
            # Only primes where missed_count is 1 are critical.
            # If missed_count >= 2, adding 1 point cannot make it 0.
            # If missed_count == 0, it's already invalid (shouldn't happen if points is admissible, 
            # except for p > current_len which might not have been constrained yet).
            # Wait.
            # If p <= current_len, points is admissible, so missed_count[p] >= 1.
            # If p > current_len, points was not required to be admissible mod p.
            # So missed_count[p] could be 0.
            # If missed_count[p] == 0, adding cand won't help (it will still be 0).
            # So if p is a new constraint (p == new_len and prime) and missed_count[p] == 0, 
            # we cannot add cand?
            # No, if missed_count[p] == 0, it means ALL residues are covered by `points`.
            # Adding `cand` (which is in one of those covered residues) will keep all residues covered.
            # So `new_points` will cover all residues mod p.
            # Thus `new_points` is NOT admissible.
            # So we must NOT add cand if there exists a prime p <= new_len such that:
            # 1. p <= current_len: missed_count[p] == 1 AND cand % p == the unique missed residue.
            # 2. p == new_len (and prime): missed_count[p] == 0 OR (missed_count[p] == 1 AND cand % p == unique missed residue).
            
            # Let's iterate primes.
            # To speed up, we can pre-filter primes.
            # But pi(N) is small. Just loop.
            
            for p in primes:
                if p > new_len:
                    break
                
                m_set = missed[p]
                m_size = len(m_set)
                
                if m_size == 0:
                    # All residues covered. Adding cand keeps them covered.
                    # This is only a problem if p is a constraint.
                    # p <= new_len is a constraint.
                    # So if m_size == 0, we cannot add cand?
                    # Wait. If p <= current_len, m_size cannot be 0 because points is admissible.
                    # So m_size == 0 only happens if p > current_len.
                    # This implies p == new_len (since p is prime and we break at p > new_len).
                    # So if new_len is prime and missed[new_len] is empty, we cannot add cand.
                    can_add = False
                    break
                    
                elif m_size == 1:
                    # Only one missed residue.
                    # If cand is in that residue, it fills it, making m_size 0.
                    # Then new_points covers all residues mod p.
                    # So new_points is not admissible.
                    if cand % p in m_set:
                        can_add = False
                        break
                # else m_size >= 2: safe.
                
            if can_add:
                # Add cand
                points.append(cand)
                in_set.add(cand)
                current_len = new_len
                improved = True
                
                # Update counts and missed
                for p in primes:
                    if p > current_len:
                        break # Only need to update up to current_len? 
                        # Actually, we need missed[p] for p up to current_len+1 for next check.
                        # So update all p in primes list that are <= current_len+1?
                        # Just update all, it's fast enough.
                    
                    r = cand % p
                    counts[p][r] += 1
                    if r in missed[p]:
                        missed[p].discard(r)
                        
    return points

# EVOLVE-BLOCK-END
