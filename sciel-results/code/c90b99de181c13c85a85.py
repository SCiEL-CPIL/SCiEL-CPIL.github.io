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
    """
    Fast heuristic to get a decent starting point.
    Start with all numbers [0..N], and for each prime p <= N,
    if all p classes are covered, remove the least populated class.
    This is O(N * pi(N)) which is fast for N=5000.
    """
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
    current_len = len(points)
    
    # 2. Preparation for Local Search
    # Precompute primes up to N+1
    primes = _primes(N + 1)
    
    # Initialize counts and missed sets
    # We need stats for primes p <= current_len + 1 potentially, 
    # but as we add points, len increases.
    # Let's compute stats for all primes in `primes` list initially.
    # This is O(N * pi(N)) ~ 3M ops. Fast.
    
    counts = {}
    missed = {}
    
    for p in primes:
        counts[p] = [0] * p
        for v in points:
            counts[p][v % p] += 1
        missed[p] = set()
        for r in range(p):
            if counts[p][r] == 0:
                missed[p].add(r)
                
    # 3. Local Search
    # We try to insert candidates from [0, N] that are not in the set.
    # To maximize the chance of finding a valid insertion, we prioritize candidates
    # that do NOT fill the "unique missed residue" for any critical prime.
    
    candidates = [i for i in range(N + 1) if i not in in_set]
    
    # Optimization: Sort candidates by "safety" initially?
    # Or just shuffle and iterate.
    # A smarter approach: For each candidate, check if it's "safe".
    # A candidate is safe if for all primes p <= current_len + 1:
    #   - If missed[p] has size 1, cand % p is NOT in missed[p].
    #   - If missed[p] has size 0, then p must be > current_len + 1 (otherwise we couldn't add anything).
    #     Wait, if missed[p] has size 0, it means all residues are covered.
    #     If p <= current_len, the set is already invalid (shouldn't happen).
    #     If p == current_len + 1 (and prime), adding any cand keeps it invalid.
    #     So if any prime p <= current_len + 1 has missed[p] size 0, we are stuck.
    
    # Let's check if we are stuck immediately.
    stuck = False
    for p in primes:
        if p > current_len + 1:
            break
        if len(missed[p]) == 0:
            stuck = True
            break
            
    if not stuck:
        # We can try to insert.
        # We will iterate through passes.
        # In each pass, we try to insert candidates.
        # To be efficient, we can pre-filter candidates that are "safe" for the current state.
        # But the state changes as we insert.
        
        # Let's use a priority queue or just sort by some heuristic?
        # Simpler: Shuffle and iterate.
        
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 5):
            improved = False
            random.shuffle(candidates)
            
            # Optimization: We can check candidates in an order that is likely to succeed.
            # Candidates that are "rare" in the missed sets are better?
            # Actually, any candidate that doesn't hit a unique missed residue is safe.
            
            for cand in candidates:
                if cand in in_set:
                    continue
                
                new_len = current_len + 1
                can_add = True
                
                # Check primes p <= new_len
                for p in primes:
                    if p > new_len:
                        break
                    
                    m_set = missed[p]
                    m_size = len(m_set)
                    
                    if m_size == 0:
                        # This prime is already "full". Adding cand won't help.
                        # Since p <= new_len, this is a violation.
                        can_add = False
                        break
                    elif m_size == 1:
                        # Only one missed residue.
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
                            # We might need stats for p = current_len + 1 in the next iteration
                            # But we break here. Let's update all primes <= current_len + 1?
                            # Actually, just updating all is safer and fast enough.
                            pass
                        
                        r = cand % p
                        counts[p][r] += 1
                        if r in missed[p]:
                            missed[p].discard(r)
                        
    return points

# EVOLVE-BLOCK-END
