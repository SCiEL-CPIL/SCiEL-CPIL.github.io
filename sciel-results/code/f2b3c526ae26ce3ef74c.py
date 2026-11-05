import time
import random
from collections import defaultdict

N = 5000
DEADLINE = 325  # safety margin before 330s kill

def _primes(limit):
    """Return list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def _is_admissible(points, primes):
    """Check admissibility: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    if n == 0:
        return True
    for p in primes:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True

def _construct_greedy(primes, start_time, deadline):
    """
    Greedy removal with overlap-aware class choice.
    """
    present = bytearray(b'\x01') * (N + 1)
    choices = {}
    
    for p in primes:
        if p > N:
            break
        if time.time() - start_time > deadline:
            break

        # count surviving points per residue class mod p
        counts = [0] * p
        for v in range(N + 1):
            if present[v]:
                counts[v % p] += 1

        # if some class already empty, constraint satisfied
        if any(c == 0 for c in counts):
            # We don't strictly need to record a choice if it's already empty,
            # but for the local search, we need a consistent state.
            # If it's empty, any choice is effectively "free" in terms of removing new points?
            # No, if we assign a choice, it might remove points later if the set changes?
            # In this greedy phase, the set only shrinks. So if it's empty now, it stays empty.
            # So we can pick any residue, say 0.
            choices[p] = 0
            continue

        # choose class with minimum surviving count
        min_count = min(counts)
        # tie-break: prefer smaller class index for determinism
        target = counts.index(min_count)
        choices[p] = target
        
        for v in range(target, N + 1, p):
            present[v] = 0

    return choices

def _local_search(primes, initial_choices, start_time, deadline):
    """
    Coordinate descent local search on the residue choices.
    """
    choices = dict(initial_choices)
    
    # Precompute lists of points for each prime and residue
    # classes[p][r] = list of x in [0, N] such that x % p == r
    classes = {}
    for p in primes:
        if p > N:
            break
        classes[p] = [[] for _ in range(p)]
        for x in range(N + 1):
            classes[p][x % p].append(x)
            
    # Only consider primes <= N
    active_primes = [p for p in primes if p <= N]
    
    # Initialize removed_count
    removed_count = [0] * (N + 1)
    for p in active_primes:
        r = choices[p]
        for x in classes[p][r]:
            removed_count[x] += 1
            
    # Function to get score (number of survivors)
    def get_score():
        return sum(1 for c in removed_count if c == 0)
        
    current_score = get_score()
    best_score = current_score
    best_choices = dict(choices)
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Shuffle order of primes to avoid bias?
        # Or just iterate. Shuffling might help escape local minima.
        search_primes = active_primes[:]
        random.shuffle(search_primes)
        
        for p in search_primes:
            if time.time() - start_time > deadline:
                break
                
            r_old = choices[p]
            
            # Calculate K_p[x] = removed_count[x] - (1 if x in C_{p, r_old} else 0)
            # We don't need to store K_p explicitly, we can compute the "free" set on the fly
            # Free set S_free = { x | K_p[x] == 0 }
            # K_p[x] == 0  <=>  removed_count[x] == 0 AND x not in C_{p, r_old}
            # OR removed_count[x] == 1 AND x in C_{p, r_old} AND ... wait.
            # K_p[x] is the count if we REMOVE p's contribution.
            # So K_p[x] = removed_count[x] - (1 if x in C_{p, r_old} else 0).
            # We want x such that K_p[x] == 0.
            
            # Let's identify S_free efficiently.
            # S_free contains x where:
            # 1. removed_count[x] == 0 and x not in C_{p, r_old}
            # 2. removed_count[x] == 1 and x in C_{p, r_old}
            
            # Actually, it's easier to just iterate through candidates r' and count intersections.
            # For each r', cost = |C_{p, r'} \cap S_free|.
            # S_free is the set of points that would survive if p removed NOTHING.
            
            # Let's build S_free as a boolean array or set?
            # N=5000, so a set is fine.
            s_free = set()
            for x in range(N + 1):
                # Check if x is free without p
                # x is free if removed_count[x] - (1 if x in C_{p, r_old} else 0) == 0
                # x in C_{p, r_old} iff x % p == r_old
                if x % p == r_old:
                    if removed_count[x] == 1:
                        s_free.add(x)
                else:
                    if removed_count[x] == 0:
                        s_free.add(x)
                        
            # Now find r' that minimizes |C_{p, r'} \cap S_free|
            min_cost = float('inf')
            best_r = r_old
            
            for r in range(p):
                cost = 0
                for x in classes[p][r]:
                    if x in s_free:
                        cost += 1
                if cost < min_cost:
                    min_cost = cost
                    best_r = r
                    
            if best_r != r_old:
                # Update removed_count
                for x in classes[p][r_old]:
                    removed_count[x] -= 1
                for x in classes[p][best_r]:
                    removed_count[x] += 1
                
                choices[p] = best_r
                improved = True
                
                # Update score
                # We can recalculate score, or update incrementally
                # Incremental:
                # Points in C_{p, r_old} that became 0: +survivors
                # Points in C_{p, best_r} that became >0: -survivors
                # This is tricky. Just recalculate.
                new_score = get_score()
                if new_score > best_score:
                    best_score = new_score
                    best_choices = dict(choices)
                    
    return best_choices, best_score

def construct():
    start = time.time()
    deadline = start + DEADLINE
    primes = _primes(N)
    
    # Initial greedy
    choices = _construct_greedy(primes, start, deadline)
    
    # Local search
    best_choices, best_score = _local_search(primes, choices, start, deadline - (time.time() - start))
    
    # Reconstruct the tuple from best_choices
    # A point x is in the tuple if for all p, x % p != best_choices[p]
    # Note: We only assigned choices for p <= N.
    # Primes > N don't matter for points in [0, N] because p > N implies x % p = x, 
    # and we only need to check p <= len(tuple). 
    # If len(tuple) < p, we don't need to satisfy p.
    # However, our construction ensures admissibility for all p <= N by removing a class.
    # So the resulting set is admissible for all p <= N.
    # Since len(tuple) <= N, it is admissible for all p <= len(tuple).
    
    result = []
    for x in range(N + 1):
        is_survivor = True
        for p in primes:
            if p > N:
                break
            if x % p == best_choices[p]:
                is_survivor = False
                break
        if is_survivor:
            result.append(x)
            
    return sorted(result)

# EVOLVE-BLOCK-START
# Entry point is construct(); logic above.
# EVOLVE-BLOCK-END
