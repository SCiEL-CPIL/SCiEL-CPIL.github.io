"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time

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
    # Start from 1 and the primes above sqrt(N); repair any prime whose residue classes are all covered
    # by deleting the least-populated class, until the tuple is admissible.
    points = [1] + [p for p in _ALL_PRIMES if p * p > N]
    
    # Repair Phase
    changed = True
    while changed:
        if time.time() - start_time > 250:
            break
        changed = False
        current_len = len(points)
        for p in _ALL_PRIMES:
            if p > current_len:
                break
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                # Find least populated class
                min_size = min(len(v) for v in classes.values())
                # Remove all values in the class(es) with min size
                # To be deterministic, pick the first one encountered or smallest residue?
                # Current code: min(classes.values(), key=len) picks one.
                # Let's stick to that.
                target_class = None
                for res, vals in classes.items():
                    if len(vals) == min_size:
                        target_class = vals
                        break
                points = [v for v in points if v not in target_class]
                changed = True
                break
    
    # Addition Phase
    # Try to add back elements from [0, N] \ points
    # Precompute missing residues for current points
    # M_p = set of residues r such that count(r) == 0
    
    # We will iterate candidates.
    # To maximize additions, maybe sort candidates?
    # Candidates that are "safe" for more primes are better.
    
    point_set = set(points)
    candidates = [i for i in range(N + 1) if i not in point_set]
    
    # We need to check admissibility for p <= len(points) + 1
    # As we add points, len increases.
    
    # For efficiency, we can maintain M_p.
    # Initially, compute M_p for all p <= N.
    # M_p = {r in 0..p-1 | no v in points has v%p == r}
    
    # This is O(N * pi(N)) which is ~ 5000 * 669 = 3.3M ops. Fast.
    
    M = {}
    for p in _ALL_PRIMES:
        if p > len(points) + 10: # We might add a few, so check a bit more
            break
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        M[p] = set(r for r in range(p) if counts[r] == 0)
    
    # Now try to add candidates
    # Sort candidates by "safety"? 
    # A candidate x is safe if for all p <= len(points)+1, M[p] != {x%p}
    # We can just iterate.
    
    # To be fast, we can just iterate through candidates in random order or ascending.
    # Ascending is fine.
    
    for x in candidates:
        if time.time() - start_time > 280:
            break
            
        # Check if adding x is safe
        new_len = len(points) + 1
        safe = True
        for p in _ALL_PRIMES:
            if p > new_len:
                break
            if p not in M:
                # Need to compute M for this p? 
                # If p > initial len + 10, we didn't compute it.
                # But p <= new_len.
                # If p was not in M, it means p > initial len + 10.
                # But new_len = initial len + added.
                # If we added many, p might be relevant now.
                # Let's just compute on the fly if not in M.
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                M[p] = set(r for r in range(p) if counts[r] == 0)
                
            if M[p] == {x % p}:
                safe = False
                break
        
        if safe:
            points.append(x)
            point_set.add(x)
            # Update M for all p <= new_len
            # Adding x increases count of x%p by 1.
            # If count was 0, it becomes 1, so r is removed from M[p].
            # If count was > 0, M[p] unchanged.
            for p in _ALL_PRIMES:
                if p > new_len:
                    break
                r = x % p
                if r in M[p]:
                    M[p].remove(r)
                    
    return points
# EVOLVE-BLOCK-END
