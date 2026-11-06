import itertools
import time

def construct():
    """
    Returns a list of 101 lists. Each list contains distinct denominators in 1..10
    such that the sum of their modular inverses modulo 101 equals the index.
    Minimizes the largest row length.
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inverses = {d: pow(d, -1, MOD) for d in denominators}
    
    # For each residue, track the shortest subset that sums to it
    # best[r] = (length, list_of_denoms)
    best = [None] * MOD
    
    # Precompute all subsets by length, in order from shortest to longest
    for length in range(1, MAX_DENOM + 1):
        for combo in itertools.combinations(denominators, length):
            s = sum(inverses[d] for d in combo) % MOD
            if best[s] is None:
                best[s] = (length, list(combo))
        
        # Check if we've covered all residues
        if all(b is not None for b in best):
            break
        
        # Deadline check
        if time.time() - start_time > DEADLINE:
            break
    
    # Build results
    results = []
    for r in range(MOD):
        if best[r] is not None:
            results.append(best[r][1])
        else:
            # Fallback: find any valid subset (shouldn't happen)
            found = False
            for length in range(1, MAX_DENOM + 1):
                for combo in itertools.combinations(denominators, length):
                    s = sum(inverses[d] for d in combo) % MOD
                    if s == r:
                        results.append(list(combo))
                        found = True
                        break
                if found:
                    break
            if not found:
                results.append([1])
    
    return results
