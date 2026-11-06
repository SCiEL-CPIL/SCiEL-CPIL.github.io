import itertools
import time

def construct():
    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inverses = {d: pow(d, -1, MOD) for d in denominators}
    
    # For each residue, find the minimum-length subset that achieves it
    # Iterate by length to find minimum first
    best = [None] * MOD  # best[r] = list of denominators
    
    for length in range(1, MAX_DENOM + 1):
        for combo in itertools.combinations(denominators, length):
            s = sum(inverses[d] for d in combo) % MOD
            if best[s] is None:
                best[s] = list(combo)
        
        # Early termination: if all residues are covered, we're done
        if all(b is not None for b in best):
            break
    
    # All 101 residues should be covered by now
    # (For p=101, denom=10, this is guaranteed)
    results = []
    for r in range(MOD):
        if best[r] is not None:
            results.append(best[r])
        else:
            # Fallback: should not happen, but just in case
            results.append([1])
    
    return results
