import numpy as np

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We'll work on interval [1, MAX_L]
    MAX_L = 2000
    x = np.arange(1, MAX_L + 1)
    
    # coverage[x-1] = number of primes covering x
    coverage = np.zeros(MAX_L, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Process primes in descending order (large primes first for targeted coverage)
    # For each prime, choose residue to maximize coverage of deficit integers
    for i in range(n_primes - 1, -1, -1):
        p = primes[i]
        best_r = 0
        best_score = -1
        for r in range(p):
            # Which integers in [1, MAX_L] are ≡ r (mod p)?
            mask = (x % p) == r
            # Count how many of these are currently at coverage 1 (would become 2)
            # vs coverage 0 (would become 1)
            # We want to maximize: (# at coverage 1) + 0.1 * (# at coverage 0)
            # Actually, prioritize fixing coverage-1 integers
            cnt1 = np.sum(coverage[mask] == 1)
            cnt0 = np.sum(coverage[mask] == 0)
            score = cnt1 * 10 + cnt0
            if score > best_score:
                best_score = score
                best_r = r
        residues[i] = best_r
        # Update coverage
        mask = (x % primes[i]) == best_r
        coverage[mask] += 1
    
    # Now find the current L
    def compute_L(coverage, MAX_L):
        # Find largest L such that all x in [1,L] have coverage >= 2
        for L in range(MAX_L, 0, -1):
            if np.all(coverage[:L] >= 2):
                return L
        return 0
    
    L = compute_L(coverage, MAX_L)
    
    # Local search: try to improve by adjusting residues of primes that affect the bottleneck
    # The bottleneck is the first integer with coverage < 2
    for iteration in range(50):
        # Find first integer with coverage < 2
        bottleneck = -1
        for L_check in range(L, MAX_L + 1):
            if L_check < MAX_L and coverage[L_check] < 2:
                bottleneck = L_check
                break
        if bottleneck == -1:
            # Check if we can extend
            if L < MAX_L and coverage[L] >= 2:
                L += 1
                continue
            break
        
        # Try to fix the bottleneck by changing residue of some prime
        # Primes that could cover bottleneck: p where bottleneck % p = r_p (current)
        # We need to change r_p so that (bottleneck % p) == new_r_p
        # But this might break other coverages
        improved = False
        for i in range(n_primes):
            p = primes[i]
            target_r = bottleneck % p
            if target_r == residues[i]:
                continue
            # Try setting residues[i] = target_r
            old_r = residues[i]
            residues[i] = target_r
            # Recompute coverage change
            old_mask = (x % p) == old_r
            new_mask = (x % p) == target_r
            coverage[old_mask] -= 1
            coverage[new_mask] += 1
            # Check if L improved
            new_L = compute_L(coverage, MAX_L)
            if new_L > L:
                L = new_L
                improved = True
                break
            else:
                # Revert
                coverage[old_mask] += 1
                coverage[new_mask] -= 1
                residues[i] = old_r
        if not improved:
            break
    
    return [int(r) for r in residues]
