# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = 60.0  # safety margin
    
    # We want distinct moduli in [2, 128]
    # Two classes (r1,m1), (r2,m2) are disjoint iff r1 != r2 (mod gcd(m1,m2))
    
    # Strategy: prioritize moduli with many divisors (more "flexible" for disjointness)
    # Actually, moduli with large gcd with others are harder to be disjoint from.
    # Moduli that are prime and large have gcd=1 with most others, making disjointness easy.
    # So we want to include as many moduli as possible.
    
    # Key insight: if gcd(m1,m2)=1, then ANY residues are disjoint (since mod 1 everything is 0).
    # So coprime moduli are always compatible regardless of residues.
    
    # The hard cases are when gcd(m1,m2) > 1.
    
    # Let's try a greedy approach:
    # 1. Generate all moduli 2..128
    # 2. Sort by number of divisors descending (more composite = harder constraints)
    #    Actually, we want to place the "hard" ones first where they have fewer choices,
    #    or place "easy" ones (coprime to many) first.
    
    # Better: use backtracking/greedy with careful ordering.
    
    # Let me think differently. The maximum number is bounded by 127 (moduli 2-128).
    # But we need pairwise disjointness.
    
    # Observation: If we pick moduli that are all pairwise coprime, then ANY residues work.
    # Max pairwise coprime set from 2..128: primes + 4 + 8 + 9 + 16 + 25 + 27 + ... 
    # Actually just distinct prime powers: 2,3,4,5,7,8,9,11,13,16,17,19,23,25,27,29,31,32,37,41,43,47,49,53,59,61,64,67,71,73,79,81,83,89,97,101,103,107,109,113,121,125,127
    # That's a lot. But we can do better by including non-coprime moduli with careful residues.
    
    # Greedy: try to include ALL moduli 2-128 with appropriate residues.
    # Process moduli in order, assign residue to be disjoint from all previously assigned.
    
    moduli = list(range(2, 129))
    
    # Sort by number of common divisors with already-chosen, but we don't know yet.
    # Let's just try in natural order and see if we can assign residues.
    
    assigned = []  # list of (residue, modulus)
    
    for m in moduli:
        if time.time() - start_time > deadline:
            break
        
        found = False
        for r in range(m):
            # Check if (r, m) is disjoint from all assigned
            ok = True
            for (r2, m2) in assigned:
                g = math.gcd(m, m2)
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                assigned.append((r, m))
                found = True
                break
        
        if not found:
            # Skip this modulus
            pass
    
    return [[r, m] for (r, m) in assigned]
# EVOLVE-BLOCK-END
