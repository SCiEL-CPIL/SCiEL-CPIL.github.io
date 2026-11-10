"""Rewritten program: Fast randomized greedy for maximum disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    if n >= 0: sieve[0] = False
    if n >= 1: sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    MAX_MOD = 128
    
    moduli_list = list(range(2, MAX_MOD + 1))
    
    best_count = 0
    best_result = []
    
    # Run multiple randomized greedy passes
    for _ in range(10000):
        if time.time() > deadline:
            break
        
        # Shuffle moduli order
        current_moduli = moduli_list[:]
        random.shuffle(current_moduli)
        
        current_selected = [] # list of (r, m)
        
        for m in current_moduli:
            # Try to find a residue r for modulus m
            # Shuffle residues for randomness
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                # Check against all currently selected
                valid = True
                for r_s, m_s in current_selected:
                    if not _is_disjoint(r, m, r_s, m_s):
                        valid = False
                        break
                
                if valid:
                    current_selected.append((r, m))
                    found = True
                    break
            
            # If not found, we just skip this modulus in this pass
        
        if len(current_selected) > best_count:
            best_count = len(current_selected)
            best_result = [[r, m] for r, m in current_selected]
            
            # Early stop if we reach a very high number? 
            # The max possible is 127 (one per modulus). 
            # If we reach 127, we can stop.
            if best_count == 127:
                break
            
    # If no result was found or count is low, provide a robust fallback
    # Fallback: Use all primes up to 128 with residue 1.
    # Since primes are pairwise coprime, gcd(p_i, p_j) = 1, so (1-1)%1 == 0? 
    # Wait. _is_disjoint checks (r1 - r2) % g != 0.
    # If g=1, (r1-r2)%1 is always 0. So they are NOT disjoint?
    # Let's re-read the criterion.
    # "Every two infinite residue classes must be disjoint"
    # Classes: A = {x | x = r1 + k*m1}, B = {y | y = r2 + l*m2}.
    # They intersect if there exists x such that x = r1 mod m1 AND x = r2 mod m2.
    # By CRT, a solution exists iff r1 = r2 mod gcd(m1, m2).
    # So they are DISJOINT iff r1 != r2 mod gcd(m1, m2).
    # My _is_disjoint function: return (r1 - r2) % g != 0.
    # This returns True if they are disjoint.
    # If g=1, (r1-r2)%1 == 0. So _is_disjoint returns False.
    # This means if gcd=1, the classes ALWAYS intersect?
    # Let's check: m1=2, m2=3. r1=0, r2=0.
    # x=0 is in 0 mod 2. x=0 is in 0 mod 3. Intersection {0, 6, 12...}. Not disjoint.
    # r1=0, r2=1.
    # x=0 (mod 2), x=1 (mod 3).
    # x=4: 4=0 mod 2, 4=1 mod 3. Intersection {4, 10...}. Not disjoint.
    # Wait. If gcd(m1, m2)=1, the CRT guarantees a unique solution mod m1*m2.
    # So the classes ALWAYS intersect.
    # Therefore, we CANNOT have two classes with coprime moduli!
    # This is a crucial realization.
    # The condition for disjointness is (r1 - r2) % g != 0.
    # If g=1, this is always 0, so they are never disjoint.
    # So we must avoid pairs of moduli with gcd 1.
    # This means all chosen moduli must have gcd > 1 with each other?
    # No, just that for any pair, the residues must satisfy the condition.
    # If gcd=1, it's impossible.
    # So the set of moduli must be such that no two are coprime?
    # Or if they are coprime, we simply can't pick both?
    # Yes. So the moduli themselves form a graph where edges exist if gcd>1?
    # No, if gcd=1, we *cannot* pick both, regardless of residues.
    # So we need a subset of moduli where we can assign residues.
    # If we pick moduli that are all even, gcd is at least 2.
    # Then we need r1 != r2 mod 2.
    # So we can have at most 2 classes if all moduli are even?
    # No, if we have m1=4, m2=6. gcd=2.
    # r1=0, r2=1. (0-1)%2 = 1 != 0. Disjoint.
    # m3=10. gcd(4,10)=2. r3 must be != 0 mod 2. So r3=1.
    # gcd(6,10)=2. r3 must be != 1 mod 2. So r3=0.
    # Contradiction. r3 cannot be both 1 and 0 mod 2.
    # So we can't just pick any even moduli.
    
    # This makes the problem much harder. The "coprime" fallback is INVALID.
    # I need a valid fallback.
    # Let's try powers of 2.
    # m = 2, 4, 8, 16, 32, 64, 128.
    # gcd(2^i, 2^j) = 2^min(i,j).
    # Let's try to construct a valid set for powers of 2.
    # m=2, r=0.
    # m=4, r must be != 0 mod 2. So r=1 or 3. Pick 1.
    # m=8, r must be != 0 mod 2 (vs 2) AND != 1 mod 4 (vs 4).
    #   vs 2: r % 2 != 0 -> r is odd.
    #   vs 4: r % 4 != 1 -> r is 3, 7.
    #   Pick 3.
    # m=16, r must be:
    #   vs 2: r % 2 != 0 (odd)
    #   vs 4: r % 4 != 1
    #   vs 8: r % 8 != 3
    #   Candidates: 1,3,5,7,9,11,13,15.
    #   Odd: 1,3,5,7,9,11,13,15.
    #   != 1 mod 4: 3,5,7,9,11,13,15 (1 is out).
    #   != 3 mod 8: 3,5,7,9,11,13,15 -> 3 is out.
    #   Remaining: 5,7,9,11,13,15.
    #   Pick 5.
    # m=32, r must be:
    #   vs 2: odd
    #   vs 4: != 1 mod 4
    #   vs 8: != 3 mod 8
    #   vs 16: != 5 mod 16
    #   Candidates: 7,9,11,13,15, 23,25,27,29,31.
    #   Check 7: 7%2=1(OK), 7%4=3(OK!=1), 7%8=7(OK!=3), 7%16=7(OK!=5).
    #   Pick 7.
    # m=64, r must be:
    #   ... != 7 mod 32.
    #   Try 9? 9%2=1, 9%4=1(FAIL).
    #   Try 11? 11%2=1, 11%4=3(OK), 11%8=3(FAIL).
    #   Try 13? 13%2=1, 13%4=1(FAIL).
    #   Try 15? 15%2=1, 15%4=3(OK), 15%8=7(OK), 15%16=15(OK!=5), 15%32=15(OK!=7).
    #   Pick 15.
    # m=128, r must be:
    #   ... != 15 mod 64.
    #   Try 17? 17%2=1, 17%4=1(FAIL).
    #   Try 19? 19%2=1, 19%4=3(OK), 19%8=3(FAIL).
    #   Try 21? 21%2=1, 21%4=1(FAIL).
    #   Try 23? 23%2=1, 23%4=3(OK), 23%8=7(OK), 23%16=7(OK!=5), 23%32=23(OK!=7), 23%64=23(OK!=15).
    #   Pick 23.
    
    # So a valid set is:
    # [0, 2], [1, 4], [3, 8], [5, 16], [7, 32], [15, 64], [23, 128]
    # Size 7.
    
    if not best_result or best_count < 7:
        best_result = [[0, 2], [1, 4], [3, 8], [5, 16], [7, 32], [15, 64], [23, 128]]
        
    return best_result
# EVOLVE-BLOCK-END
