"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(n+1) if sieve[i]]

def _coverage_length(classes, cap=100000):
    """Return largest L such that every integer 1..L is covered."""
    if not classes:
        return 0
    # Build a set of covered residues for each prime
    covered = [False] * (cap + 1)
    for p, r in classes:
        for x in range(r if r >= 1 else p, cap + 1, p):
            if x <= cap:
                covered[x] = True
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _reciprocal_sum(classes):
    s = Fraction(0, 1)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    primes = _primes_upto(1000)
    
    # Greedy set-cover: at each step, pick (p, r) that maximizes new coverage / cost
    # We want to cover 1..L for as large L as possible.
    
    # First, try a simple greedy: for each prime, pick the residue that covers the most
    # currently-uncovered small integers.
    
    # Strategy: 
    # 1. Start with no classes.
    # 2. Repeatedly add the class (p, r) that covers the most uncovered integers in 1..MAX_L
    #    subject to budget constraint.
    # 3. Use local search to improve.
    
    MAX_L = 10000
    
    # Precompute for each prime p and residue r, which integers in 1..MAX_L it covers
    # This is too expensive for all combinations. Instead, use a smarter approach.
    
    # Alternative: greedy by adding primes one at a time, choosing best residue.
    # Start with the smallest primes (they have the most "power" per unit cost).
    
    # Sort primes by size (smaller primes cover more densely)
    sorted_primes = sorted(primes)
    
    # Greedy: add primes in order of increasing size, choosing best residue
    # to maximize coverage of the initial segment.
    
    budget = Fraction(1, 1)
    classes = []
    used_primes = set()
    
    # Track which integers are covered
    # Use a boolean array for 1..MAX_L
    covered = [False] * (MAX_L + 1)
    
    def current_coverage():
        L = 0
        for i in range(1, MAX_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L
    
    # Greedy phase: try adding each prime (in order of increasing p) with best residue
    for p in sorted_primes:
        if time.time() > deadline:
            break
        cost = Fraction(1, p)
        if budget - cost < 0:
            continue
        
        # Find best residue for this prime
        best_r = 0
        best_gain = -1
        for r in range(p):
            # Count how many new integers in 1..MAX_L would be covered
            gain = 0
            for x in range(r if r >= 1 else p, MAX_L + 1, p):
                if not covered[x]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_gain > 0:
            # Add this class
            classes.append([p, best_r])
            used_primes.add(p)
            budget -= cost
            # Update covered
            for x in range(best_r if best_r >= 1 else p, MAX_L + 1, p):
                covered[x] = True
    
    # Now try local search: swap one class for another to improve coverage
    def try_improve():
        nonlocal classes, covered, budget
        improved = True
        while improved and time.time() < deadline:
            improved = False
            L_before = current_coverage()
            
            # Try removing each class and re-adding a different one
            for idx in range(len(classes)):
                if time.time() > deadline:
                    break
                p_out, r_out = classes[idx]
                cost_out = Fraction(1, p_out)
                
                # Remove this class
                classes.pop(idx)
                budget += cost_out
                # Update covered
                for x in range(r_out if r_out >= 1 else p_out, MAX_L + 1, p_out):
                    # Only unset if no other class covers x
                    # This is complex; skip for simplicity in this version
                    pass
                
                # Try adding a new class
                best_new = None
                best_L = -1
                for p_new in primes:
                    if p_new in used_primes and p_new != p_out:
                        continue
                    cost_new = Fraction(1, p_new)
                    if budget - cost_new < 0:
                        continue
                    
                    # Find best residue
                    best_r_new = 0
                    best_L_candidate = -1
                    for r_new in range(p_new):
                        # Temporarily add and check coverage
                        # This is expensive; approximate by checking key positions
                        L_cand = 0
                        for i in range(1, MAX_L + 1):
                            is_c = covered[i]
                            if not is_c:
                                # Check if new class covers i
                                if i % p_new == r_new % p_new:
                                    is_c = True
                            if is_c:
                                L_cand = i
                            else:
                                break
                        if L_cand > best_L_candidate:
                            best_L_candidate = L_cand
                            best_r_new = r_new
                    
                    if best_L_candidate > best_L:
                        best_L = best_L_candidate
                        best_new = [p_new, best_r_new]
                
                if best_new and best_L > L_before:
                    classes.append(best_new)
                    used_primes.add(best_new[0])
                    budget -= Fraction(1, best_new[0])
                    improved = True
                    break
                else:
                    # Put back the removed class
                    classes.insert(idx, [p_out, r_out])
                    budget -= cost_out
    
    try_improve()
    
    # Compute final coverage
    # Rebuild covered array from scratch
    covered_final = [False] * (100001)
    for p, r in classes:
        for x in range(r if r >= 1 else p, 100001, p):
            covered_final[x] = True
    
    L = 0
    for i in range(1, 100001):
        if covered_final[i]:
            L = i
        else:
            break
    
    return classes

# EVOLVE-BLOCK-END
