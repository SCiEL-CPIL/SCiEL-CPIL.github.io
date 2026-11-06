import math
import time
import random

def construct():
    START = time.time()
    DEADLINE = START + 80
    
    MAX_MOD = 128
    
    def is_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
        
    def try_greedy(order):
        """
        Greedily add moduli in the given order.
        For each modulus, find the first valid residue.
        """
        solution = []
        for m in order:
            if time.time() > DEADLINE:
                break
            found = False
            for r in range(m):
                ok = True
                for r_i, m_i in solution:
                    if not is_disjoint(r, m, r_i, m_i):
                        ok = False
                        break
                if ok:
                    solution.append((r, m))
                    found = True
                    break
            # If no residue found, skip this modulus
        return solution

    # Strategy 1: Base p=11 family
    base_11 = [(i, 11 * (i + 1)) for i in range(11)]
    best = base_11
    
    # Strategy 2: Greedy on all moduli, sorted descending
    # Larger moduli have more residue options, might be easier to place?
    # Or smaller moduli have fewer options, so place them first?
    # Let's try both.
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    # Order 1: Descending
    res1 = try_greedy(sorted(all_mods, reverse=True))
    if len(res1) > len(best):
        best = res1
        
    # Order 2: Ascending
    if time.time() < DEADLINE - 10:
        res2 = try_greedy(sorted(all_mods))
        if len(res2) > len(best):
            best = res2
            
    # Order 3: Multiples of 2 first, then 3, etc?
    # Or just random restarts with a good heuristic
    # Let's try a heuristic: Sort by number of divisors (more divisors = more constraints? or more gcds?)
    # Actually, let's try to extend the p=7 family specifically.
    
    def try_prime_family(p):
        """
        Try to construct a large set using multiples of p.
        """
        mods = [m for m in range(2, MAX_MOD + 1) if m % p == 0]
        # Sort mods by value
        res = try_greedy(mods)
        return res

    # Try p=2, 3, 5, 7
    for p in [2, 3, 5, 7]:
        if time.time() > DEADLINE - 5:
            break
        res_p = try_prime_family(p)
        if len(res_p) > len(best):
            best = res_p
            
    # Try a mixed greedy with randomization
    # Use a simple local search? No, time is tight.
    # Just a few random permutations of all_mods
    
    for _ in range(10):
        if time.time() > DEADLINE - 5:
            break
        random.shuffle(all_mods)
        res_rand = try_greedy(all_mods)
        if len(res_rand) > len(best):
            best = res_rand
            
    # Verify best
    n = len(best)
    valid = True
    for i in range(n):
        if time.time() > DEADLINE:
            break
        for j in range(i + 1, n):
            r1, m1 = best[i]
            r2, m2 = best[j]
            if not is_disjoint(r1, m1, r2, m2):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to known good
        return [[r, m] for r, m in base_11]
        
    return [[r, m] for r, m in best]
