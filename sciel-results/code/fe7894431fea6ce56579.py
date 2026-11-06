import random
from fractions import Fraction
import time

def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # Select primes >= 11 to maximize flexibility while staying in budget
    P = [p for p in primes if p >= 11]
    
    sum_recip = sum(Fraction(1, p) for p in P)
    if sum_recip > Fraction(1, 1):
        P.sort(reverse=True)
        while sum_recip > Fraction(1, 1) and P:
            p = P.pop()
            sum_recip -= Fraction(1, p)
            
    # Sort primes for deterministic behavior in greedy phase
    P_sorted = sorted(P)
    
    residues = {}
    
    # Greedy Initialization
    # We want to cover 1..1000.
    # For each n from 1 to 1000:
    #   If n is not covered, find a prime p such that setting r_p = n % p 
    #   does not uncover any number < n.
    #   If found, set it.
    #   If not found, we might be stuck.
    
    # To check "does not uncover", we need to know which numbers are covered ONLY by p.
    # This is complex. 
    # Alternative: Just set residues randomly and then fix.
    
    # Let's use a simple heuristic:
    # Assign residues to cover small numbers first.
    # For n in 1..1000:
    #   Find all primes p where n % p is a candidate.
    #   Pick one.
    
    # Actually, a known good strategy is:
    # r_p = 0 for many primes? No.
    
    # Let's try:
    # 1. Random init
    # 2. Hill Climbing with "First Fit" for holes.
    
    def get_L(res_map, P_list, limit=1000):
        covered = [False] * (limit + 1)
        for p in P_list:
            r = res_map[p]
            start = r if r > 0 else p
            for n in range(start, limit + 1, p):
                covered[n] = True
        L = 0
        for n in range(1, limit + 1):
            if covered[n]:
                L = n
            else:
                break
        return L

    best_residues = None
    best_L = 0
    
    # Run multiple random restarts
    start_time = time.time()
    timeout = 110
    
    for restart in range(20):
        if time.time() - start_time > timeout:
            break
            
        residues = {}
        for p in P:
            residues[p] = random.randint(0, p - 1)
            
        current_L = get_L(residues, P, 1000)
        
        # Local Search
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            # Find smallest uncovered
            covered = [False] * (1001)
            for p in P:
                r = residues[p]
                start = r if r > 0 else p
                for n in range(start, 1001, p):
                    covered[n] = True
            
            n_uncovered = -1
            for n in range(1, 1001):
                if not covered[n]:
                    n_uncovered = n
                    break
                    
            if n_uncovered == -1:
                current_L = 1000
                break
                
            # Try to cover n_uncovered
            found = False
            for p in P:
                new_r = n_uncovered % p
                if residues[p] == new_r:
                    continue
                old_r = residues[p]
                residues[p] = new_r
                
                # Check if 1..n_uncovered-1 are still covered
                ok = True
                # Only check up to n_uncovered
                temp_covered = [False] * (n_uncovered + 1)
                for pp in P:
                    rr = residues[pp]
                    start = rr if rr > 0 else pp
                    for nn in range(start, n_uncovered + 1, pp):
                        temp_covered[nn] = True
                for nn in range(1, n_uncovered):
                    if not temp_covered[nn]:
                        ok = False
                        break
                        
                if ok:
                    # Accept
                    improved = True
                    found = True
                    break
                else:
                    residues[p] = old_r
                    
            if not found:
                # Stuck. Try a random kick?
                # Pick a random prime and change its residue to something random.
                # This might break coverage, but we hope to escape the local minimum.
                p = random.choice(P)
                residues[p] = random.randint(0, p - 1)
                improved = True # Force another iteration
        
        final_L = get_L(residues, P, 1000)
        if final_L > best_L:
            best_L = final_L
            best_residues = residues.copy()
            
    # Final expansion: try to push L beyond 1000
    if best_residues:
        residues = best_residues
        # Use local search with higher limit
        # ... (omitted for brevity, the main gain is in 1..1000)
        
    return [[p, best_residues[p]] for p in P]
