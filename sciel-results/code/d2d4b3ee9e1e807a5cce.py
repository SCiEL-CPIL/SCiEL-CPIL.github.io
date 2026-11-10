import math
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin
    
    # Moduli range 2..128
    max_mod = 128
    all_moduli = list(range(2, max_mod + 1))
    
    # Precompute gcds for speed
    # gcd_matrix[i][j] = gcd(i, j)
    # We only need this for selected moduli, but precomputing for all is fine (128x128)
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def is_valid_assignment(residues, moduli):
        n = len(moduli)
        for i in range(n):
            for j in range(i + 1, n):
                g = gcd(moduli[i], moduli[j])
                if residues[i] % g == residues[j] % g:
                    return False
        return True

    def find_solution(moduli_set):
        # Backtracking to assign residues
        n = len(moduli_set)
        residues = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            if time.time() > deadline:
                return False
            
            m = moduli_set[idx]
            # Try all possible residues for m
            # Optimization: Shuffle residues to find solution faster
            candidates = list(range(m))
            random.shuffle(candidates)
            
            for r in candidates:
                residues[idx] = r
                # Check against all previous
                valid = True
                for j in range(idx):
                    g = gcd(m, moduli_set[j])
                    if r % g == residues[j] % g:
                        valid = False
                        break
                if valid:
                    if backtrack(idx + 1):
                        return True
            residues[idx] = None
            return False
        
        if backtrack(0):
            return [(residues[i], moduli_set[i]) for i in range(n)]
        return None

    # Strategy: 
    # 1. Try to find a large set of moduli that are "compatible".
    # Compatibility means they share factors.
    # We group moduli by their smallest prime factor?
    # Or just try random subsets of moduli, sorted by size.
    
    best_solution = []
    
    # Heuristic: Moduli with more small prime factors are more likely to be compatible with others.
    # Let's try to build a set using primes 2, 3, 5.
    
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    # Filter moduli to those composed of {2, 3, 5}
    # This ensures high connectivity in the gcd graph
    candidate_moduli = [m for m in all_moduli if get_prime_factors(m).issubset({2, 3, 5})]
    
    # Sort by number of factors? Or just random order?
    # Let's try greedy addition.
    
    current_set = []
    current_residues = []
    
    # We want to maximize size.
    # Try adding moduli one by one.
    # If we can't add a modulus, skip it.
    
    # To get a better set, we can try multiple random orderings.
    
    for _ in range(100): # 100 restarts
        if time.time() > deadline:
            break
            
        random.shuffle(candidate_moduli)
        
        temp_set = []
        temp_res = []
        
        for m in candidate_moduli:
            if time.time() > deadline:
                break
            
            # Try to find a residue for m that is compatible with temp_set
            success = False
            candidates = list(range(m))
            random.shuffle(candidates)
            
            for r in candidates:
                valid = True
                for j in range(len(temp_set)):
                    g = gcd(m, temp_set[j])
                    if r % g == temp_res[j] % g:
                        valid = False
                        break
                if valid:
                    temp_set.append(m)
                    temp_res.append(r)
                    success = True
                    break
            
            if success:
                if len(temp_set) > len(best_solution):
                    best_solution = list(zip(temp_res, temp_set))
                    
    # Fallback: Powers of 2
    fallback = [[2**(k-1), 2**k] for k in range(1, 8)]
    if len(fallback) > len(best_solution):
        best_solution = fallback

    return best_solution
