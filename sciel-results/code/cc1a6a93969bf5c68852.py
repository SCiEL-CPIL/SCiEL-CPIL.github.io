import time
import math

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_disjoint(r1, m1, r2, m2):
    g = _gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Filter moduli: Exclude primes > 64
    all_moduli = []
    for m in range(2, 129):
        is_prime = True
        for i in range(2, int(math.isqrt(m)) + 1):
            if m % i == 0:
                is_prime = False
                break
        if is_prime and m > 64:
            continue
        all_moduli.append(m)
        
    # Sort by descending modulus (large moduli first)
    all_moduli.sort(reverse=True)
    
    best_result = []
    best_size = 0
    
    def backtrack(index, current_residues, current_moduli):
        nonlocal best_result, best_size
        if time.time() > deadline:
            return
        
        if len(current_moduli) > best_size:
            best_size = len(current_moduli)
            best_result = list(zip(current_residues, current_moduli))
            
        if index == len(all_moduli):
            return
            
        # Try to add all_moduli[index]
        m = all_moduli[index]
        found = False
        for r in range(m):
            valid = True
            for i in range(len(current_moduli)):
                if not _is_disjoint(r, m, current_residues[i], current_moduli[i]):
                    valid = False
                    break
            if valid:
                found = True
                current_residues.append(r)
                current_moduli.append(m)
                backtrack(index + 1, current_residues, current_moduli)
                current_residues.pop()
                current_moduli.pop()
                # If we found a solution, we could continue to find larger ones,
                # but we want the MAXIMUM. So we must explore all branches?
                # No, we want the max size. We should continue searching for larger sets.
                # However, if we find a set of size K, we need to see if K+1 is possible.
                # Backtracking will handle this.
                
        # Try to skip all_moduli[index]
        backtrack(index + 1, current_residues, current_moduli)

    backtrack(0, [], [])
    
    best_result.sort(key=lambda x: x[1])
    return [[r, m] for r, m in best_result]
