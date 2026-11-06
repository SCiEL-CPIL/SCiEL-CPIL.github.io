import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 80  # safety margin before 90s
    
    MAX_MOD = 128
    # All candidate moduli from 2 to 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd table for speed
    gcd_table = {}
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            gcd_table[(i, j)] = gcd(i, j)
    
    def get_gcd(a, b):
        if a <= b:
            return gcd_table.get((a, b), gcd(a, b))
        else:
            return gcd_table.get((b, a), gcd(a, b))
    
    # Check if adding (r, m) is compatible with existing (r_i, m_i) for all i
    def is_compatible(r, m, selected):
        for (r_i, m_i) in selected:
            g = get_gcd(m, m_i)
            if g > 1:
                # They intersect if g divides (r - r_i)
                if (r - r_i) % g == 0:
                    return False
            else:
                # gcd == 1, always intersect, so incompatible
                return False
        return True
    
    # Backtracking with ordering heuristic
    def backtrack(moduli_list, selected, best, depth=0):
        if time.time() > deadline:
            return
        if len(selected) > len(best):
            best[:] = selected[:]
        
        if not moduli_list:
            return
        
        # Try each remaining modulus
        for idx in range(len(moduli_list)):
            m = moduli_list[idx]
            remaining = moduli_list[:idx] + moduli_list[idx+1:]
            
            # Try residues 0 to m-1
            # Heuristic: try residues that are "far" from existing ones
            # For speed, try a subset first
            found = False
            # Try all residues (m can be up to 128, so at most 128 tries)
            for r in range(m):
                if is_compatible(r, m, selected):
                    selected.append((r, m))
                    backtrack(remaining, selected, best, depth + 1)
                    selected.pop()
                    found = True
                    break  # Found one valid residue, continue with next modulus
            
            if not found:
                # This modulus can't be added, skip it
                backtrack(remaining, selected, best, depth)
    
    # Better approach: use a greedy + local search or smarter backtracking
    # First, let's try a greedy approach with multiple orderings
    
    def greedy_solve(ordering):
        selected = []
        for m in ordering:
            # Find a residue r in [0, m-1] that's compatible
            best_r = None
            for r in range(m):
                if is_compatible(r, m, selected):
                    best_r = r
                    break
            if best_r is not None:
                selected.append((best_r, m))
        return selected
    
    # Try multiple orderings
    best_result = []
    
    # Ordering 1: descending (larger moduli first - more flexible)
    best_result = greedy_solve(sorted(all_moduli, reverse=True))
    
    # Ordering 2: ascending
    r2 = greedy_solve(sorted(all_moduli))
    if len(r2) > len(best_result):
        best_result = r2
    
    # Ordering 3: even moduli first, then odd
    evens = [m for m in all_moduli if m % 2 == 0]
    odds = [m for m in all_moduli if m % 2 == 1]
    r3 = greedy_solve(evens + odds)
    if len(r3) > len(best_result):
        best_result = r3
    
    # Ordering 4: by number of prime factors (more factors = more compatible)
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    r4 = greedy_solve(sorted(all_moduli, key=num_prime_factors, reverse=True))
    if len(r4) > len(best_result):
        best_result = r4
    
    # Now try to improve with local search / backtracking on the best found
    # Convert best_result to list of [r, m]
    if best_result:
        # Try adding more moduli via backtracking
        used_moduli = set(m for _, m in best_result)
        remaining_moduli = [m for m in all_moduli if m not in used_moduli]
        
        # Try to add each remaining modulus
        improved = list(best_result)
        for m in remaining_moduli:
            if time.time() > deadline:
                break
            for r in range(m):
                if is_compatible(r, m, improved):
                    improved.append((r, m))
                    break
        
        if len(improved) > len(best_result):
            best_result = improved
    
    # Convert to required format: list of [residue, modulus]
    result = [[r, m] for r, m in best_result]
    
    # Ensure nonempty
    if not result:
        result = [[1, 2]]
    
    return result
