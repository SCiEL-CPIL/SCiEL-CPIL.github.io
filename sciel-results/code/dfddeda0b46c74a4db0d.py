"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def _gcd(a, b):
    return math.gcd(a, b)

def _is_disjoint(r1, m1, r2, m2):
    """Check if residue class r1 mod m1 is disjoint from r2 mod m2."""
    g = _gcd(m1, m2)
    return (r1 - r2) % g != 0

def _count_violations(residues, moduli):
    """Count number of pairwise disjointness violations."""
    violations = 0
    n = len(moduli)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_disjoint(residues[i], moduli[i], residues[j], moduli[j]):
                violations += 1
    return violations

def _local_search_residues(moduli, time_limit=5.0):
    """
    Use Min-Conflicts Local Search to find residues for the given moduli
    such that all pairs are disjoint.
    """
    n = len(moduli)
    if n == 0:
        return []
    
    # Initialize residues randomly
    residues = [random.randint(0, m - 1) for m in moduli]
    
    start_time = time.time()
    best_residues = residues[:]
    best_violations = _count_violations(residues, moduli)
    
    if best_violations == 0:
        return residues
    
    while time.time() - start_time < time_limit:
        # Find a variable with violations
        # For efficiency, we can just pick a random variable and check if it has violations
        # Or better: pick a random pair that is violated, and change one of them.
        
        # Let's try to find a violated pair
        violated_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if not _is_disjoint(residues[i], moduli[i], residues[j], moduli[j]):
                    violated_pairs.append((i, j))
        
        if not violated_pairs:
            return residues
        
        # Pick a random violated pair
        i, j = random.choice(violated_pairs)
        
        # Try to change r_i or r_j to reduce violations
        # Let's try changing r_i first
        best_new_r = residues[i]
        best_local_violations = _count_violations(residues, moduli) # Current total
        
        # Actually, calculating total violations every time is O(N^2).
        # We can just count violations involving i.
        
        def count_violations_for_index(idx, res_list, mod_list):
            count = 0
            for k in range(len(mod_list)):
                if k != idx:
                    if not _is_disjoint(res_list[idx], mod_list[idx], res_list[k], mod_list[k]):
                        count += 1
            return count

        current_viol_i = count_violations_for_index(i, residues, moduli)
        
        # Try all possible values for r_i
        for new_r in range(moduli[i]):
            residues[i] = new_r
            new_viol_i = count_violations_for_index(i, residues, moduli)
            if new_viol_i < best_local_violations:
                best_local_violations = new_viol_i
                best_new_r = new_r
        residues[i] = best_new_r
        
        # Check if total violations decreased
        total_violations = _count_violations(residues, moduli)
        if total_violations < best_violations:
            best_violations = total_violations
            best_residues = residues[:]
            
        if best_violations == 0:
            return best_residues

    return best_residues if best_violations == 0 else None

def construct():
    """
    Maximize the number of pairwise disjoint infinite residue classes with distinct moduli in [2, 128].
    Strategy:
    1. Identify the largest set of pairwise non-coprime moduli.
       The set of all even numbers in [2, 128] forms such a set of size 64.
    2. Solve the CSP for residues on this set using Local Search.
    3. If successful, return the 64 pairs.
    4. If not, try subsets or other prime-based sets (less likely to be larger).
    """
    start_time = time.time()
    time_limit = 55.0
    
    # Candidate moduli sets
    # 1. All even numbers
    evens = list(range(2, 129, 2))
    
    # Try to solve for evens
    residues = _local_search_residues(evens, time_limit=10.0)
    if residues is not None:
        # Verify
        valid = True
        for i in range(len(evens)):
            for j in range(i+1, len(evens)):
                if not _is_disjoint(residues[i], evens[i], residues[j], evens[j]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return [[r, m] for r, m in zip(residues, evens)]
            
    # Fallback: Greedy random search similar to original but smarter
    # Just in case the CSP fails (unlikely for evens)
    # We can also try multiples of 3 (size 42) if evens fail, but 64 > 42.
    
    # If evens failed, try a generic local search on the full space?
    # No, the structural constraint is key.
    
    # Let's try a randomized restart for the CSP on evens
    for _ in range(5):
        residues = _local_search_residues(evens, time_limit=5.0)
        if residues is not None:
            valid = True
            for i in range(len(evens)):
                for j in range(i+1, len(evens)):
                    if not _is_disjoint(residues[i], evens[i], residues[j], evens[j]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return [[r, m] for r, m in zip(residues, evens)]
                
    # If all else fails, return a small valid set (e.g. just one pair)
    # This should never happen.
    return [[0, 2]]

# EVOLVE-BLOCK-END
