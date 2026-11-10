import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Heuristic: Sort by descending number of divisors
    # Numbers with many divisors have many constraints
    def num_divisors(n):
        count = 0
        i = 2
        while i * i <= n:
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
            i += 1
        return count + 1 # +1 for 1 and n
    
    # Precompute gcds and divisors?
    # Let's just sort once
    moduli.sort(key=num_divisors, reverse=True)
    
    best_solution = []
    
    # Greedy with backtracking / local search
    # We'll try multiple random orders or perturbations
    
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Shuffle moduli slightly or use a different heuristic order
        # Actually, let's stick to the divisor order but randomize ties or start index
        current_moduli = moduli[:]
        random.shuffle(current_moduli) # Maybe too random?
        # Better: Sort by a random weight + divisor count
        
        selected = [] # list of (residue, modulus)
        
        for m in current_moduli:
            if time.time() > deadline:
                break
            # Find a valid residue for m
            # Constraints: for all (r_j, m_j) in selected, r % gcd(m, m_j) != r_j % gcd(m, m_j)
            
            # Optimization: If we can't find a residue, skip m
            # Try residues in random order to break symmetry
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                valid = True
                for r_j, m_j in selected:
                    g = gcd(m, m_j)
                    if r % g == r_j % g:
                        valid = False
                        break
                if valid:
                    selected.append((r, m))
                    found = True
                    break
            
            if found:
                pass
            else:
                # Skip m
                pass
        
        if len(selected) > len(best_solution):
            best_solution = selected[:]
            
        # Local Search: Try to add skipped moduli by removing some selected ones?
        # Or just rely on the greedy with different orders.
        
    # Convert to required format
    # The task asks for a list of [residue, modulus] pairs
    # We should sort them or just return as is.
    # The verifier checks pairwise disjointness.
    
    # Ensure non-empty
    if not best_solution:
        # Fallback: Just one class
        return [[0, 2]]
        
    return [[r, m] for r, m in best_solution]
