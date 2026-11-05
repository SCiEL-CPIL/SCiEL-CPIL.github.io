import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit for 90s total
    
    max_modulus = 128
    # Precompute all valid pairs
    # We will select a subset of these
    
    # Heuristic: Try to include as many moduli as possible.
    # Sort moduli by some heuristic, e.g., number of divisors, or just random.
    
    best_solution = []
    
    # Try several random seeds/orderings
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Create a list of all possible (r, m) pairs
        # For each m, we can choose any r in [0, m)
        # To maximize count, we should try to fit as many m as possible.
        
        # Strategy: Randomly permute moduli 2..128.
        # For each modulus in order, try to find a residue r such that 
        # it is disjoint from all previously selected classes.
        
        moduli = list(range(2, max_modulus + 1))
        random.shuffle(moduli)
        
        current_solution = []
        
        for m in moduli:
            # Try to find a residue r for this m
            # We can try all r in [0, m) or a random subset
            # To be fast, try a few random r's or specific ones
            
            found = False
            # Try r = 1, m-1, m//2, etc first? Or random?
            # Let's try all r if m is small, random if large.
            # Actually, m is at most 128, so trying all r is 128 checks.
            # 128 moduli * 128 residues * 128 checks = ~2M ops per attempt.
            # 100 attempts = 200M ops. Might be too slow in Python.
            
            # Optimization: 
            # For a new (r, m), check against all (r_i, m_i) in current_solution.
            # Condition: r not congruent r_i mod gcd(m, m_i)
            
            # Let's try a limited set of residues for each m to speed up.
            # Or just try all r, but break early if found.
            
            for r in range(m):
                compatible = True
                for r_i, m_i in current_solution:
                    d = gcd(m, m_i)
                    if r % d == r_i % d:
                        compatible = False
                        break
                if compatible:
                    current_solution.append([r, m])
                    found = True
                    break
            
            # If no residue found for this m, we skip this m.
            # This greedy approach might miss better solutions.
        
        if len(current_solution) > len(best_solution):
            best_solution = current_solution
            
        # Early stop if we found a very good solution?
        # Max possible is 127 (moduli 2..128).
        # If we get close to that, stop.
        if len(best_solution) > 50:
             break

    return best_solution
