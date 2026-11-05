import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    max_m = 128
    # Candidate moduli: all even numbers
    moduli = [m for m in range(2, max_m + 1) if m % 2 == 0]
    
    best_solution = None
    best_count = 0
    
    # Try to find a valid assignment for a subset of moduli
    # We will use a greedy removal strategy combined with local search for residues
    
    # Initialize with all even moduli
    current_moduli = moduli[:]
    
    # Function to check if a set of moduli with given residues is valid
    def is_valid(mods, res):
        n = len(mods)
        for i in range(n):
            for j in range(i + 1, n):
                g = math.gcd(mods[i], mods[j])
                if g == 1:
                    return False
                # Disjoint iff r_i not congruent r_j mod g
                # i.e., (r_i - r_j) % g != 0
                if (res[i] - res[j]) % g == 0:
                    return False
        return True

    # Function to try to find residues for a given set of moduli
    def find_residues(mods):
        # Random initialization
        res = [random.randint(0, m - 1) for m in mods]
        
        # Local search
        for _ in range(1000):
            if is_valid(mods, res):
                return res
            # Pick a random conflicting pair and fix it
            # To be faster, just pick a random index and change its residue
            idx = random.randint(0, len(mods) - 1)
            res[idx] = random.randint(0, mods[idx] - 1)
        return None

    # Strategy: Start with all even moduli.
    # If find_residues fails, remove the most "conflicting" modulus.
    # How to measure conflict? Just remove randomly or based on degree in conflict graph.
    # Simpler: Remove randomly until it works.
    
    # To be smarter, we can try to keep as many as possible.
    # We'll iterate, removing one at a time, and trying to solve.
    
    # Optimization: Pre-compute gcds?
    
    # Let's try a few random restarts of the removal process
    for restart in range(100):
        if time.time() > deadline:
            break
            
        mods = moduli[:]
        random.shuffle(mods)
        
        # Try to solve for mods
        # If fails, remove the last element (randomly chosen) and retry
        # This is slow if we restart full search each time.
        
        # Better:
        # Start with all.
        # While not valid and time remains:
        #   Remove a modulus.
        #   Try to find residues for the remaining.
        
        current_mods = moduli[:]
        random.shuffle(current_mods)
        
        # We will try to solve for the full set first
        # If it fails, we remove moduli one by one from the end of the shuffled list
        
        # To speed up, we can check validity incrementally?
        # No, just try find_residues.
        
        # If find_residues is too slow, we need a better heuristic.
        # Let's assume find_residues is fast enough for ~60 items.
        
        res = find_residues(current_mods)
        if res:
            if len(current_mods) > best_count:
                best_count = len(current_mods)
                best_solution = [[res[i], current_mods[i]] for i in range(len(current_mods))]
            continue
        
        # If full set fails, remove one and retry
        while current_mods and time.time() < deadline:
            # Remove a random modulus
            idx = random.randint(0, len(current_mods) - 1)
            removed_mod = current_mods.pop(idx)
            
            # Try to find residues for the reduced set
            # Note: We can reuse previous residues? No, simpler to restart or adjust.
            # Given the small size, restart is fine.
            res = find_residues(current_mods)
            if res:
                if len(current_mods) > best_count:
                    best_count = len(current_mods)
                    best_solution = [[res[i], current_mods[i]] for i in range(len(current_mods))]
                break
            # If not, loop removes another
        
    if best_solution:
        return best_solution
    else:
        # Fallback to seed
        return [[(1<<k)-1,1<<(k+1)] for k in range(7)]
