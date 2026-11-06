import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Precompute GCDs
    max_m = 128
    gcds = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(2, max_m + 1):
        for j in range(2, max_m + 1):
            gcds[i][j] = math.gcd(i, j)
            
    # We know the max size is likely 64 (all evens).
    # We will try to find a valid assignment for all 64 even numbers.
    # If we fail, we try to find a subset.
    
    even_moduli = list(range(2, max_m + 1, 2))
    
    def solve_csp(moduli, assigned, remaining):
        if time.time() > deadline:
            return None
            
        if not remaining:
            return assigned
            
        # MRV Heuristic: Pick modulus with fewest valid residues
        best_m = None
        best_valid = None
        min_count = float('inf')
        
        for m in remaining:
            valid_r = []
            for r in range(m):
                ok = True
                for r_i, m_i in assigned:
                    g = gcds[m][m_i]
                    if (r - r_i) % g == 0:
                        ok = False
                        break
                if ok:
                    valid_r.append(r)
            if len(valid_r) < min_count:
                min_count = len(valid_r)
                best_m = m
                best_valid = valid_r
            if min_count == 0:
                return None
            if min_count == 1:
                break
                
        if best_m is None:
            return None
            
        # Try residues for best_m
        # Randomize order of residues to avoid local minima?
        # Or just try the first one?
        # For CSP, trying all is better, but we have time limit.
        # Let's try a few random permutations.
        
        # Backtrack
        remaining.remove(best_m)
        
        for r in best_valid:
            new_assigned = assigned + [[r, best_m]]
            res = solve_csp(moduli, new_assigned, remaining)
            if res is not None:
                return res
                
        remaining.add(best_m)
        return None

    # Try to solve for all even moduli
    # Use a set for remaining for O(1) removal
    # But list is fine for 64 items
    
    best_solution = []
    
    # Try multiple random restarts with different residue orderings
    for _ in range(1000):
        if time.time() > deadline:
            break
        # Shuffle moduli order? MRV handles it.
        # Just run the CSP solver.
        # The CSP solver above is deterministic if it picks the first valid residue.
        # To make it randomized, we should shuffle the valid residues.
        
        # Let's implement a randomized version inside the loop
        pass

    # Simpler: Just try to assign residues to all evens using a greedy randomized approach
    # and keep the best.
    
    best_count = 0
    best_pairs = []
    
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        assigned = []
        moduli = list(range(2, max_m + 1, 2))
        random.shuffle(moduli)
        
        for m in moduli:
            valid_r = []
            for r in range(m):
                ok = True
                for r_i, m_i in assigned:
                    if (r - r_i) % gcds[m][m_i] == 0:
                        ok = False
                        break
                if ok:
                    valid_r.append(r)
            
            if valid_r:
                r = random.choice(valid_r)
                assigned.append([r, m])
                
        if len(assigned) > best_count:
            best_count = len(assigned)
            best_pairs = assigned
            
            if best_count == 64:
                break
                
    return best_pairs
