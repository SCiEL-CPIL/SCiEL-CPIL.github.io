import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit for safety
    
    moduli = list(range(2, 129))
    
    # Sort moduli to improve search efficiency
    # Heuristic: Sort by descending number of divisors or descending modulus?
    # Let's try descending modulus first, as larger moduli have more freedom?
    # Or ascending?
    # In my manual trace, I added 24, 48 to {8,16,32,64,96,128}.
    # Let's use a greedy + backtracking approach.
    
    best_solution = []
    
    # Precompute gcds
    gcds = [[0]*129 for _ in range(129)]
    for i in range(2, 129):
        for j in range(i, 129):
            gcds[i][j] = math.gcd(i, j)
            gcds[j][i] = gcds[i][j]

    def backtrack(current_set, current_residues, index, moduli_order):
        nonlocal best_solution
        if time.time() > deadline:
            return
        
        if len(current_set) > len(best_solution):
            best_solution = [ [r, m] for m, r in zip(current_set, current_residues) ]
            
        if index == len(moduli_order):
            return
            
        m = moduli_order[index]
        
        # Try to find a valid residue for m
        # Forbidden residues logic
        # We need r in [0, m) such that for all (m', r') in current_set:
        # r != r' % gcd(m, m')
        
        # Optimization: If current_set is empty, pick 0
        if not current_set:
            backtrack(current_set + [m], current_residues + [0], index + 1, moduli_order)
            # Also try other residues? No, 0 is as good as any for the first one.
            # But maybe 0 is bad for future?
            # For simplicity, try a few candidates.
            
        # Check all possible residues
        # To speed up, we can check constraints incrementally.
        # But m is up to 128, so 128 checks is fast.
        
        # Heuristic: Try residues that are "central" or random?
        # Let's just iterate 0 to m-1.
        for r in range(m):
            valid = True
            for i, m_prime in enumerate(current_set):
                g = gcds[m][m_prime]
                r_prime = current_residues[i]
                if r % g == r_prime % g:
                    valid = False
                    break
            if valid:
                backtrack(current_set + [m], current_residues + [r], index + 1, moduli_order)
                # Optional: Prune if we found a solution? No, we want max size.
                # But we can break if we found a valid r, because we just want ANY valid set of this size?
                # No, we want to explore if a different r allows more future moduli.
                # However, for a fixed set of moduli, if one r works, does another?
                # The future constraints depend on r.
                # So we should try multiple r's.
                # But to save time, we can try a limited number of candidates.
                # Let's try the first 3 valid residues.
                if r > 2: 
                    # Heuristic: don't explore too many branches for the same modulus
                    # This makes it a heuristic search, not exact.
                    # Given the time limit, this is acceptable.
                    break

    # Order moduli: 
    # Maybe group by prime factors?
    # Let's try sorting by descending modulus.
    moduli_order = sorted(moduli, reverse=True)
    
    # Also try ascending
    # And a mixed order?
    
    backtrack([], [], 0, moduli_order)
    
    if time.time() < deadline:
        # Try ascending
        moduli_order = sorted(moduli)
        backtrack([], [], 0, moduli_order)
        
    if not best_solution:
        # Fallback
        return [[1, 2]]
        
    return best_solution
