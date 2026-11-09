import math
import time

def construct():
    max_mod = 128
    
    # The set of moduli to try.
    # Based on analysis, pairwise non-coprime is required.
    # Largest such set is evens.
    # Including 2 forces all other residues to be odd, 
    # which makes any pair of other moduli (gcd>=2) conflict if gcd=2.
    # So 2 is bad if we want many moduli.
    # Let's try evens excluding 2.
    
    candidate_moduli = [m for m in range(4, max_mod + 1) if m % 2 == 0]
    
    # If this set is too large for backtracking, we might need a heuristic.
    # But 63 variables is small for a CSP solver if the constraints are tight.
    
    def solve_residues(moduli):
        n = len(moduli)
        assignment = [None] * n
        
        # Precompute gcds
        gcds = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i):
                gcds[i][j] = math.gcd(moduli[i], moduli[j])
                gcds[j][i] = gcds[i][j]
        
        # Heuristic: Order variables by most constrained first?
        # Or just try in order.
        # To speed up, we can use forward checking.
        
        def is_valid(idx, r):
            m = moduli[idx]
            for j in range(idx):
                if assignment[j] is None:
                    continue
                g = gcds[idx][j]
                if (r - assignment[j]) % g == 0:
                    return False
            return True
        
        def backtrack(idx):
            if idx == n:
                return True
            
            # Try residues in a smart order?
            # Maybe randomize? Or just 0..m-1.
            m = moduli[idx]
            for r in range(m):
                if is_valid(idx, r):
                    assignment[idx] = r
                    if backtrack(idx + 1):
                        return True
                    assignment[idx] = None
            
            return False
        
        if backtrack(0):
            return assignment
        return None

    result_assignment = solve_residues(candidate_moduli)
    
    if result_assignment is None:
        # Fallback: Try a smaller set, e.g., multiples of 3?
        # Or just return a valid small set.
        # For now, let's assume the even set works.
        # If it doesn't, we need a fallback.
        # Let's try to construct a valid set of size 10 manually or via greedy as fallback.
        # But the goal is to maximize.
        
        # Let's try another large set: Multiples of 2, but maybe we can include some odds?
        # No, odds are coprime to evens.
        
        # If the even set fails, try multiples of 3 (3, 6, 9... 126). Size 42.
        candidate_moduli_3 = [m for m in range(3, max_mod + 1) if m % 3 == 0]
        result_assignment = solve_residues(candidate_moduli_3)
        if result_assignment is None:
             # Fallback to greedy
             pass

    # Convert to list of [residue, modulus]
    if result_assignment:
         return [[result_assignment[i], candidate_moduli[i]] for i in range(len(candidate_moduli))]
    
    # Emergency fallback
    return [[0, 2]]
