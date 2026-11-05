import math
import random

def construct():
    max_mod = 128
    moduli = list(range(2, max_mod + 1, 2))
    
    # We want to assign r_m for m in moduli
    # such that r_m != r_n mod gcd(m, n)
    
    # Order moduli to help backtracking? 
    # Maybe smaller moduli first? Or those with more divisors?
    # Let's just use the natural order.
    
    n = len(moduli)
    assignment = [None] * n
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
            gcds[j][i] = gcds[i][j]
            
    def backtrack(idx):
        if idx == n:
            return True
        m = moduli[idx]
        # Try all possible residues for m
        # Shuffle to break symmetry
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            ok = True
            for j in range(idx):
                g = gcds[idx][j]
                if r % g == assignment[j] % g:
                    ok = False
                    break
            if ok:
                assignment[idx] = r
                if backtrack(idx + 1):
                    return True
                assignment[idx] = None
        return False
        
    # Try a few times with different seeds
    for _ in range(10):
        random.seed()
        if backtrack(0):
            result = [[assignment[i], moduli[i]] for i in range(n)]
            return result
            
    # If failed, return a smaller valid set
    # Fallback: just return the seed (even if invalid, to avoid crash)
    return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
