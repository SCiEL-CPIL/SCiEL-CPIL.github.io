import math
import time

def construct():
    start_time = time.time()
    deadline = 60.0
    
    # Candidate moduli: all even numbers from 2 to 128
    moduli = [2 * i for i in range(1, 65)]
    
    # Precompute gcds
    n = len(moduli)
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
            gcds[j][i] = gcds[i][j]
            
    # Backtracking solver
    residues = [None] * n
    
    def is_valid(i, r):
        # Check if residue r for moduli[i] is valid against assigned residues
        for j in range(n):
            if residues[j] is not None:
                g = gcds[i][j]
                if r % g == residues[j] % g:
                    return False
        return True

    def backtrack(idx):
        if time.time() - start_time > deadline - 5:
            return False
        
        if idx == n:
            return True
            
        # Try residues for moduli[idx]
        m = moduli[idx]
        for r in range(m):
            if is_valid(idx, r):
                residues[idx] = r
                if backtrack(idx + 1):
                    return True
                residues[idx] = None
                
        return False

    # Order moduli by size of domain? All small.
    # Or by number of constraints?
    # Let's just try in order.
    
    if backtrack(0):
        result = [[residues[i], moduli[i]] for i in range(n)]
        return result
    else:
        # Fallback: just return the seed or a smaller valid set
        # If 64 fails, try 63?
        # For now, return the seed to be safe?
        # But we expect 64 to work.
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
