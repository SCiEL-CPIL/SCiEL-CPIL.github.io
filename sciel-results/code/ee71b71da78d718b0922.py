import math
import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Candidate moduli: 2 to 128
    all_moduli = list(range(2, 129))
    
    # Strategy:
    # 1. Try the set of all even numbers.
    # 2. If that fails, try removing elements from it.
    # 3. Use backtracking to find a valid residue assignment.
    
    evens = [m for m in all_moduli if m % 2 == 0]
    
    def solve_residues(moduli):
        """
        Finds a valid residue assignment for the given list of moduli.
        Returns a list of residues or None if no solution found within time.
        """
        n = len(moduli)
        if n == 0:
            return []
        
        # Precompute gcds
        gcds = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                g = _gcd(moduli[i], moduli[j])
                gcds[i][j] = g
                gcds[j][i] = g
        
        # Order moduli to optimize backtracking
        # Heuristic: Moduli with smaller values or higher "constraint density" first?
        # Let's just sort by value ascending. Small moduli have fewer residues to try.
        order = sorted(range(n), key=lambda i: moduli[i])
        
        # Map original index to position in order
        pos_in_order = {orig: pos for pos, orig in enumerate(order)}
        
        # Residues array, initialized to -1
        residues = [-1] * n
        
        def is_valid(i, r):
            """Check if residue r for modulus i is valid with already assigned residues."""
            for j in range(n):
                if residues[j] != -1 and j != i:
                    g = gcds[i][j]
                    if (r - residues[j]) % g == 0:
                        return False
            return True
            
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            i = order[idx]
            m = moduli[i]
            
            # Try residues 0 to m-1
            # Optimization: Shuffle? No, deterministic is fine for now.
            # If m is small, it's fast.
            for r in range(m):
                if is_valid(i, r):
                    residues[i] = r
                    if backtrack(idx + 1):
                        return True
                    residues[i] = -1
                    
            return False
            
        if backtrack(0):
            return residues
        return None

    # Try with all evens
    current_moduli = evens[:]
    result = solve_residues(current_moduli)
    
    if result is not None:
        return [[result[i], current_moduli[i]] for i in range(len(current_moduli))]
    
    # If failed, try removing elements
    # This might be slow if we do it naively.
    # Given the high likelihood of success for evens, this fallback might not be needed.
    # But to be safe, let's try a smaller set of evens if the first one fails.
    # Or just return a known small valid set if time is running out.
    
    # Fallback: return a simple valid set
    # (1, 2) is valid alone.
    # (1, 2), (3, 4)? Gcd 2. 1-3 = -2. Div by 2. Fail.
    # (1, 2), (1, 4)? Gcd 2. 1-1=0. Div by 2. Fail.
    # (0, 2), (1, 4)? Gcd 2. 0-1 = -1. Not div by 2. OK.
    # So [[0, 2], [1, 4]] is valid.
    
    if time.time() > deadline - 5:
        return [[0, 2], [1, 4]]
        
    # Try a greedy approach to build a set if backtracking failed
    # This is unlikely to be better than the backtracking on evens.
    
    return [[0, 2], [1, 4]]
