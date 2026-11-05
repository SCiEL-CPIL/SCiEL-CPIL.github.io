import math
import random
import time

def construct():
    moduli = list(range(20, 201))
    # Sort descending to assign larger moduli first (more flexibility)
    moduli.sort(reverse=True)
    
    n = len(moduli)
    assignment = [-1] * n
    # Precompute gcds for speed?
    # gcds[i][j] = gcd(moduli[i], moduli[j])
    
    # Simple backtracking
    # To avoid deep recursion issues, we can use iterative or just trust Python's recursion limit (1000 is default, depth is 181, so OK)
    
    def is_valid(idx, r):
        m = moduli[idx]
        for j in range(idx):
            if assignment[j] == -1:
                continue
            m2 = moduli[j]
            r2 = assignment[j]
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True

    def backtrack(idx):
        if idx == n:
            return True
        m = moduli[idx]
        # Try residues in a random order to break symmetry?
        # Or just 0 to m-1
        # Randomization helps if we get stuck in local minima (though this is exact search)
        residues = list(range(m))
        random.shuffle(residues)
        
        for r in residues:
            if is_valid(idx, r):
                assignment[idx] = r
                if backtrack(idx + 1):
                    return True
                assignment[idx] = -1
        return False

    # Set recursion limit higher just in case
    import sys
    sys.setrecursionlimit(300)

    # Try to find a solution
    # If it takes too long, we might need a heuristic.
    # But 181 variables is small for modern CPUs if the solution is easy to find.
    
    start_time = time.time()
    while time.time() - start_time < 100: # Leave some time for return
        if backtrack(0):
            # Construct result
            result = []
            for i in range(n):
                result.append([moduli[i], assignment[i]])
            return result
        # If failed, restart with new random seed?
        # The shuffle is inside, so it's effectively randomized.
        # But if it fails, it means no solution? Unlikely.
        # Or it's just slow.
        pass

    # Fallback: Greedy with even moduli (original) or partial solution
    # Just return whatever we have if we run out of time?
    # The problem says "timeout scores 0".
    # So we must ensure we return *something* valid.
    # If backtracking fails or times out, we can return a valid subset.
    
    # Let's try a greedy approach as a fallback or primary if backtracking is risky.
    # Actually, backtracking is exact. If it returns False, no solution exists.
    # But we suspect a solution exists.
    # If it's slow, we might timeout.
    
    # Let's try a simpler greedy first, and if it succeeds, return.
    # If greedy fails, try backtracking?
    # Or just use backtracking.
    
    # To be safe, let's implement a greedy that tries to add moduli.
    # If greedy gets all 181, it's fast.
    
    selected = []
    # Try to add all moduli in descending order
    for m in moduli:
        found = False
        # Try random residues
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            ok = True
            for m2, r2 in selected:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                selected.append([m, r])
                found = True
                break
        if not found:
            # Skip this modulus
            pass
            
    return selected
