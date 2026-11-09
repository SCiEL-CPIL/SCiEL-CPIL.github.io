import time
from math import gcd

def construct():
    deadline = time.time() + 85
    
    # Use the set of even moduli, which is known to be a large pairwise non-coprime set
    moduli = [m for m in range(2, 129) if m % 2 == 0]
    n = len(moduli)
    
    # Precompute gcds
    gcd_mat = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_mat[i][j] = gcd(moduli[i], moduli[j])
            
    # Heuristic: Order variables by descending modulus? Or maybe by number of constraints?
    # Since it's a complete graph, maybe just index order or random.
    # Let's try to solve for all n.
    
    def solve():
        assignment = [None] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            m = moduli[idx]
            # Try residues in a specific order?
            # Maybe try residues that are "different" from previous ones.
            for r in range(m):
                # Check constraints with all previous assigned variables
                valid = True
                for j in range(idx):
                    if assignment[j] is None:
                        continue
                    g = gcd_mat[idx][j]
                    if (r - assignment[j]) % g == 0:
                        valid = False
                        break
                if valid:
                    assignment[idx] = r
                    if backtrack(idx + 1):
                        return True
                    assignment[idx] = None
            return False
            
        return backtrack(0), assignment

    # First, try to solve for all even moduli
    success, assignment = solve()
    if success:
        return [[assignment[i], moduli[i]] for i in range(n)]
        
    # If it fails, try smaller subsets?
    # Or maybe random restarts with different orders?
    
    # For now, if the full set fails, we can try to add moduli one by one to a smaller set
    # But 64 is likely solvable.
    
    return [[0, 2]]
