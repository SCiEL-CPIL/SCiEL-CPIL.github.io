"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W):
        r = len(U[0]) if U and U[0] else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Use a structured search based on the algebraic structure of matmul
    # Key insight: we can use block matrices and known low-rank decompositions
    # For 2x5x5, we can think of it as blocks
    
    # Try to find a decomposition by fixing the structure
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Strategy: use random linear combinations and solve
    # For a given rank r, randomly initialize U and V, then solve for W
    # The equations are: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is linear in W entries for fixed U,V
    
    def solve_for_W(U, V, r):
        """Given U and V, solve linear system for W."""
        # Variables: W[k*n+i, c] for k in range(p), i in range(n), c in range(r)
        # Total variables: p*n*r
        # Equations: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Note: each equation only involves W entries with specific (k,i)
        # So equations for different (k,i) are independent!
        
        # Group equations by (k,i):
        # For fixed (k,i): sum over j of: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_j 1 = m
        # Wait, that's not right. Each (i,j,k) gives one equation = 1.
        # For fixed (k,i), the equations are over j=0..m-1:
        #   sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for each j
        # This is a system of m equations in r unknowns (W[k*n+i, c] for c=0..r-1)
        
        # Build the system: A[j,c] * W[k*n+i,c] = 1 for j=0..m-1
        # where A[j,c] = U[i*m+j,c] * V[j*p+k,c]
        
        W = [[0]*r for _ in range(pn)]
        for k in range(p):
            for i in range(n):
                # Build m x r matrix A
                A = [[0]*r for _ in range(m)]
                for j in range(m):
                    for c in range(r):
                        A[j][c] = U[i*m+j][c] * V[j*p+k][c]
                # Solve A * w = 1 (vector of ones)
                # Use simple Gaussian elimination with fractions
                # Augmented matrix
                aug = [[Fraction(A[j][c]) for c in range(r)] + [Fraction(1)] for j in range(m)]
                
                # Gaussian elimination
                rows = m
                cols = r
                rank_found = 0
                for col in range(cols):
                    if rank_found >= rows:
                        break
                    # Find pivot
                    pivot = -1
                    for row in range(rank_found, rows):
                        if aug[row][col] != 0:
                            pivot = row
                            break
                    if pivot == -1:
                        continue
                    aug[rank_found], aug[pivot] = aug[pivot], aug[rank_found]
                    # Eliminate
                    pivot_val = aug[rank_found][col]
                    for row in range(rows):
                        if row == rank_found:
                            continue
                        factor = aug[row][col] / pivot_val
                        for cc in range(col, cols+1):
                            aug[row][cc] -= factor * aug[rank_found][cc]
                    rank_found += 1
                
                # Check consistency
                for row in range(rank_found, rows):
                    if any(aug[row][c] != 0 for c in range(cols)) and aug[row][cols] != 0:
                        return None  # Inconsistent
                
                # Back substitute to find a solution
                w = [Fraction(0) for _ in range(cols)]
                for row in range(rank_found-1, -1, -1):
                    # Find the column where this row has its pivot
                    pivot_col = -1
                    for c in range(cols-1, -1, -1):
                        if aug[row][c] != 0:
                            pivot_col = c
                            break
                    if pivot_col == -1:
                        continue
                    val = aug[row][cols]
                    for c in range(pivot_col+1, cols):
                        val -= aug[row][c] * w[c]
                    w[pivot_col] = val / aug[row][pivot_col]
                
                for c in range(r):
                    W[k*n+i][c] = w[c]
        
        return W
    
    # Now search over ranks
    # For each rank r, try random U,V and solve for W
    for r in range(25, best_rank):
        if time.time() > deadline - 15:
            break
        found = False
        for attempt in range(100):
            if time.time() > deadline - 15:
                break
            random.seed(1000 + r * 100 + attempt)
            # Random U and V with small integer entries
            scale = 3
            U = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(nm)]
            V = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(mp)]
            
            W = solve_for_W(U, V, r)
            if W is not None:
                # Verify
                if verify(U, V, W):
                    # Convert to integers if possible
                    U_out = [[int(u) if u.denominator == 1 else float(u) for u in row] for row in U]
                    V_out = [[int(v) if v.denominator == 1 else float(v) for v in row] for row in V]
                    W_out = [[int(w) if w.denominator == 1 else float(w) for w in row] for row in W]
                    best = {"U": U_out, "V": V_out, "W": W_out}
                    best_rank = r
                    found = True
                    break
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
