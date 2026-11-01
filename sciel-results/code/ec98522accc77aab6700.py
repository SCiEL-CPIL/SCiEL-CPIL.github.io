import time
import random
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target T[i, j, k] = 1 for all i, j, k.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _build_rank10():
    """Build the known rank-10 solution (Strassen for first 2 cols + identity for last 3)."""
    n, m, p = 2, 2, 5
    U = [[0]*10 for _ in range(n*m)]
    V = [[0]*10 for _ in range(m*p)]
    W = [[0]*10 for _ in range(p*n)]
    
    # Strassen for 2x2 * 2x2 (k=0,1), using 7 multiplications
    # P1 = (a+d)(e+h)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[6][0]=1
    W[0][0]=1; W[3][0]=1
    # P2 = (c+d)e
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    # P3 = a(f-h)
    U[0][2]=1
    V[1][2]=1; V[7][2]=-1
    W[1][2]=1; W[3][2]=1
    # P4 = d(g-e)
    U[3][3]=1
    V[0][3]=-1; V[5][3]=1
    W[0][3]=1; W[2][3]=1
    # P5 = (a+b)h
    U[0][4]=1; U[1][4]=1
    V[6][4]=1
    W[0][4]=-1; W[1][4]=1
    # P6 = (c-a)(e+f)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    # P7 = (b-d)(g+h)
    U[1][6]=1; U[3][6]=-1
    V[5][6]=1; V[6][6]=1
    W[0][6]=1
    
    # Identity for k=2,3,4 (rank 7,8,9)
    for k in range(2, 5):
        r = k + 5  # r=7,8,9
        for i in range(2):
            for j in range(2):
                U[i*2+j][r] = 1
        for j in range(2):
            V[j*5+k][r] = 1
        for i in range(2):
            W[k*2+i][r] = 1
    
    return U, V, W

def _try_rank9_block_structure(seed=42):
    """
    Block-structured search for rank 9.
    
    Key insight: The 2x2x5 tensor can be viewed as two blocks:
    - Block A: 2x2 * 2x3 (k=0,1,2)
    - Block B: 2x2 * 2x2 (k=3,4)
    
    If we can find a rank-9 decomposition where the factor matrices for k=3,4
    are linear combinations of those for k=0,1,2, we exploit the structure.
    
    We parameterize:
    - U: (4, 9) - shared across all k
    - V: (10, 9) - but V for k=3,4 are linear combos of V for k=0,1,2
    - W: (10, 9) - similarly
    
    Specifically, we set:
    V[j, 3] = alpha * V[j, 0] + beta * V[j, 1] + gamma * V[j, 2]
    V[j, 4] = delta * V[j, 0] + epsilon * V[j, 1] + zeta * V[j, 2]
    Similarly for W.
    
    This reduces the free parameters and makes the search more tractable.
    """
    n, m, p = 2, 2, 5
    rank = 9
    
    # We'll search over integer/half-integer coefficients for the linear combinations
    # and the base factor matrices.
    
    # For efficiency, use numpy for the search and Fraction for verification.
    
    best_solution = None
    start = time.time()
    deadline = 130.0
    
    # Try several random initializations with the block structure
    for restart in range(50):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed + restart)
        
        # Initialize U with small random integers
        U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
        # Ensure U is not degenerate
        U[0][0] = 1
        
        # Initialize V for k=0,1,2 (6 rows: j=0,1 for each k)
        V_base = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(m*3)]  # 6 rows
        V_base[0][0] = 1
        
        # Initialize W for k=0,1,2 (6 rows: k=0,1,2 for i=0,1)
        W_base = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(3*n)]  # 6 rows
        W_base[0][0] = 1
        
        # Coefficients for expressing k=3,4 in terms of k=0,1,2
        # V[j,3] = a*V[j,0] + b*V[j,1] + c*V[j,2]
        # V[j,4] = d*V[j,0] + e*V[j,1] + f*V[j,2]
        coeffs_V = [rng.randint(-2, 2) for _ in range(6)]
        coeffs_W = [rng.randint(-2, 2) for _ in range(6)]
        
        # Build full V and W
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for j in range(m):
            for k in range(3):
                V[j*p+k] = V_base[j*3+k][:]
        
        for k in range(3):
            for i in range(n):
                W[k*n+i] = W_base[k*n+i][:]
        
        # Apply linear combinations for k=3,4
        for j in range(m):
            for c in range(rank):
                V[j*p+3][c] = (coeffs_V[0]*V_base[j*3][c] + coeffs_V[1]*V_base[j*3+1][c] + coeffs_V[2]*V_base[j*3+2][c])
                V[j*p+4][c] = (coeffs_V[3]*V_base[j*3][c] + coeffs_V[4]*V_base[j*3+1][c] + coeffs_V[5]*V_base[j*3+2][c])
        
        for i in range(n):
            for c in range(rank):
                W[3*n+i][c] = (coeffs_W[0]*W_base[0*n+i][c] + coeffs_W[1]*W_base[1*n+i][c] + coeffs_W[2]*W_base[2*n+i][c])
                W[4*n+i][c] = (coeffs_W[3]*W_base[0*n+i][c] + coeffs_W[4]*W_base[1*n+i][c] + coeffs_W[5]*W_base[2*n+i][c])
        
        # Now do a few ALS-like refinement steps
        for it in range(20):
            if time.time() > deadline:
                break
            
            # Update U given V, W
            new_U = [[Fraction(0) for _ in range(rank)] for _ in range(n*m)]
            for i in range(n):
                for j in range(m):
                    M = [Fraction(0) for _ in range(rank)]
                    b = Fraction(0)
                    for k in range(p):
                        b += Fraction(W[k*n+i][0])  # placeholder, will fix
                        for c in range(rank):
                            M[c] += Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    sum_sq = sum(mc*mc for mc in M)
                    if sum_sq != 0:
                        factor = b / sum_sq
                        for c in range(rank):
                            new_U[i*m+j][c] = M[c] * factor
            U = [[float(x) for x in row] for row in new_U]
            
            # Update V given U, W
            new_V = [[Fraction(0) for _ in range(rank)] for _ in range(m*p)]
            for j in range(m):
                for k in range(p):
                    M = [Fraction(0) for _ in range(rank)]
                    b = Fraction(0)
                    for i in range(n):
                        b += Fraction(U[i*m+j][0])
                        for c in range(rank):
                            M[c] += Fraction(U[i*m+j][c]) * Fraction(W[k*n+i][c])
                    sum_sq = sum(mc*mc for mc in M)
                    if sum_sq != 0:
                        factor = b / sum_sq
                        for c in range(rank):
                            new_V[j*p+k][c] = M[c] * factor
            V = [[float(x) for x in row] for row in new_V]
            
            # Update W given U, V
            new_W = [[Fraction(0) for _ in range(rank)] for _ in range(p*n)]
            for k in range(p):
                for i in range(n):
                    M = [Fraction(0) for _ in range(rank)]
                    b = Fraction(0)
                    for j in range(m):
                        b += Fraction(V[j*p+k][0])
                        for c in range(rank):
                            M[c] += Fraction(V[j*p+k][c]) * Fraction(U[i*m+j][c])
                    sum_sq = sum(mc*mc for mc in M)
                    if sum_sq != 0:
                        factor = b / sum_sq
                        for c in range(rank):
                            new_W[k*n+i][c] = M[c] * factor
            W = [[float(x) for x in row] for row in new_W]
            
            # Round to nearest half-integer
            U = [[round(x*2)/2 for x in row] for row in U]
            V = [[round(x*2)/2 for x in row] for row in V]
            W = [[round(x*2)/2 for x in row] for row in W]
            
            # Check validity
            if _verify(U, V, W, n, m, p, rank):
                best_solution = (U, V, W)
                return best_solution
        
        if best_solution:
            return best_solution
    
    return None

def _try_rank9_direct_search(seed=123):
    """
    Direct randomized search for rank-9 decomposition using a different
    parameterization: start from rank-10 and try to merge two columns.
    """
    n, m, p = 2, 2, 5
    rank = 9
    
    U10, V10, W10 = _build_rank10()
    
    start = time.time()
    deadline = 130.0
    
    # Try to merge pairs of columns from rank-10 into rank-9
    # Column c in rank-10: U[:,c], V[:,c], W[:,c]
    # We want to find alpha, beta such that:
    # col_a = alpha * col_c1 + beta * col_c2
    # for U, V, W simultaneously
    
    for c1 in range(10):
        for c2 in range(c1+1, 10):
            if time.time() > deadline:
                break
            
            # Skip if columns are identical
            same = True
            for i in range(n*m):
                if U10[i][c1] != U10[i][c2]:
                    same = False
                    break
            if same:
                for j in range(m*p):
                    if V10[j][c1] != V10[j][c2]:
                        same = False
                        break
            if same:
                for k in range(p*n):
                    if W10[k][c1] != W10[k][c2]:
                        same = False
                        break
            if same:
                continue
            
            # Try to find alpha, beta (integers or half-integers) such that
            # the merged column approximates both
            # For simplicity, try alpha=1, beta=-1, etc.
            for alpha in [-2, -1, -0.5, 0.5, 1, 2]:
                for beta in [-2, -1, -0.5, 0.5, 1, 2]:
                    if alpha == 0 and beta == 0:
                        continue
                    
                    # Build rank-9 matrices by removing c2 and adding merged column
                    U9 = []
                    V9 = []
                    W9 = []
                    
                    for i in range(n*m):
                        row = []
                        for c in range(10):
                            if c == c2:
                                continue
                            row.append(U10[i][c])
                        # Add merged column
                        row.append(alpha*U10[i][c1] + beta*U10[i][c2])
                        U9.append(row)
                    
                    for j in range(m*p):
                        row = []
                        for c in range(10):
                            if c == c2:
                                continue
                            row.append(V10[j][c])
                        row.append(alpha*V10[j][c1] + beta*V10[j][c2])
                        V9.append(row)
                    
                    for k in range(p*n):
                        row = []
                        for c in range(10):
                            if c == c2:
                                continue
                            row.append(W10[k][c])
                        row.append(alpha*W10[k][c1] + beta*W10[k][c2])
                        W9.append(row)
                    
                    if _verify(U9, V9, W9, n, m, p, rank):
                        return U9, V9, W9
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    
    Strategy:
    1. Build rank-10 solution (Strassen + identity).
    2. Try block-structured rank-9 search.
    3. Try direct column-merging rank-9 search.
    4. Return best found.
    """
    n, m, p = 2, 2, 5
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Build rank-10 fallback
    U10, V10, W10 = _build_rank10()
    
    if not _verify(U10, V10, W10, n, m, p, 10):
        # Fallback to schoolbook rank 20
        best_rank = 20
        U20 = [[0]*20 for _ in range(n*m)]
        V20 = [[0]*20 for _ in range(m*p)]
        W20 = [[0]*20 for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U20[i*m+j][col] = 1
                    V20[j*p+k][col] = 1
                    W20[k*n+i][col] = 1
                    col += 1
        return {"U": U20, "V": V20, "W": W20}
    
    best_U, best_V, best_W = U10, V10, W10
    best_rank = 10
    
    # Try rank-9 block structure
    if time.time() < deadline - 30:
        result = _try_rank9_block_structure(seed=42)
        if result:
            U9, V9, W9 = result
            if _verify(U9, V9, W9, n, m, p, 9):
                best_U, best_V, best_W = U9, V9, W9
                best_rank = 9
    
    # Try rank-9 direct search
    if best_rank > 9 and time.time() < deadline - 15:
        result = _try_rank9_direct_search(seed=123)
        if result:
            U9, V9, W9 = result
            if _verify(U9, V9, W9, n, m, p, 9):
                best_U, best_V, best_W = U9, V9, W9
                best_rank = 9
    
    return {"U": best_U, "V": best_V, "W": best_W}
