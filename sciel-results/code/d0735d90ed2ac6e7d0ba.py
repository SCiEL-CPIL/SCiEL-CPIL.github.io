import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 140  # Leave 10s margin before 150s kill

    # Precompute the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W, R):
        """Verify exact decomposition with integer/half-integer entries."""
        for arr in [U, V, W]:
            for val in arr:
                if not (val * 2 == int(val * 2)):
                    return False
        Uf = np.array(U, dtype=np.float64)
        Vf = np.array(V, dtype=np.float64)
        Wf = np.array(W, dtype=np.float64)
        T_rec = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
        if not np.allclose(T_rec, T, atol=1e-10):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0)
                    for r in range(R):
                        s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if s != 1:
                        return False
        return True

    # Build schoolbook as fallback
    best_rank = n * m * p
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school, best_rank)

    # Constructive approach: Use structured decomposition
    # The key insight: for 2x3x6, we can use the fact that the tensor
    # T_{i,j,k} = delta at position (i*m+j, j*p+k, k*n+i)
    # 
    # Strategy: Use a decomposition where we share factors across j values
    # For each pair (i,k), the factor W[k*n+i] appears in all j blocks
    # We can use this to reduce rank.
    #
    # Let's try rank 18: 3 blocks of 6, sharing structure
    
    # Approach: For each k in 0..5, use 3 terms (one per j) but with shared W factors
    # Actually, let's think differently.
    #
    # The tensor has 36 non-zero entries. We want to decompose it as sum of R rank-1 tensors.
    # A rank-1 tensor is u ⊗ v ⊗ w where u is in R^{nm}, v in R^{mp}, w in R^{pn}
    #
    # Let's try a constructive decomposition with rank 18:
    # For each j (0,1,2) and each k (0..5), we have 2 entries (for i=0,1)
    # Total: 3 * 6 * 2 = 36 entries
    #
    # Idea: Use the fact that for fixed (j,k), the two entries (i=0,1) can be
    # represented with 2 rank-1 terms, but we might be able to share across j.
    #
    # Let's try: For each k, use 6 terms (2 per j, but shared W)
    # That gives 6 * 6 = 36, no improvement.
    #
    # Alternative: Use polynomial/interpolation ideas
    # For 2x6 matrix mult, we can use Karatsuba-like methods
    # 
    # Let me try a different structural approach:
    # Group by i (row of first matrix)
    # For i=0: entries at (j, j*p+k, k*2+0) = (j, j*p+k, 2k)
    # For i=1: entries at (j, j*p+k, k*2+1) = (j, j*p+k, 2k+1)
    #
    # So W factors are at even positions (for i=0) and odd positions (for i=1)
    # This suggests we can separate by parity!
    #
    # Let's define:
    # For i=0: W_0[c] = 1 if c is even (c=2k), else 0
    # For i=1: W_1[c] = 1 if c is odd (c=2k+1), else 0
    #
    # Then for each (i, j, k):
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Let's try rank 12: 2 (for i) * 6 (for k) terms, shared across j
    # But we need 3 different j values...
    
    # Let me try a systematic constructive approach with rank R
    # and verify it works
    
    def try_constructive(R):
        """Try to construct a rank-R decomposition using structure."""
        if R < 12:
            return None
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        # Strategy: For each (i, k) pair, assign a term index
        # i in 0..1, k in 0..5 => 12 pairs
        # For each j in 0..2, we need the term to contribute to (i*m+j, j*p+k, k*n+i)
        #
        # If we use one term per (i,k) pair, that's 12 terms, but each term
        # would need to work for all 3 j values simultaneously, which is impossible
        # with rank-1 tensors.
        #
        # So we need at least 3 terms per (i,k) pair if we don't share.
        # That's 36 total.
        #
        # To reduce, we need to share across j. But rank-1 tensors can't easily
        # share across different j in a way that works for all 3.
        #
        # Let's try a different approach: use the actual algebraic structure.
        # 
        # The matrix multiplication (2x3)(3x6) can be viewed as:
        # C = A * B where A is 2x3, B is 3x6
        # C[i,k] = sum_j A[i,j] * B[j,k]
        #
        # This is equivalent to computing 2x6 = 12 products, each being a sum of 3 products.
        # Total "scalar multiplications" = 12 * 3 = 36.
        #
        # To reduce, we can use the fact that some combinations can be shared.
        # For example, if we compute (A[i,0] + A[i,1]) * (B[0,k] + B[1,k])
        # we get A[i,0]*B[0,k] + A[i,0]*B[1,k] + A[i,1]*B[0,k] + A[i,1]*B[1,k]
        # which covers 4 of the 9 products for row i.
        #
        # This is the Karatsuba idea. Let's apply it.
        #
        # For 3x3 matrix mult, we can do it in 7 multiplications (Strassen).
        # For 2x3 by 3x6, we can potentially do better than 36.
        
        # Let's use a known result: the rank of matrix multiplication tensor
        # M_{m,n,p} is at most m*n*p, and for small cases we can find better.
        #
        # For 2x3x6, let's try to find a decomposition with rank 24 or less.
        
        # I'll use a heuristic: try to find a decomposition by solving a system
        # of equations.
        
        return None
    
    # Let's try a direct constructive approach with specific patterns
    # 
    # Idea: Use the fact that the output index c = k*n + i depends on k and i,
    # not on j. So for fixed (i,k), all j values write to the same c.
    # This means we can potentially share W factors.
    #
    # Let's try: For each (i, k), use 3 terms (one per j) but with the same W factor.
    # That still gives 36 terms.
    #
    # Alternative: Can we use fewer terms by having each term contribute to
    # multiple (i,j,k) entries?
    #
    # A rank-1 term u⊗v⊗w contributes u[a]*v[b]*w[c] to T[a,b,c].
    # For it to contribute to multiple target entries, we need:
    # u[i*m+j]*v[j*p+k]*w[k*n+i] = 1 for multiple (i,j,k)
    #
    # This is hard to achieve with integer entries.
    
    # Let's try a completely different approach: use linear algebra to find
    # the minimal rank by solving a system of equations.
    
    # For rank R, we need to find U (nm x R), V (mp x R), W (pn x R) such that
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all (a,b,c)
    #
    # This is a system of (nm*mp*pn) equations in (nm*R + mp*R + pn*R) unknowns.
    # For R=18: 36*18 + 18*18 + 12*18 = 648 + 324 + 216 = 1188 unknowns
    # Number of equations: 36*18*12 = 7776
    # This is underdetermined in a different way...
    
    # Actually, the equations are nonlinear (trilinear). We need a different approach.
    
    # Let's use a iterative approach: fix two matrices, solve for the third.
    
    def solve_for_W(U, V, R):
        """Given U and V, solve for W such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]"""
        W = np.zeros((p * n, R))
        
        # For each c, we need: sum_r (sum_{a,b} T[a,b,c] * U[a,r] * V[b,r]) = W[c,r] * (something)
        # This doesn't directly work because of the trilinear structure.
        
        # Instead, for each (a,b) pair where T[a,b,c] is non-zero:
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        #
        # For fixed (a,b), this gives us a linear equation in W[c,·] for each c where T[a,b,c]!=0
        # But T[a,b,c] is only non-zero for one c (c = k*n+i where a=i*m+j, b=j*p+k)
        
        # So for each (a,b) with T[a,b,c]=1:
        # 1 = sum_r U[a,r] * V[b,r] * W[c,r]
        #
        # This is one equation per (a,b) pair, giving us nm*mp = 648 equations
        # in pn*R = 12*R unknowns.
        #
        # For R=18: 12*18 = 216 unknowns, 648 equations. Overdetermined.
        
        # Let's set up the linear system properly.
        # For each (a, b, c) where T[a,b,c] = 1:
        # sum_r U[a,r] * V[b,r] * W[c,r] = 1
        #
        # This is linear in W if U and V are fixed.
        # We have 36 such equations (one per non-zero T entry).
        # Unknowns: W[c,r] for c in 0..11, r in 0..R-1 => 12*R unknowns.
        
        # For R=3: 36 unknowns, 36 equations. Might be solvable!
        # For R=6: 72 unknowns, 36 equations. Underdetermined.
        
        # Let's try R=3 first.
        
        if R == 3:
            # 36 equations, 36 unknowns
            A_mat = np.zeros((36, 36))
            b_vec = np.zeros(36)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        
                        for r in range(R):
                            col = c * R + r
                            A_mat[eq_idx, col] = U[a, r] * V[b, r]
                        b_vec[eq_idx] = 1
                        eq_idx += 1
            
            try:
                W_sol = np.linalg.solve(A_mat, b_vec)
                return W_sol
            except np.linalg.LinAlgError:
                return None
        
        elif R == 6:
            # 36 equations, 72 unknowns. Use least squares.
            A_mat = np.zeros((36, 72))
            b_vec = np.zeros(36)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        
                        for r in range(R):
                            col = c * R + r
                            A_mat[eq_idx, col] = U[a, r] * V[b, r]
                        b_vec[eq_idx] = 1
                        eq_idx += 1
            
            try:
                W_sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                # Check if it's an exact solution
                if np.allclose(A_mat @ W_sol, b_vec, atol=1e-10):
                    return W_sol
            except:
                pass
            return None
        
        else:
            return None
    
    # Try to find a solution with small R
    # First, try R=3 with random U, V
    
    import random
    random.seed(123)
    
    for R in [3, 4, 5, 6, 8, 10, 12]:
        if time.time() - start_time > timeout:
            break
        
        found = False
        for attempt in range(100):
            if time.time() - start_time > timeout:
                break
            
            # Random initialization for U and V
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            
            for r in range(R):
                # Sparse random initialization
                for _ in range(2):
                    U[random.randint(0, n * m - 1), r] = random.choice([-1, 1])
                for _ in range(2):
                    V[random.randint(0, m * p - 1), r] = random.choice([-1, 1])
            
            W = solve_for_W(U, V, R)
            if W is None:
                continue
            
            # Verify
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_rec, T, atol=1e-10):
                # Check for integer/half-integer entries
                valid = True
                for arr in [U, V, W]:
                    for val in arr:
                        if abs(val * 2 - round(val * 2)) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    U_list = [[int(round(U[a, r])) for r in range(R)] for a in range(n * m)]
                    V_list = [[int(round(V[b, r])) for r in range(R)] for b in range(m * p)]
                    W_list = [[int(round(W[c, r])) for r in range(R)] for c in range(p * n)]
                    if verify(U_list, V_list, W_list, R):
                        best_decomp = (U_list, V_list, W_list, R)
                        found = True
                        break
        
        if found:
            break
    
    # If we found something better than schoolbook, use it
    if best_decomp[3] < n * m * p:
        return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
    
    # Otherwise, return schoolbook
    return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
