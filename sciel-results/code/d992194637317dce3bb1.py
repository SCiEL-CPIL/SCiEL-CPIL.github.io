import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    def check_solution(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def to_half_integers(U, V, W):
        """Round to nearest half-integer."""
        def round_half(x):
            return int(round(x * 2)) / 2.0
        U2 = [[round_half(U[i][r]) for r in range(len(U[0]))] for i in range(len(U))]
        V2 = [[round_half(V[i][r]) for r in range(len(V[0]))] for i in range(len(V))]
        W2 = [[round_half(W[i][r]) for r in range(len(W[0]))] for i in range(len(W))]
        return U2, V2, W2
    
    def als_decompose(rank, max_iters=200, tol=1e-10):
        """ALS to find rank-r decomposition."""
        np.random.seed(int(time.time() * 1000) % 2**31)
        
        # Initialize
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            # The equation is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
            # Let A_k[r] = V[j*p+k,r] * W[k*n+i,r], then sum_r U[i*m+j,r] * A_k[r] = 1 for all k
            # This is a linear system: U[i*m+j, :] @ A_k = 1 for all k
            # Stack: for fixed (i,j), we have p equations in rank unknowns
            # M[i,j] = [A_0; A_1; ...; A_{p-1}] is p x rank
            # We want M[i,j] @ U[i*m+j, :]^T = 1 (vector of ones)
            
            for i in range(n):
                for j in range(m):
                    # Build the p x rank matrix
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Solve A @ x = 1 (least squares)
                    try:
                        x, residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
            # Let B_i[r] = U[i*m+j,r] * W[k*n+i,r], then sum_r V[j*p+k,r] * B_i[r] = 1 for all i
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    try:
                        x, residuals, rank_s, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
            # Let C_j[r] = U[i*m+j,r] * V[j*p+k,r], then sum_r W[k*n+i,r] * C_j[r] = 1 for all j
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    try:
                        x, residuals, rank_s, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass
            
            # Check convergence
            if it % 10 == 0:
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_err = max(max_err, abs(s - 1))
                if max_err < tol:
                    break
        
        # Check final solution
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = max(max_err, abs(s - 1))
        
        if max_err < 1e-8:
            return U, V, W, max_err
        return None
    
    # Try ranks from low to high
    for rank in range(50, 101):
        if time.time() > deadline:
            break
        
        result = als_decompose(rank)
        if result is not None:
            U, V, W, err = result
            # Round to half-integers
            U_list = [[float(U[i][r]) for r in range(rank)] for i in range(nm)]
            V_list = [[float(V[i][r]) for r in range(rank)] for i in range(mp)]
            W_list = [[float(W[i][r]) for r in range(rank)] for i in range(pn)]
            U_h, V_h, W_h = to_half_integers(U_list, V_list, W_list)
            
            if check_solution(U_h, V_h, W_h, rank):
                return {"U": U_h, "V": V_h, "W": W_h}
            
            # If rounding didn't work, try the unrounded version if it's close enough
            # But the problem requires integer/half-integer, so we can't return non-half-integer
            # Try a few more ALS iterations or different initialization
    
    return best
