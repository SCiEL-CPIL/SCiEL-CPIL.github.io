"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # The tensor equation: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 
    # We search for the minimum rank r such that a solution exists.
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    def check_solution(U, V, W):
        """Check if U, V, W satisfy the tensor equations."""
        # U: (nm, r), V: (mp, r), W: (pn, r)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    def try_find_solution(rank, num_restarts=50, max_iter=200):
        """Try to find a solution of given rank using random init + local search."""
        np.random.seed(42)
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            # Random initialization
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Use a simple iterative method: fix two matrices, solve for the third
            # This is a rank-1 update approach
            success = False
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                # Fix V and W, solve for U
                # For each (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                # This is: U[i*m+j, :] @ diag(V[j*p+k,:] * W[k*n+i,:]) ... not quite
                # Actually: for fixed (i,j), we need for all k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # Let A_k[r] = V[j*p+k,r] * W[k*n+i,r], then sum_r U[i*m+j,r] * A_k[r] = 1 for all k
                # This is: U[i*m+j, :] @ A = 1 (vector of ones)
                # A is p x rank, so we need U[i*m+j,:] such that U[i*m+j,:] @ A = ones(p)
                
                # Build A: A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # For each (i,j), solve U[i*m+j,:] @ A = ones(p)
                # A is p x rank. If rank >= p, we can solve this.
                
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve U[i*m+j, :] @ A = ones(p)
                        # This is a linear system: A.T @ x = ones(p), where x = U[i*m+j,:]
                        # A.T is rank x p
                        try:
                            x, residuals, rank_sol, sv = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                            U[i*m+j, :] = x
                        except:
                            continue
                            
                # Fix U and W, solve for V
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, rank))
                        for i in range(n):
                            B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        # Solve V[j*p+k, :] @ B = ones(n)
                        try:
                            x, residuals, rank_sol, sv = np.linalg.lstsq(B.T, np.ones(n), rcond=None)
                            V[j*p+k, :] = x
                        except:
                            continue
                            
                # Fix U and V, solve for W
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, rank))
                        for j in range(m):
                            C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        # Solve W[k*n+i, :] @ C = ones(m)
                        try:
                            x, residuals, rank_sol, sv = np.linalg.lstsq(C.T, np.ones(m), rcond=None)
                            W[k*n+i, :] = x
                        except:
                            continue
                            
                # Check if we've converged
                if check_solution(U, V, W):
                    success = True
                    break
                    
            if success:
                return U, V, W
                
        return None
    
    # Try ranks from the lower bound upward
    min_rank = max(nm, mp, pn)  # 21
    for rank in range(min_rank, min_rank + 15):
        if time.time() > deadline - 10:
            break
        result = try_find_solution(rank, num_restarts=30, max_iter=100)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists with integer/half-integer entries
            # Round to nearest 0.5
            U_out = [[round(U[i][j] * 2) / 2 for j in range(rank)] for i in range(nm)]
            V_out = [[round(V[i][j] * 2) / 2 for j in range(rank)] for i in range(mp)]
            W_out = [[round(W[i][j] * 2) / 2 for j in range(rank)] for i in range(pn)]
            
            # Verify the rounded solution
            U_np = np.array(U_out)
            V_np = np.array(V_out)
            W_np = np.array(W_out)
            if check_solution(U_np, V_np, W_np):
                return {"U": U_out, "V": V_out, "W": W_out}
            else:
                # Return the unrounded solution if it's close enough
                # Actually, the task says integer/half-integer entries
                # Let's try rounding to integers first, then half-integers
                U_int = [[round(U[i][j]) for j in range(rank)] for i in range(nm)]
                V_int = [[round(V[i][j]) for j in range(rank)] for i in range(mp)]
                W_int = [[round(W[i][j]) for j in range(rank)] for i in range(pn)]
                U_np = np.array(U_int, dtype=float)
                V_np = np.array(V_int, dtype=float)
                W_np = np.array(W_int, dtype=float)
                if check_solution(U_np, V_np, W_np):
                    return {"U": U_int, "V": V_int, "W": W_int}
                else:
                    # Return half-integer version
                    return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
