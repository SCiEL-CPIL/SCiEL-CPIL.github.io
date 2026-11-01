"""Improved program for matrix multiplication 2x2 by 2x11 rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 11]
DEADLINE = 135  # safety margin before 150s kill

def _verify(U, V, W):
    """Verify that U, V, W form an exact tensor decomposition of matrix mult."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the matrix multiplication tensor T[i,j,k] -> product
    # The tensor for multiplying (n x m) by (m x p) gives (n x p) result
    # T[i, j, k] = 1 if this term contributes to C[i,k] via A[i,j]*B[j,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == T[i,j,k]
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                expected = 1.0
                if abs(val - expected) > 1e-9:
                    return False
    return True

def _make_schoolbook():
    """Return the rank n*m*p schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_random_search(max_rank, time_budget):
    """Try to find a rank decomposition below max_rank using random initialization + refinement."""
    n, m, p = DIMENSIONS
    start = time.time()
    
    for trial_rank in range(2 * n * p, max_rank):  # lower bound is 2*n*p=44? No...
        # Actually lower bound for 2x2 * 2x11: rank >= max(n*m, m*p, p*n) = max(4,22,22)=22
        # But also rank >= n*p = 22 for the output dimension... 
        # Strassen lower bound: rank >= n*m*p / (something). 
        # Simple lower bound: rank >= max(n*m, m*p, n*p) = 22
        pass
    
    # Let's try ranks from 22 up to max_rank-1
    for trial_rank in range(22, max_rank):
        if time.time() - start > time_budget:
            break
        for attempt in range(50):
            if time.time() - start > time_budget:
                break
            np.random.seed(attempt * 1000 + trial_rank)
            U = np.random.randn(n * m, trial_rank)
            V = np.random.randn(m * p, trial_rank)
            W = np.random.randn(p * n, trial_rank)
            
            # Iterative refinement: alternate fixing two matrices and solving for the third
            for it in range(200):
                if time.time() - start > time_budget:
                    break
                
                # Fix V, W; solve for U
                # For each (i,j): U[i*m+j, :] should satisfy sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all k
                # This is a linear system: for fixed (i,j), we need vector u such that
                # M_k = sum_r u_r * V[j*p+k,r]*W[k*n+i,r] = 1 for all k
                # This is: (V[j,:] * W_col) contracted with u = 1
                # Let A[k,r] = V[j*p+k, r] * W[k*n+i, r]
                # Then sum_r u[r] * A[k,r] = 1 for all k
                # A^T u = 1 (if A is p x rank, we need A^T (rank x p) times... wait)
                # Actually: for each k: dot(u, A[k,:]) = 1
                # So A @ u = ones(p)  where A is p x rank
                # If rank >= p, this is overdetermined; use least squares
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, trial_rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Fix U, W; solve for V
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n * m, trial_rank))  # wait, need to be careful
                        # For fixed (j,k): need v such that sum_r U[i*m+j,r]*v_r*W[k*n+i,r] = 1 for all i
                        # Let C[i,r] = U[i*m+j,r]*W[k*n+i,r]
                        # sum_r C[i,r]*v_r = 1 for all i in range(n)
                        # C @ v = ones(n)
                        C = np.zeros((n, trial_rank))
                        for i in range(n):
                            C[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(C, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Fix U, V; solve for W
                for k in range(p):
                    for i in range(n):
                        D = np.zeros((m, trial_rank))
                        for j in range(m):
                            D[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(D, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += abs(val - 1.0)
                if err < 1e-8:
                    # Found a solution! Convert to list of lists with rounding
                    U_list = [[float(round(x, 6)) for x in row] for row in U]
                    V_list = [[float(round(x, 6)) for x in row] for row in V]
                    W_list = [[float(round(x, 6)) for x in row] for row in W]
                    if _verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list, "rank": trial_rank}
                    # Try without rounding
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if _verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list, "rank": trial_rank}
    
    return None

def construct():
    start = time.time()
    
    # Start with schoolbook (rank 44)
    best = _make_schoolbook()
    best_rank = 44
    
    # Try to find lower rank
    # Allocate ~120 seconds for search
    time_budget = 115  # leave 20s margin
    
    result = _try_random_search(best_rank, time_budget)
    if result is not None and result["rank"] < best_rank:
        best_rank = result["rank"]
        best = {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Final time check
    if time.time() - start > DEADLINE:
        pass
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
