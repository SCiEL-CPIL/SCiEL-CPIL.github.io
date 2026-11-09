"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

import time
import random
import numpy as np
from fractions import Fraction

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _verify(U, V, W, n, m, p):
    """Check if decomposition is valid.
    Condition: for all i,k,j1,j2: sum_c W[k*n+i,c]*U[i*m+j1,c]*V[j2*p+k,c] = delta_{j1,j2}
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += W_np[k*n+i, c] * U_np[i*m+j1, c] * V_np[j2*p+k, c]
                    expected = 1.0 if j1 == j2 else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _try_rank(n, m, p, target_rank, seed=0, time_limit=10):
    """Try to find a valid decomposition of given rank using structured search.
    Condition: for all i,k,j1,j2: sum_c W[k*n+i,c]*U[i*m+j1,c]*V[j2*p+k,c] = delta_{j1,j2}
    """
    np_rng = np.random.default_rng(seed)
    start = time.time()
    
    for attempt in range(200):
        if time.time() - start > time_limit:
            break
        
        # Random initialization with small integer values
        U_init = np_rng.integers(-2, 3, size=(n*m, target_rank)).astype(float)
        V_init = np_rng.integers(-2, 3, size=(m*p, target_rank)).astype(float)
        
        # For each (k,i), we need W[k*n+i,:] such that:
        # For all j1,j2: sum_c W[k*n+i,c] * U[i*m+j1,c] * V[j2*p+k,c] = delta_{j1,j2}
        # Let M[j1,j2] = sum_c U[i*m+j1,c] * V[j2*p+k,c] * W[k*n+i,c] = delta_{j1,j2}
        # This is: (U_block * diag(w) * V_block^T) = I where U_block is m x R, V_block is m x R
        # i.e., U_block @ diag(w) @ V_block.T = I_m
        # Let P = U_block, Q = V_block. We need P @ diag(w) @ Q.T = I
        # This means P @ diag(w) = Q^{-T} (if Q is invertible)
        # So w_c = (P @ diag(w) @ Q.T)[c,c]... this is complex.
        # 
        # Alternative: P @ diag(w) @ Q.T = I
        # Let X = P @ diag(w). Then X @ Q.T = I, so X = Q^{-T} (if Q invertible)
        # So P @ diag(w) = Q^{-T}, which means for each row r of P:
        # sum_c P[r,c] * w_c * Q.T[c,s] = Q^{-T}[r,s] for all s
        # This is: P @ diag(w) = Q^{-T}
        # So diag(w) = P^{-1} @ Q^{-T} (if P invertible)
        # w_c = (P^{-1} @ Q^{-T})[c,c]
        
        W = np.zeros((p*n, target_rank))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Extract m x R blocks
                P = U_init[i*m:(i+1)*m, :]  # m x R
                Q = V_init[k*p:(k+1)*p + k, :]  # This is wrong indexing
                
                # V has shape (m*p, R). V[j*p+k, c] for j in range(m)
                # So for fixed k, the rows are V[0*p+k], V[1*p+k], ..., V[(m-1)*p+k]
                Q = np.array([V_init[j*p+k, :] for j in range(m)])  # m x R
                
                # We need P @ diag(w) @ Q.T = I_m
                # Check if P and Q are invertible (they are m x R, R >= m)
                if target_rank < m:
                    valid = False
                    break
                
                # P @ diag(w) = Q^{-T}
                # Let's try: solve P @ diag(w) = inv(Q.T)
                # This is equivalent to: for each column c, P[:,c] * w_c contributes
                # Actually: (P @ diag(w))[r,s] = P[r,s] * w_s
                # So P @ diag(w) is element-wise: P[r,s] * w_s
                # We need (P * w[None,:]) @ Q.T = I
                # Let B = P * w[None,:] (each column s of P scaled by w_s)
                # B @ Q.T = I
                # So B = (Q.T)^{-1} = inv(Q.T)
                # P * w[None,:] = inv(Q.T)
                # w_s = inv(Q.T)[r,s] / P[r,s] for any r where P[r,s] != 0
                # This must be consistent across r.
                
                # Better approach: B @ Q.T = I where B[r,s] = P[r,s] * w_s
                # So B = inv(Q.T)
                # P[r,s] * w_s = inv(Q.T)[r,s]
                # w_s = inv(Q.T)[r,s] / P[r,s]
                # For consistency, this must be the same for all r.
                
                try:
                    Q_T_inv = np.linalg.inv(Q.T)  # R x m... wait Q is m x R, Q.T is R x m
                    # Q.T is R x m, which is not square if R != m
                    # This approach doesn't work directly when R > m.
                except:
                    valid = False
                    break
                
                # Let me use a different approach:
                # We need: sum_c P[r,c] * w_c * Q[s,c] = delta_{r,s} for all r,s
                # This is a system of m^2 equations in R unknowns (w_0, ..., w_{R-1})
                # When R >= m^2, we can have a solution.
                # When R < m^2, it's overdetermined.
                
                if target_rank < m * m:
                    valid = False
                    break
                
                # Build the linear system: A @ w = b
                # A has m^2 rows, R columns
                # A[row, c] = P[r,c] * Q[s,c] where row = r*m + s
                # b[row] = delta_{r,s}
                A_mat = np.zeros((m*m, target_rank))
                b_vec = np.zeros(m*m)
                for r in range(m):
                    for s in range(m):
                        row = r * m + s
                        A_mat[row, :] = P[r, :] * Q[s, :]
                        b_vec[row] = 1.0 if r == s else 0.0
                
                # Solve least squares
                try:
                    w, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    valid = False
                    break
                
                # Check exactness
                if np.linalg.norm(A_mat @ w - b_vec) > 1e-8:
                    valid = False
                    break
                
                W[k*n+i, :] = w
        
        if not valid:
            continue
        
        # Round to nearest half-integer and verify with full condition
        U_rounded = np.round(U_init * 2) / 2
        V_rounded = np.round(V_init * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Full verification
        ok = True
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(target_rank):
                            s += W_rounded[k*n+i, c] * U_rounded[i*m+j1, c] * V_rounded[j2*p+k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > 1e-6:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            U_list = [[int(x*2)/2 for x in U_rounded[row]] for row in range(n*m)]
            V_list = [[int(x*2)/2 for x in V_rounded[row]] for row in range(m*p)]
            W_list = [[int(x*2)/2 for x in W_rounded[row]] for row in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # For 3x3x6: m=3, so we need target_rank >= m*m = 9 for the linear system to be solvable
    # Try ranks from 9 up to 53
    # The schoolbook is 54. Let's try to find something better.
    for target_rank in range(9, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = _try_rank(n, m, p, target_rank, seed=seed, time_limit=5)
            if result is not None:
                best = result
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Found a valid rank, continue trying lower ranks
            continue
    
    return best
# EVOLVE-BLOCK-END
