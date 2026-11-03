import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _build_tensor(n, m, p):
    """Return the matrix multiplication tensor as a 3D numpy array of floats."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W (numpy arrays) form a valid rank-r decomposition."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(arr, tol=1e-6):
    """Round array entries to nearest half-integer if close enough."""
    rounded = np.round(arr * 2) / 2
    if np.max(np.abs(arr - rounded)) < tol:
        return rounded
    return arr

def _schoolbook(n, m, p):
    """Return schoolbook decomposition as list-of-lists."""
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
    """Check if U,V,W (list-of-lists or numpy) form a valid rank-r decomposition of matrix mult."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    return _verify_exact(U_arr, V_arr, W_arr, n, m, p)

def _gaussian_pivots(mat):
    """
    Perform exact Gaussian elimination on a matrix (list of lists of Fraction).
    Return list of pivot row indices and the number of pivots (= rank).
    """
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        return [], 0
    
    # Work on a copy
    A = [row[:] for row in mat]
    n_rows = len(A)
    n_cols = len(A[0])
    
    pivot_rows = []
    rank = 0
    for col in range(n_cols):
        # Find pivot
        pivot_row = -1
        for row in range(rank, n_rows):
            if A[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        
        # Swap
        if pivot_row != rank:
            A[rank], A[pivot_row] = A[pivot_row], A[rank]
        
        pivot_rows.append(rank)
        pivot_val = A[rank][col]
        
        # Eliminate below
        for row in range(rank + 1, n_rows):
            if A[row][col] != 0:
                factor = A[row][col] / pivot_val
                for c in range(col, n_cols):
                    A[row][c] -= factor * A[rank][c]
        
        rank += 1
        if rank == n_rows:
            break
    
    return pivot_rows, rank

def construct():
    """
    Search for a lower-rank exact tensor decomposition using NLP (SLSQP) to minimize
    reconstruction error. Fall back to schoolbook if no improvement found.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    T = _build_tensor(n, m, p)
    
    # Try ranks from the theoretical lower bound up to schoolbook
    lower_bound = max(nm, mp, pn)  # 21
    
    for r in range(lower_bound, min(best_rank, lower_bound + 8)):
        if time.time() > deadline:
            break
        
        # Use SLSQP to minimize Frobenius norm of reconstruction error
        # Variables: flatten U (nm*r), V (mp*r), W (pn*r)
        total_vars = nm * r + mp * r + pn * r
        
        # Initialize with a structured guess based on schoolbook
        U0 = np.zeros((nm, r))
        V0 = np.zeros((mp, r))
        W0 = np.zeros((pn, r))
        
        # Try to use first r of the 42 schoolbook terms
        for l in range(min(r, n * m * p)):
            i = l // (m * p)
            j = (l // p) % m
            k = l % p
            U0[i * m + j, l] = 1.0
            V0[j * p + k, l] = 1.0
            W0[k * n + i, l] = 1.0
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def objective(x):
            U = x[:nm*r].reshape(nm, r)
            V = x[nm*r:nm*r+mp*r].reshape(mp, r)
            W = x[nm*r+mp*r:].reshape(pn, r)
            # Compute reconstruction error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err += (s - 1.0) ** 2
            return err
        
        def gradient(x):
            U = x[:nm*r].reshape(nm, r)
            V = x[nm*r:nm*r+mp*r].reshape(mp, r)
            W = x[nm*r+mp*r:].reshape(pn, r)
            
            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        d = 2.0 * (s - 1.0)
                        gU[i*m+j] += d * (V[j*p+k] * W[k*n+i])
                        gV[j*p+k] += d * (U[i*m+j] * W[k*n+i])
                        gW[k*n+i] += d * (U[i*m+j] * V[j*p+k])
            
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, x0, method='SLSQP', jac=gradient,
                             options={'maxiter': 200, 'ftol': 1e-10})
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        x_opt = result.x
        U_opt = x_opt[:nm*r].reshape(nm, r)
        V_opt = x_opt[nm*r:nm*r+mp*r].reshape(mp, r)
        W_opt = x_opt[nm*r+mp*r:].reshape(pn, r)
        
        # Round to half-integers
        U_rounded = _round_to_half_integer(U_opt)
        V_rounded = _round_to_half_integer(V_opt)
        W_rounded = _round_to_half_integer(W_opt)
        
        # Verify exactness
        if _verify_exact(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-9):
            # Convert to list-of-lists
            U_list = [[float(U_rounded[i, l]) for l in range(r)] for i in range(nm)]
            V_list = [[float(V_rounded[i, l]) for l in range(r)] for i in range(mp)]
            W_list = [[float(W_rounded[i, l]) for l in range(r)] for i in range(pn)]
            
            # Double-check with exact verification
            if _verify(U_list, V_list, W_list, n, m, p):
                return {'U': U_list, 'V': V_list, 'W': W_list}
        
        # Also try without rounding (in case entries are already close to half-integers)
        if _verify_exact(U_opt, V_opt, W_opt, n, m, p, tol=1e-8):
            U_list = [[float(U_opt[i, l]) for l in range(r)] for i in range(nm)]
            V_list = [[float(V_opt[i, l]) for l in range(r)] for i in range(mp)]
            W_list = [[float(W_opt[i, l]) for l in range(r)] for i in range(pn)]
            if _verify(U_list, V_list, W_list, n, m, p):
                return {'U': U_list, 'V': V_list, 'W': W_list}
    
    return best
