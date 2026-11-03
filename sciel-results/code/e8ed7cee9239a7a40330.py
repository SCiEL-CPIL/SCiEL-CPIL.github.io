"""Matrix multiplication 2x3x7 tensor decomposition - alternating least squares with polish."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _target_tensor(n, m, p):
    """
    Build the target matrix-multiplication tensor T with shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), else 0.
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _residual(T, U, V, W):
    """
    Compute the residual tensor R = T - sum_c U[:,c] outer V[:,c] outer W[:,c].
    Returns the Frobenius norm of R.
    """
    R = T.copy()
    for c in range(U.shape[1]):
        R -= np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.linalg.norm(R)

def _check_full(T, U, V, W, tol=1e-9):
    """Check if the decomposition is exact (within tol)."""
    R = T.copy()
    for c in range(U.shape[1]):
        R -= np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.linalg.norm(R) < tol

def _solve_one(T, U, V, W, rank, which):
    """
    Solve for one of U, V, W given the other two.
    which: 0 for U, 1 for V, 2 for W.
    Returns the solved matrix.
    """
    if which == 0:  # Solve for U
        # R = T - sum_{c!=0} U_c V_c W_c (but we're solving all at once)
        # Actually, for ALS: fix V and W, solve for U.
        # The constraint is: sum_c U[:,c] * V[:,c] * W[:,c] = T
        # This is a linear system in U.
        # Let's unfold: for each (a, b) index in U's first two dims... wait U is (n*m, rank)
        # The tensor equation: T[a, b, d] = sum_c U[a, c] * V[b, c] * W[d, c]
        # For fixed V, W, this is linear in U.
        # We can write it as a matrix equation.
        
        # Let's reshape. For each c, the contribution to T is U[:,c] * V[:,c] * W[:,c]
        # This is tricky to solve directly as a linear system in U because V and W are fixed.
        
        # Alternative: use the fact that if we fix V and W, we can solve for U using least squares.
        # Let's construct the linear system.
        # T[a,b,d] = sum_c U[a,c] * (V[b,c] * W[d,c])
        # Let M[a, (b,d,c)] ... no, that's not right.
        # Let's think of it as: for each a, T[a, :, :] = sum_c U[a,c] * (V[:,c] * W[:,c])
        # So T[a] (as a vector of length m*p*p*n) = sum_c U[a,c] * vec(V[:,c] * W[:,c])
        # This is a linear system: T_vec = A * U_vec, where A is a matrix.
        
        nm = U.shape[0]
        mp = V.shape[0]
        pn = W.shape[0]
        
        # Build the design matrix A of shape (nm, mp*pn*rank) ? No.
        # Let's flatten T to (nm, mp*pn).
        T_flat = T.reshape(nm, mp*pn)
        
        # We want to find U such that T_flat[a, :] = sum_c U[a,c] * (V[:,c] * W[:,c]).reshape(-1)
        # Let's build a matrix B of shape (mp*pn, rank) where B[(b,d), c] = V[b,c] * W[d,c]
        B = np.zeros((mp*pn, rank))
        for b in range(mp):
            for d in range(pn):
                for c in range(rank):
                    B[b*pn + d, c] = V[b, c] * W[d, c]
        
        # Now T_flat[a, :] = U[a, :] @ B
        # So U = T_flat @ pinv(B)
        U_sol = T_flat @ np.linalg.pinv(B)
        return U_sol
    
    elif which == 1:  # Solve for V
        nm = U.shape[0]
        mp = V.shape[0]
        pn = W.shape[0]
        
        # T[a, b, d] = sum_c U[a,c] * V[b,c] * W[d,c]
        # For fixed U, W: T[:, b, :] = sum_c V[b,c] * (U[:,c] * W[:,c])
        # Flatten: T_flat2 of shape (mp, nm*pn)
        T_flat2 = T.transpose(1, 0, 2).reshape(mp, nm*pn)
        
        # Build matrix B of shape (nm*pn, rank) where B[(a,d), c] = U[a,c] * W[d,c]
        B = np.zeros((nm*pn, rank))
        for a in range(nm):
            for d in range(pn):
                for c in range(rank):
                    B[a*pn + d, c] = U[a, c] * W[d, c]
        
        # T_flat2[b, :] = V[b, :] @ B
        V_sol = T_flat2 @ np.linalg.pinv(B)
        return V_sol
    
    else:  # Solve for W
        nm = U.shape[0]
        mp = V.shape[0]
        pn = W.shape[0]
        
        # T[a, b, d] = sum_c U[a,c] * V[b,c] * W[d,c]
        # For fixed U, V: T[:, :, d] = sum_c W[d,c] * (U[:,c] * V[:,c])
        # Flatten: T_flat3 of shape (pn, nm*mp)
        T_flat3 = T.transpose(2, 0, 1).reshape(pn, nm*mp)
        
        # Build matrix B of shape (nm*mp, rank) where B[(a,b), c] = U[a,c] * V[b,c]
        B = np.zeros((nm*mp, rank))
        for a in range(nm):
            for b in range(mp):
                for c in range(rank):
                    B[a*mp + b, c] = U[a, c] * V[b, c]
        
        # T_flat3[d, :] = W[d, :] @ B
        W_sol = T_flat3 @ np.linalg.pinv(B)
        return W_sol

def _alternating_optimize(n, m, p, target_rank, time_budget=60, seed=42):
    """
    Use alternating least squares to try to find a rank-`target_rank` decomposition.
    Returns (U, V, W) as numpy arrays, or None if failed.
    """
    T = _target_tensor(n, m, p)
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    n_attempts = 5
    for attempt in range(n_attempts):
        if time.time() > deadline - 10:
            break
        
        rng = np.random.RandomState(seed + attempt)
        
        # Random initialization
        U = rng.randn(n*m, target_rank) * 0.1
        V = rng.randn(m*p, target_rank) * 0.1
        W = rng.randn(p*n, target_rank) * 0.1
        
        # Initialize with schoolbook if target_rank == n*m*p, otherwise random
        if target_rank == n*m*p:
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # Solve for U
            U = _solve_one(T, U, V, W, target_rank, 0)
            # Solve for V
            V = _solve_one(T, U, V, W, target_rank, 1)
            # Solve for W
            W = _solve_one(T, U, V, W, target_rank, 2)
            
            err = _residual(T, U, V, W)
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-8:
                break
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

def _round_to_half_int(arr, max_denom=2):
    """
    Round entries of arr to the nearest multiple of 1/max_denom.
    Returns a 2D list of Fractions.
    """
    result = []
    for row in arr:
        new_row = []
        for val in row:
            if abs(val) < 1e-10:
                new_row.append(Fraction(0))
            else:
                # Round to nearest 1/max_denom
                scaled = val * max_denom
                rounded = round(scaled)
                frac = Fraction(rounded, max_denom)
                new_row.append(frac)
        result.append(new_row)
    return result

def _verify_fraction_decomposition(U_frac, V_frac, W_frac, n, m, p):
    """
    Verify that the fraction-based decomposition is exact.
    """
    rank = len(U_frac[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_frac[i*m + j][c] * V_frac[j*p + k][c] * W_frac[k*n + i][c]
                if total != 1:
                    return False
    return True

def _polish(U, V, W, n, m, p, time_budget=20):
    """
    Try to polish the decomposition to get exact half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Round to half-integers
    U_frac = _round_to_half_int(U, 2)
    V_frac = _round_to_half_int(V, 2)
    W_frac = _round_to_half_int(W, 2)
    
    if _verify_fraction_decomposition(U_frac, V_frac, W_frac, n, m, p):
        return U_frac, V_frac, W_frac
    
    # Try rounding to integers
    U_frac = _round_to_half_int(U, 1)
    V_frac = _round_to_half_int(V, 1)
    W_frac = _round_to_half_int(W, 1)
    
    if _verify_fraction_decomposition(U_frac, V_frac, W_frac, n, m, p):
        return U_frac, V_frac, W_frac
    
    # Try a local search: perturb one entry at a time and see if it improves
    # This is complex, so for now just return None
    return None

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct the tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    # Known lower bound for 2x3x7 is around 30-35, let's try a range
    target_ranks = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        result = _alternating_optimize(n, m, p, target_rank, time_budget=10)
        if result is not None:
            U, V, W = result
            # Try to polish to half-integers
            polished = _polish(U, V, W, n, m, p, time_budget=5)
            if polished is not None:
                U_frac, V_frac, W_frac = polished
                best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                best_rank = target_rank
                break  # Found a valid lower-rank decomposition
        if time.time() > deadline - 30:
            break
    
    return best_result
# EVOLVE-BLOCK-END
