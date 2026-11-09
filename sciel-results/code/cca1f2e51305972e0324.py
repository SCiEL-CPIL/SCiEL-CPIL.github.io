"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _to_rational_array(mat, tol=1e-8):
    """Convert a numeric matrix to a matrix of Fractions (exact rationals)."""
    rows, cols = mat.shape
    result = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            val = mat[i, j]
            if abs(val) < tol:
                result[i][j] = Fraction(0)
                continue
            found = False
            for den in range(1, 201):
                num = round(val * den)
                if abs(val - num / den) < tol:
                    result[i][j] = Fraction(num, den)
                    found = True
                    break
            if not found:
                result[i][j] = Fraction(val).limit_denominator(1000000)
    return result

def _verify_exact(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify decomposition exactly using vectorized operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build full tensor from decomposition and compare
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Efficient: use einsum
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    if np.max(np.abs(T_decomp - T_target)) > tol:
        return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Build the schoolbook solution as fallback
    def make_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = schoolbook_rank
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    import random
    random.seed(12345)
    np.random.seed(12345)
    
    lower_bound = max(n*m, m*p, p*n)  # 12
    
    # Precompute Kronecker product matrices for fast ALS
    # For solving U: B = kron(V.T, W.T) reshaped appropriately
    # T_flat[a, bc] = sum_r U[a,r] * V[b,r] * W[c,r]
    # B[bc, r] = V[b,r] * W[c,r] where bc = b*(p*n) + c
    # This is: B = np.kron(W, V) but need to be careful with ordering
    # Actually B[bc, r] = V[b,r]*W[c,r]
    # If we flatten bc as b*(p*n)+c, then B = np.kron(np.eye(p*n)...) no
    # Let's use: for each r, column r of B is vec(V[:,r] * W[:,r]) where vec is column-major over (b,c)
    # B[bc, r] where bc = b*(p*n) + c means B is built as:
    # B = np.zeros((m*p*p*n, rank))
    # B[b*(p*n)+c, r] = V[b,r]*W[c,r]
    # This equals: B = (V.T @ something) ... actually it's a Hadamard-like structure
    # We can compute it as: for each r, B[:,r] = V[:,r] tiled and W[:,r] repeated
    # B[:,r] = np.tile(V[:,r], (p*n,1)).reshape(-1) * np.repeat(W[:,r], m*p)
    # Wait: B[b*(p*n)+c, r] = V[b,r]*W[c,r]
    # For fixed r: the vector is [V[0,r]*W[0,r], V[0,r]*W[1,r], ..., V[0,r]*W[p*n-1,r], V[1,r]*W[0,r], ...]
    # = np.kron(V[:,r], W[:,r]) if we think of it as outer product flattened
    # Actually np.kron(V[:,r], W[:,r]) gives [V[0,r]*W[0,r], V[0,r]*W[1,r], ..., V[0,r]*W[p*n-1,r], V[1,r]*W[0,r], ...]
    # Yes! That's exactly what we want.
    
    def build_B_VW(V, W, rank):
        """Build B matrix where B[b*(p*n)+c, r] = V[b,r]*W[c,r]"""
        # B[:,r] = np.kron(V[:,r], W[:,r])
        B = np.zeros((m*p*p*n, rank))
        for r in range(rank):
            B[:, r] = np.kron(V[:, r], W[:, r])
        return B
    
    def build_D_UW(U, W, rank):
        """Build D matrix where D[a*(p*n)+c, r] = U[a,r]*W[c,r]"""
        D = np.zeros((n*m*p*n, rank))
        for r in range(rank):
            D[:, r] = np.kron(U[:, r], W[:, r])
        return D
    
    def build_D_UV(U, V, rank):
        """Build D matrix where D[a*(m*p)+b, r] = U[a,r]*V[b,r]"""
        D = np.zeros((n*m*m*p, rank))
        for r in range(rank):
            D[:, r] = np.kron(U[:, r], V[:, r])
        return D
    
    # Precompute reshaped tensors
    T_flat_U = T.reshape(n*m, m*p*p*n)  # For solving U
    T_flat_V = T.transpose(2, 1, 0).reshape(n*m*p*n, m*p)  # For solving V: T_reshaped[(a,c), b] = T[a,b,c]
    T_flat_W = T.transpose(1, 2, 0).reshape(n*m*m*p, p*n)  # For solving W: T_reshaped[(a,b), c] = T[a,b,c]
    
    def als_solve_U(V, W, rank):
        B = build_B_VW(V, W, rank)
        try:
            U_sol = T_flat_U @ np.linalg.pinv(B.T)
        except:
            U_sol = np.random.randn(n*m, rank)
        return U_sol
    
    def als_solve_V(U, W, rank):
        D = build_D_UW(U, W, rank)
        try:
            V_T = np.linalg.pinv(D) @ T_flat_V
            V_sol = V_T.T
        except:
            V_sol = np.random.randn(m*p, rank)
        return V_sol
    
    def als_solve_W(U, V, rank):
        D = build_D_UV(U, V, rank)
        try:
            W_T = np.linalg.pinv(D) @ T_flat_W
            W_sol = W_T.T
        except:
            W_sol = np.random.randn(p*n, rank)
        return W_sol
    
    # Try ranks from lower bound up to schoolbook
    for target_rank in range(lower_bound, schoolbook_rank):
        if time.time() > DEADLINE - 25:
            break
        
        max_tries = 40
        found = False
        
        for trial in range(max_tries):
            if time.time() > DEADLINE - 15:
                break
            
            # Random initialization
            U_al = np.random.randn(n * m, target_rank)
            V_al = np.random.randn(m * p, target_rank)
            W_al = np.random.randn(p * n, target_rank)
            
            # Run ALS iterations
            prev_err = np.inf
            for it in range(60):
                if time.time() > DEADLINE - 10:
                    break
                U_al = als_solve_U(V_al, W_al, target_rank)
                V_al = als_solve_V(U_al, W_al, target_rank)
                W_al = als_solve_W(U_al, V_al, target_rank)
                
                # Check convergence
                T_decomp = np.einsum('ar,br,cr->abc', U_al, V_al, W_al)
                err = np.max(np.abs(T_decomp - T))
                if err < 1e-4:
                    break
                if abs(err - prev_err) < 1e-10:
                    break
                prev_err = err
            
            if time.time() > DEADLINE - 5:
                break
            
            # Verify
            valid = _verify_exact(U_al, V_al, W_al, n, m, p, target_rank, tol=1e-4)
            if valid:
                # Convert to rational
                U_rat = _to_rational_array(U_al)
                V_rat = _to_rational_array(V_al)
                W_rat = _to_rational_array(W_al)
                
                # Verify with rationals
                valid_rat = _verify_exact(U_rat, V_rat, W_rat, n, m, p, target_rank, tol=1e-6)
                if valid_rat:
                    best_result = {"U": U_rat, "V": V_rat, "W": W_rat}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
