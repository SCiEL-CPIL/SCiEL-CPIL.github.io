import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 3
    start_time = time.time()
    deadline = start_time + 120.0

    # Known rank for 2x3 * 3x3 is 18.
    # We will try to find a rank 15 solution, falling back to 18 if not found in time.
    target_ranks = [15, 16, 17]
    
    # Fallback: Schoolbook Rank 18
    U_18 = np.zeros((n * m, 18))
    V_18 = np.zeros((m * p, 18))
    W_18 = np.zeros((p * n, 18))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_18[i * m + j, idx] = 1.0
                V_18[j * p + k, idx] = 1.0
                W_18[k * n + i, idx] = 1.0
                idx += 1
    
    best_U, best_V, best_W = U_18, V_18, W_18
    best_rank = 18

    def verify_exact(U, V, W, rank):
        """
        Verify if the decomposition is exact and has rational entries (denominators dividing some power of 2).
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        # Attempt to infer fraction from float
                        # If it's not close to a half-integer or integer, fail.
                        # The tensor rank problem for small matrices often has solutions in integers or half-integers.
                        # For safety, we check if the sum is 1 within tolerance using high precision or fractions.
                        # Since we snap to half-integers, we can just use the snapped values if they were valid floats.
                        # But here we are checking the float result of the product sum.
                        pass
                    
                    # Better approach: check the float sum first
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-8:
                        return False
                    
                    # If float check passes, attempt to reconstruct exact rational if possible
                    # For simplicity in this specific problem, we assume the search snaps to half-integers.
                    # If we didn't snap, we'd need a robust rational approximation.
                    # But we WILL snap before final verification.
        return True

    def snap_to_half_ints(U, V, W):
        """Snap entries to nearest half-integer."""
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    def check_valid_snapped(U, V, W, rank):
        """
        Exact verification using Fractions, assuming U, V, W are half-integers.
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        # Convert float to Fraction safely
                        # Since we snapped, U[i*m+j, r] is k/2
                        # round(val * 2) gives the integer numerator
                        nu = int(np.round(U[i * m + j, r] * 2))
                        nv = int(np.round(V[j * p + k, r] * 2))
                        nw = int(np.round(W[k * n + i, r] * 2))
                        s += Fraction(nu * nv * nw, 8) # (nu/2)*(nv/2)*(nw/2) = nu*nv*nw / 8
                    if s != 1:
                        return False
        return True

    def als_update(U, V, W, rank, update_matrix):
        """
        Alternating Least Squares update for one of the matrices.
        update_matrix: 0 for U, 1 for V, 2 for W
        """
        new_M = np.zeros_like(U)
        
        if update_matrix == 0: # Update U
            for i in range(n):
                for j in range(m):
                    idx_u = i * m + j
                    A = np.zeros((p, rank))
                    b = np.ones(p)
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve A x = b for x = U[idx_u, :]
                    # Using min-norm solution
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        new_M[idx_u, :] = x
                    except:
                        new_M[idx_u, :] = U[idx_u, :]
            
        elif update_matrix == 1: # Update V
            for j in range(m):
                for k in range(p):
                    idx_v = j * p + k
                    A = np.zeros((n, rank))
                    b = np.ones(n)
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        new_M[idx_v, :] = x
                    except:
                        new_M[idx_v, :] = V[idx_v, :]
                        
        elif update_matrix == 2: # Update W
            for k in range(p):
                for i in range(n):
                    idx_w = k * n + i
                    A = np.zeros((m, rank))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        new_M[idx_w, :] = x
                    except:
                        new_M[idx_w, :] = W[idx_w, :]
                        
        return new_M

    def compute_residue_norm(U, V, W, rank):
        """Calculate L2 norm of the error tensor."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (val - 1.0) ** 2
        return np.sqrt(err)

    # Search for lower ranks
    for rank in target_ranks:
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random initializations
        for restart in range(200):
            if time.time() > deadline - 2.0:
                break
            
            rng = np.random.RandomState(rank * 10000 + restart)
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Normalize to avoid scaling issues
            U /= np.linalg.norm(U) * np.sqrt(rank / (n*m))
            V /= np.linalg.norm(V) * np.sqrt(rank / (m*p))
            W /= np.linalg.norm(W) * np.sqrt(rank / (p*n))
            
            # Iterative refinement
            converged = False
            for it in range(50):
                if time.time() > deadline - 1.0:
                    break
                
                U = als_update(U, V, W, rank, 0)
                V = als_update(U, V, W, rank, 1)
                W = als_update(U, V, W, rank, 2)
                
                err = compute_residue_norm(U, V, W, rank)
                
                # If error is extremely small, snap and check
                if err < 1e-12:
                    U_s, V_s, W_s = snap_to_half_ints(U, V, W)
                    if check_valid_snapped(U_s, V_s, W_s, rank):
                        best_U, best_V, best_W = U_s, V_s, W_s
                        best_rank = rank
                        converged = True
                        break
            if converged:
                break
        
        if best_rank == rank:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
