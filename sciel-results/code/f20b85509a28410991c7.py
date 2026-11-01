import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Build the target tensor T of shape (n*p, n*m, m*p)
    # T[i*p+k, i*m+j, j*p+k] = 1
    T = np.zeros((n * p, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * p + k, i * m + j, j * p + k] = 1.0
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose T."""
        rank = U.shape[1]
        S = np.zeros_like(T)
        for r in range(rank):
            S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(S, T, atol=tol)
    
    def to_half_integer(M):
        """Try to round matrix to half-integers and check validity."""
        M_rounded = np.round(M * 2) / 2.0
        return M_rounded
    
    def check_half_integer_solution(U, V, W):
        """Check if rounding to half-integers gives exact solution."""
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        return verify(U_r, V_r, W_r)
    
    # Strategy: Try structured decompositions
    # For matrix multiplication, we can use the fact that each output element
    # is a sum of products. We can try to share terms.
    
    # Approach 1: Try to find low-rank solutions via random search with refinement
    # Approach 2: Use known algebraic structures
    
    # Let's try ranks from 20 down to 24 (schoolbook is 24)
    # Actually, let's be more aggressive and try lower ranks
    
    best_solution = None
    
    # First, try the schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_solution = (U_sb, V_sb, W_sb, rank_schoolbook)
    
    # Try to find better solutions
    # Use a structured approach: for each possible rank, try multiple random initializations
    # and use ALS with exact verification
    
    for target_rank in range(23, 15, -1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(50):
            if time.time() > deadline:
                break
            if found:
                break
                
            np.random.seed(seed)
            
            # Random initialization
            U = np.random.randn(n * p, target_rank) * 0.5
            V = np.random.randn(n * m, target_rank) * 0.5
            W = np.random.randn(m * p, target_rank) * 0.5
            
            # ALS optimization
            max_iters = 200
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                for r in range(target_rank):
                    Residual = T.copy()
                    for r2 in range(target_rank):
                        if r2 == r:
                            continue
                        Residual -= np.einsum('i,j,k->ijk', U[:, r2], V[:, r2], W[:, r2])
                    vw = np.kron(V[:, r], W[:, r])
                    Res_2d = Residual.reshape(n * p, n * m * m * p)
                    denom = np.dot(vw, vw)
                    if denom > 1e-12:
                        U[:, r] = (Res_2d @ vw) / denom
                
                # Update V
                for r in range(target_rank):
                    Residual = T.copy()
                    for r2 in range(target_rank):
                        if r2 == r:
                            continue
                        Residual -= np.einsum('i,j,k->ijk', U[:, r2], V[:, r2], W[:, r2])
                    Res_2d = Residual.transpose(1, 0, 2).reshape(n * m, n * p * m * p)
                    uw = np.kron(U[:, r], W[:, r])
                    denom = np.dot(uw, uw)
                    if denom > 1e-12:
                        V[:, r] = (Res_2d @ uw) / denom
                
                # Update W
                for r in range(target_rank):
                    Residual = T.copy()
                    for r2 in range(target_rank):
                        if r2 == r:
                            continue
                        Residual -= np.einsum('i,j,k->ijk', U[:, r2], V[:, r2], W[:, r2])
                    Res_2d = Residual.transpose(2, 0, 1).reshape(m * p, n * p * n * m)
                    uv = np.kron(U[:, r], V[:, r])
                    denom = np.dot(uv, uv)
                    if denom > 1e-12:
                        W[:, r] = (Res_2d @ uv) / denom
                
                # Check convergence
                S = np.zeros_like(T)
                for r in range(target_rank):
                    S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                err = np.linalg.norm(S - T)
                
                if err < 1e-8:
                    # Try to round to half-integers
                    if check_half_integer_solution(U, V, W):
                        U_r = np.round(U * 2) / 2.0
                        V_r = np.round(V * 2) / 2.0
                        W_r = np.round(W * 2) / 2.0
                        best_solution = (U_r, V_r, W_r, target_rank)
                        found = True
                        break
                elif err > prev_err * 1.1:
                    # Diverging, break early
                    break
                prev_err = err
        
        if found:
            break
    
    # Convert to lists
    U_out, V_out, W_out, rank = best_solution
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

