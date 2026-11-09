"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Build the target tensor T[i,j,k] = 1 if this corresponds to a valid multiplication term
    # For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all valid (i,j,k) - this is the rank-60 schoolbook
    # Actually the tensor for matrix mult is: for each output (i,k), sum over j of A[i,j]*B[j,k]
    # The tensor T is defined such that T[i,j,k] = 1 if we're computing C[i,k] using A[i,j] and B[j,k]
    # So T has shape (n, m, p) with T[i,j,k] = 1 for all i,j,k (it's all ones in the valid range)
    # Wait, that's not right. The tensor is:
    # T[i,j,k] = 1 if this term contributes to C[i,k], which is always true for valid indices
    # So the tensor is all ones of shape (n, m, p) = (3, 4, 5)
    # But that's the schoolbook. We want to find a rank decomposition.
    
    # The matrix multiplication tensor T is defined as:
    # T[i, j, k] = 1 for all i in range(n), j in range(m), k in range(p)
    # This is because C[i,k] = sum_j A[i,j] B[j,k], and each term (i,j,k) contributes.
    # So T is the all-ones tensor of shape (3,4,5).
    
    # We want to decompose T as sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # i.e., T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Start with schoolbook rank 60
    rank = n * m * p  # 60
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: try to express one column as a linear combination of others
    # and eliminate it if possible
    
    def check_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if U*V*W decomposes the all-ones tensor of shape (n,m,p)"""
        # Reconstruct tensor
        U_mat = np.asarray(U_mat)
        V_mat = np.asarray(V_mat)
        W_mat = np.asarray(W_mat)
        r = U_mat.shape[1]
        
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U_mat[i*m+j, r_idx] * V_mat[j*p+k, r_idx] * W_mat[k*n+i, r_idx]
                    T_recon[i,j,k] = val
        
        # Target is all ones
        target = np.ones((n, m, p))
        return np.allclose(T_recon, target, atol=tol)
    
    def try_reduce_rank(U_mat, V_mat, W_mat, max_iter=50):
        """Try to reduce rank by eliminating linearly dependent columns"""
        U_mat = U_mat.copy()
        V_mat = V_mat.copy()
        W_mat = W_mat.copy()
        current_rank = U_mat.shape[1]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            reduced = False
            for c in range(current_rank):
                # Try to eliminate column c
                if current_rank <= 1:
                    break
                # Check if column c is in the span of other columns (approximately)
                # We need to check if there exist alpha such that for all (i,j,k):
                # sum_r != c U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * alpha_r = 
                # U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                
                # This is a system of equations. Let's set up the linear system.
                # For each (i,j,k): sum_{r != c} alpha_r * U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                
                # Number of equations: n*m*p = 60
                # Number of unknowns: current_rank - 1
                # If current_rank - 1 >= 60, we might have a solution
                
                if current_rank - 1 < n * m * p:
                    continue  # Not enough degrees of freedom to satisfy all equations generically
                
                # Build the linear system
                A_sys = []
                b_sys = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = []
                            for r_idx in range(current_rank):
                                if r_idx == c:
                                    continue
                                row.append(U_mat[i*m+j, r_idx] * V_mat[j*p+k, r_idx] * W_mat[k*n+i, r_idx])
                            A_sys.append(row)
                            b_sys.append(U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c])
                
                A_arr = np.array(A_sys)
                b_arr = np.array(b_sys)
                
                # Solve least squares
                try:
                    alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_arr, b_arr, rcond=None)
                except:
                    continue
                
                # Check if the solution is good
                if np.allclose(A_arr @ alpha, b_arr, atol=1e-8):
                    # Eliminate column c
                    idx = [r for r in range(current_rank) if r != c]
                    U_mat = U_mat[:, idx]
                    V_mat = V_mat[:, idx]
                    W_mat = W_mat[:, idx]
                    current_rank -= 1
                    reduced = True
                    break
            
            if not reduced:
                break
        
        return U_mat, V_mat, W_mat, current_rank
    
    # Try rank reduction
    U_red, V_red, W_red, new_rank = try_reduce_rank(U, V, W, max_iter=30)
    
    if new_rank < best_rank and check_decomposition(U_red, V_red, W_red):
        best_U, best_V, best_W = U_red, V_red, W_red
        best_rank = new_rank
    
    # Try even more aggressive: random restarts with lower initial rank
    # Use a heuristic: try rank 55, 50, 45, 40, etc. with random initialization
    
    for target_rank in [55, 50, 45, 42, 40, 38, 36, 35, 34, 33, 32, 30, 28, 27, 26, 25, 24, 22, 20, 18, 16, 15, 14, 12, 10]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try random initialization and optimization
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            U_rand = np.random.randn(n*m, target_rank) * 0.1
            V_rand = np.random.randn(m*p, target_rank) * 0.1
            W_rand = np.random.randn(p*n, target_rank) * 0.1
            
            # Use gradient descent to minimize ||T_recon - ones||^2
            # This is non-convex, so we'll do a few iterations
            
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Compute gradient
                # T_recon[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # dT/dU[i*m+j,r] = V[j*p+k,r]*W[k*n+i,r] for each k
                # This is complex. Let's use a simpler approach: just check if random init works (unlikely)
                # or use a structured initialization.
                
                # Actually, let's try a different approach: use the fact that the tensor is all ones.
                # A rank-1 decomposition of all-ones tensor of shape (3,4,5) would be:
                # U[i*m+j,0] = a[i,j], V[j*p+k,0] = b[j,k], W[k*n+i,0] = c[k,i]
                # such that sum_r a[i,j]*b[j,k]*c[k,i] = 1 for all i,j,k
                # This is not generally possible with rank 1.
                
                # Let's just break and rely on the schoolbook + reduction
                break
            
            # Check if the random init happens to work (very unlikely)
            if check_decomposition(U_rand, V_rand, W_rand):
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = target_rank
                found = True
                break
    
    # Convert to list format with half-integer entries if needed
    # For now, use the best decomposition found
    U_final = best_U
    V_final = best_V
    W_final = best_W
    final_rank = best_rank
    
    # Convert numpy arrays to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    # Round to nearest half-integer if close
    def round_to_half(x):
        return round(2*x)/2
    
    U_list = [[round_to_half(v) for v in row] for row in U_list]
    V_list = [[round_to_half(v) for v in row] for row in V_list]
    W_list = [[round_to_half(v) for v in row] for row in W_list]
    
    # Verify the rounded version still works
    U_check = np.array(U_list)
    V_check = np.array(V_list)
    W_check = np.array(W_list)
    
    if not check_decomposition(U_check, V_check, W_check, tol=1e-6):
        # Use the unrounded version
        U_list = U_final.tolist()
        V_list = V_final.tolist()
        W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
