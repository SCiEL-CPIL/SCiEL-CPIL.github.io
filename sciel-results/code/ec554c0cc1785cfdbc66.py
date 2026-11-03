import numpy as np
import time
from itertools import combinations

def construct():
    n, m, p = 3, 5, 7
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - time.time()
    
    # Build the matrix multiplication tensor
    # T[i,j,k] = 1 for all i,j,k (since (AB)_{ik} = sum_j A_{ij} B_{jk})
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-8):
        """Verify that the decomposition exactly represents matrix multiplication"""
        R = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Strategy: Start with schoolbook (rank 105) and try to reduce rank
    # by finding dependencies. We'll use a greedy approach:
    # For each pair of terms, check if one can be eliminated by adjusting others.
    
    # A better approach: use the fact that we can try to find a rank-R decomposition
    # by solving a system of equations. We'll use a structured search.
    
    # Key insight: The matrix multiplication tensor has a known structure.
    # For 3x5x7, we can try to exploit the rectangular nature.
    
    # Let's try a more systematic approach:
    # 1. Start with rank 105
    # 2. Try to find a rank 104 decomposition by checking if any term is redundant
    # 3. Continue reducing
    
    # For each candidate rank R, we can try to find a decomposition
    # using a randomized approach with local search.
    
    # Actually, let's use a more principled approach:
    # We know that the tensor T_{ijk} for matrix multiplication is:
    # T_{ijk} = 1 if we're computing (AB)_{ik} = sum_j A_{ij} B_{jk}
    # 
    # A rank-1 term corresponds to choosing specific i, j, k and setting
    # U_{ij,r}=1, V_{jk,r}=1, W_{ki,r}=1
    #
    # To reduce rank, we need to find a set of rank-1 terms that sum to T.
    
    # Let's try a different approach: use the Coppersmith-Winograd style
    # decomposition adapted for rectangular matrices.
    
    # For now, let's try a practical approach:
    # 1. Use the schoolbook as a fallback
    # 2. Try to find a lower rank using a structured search
    
    # The key is to find a valid decomposition. Let's try:
    # - Start with schoolbook
    # - Try to merge terms by solving linear systems
    
    # Actually, for this problem size, let's use a more direct approach:
    # We'll try to find a rank-R decomposition by treating it as a
    # polynomial system and using Newton's method or similar.
    
    # But that's complex. Let's try a simpler heuristic:
    # For each rank R from 104 down, try random initializations and
    # use a local search to minimize the error.
    
    np.random.seed(42)
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Precompute indices for faster verification
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def compute_error(U_mat, V_mat, W_mat):
        """Compute the maximum error in the decomposition"""
        R = U_mat.shape[1]
        max_err = 0.0
        for (i, j, k) in indices:
            val = np.dot(U_mat[i*m+j, :] * V_mat[j*p+k, :], W_mat[k*n+i, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        return max_err
    
    # Try to find a lower rank decomposition
    # We'll use a random restart approach with gradient descent
    
    def try_rank_reduction(R, max_iters=2000, lr=0.01):
        """Try to find a rank-R decomposition using gradient descent"""
        if time_left() < 5:
            return None, None, None, None
        
        # Initialize with schoolbook truncated + random
        U_init = np.zeros((n*m, R), dtype=float)
        V_init = np.zeros((m*p, R), dtype=float)
        W_init = np.zeros((p*n, R), dtype=float)
        
        # Copy first R columns of schoolbook if R < rank_school
        if R <= rank_school:
            U_init[:, :R] = U_school[:, :R]
            V_init[:, :R] = V_school[:, :R]
            W_init[:, :R] = W_school[:, :R]
        else:
            U_init = np.random.randn(n*m, R) * 0.1
            V_init = np.random.randn(m*p, R) * 0.1
            W_init = np.random.randn(p*n, R) * 0.1
        
        U_cur = U_init.copy()
        V_cur = V_init.copy()
        W_cur = W_init.copy()
        
        # Compute gradients using automatic differentiation-like approach
        # For each (i,j,k), the residual is:
        # res[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1
        #
        # d(res[i,j,k])/dU[i*m+j,r] = V[j*p+k,r] * W[k*n+i,r]
        # d(res[i,j,k])/dV[j*p+k,r] = U[i*m+j,r] * W[k*n+i,r]
        # d(res[i,j,k])/dW[k*n+i,r] = U[i*m+j,r] * V[j*p+k,r]
        
        for it in range(max_iters):
            if time_left() < 5:
                break
            
            # Compute residuals
            residuals = np.zeros(len(indices))
            for idx, (i, j, k) in enumerate(indices):
                residuals[idx] = np.dot(U_cur[i*m+j, :] * V_cur[j*p+k, :], W_cur[k*n+i, :]) - 1.0
            
            err = np.max(np.abs(residuals))
            if err < 1e-8:
                return U_cur.copy(), V_cur.copy(), W_cur.copy(), err
            
            # Compute gradients
            grad_U = np.zeros_like(U_cur)
            grad_V = np.zeros_like(V_cur)
            grad_W = np.zeros_like(W_cur)
            
            for idx, (i, j, k) in enumerate(indices):
                res = residuals[idx]
                Ui = U_cur[i*m+j, :]
                Vj = V_cur[j*p+k, :]
                Wk = W_cur[k*n+i, :]
                
                grad_U[i*m+j, :] += res * Vj * Wk
                grad_V[j*p+k, :] += res * Ui * Wk
                grad_W[k*n+i, :] += res * Ui * Vj
            
            # Gradient descent step
            U_cur -= lr * grad_U
            V_cur -= lr * grad_V
            W_cur -= lr * grad_W
        
        final_err = compute_error(U_cur, V_cur, W_cur)
        if final_err < 1e-8:
            return U_cur.copy(), V_cur.copy(), W_cur.copy(), final_err
        return None, None, None, final_err
    
    # Try ranks from 104 down to 90
    for R in range(104, 89, -1):
        if time_left() < 10:
            break
        
        # Try multiple random restarts
        for trial in range(20):
            if time_left() < 10:
                break
            
            np.random.seed(42 + R * 100 + trial)
            U_res, V_res, W_res, err = try_rank_reduction(R, max_iters=1000, lr=0.05)
            
            if U_res is not None and err < 1e-8:
                best_rank = R
                best_U = U_res
                best_V = V_res
                best_W = W_res
                break
        
        if best_rank < rank_school:
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_school:
        U_out = best_U
        V_out = best_V
        W_out = best_W
    else:
        U_out = U_school
        V_out = V_school
        W_out = W_school
    
    # Final verification
    if not verify_decomposition(U_out, V_out, W_out):
        # Fall back to schoolbook
        U_out = U_school
        V_out = V_school
        W_out = W_school
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
