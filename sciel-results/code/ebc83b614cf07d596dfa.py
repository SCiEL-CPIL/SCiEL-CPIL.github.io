"""Matrix multiplication 5x5x5: search for low-rank exact decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin under 120s

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return TIME_LIMIT - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the multiplication tensor T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # T is a 3-tensor of shape (n, m, p). We want T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually, the standard formulation: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = 1 for all valid (i,j,k) since A[i,j]*B[j,k] always contributes.
    # Wait, that's not right. The tensor is: for each (i,j,k), the coefficient is 1.
    # So T is an all-ones tensor of shape (n, m, p) = (5,5,5).
    # We need: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  (with proper index mapping)
    
    # Actually let me re-read the convention:
    # U: (n*m, rank) -> U[i*m + j, r]
    # V: (m*p, rank) -> V[j*p + k, r]  
    # W: (p*n, rank) -> W[k*n + i, r]
    # And the decomposition says: for the multiplication tensor,
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 (since A[i,j]*B[j,k] always contributes to C[i,k] with coefficient 1)
    
    # So T is an all-ones (5,5,5) tensor. We need rank-r decomposition of the all-ones tensor.
    # The rank of the all-ones tensor of shape (n,m,p) is known to be n (for the standard matmul tensor).
    # Wait no - the all-ones tensor has rank 1 if we allow arbitrary tensors, but the matmul tensor is special.
    
    # Let me reconsider. The matrix multiplication tensor for n x m * m x p -> n x p is:
    # T[i,j,k] = 1 for all i,j,k (it's the structure tensor)
    # And we decompose it as T = sum_r u_r v_r w_r where u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # with u_r[i*m+j] = U[i*m+j, r], etc.
    
    # The schoolbook rank is n*m*p = 125.
    # The known best rank for 5x5x5 matmul is 125 (it's open whether it can be done better).
    # Actually, for 5x5, the best known is 125 (no improvement known).
    # But wait - the task says "find a rank-minimal exact tensor decomposition". 
    # The rank of the 5x5x5 matmul tensor is known to be exactly 125? No, it's an open problem.
    # The best upper bound is 125 (schoolbook) and lower bound is 125? 
    # Actually for n=5, the tensor rank is unknown. The best known upper bound might be less than 125.
    
    # Let me just try to find a good decomposition. Start with schoolbook (rank 125) and try to reduce.
    
    # Strategy: Use random initialization + optimization to find lower-rank decompositions.
    # Since exact integer/half-integer entries are required, we need to be careful.
    
    # First, let's verify the schoolbook works and return it as baseline.
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[mat[i, j] for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    best_rank = rank_schoolbook
    best_U = to_list(U_sb)
    best_V = to_list(V_sb)
    best_W = to_list(W_sb)
    
    # Try to find lower rank decompositions using optimization
    # The tensor T is all-ones (5,5,5). We want T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # Let's use a simple approach: try ranks from 124 down, use random init + gradient descent
    # with a smooth objective, then round to nearest half-integer and verify.
    
    T_flat = np.ones(n * m * p)  # all ones
    
    def compute_residual(U_mat, V_mat, W_mat):
        """Compute the residual between T and the decomposition."""
        # U_mat: (nm, r), V_mat: (mp, r), W_mat: (pn, r)
        r = U_mat.shape[1]
        # Reshape to tensors
        U_t = U_mat.reshape(n, m, r)  # U[i,j,r]
        V_t = V_mat.reshape(m, p, r)  # V[j,k,r]
        W_t = W_mat.reshape(p, n, r)  # W[k,i,r]
        
        # Contract: result[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a bit tricky. Let's do it step by step.
        # temp[i,j,k] = sum_r U[i,j,r] * V[j,k,r]
        # result[i,j,k] = sum_k' temp[i,j,k'] * W[k',i,k'] ... no that's wrong.
        
        # Actually: result[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Let's compute it as: for each r, outer product U[:,r] reshaped * V[:,r] reshaped * W[:,r] reshaped
        
        # More efficient: 
        # A[i,j,k] = U[i*m+j, :] dot V[j*p+k, :] ... no, that gives a scalar per (i,j,k) but we need element-wise product then sum over r.
        
        # result[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's use broadcasting:
        # U_exp: (n, m, p, r) where U_exp[i,j,k,r] = U[i*m+j, r] (independent of k)
        # V_exp: (n, m, p, r) where V_exp[i,j,k,r] = V[j*p+k, r] (independent of i)
        # W_exp: (n, m, p, r) where W_exp[i,j,k,r] = W[k*n+i, r] (independent of j)
        
        # This is memory heavy. For n=m=p=5, r up to 125: 5*5*5*125 = 78125 elements. Fine.
        
        U_exp = np.repeat(U_mat.reshape(n, m, 1, r), p, axis=2)  # (n,m,p,r)
        V_exp = np.repeat(V_mat.reshape(1, m, p, r), n, axis=0)  # (n,m,p,r)
        W_exp = W_mat.reshape(p, 1, n, r).transpose(2, 0, 1)  # need (n, m, p, r) where W_exp[i,j,k,r] = W[k*n+i, r]
        # W_mat has shape (pn, r) = (p*n, r). W[k*n+i, r].
        # W_mat.reshape(p, n, r) gives W_mat[k*n+i, r] = W_mat_reshaped[k, i, r]
        # We want W_exp[i, j, k, r] = W_mat_reshaped[k, i, r]
        W_reshaped = W_mat.reshape(p, n, r)  # W_reshaped[k, i, r] = W[k*n+i, r]
        W_exp = W_reshaped.transpose(1, 0, 2)  # (n, p, r)
        W_exp = np.repeat(W_exp, m, axis=1)  # (n, m, p, r) -- wait, this repeats along axis 1 (m)
        # Actually W_exp should be (n, m, p, r) with W_exp[i,j,k,r] = W_reshaped[k,i,r]
        # W_reshaped is (p, n, r). We want to index as [k, i, r] and put it at [i, j, k, r].
        # So: W_exp[i,j,k,r] = W_reshaped[k,i,r]
        # W_reshaped.transpose(1,0,2) gives (n, p, r) with element [i, k, r] = W_reshaped[k, i, r]. Good.
        # Then we need to expand along j: W_exp = W_reshaped.transpose(1,0,2)[:, np.newaxis, :, :] broadcast to (n, m, p, r)
        W_exp = W_reshaped.transpose(1, 0, 2)[:, np.newaxis, :, :]  # (n, 1, p, r)
        
        # Now result[i,j,k] = sum_r U_exp[i,j,k,r] * V_exp[i,j,k,r] * W_exp[i,j,k,r]
        product = U_exp * V_exp * W_exp  # (n, m, p, r)
        result = np.sum(product, axis=3)  # (n, m, p)
        
        return result - T_flat.reshape(n, m, p)
    
    def objective(U_mat, V_mat, W_mat):
        res = compute_residual(U_mat, V_mat, W_mat)
        return np.sum(res**2)
    
    # Try various ranks with random restarts
    rng = np.random.RandomState(42)
    
    for target_rank in range(124, 119, -1):  # Try 124 down to 120
        if _remaining() < 5.0:
            break
            
        for restart in range(20):
            if _remaining() < 3.0:
                break
            
            # Random initialization
            U_mat = rng.randn(nm, target_rank) * 0.1
            V_mat = rng.randn(mp, target_rank) * 0.1
            W_mat = rng.randn(pn, target_rank) * 0.1
            
            # Gradient descent with adaptive learning rate
            lr = 0.01
            best_obj = float('inf')
            best_U = U_mat.copy()
            best_V = V_mat.copy()
            best_W = W_mat.copy()
            
            for step in range(500):
                if _remaining() < 3.0:
                    break
                
                # Compute gradients numerically (or analytically)
                # Let's do analytical gradient
                # obj = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                # d(obj)/dU[i*m+j,r'] = 2 * sum_{k} (res[i,j,k]) * V[j*p+k,r'] * W[k*n+i,r']
                # where res[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
                
                res = compute_residual(U_mat, V_mat, W_mat)  # (n, m, p)
                
                # Gradient for U
                dU = np.zeros_like(U_mat)
                dV = np.zeros_like(V_mat)
                dW = np.zeros_like(W_mat)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r_idx = np.arange(target_rank)
                            # dU[i*m+j, r] = 2 * res[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                            dU[i*m + j, :] += 2 * res[i, j, k] * V_mat[j*p + k, :] * W_mat[k*n + i, :]
                            dV[j*p + k, :] += 2 * res[i, j, k] * U_mat[i*m + j, :] * W_mat[k*n + i, :]
                            dW[k*n + i, :] += 2 * res[i, j, k] * U_mat[i*m + j, :] * V_mat[j*p + k, :]
                
                # Update
                U_mat -= lr * dU
                V_mat -= lr * dV
                W_mat -= lr * dW
                
                obj = objective(U_mat, V_mat, W_mat)
                if obj < best_obj:
                    best_obj = obj
                    best_U = U_mat.copy()
                    best_V = V_mat.copy()
                    best_W = W_mat.copy()
                
                if best_obj < 1e-10:
                    break
                
                # Adaptive lr
                if step % 50 == 0 and step > 0:
                    lr *= 0.9
                
                if _remaining() < 3.0:
                    break
            
            # Check if we got close to zero
            if best_obj < 1e-6:
                # Try to round to half-integers and verify
                U_rounded = np.round(best_U * 2) / 2
                V_rounded = np.round(best_V * 2) / 2
                W_rounded = np.round(best_W * 2) / 2
                
                res_rounded = compute_residual(U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(res_rounded)) < 1e-9:
                    # Verify exactness with fractions
                    from fractions import Fraction
                    exact_ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(
                                    Fraction(int(U_rounded[i*m+j, r] * 2), 2) * 
                                    Fraction(int(V_rounded[j*p+k, r] * 2), 2) * 
                                    Fraction(int(W_rounded[k*n+i, r] * 2), 2)
                                    for r in range(target_rank)
                                )
                                if val != 1:
                                    exact_ok = False
                                    break
                        if not exact_ok:
                            break
                    if exact_ok:
                        best_rank = target_rank
                        best_U = to_list(U_rounded)
                        best_V = to_list(V_rounded)
                        best_W = to_list(W_rounded)
                        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If no improvement found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
