import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def _schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
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
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0])
    for r in range(rank):
        # Each rank-1 term: U[:,r] * V[:,r] * W[:,r]
        pass
    # Check tensor reconstruction
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix mult, T[i,j,k] = 1 if this term contributes, 0 otherwise
    # Actually the tensor for matmul(2,3,9) has T[i,j,k] = 1 for all i,j,k
    # Wait no - the tensor is T[i,j,k,l] but for bilinear it's structured.
    # The matrix multiplication tensor T has T[i,j,k] = 1 for the trilinear form
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]
    # For standard matmul, T[i,j,k] = 1 for all i,j,k in range.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    u = U[i * m + j][r]
                    v = V[j * p + k][r]
                    w = W[k * n + i][r]
                    val += u * v * w
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _try_rank17_233(deadline):
    """Try to find rank-17 decomposition for 2x3x3 matmul."""
    n, m, p = 2, 3, 3
    target_rank = 17
    start = time.time()
    
    # The tensor for 2x3x3: T[i,j,k] = 1 for i in 0..1, j in 0..2, k in 0..2
    # We need U (6 x 17), V (9 x 17), W (6 x 17)
    # such that for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # This is hard. Let me try a structured approach.
    # Known: rank(2,3,3) = 17. Let me try to construct it.
    
    # Approach: Start with schoolbook rank 18 and try to merge two terms.
    # Schoolbook: 18 terms, one for each (i,j,k).
    # U[i*3+j, r] = 1 if r corresponds to (i,j,k)
    # V[j*3+k, r] = 1 if r corresponds to (i,j,k)  
    # W[k*2+i, r] = 1 if r corresponds to (i,j,k)
    
    # To get rank 17, we need a linear combination that still works.
    # This is equivalent to finding a 17-term bilinear algorithm.
    
    # Let me try a search: parameterize U, V, W with random initialization
    # and use gradient descent on the constraint that T is reconstructed.
    
    # Define loss: sum over (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    # + sum over (i,j,k) where T should be 0 of (sum_r ...)^2
    # But for matmul, T[i,j,k]=1 for ALL i,j,k, so just the first term.
    
    # Actually wait - for matmul (2,3,3), the tensor is T[i,j,k] = 1 for all 
    # i in {0,1}, j in {0,1,2}, k in {0,1,2}. That's 18 equations.
    # We have 3 * 6 * 17 = 306 variables, 18 equations. Very underdetermined.
    # But we also need the decomposition to be "exact" in the sense that it 
    # computes the product. The tensor condition is necessary and sufficient.
    
    # Let me just try to find ANY solution with rank 17.
    
    np.random.seed(42)
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Gradient descent
    lr = 0.01
    for it in range(5000):
        if time.time() > deadline:
            break
        # Compute residuals
        loss = 0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] should be 1
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    residual = val - 1
                    loss += residual ** 2
                    # Gradient w.r.t. U[i*m+j, r] = residual * V[j*p+k, r] * W[k*n+i, r]
                    grad_U[i*m+j] += residual * V[j*p+k] * W[k*n+i]
                    grad_V[j*p+k] += residual * U[i*m+j] * W[k*n+i]
                    grad_W[k*n+i] += residual * U[i*m+j] * V[j*p+k]
        
        # Scale gradients (they're sum of products, can be large)
        scale = max(1.0, np.max(np.abs(grad_U)) + np.max(np.abs(grad_V)) + np.max(np.abs(grad_W)))
        lr_eff = lr / scale
        
        U -= lr_eff * grad_U
        V -= lr_eff * grad_V
        W -= lr_eff * grad_W
        
        if loss < 1e-12:
            break
    
    # Check if we found a valid solution
    if loss < 1e-9:
        # Convert to integer/half-integer if possible
        # For now, return the float solution - but task says integer/half-integer
        # Let me try to round
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Verify
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_int[i*m+j] * V_int[j*p+k] * W_int[k*n+i])
                    if abs(val - 1) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return (U_int.tolist(), V_int.tolist(), W_int.tolist())
    
    return None

def _build_239_from_233(U_233, V_233, W_233):
    """Build 2x3x9 decomposition from 2x3x3 decomposition.
    
    The 2x3x9 matmul: C[i,k] = sum_j A[i,j] B[j,k] for i in 0,1; k in 0..8; j in 0,1,2.
    Split k into 3 groups of 3: k = 3*q + k' where q in 0,1,2; k' in 0,1,2.
    Then C[i, 3q+k'] = sum_j A[i,j] B[j, 3q+k'].
    
    The tensor for 2x3x9 has T[i,j,k] = 1 for all i,j,k.
    Using the 2x3x3 decomposition, for each q, we have:
    sum_{r=0}^{16} U_233[i*3+j, r] * V_233[j*3+k', r] * W_233[k'*2+i, r] = 1
    
    But this gives us a decomposition for 2x3x3, not 2x3x9.
    
    For 2x3x9, we need: for each (i,j,k) with k in 0..8, the tensor entry is 1.
    
    If we use the 2x3x3 decomposition three times (for k' = 0,1,2 with different k groups),
    we get rank 3*17 = 51.
    
    Construction:
    For q in 0,1,2 and r in 0..16:
      U_239[i*3+j, 3*q+r] = U_233[i*3+j, r]
      V_239[j*9+k, 3*q+r] = V_233[j*3+(k%3), r] if k//3 == q else 0
      W_239[k*2+i, 3*q+r] = W_233[(k%3)*2+i, r] if k//3 == q else 0
    
    Wait, that's not right. Let me think again.
    
    The 2x3x3 decomposition satisfies:
    sum_r U_233[i*3+j, r] * V_233[j*3+k', r] * W_233[k'*2+i, r] = 1 for all i,j,k'
    
    For 2x3x9, k = 3q + k'. We want:
    sum_{r_total} U_239[i*3+j, r_total] * V_239[j*9+k, r_total] * W_239[k*2+i, r_total] = 1
    
    If we set for r_total = 3q + r:
      U_239[i*3+j, 3q+r] = U_233[i*3+j, r]
      V_239[j*9+k, 3q+r] = V_233[j*3+k', r] if k' = k%3 and q = k//3, else 0
      W_239[k*2+i, 3q+r] = W_233[k'*2+i, r] if k' = k%3 and q = k//3, else 0
    
    Then for a fixed (i,j,k) with k=3q+k':
    sum_{r_total} U_239[i*3+j, r_total] * V_239[j*9+k, r_total] * W_239[k*2+i, r_total]
    = sum_{r=0}^{16} U_233[i*3+j, r] * V_233[j*3+k', r] * W_233[k'*2+i, r]  (only q=k//3 terms survive)
    = 1  (by the 2x3x3 decomposition)
    
    But wait, V_239[j*9+k, 3q'+r] is nonzero only if q' = k//3 and k' = k%3.
    Similarly W_239[k*2+i, 3q'+r] is nonzero only if q' = k//3 and k' = k%3.
    
    So yes, this works! Rank = 3 * 17 = 51.
    """
    n, m, p = 2, 3, 9
    rank_233 = len(U_233[0])
    rank_239 = 3 * rank_233
    
    U_239 = [[0] * rank_239 for _ in range(n * m)]
    V_239 = [[0] * rank_239 for _ in range(m * p)]
    W_239 = [[0] * rank_239 for _ in range(p * n)]
    
    for q in range(3):
        for r in range(rank_233):
            for i in range(n):
                for j in range(m):
                    U_239[i*m+j][3*q+r] = U_233[i*m+j][r]
            for j in range(m):
                for k in range(p):
                    k_group = k // 3
                    k_prime = k % 3
                    if k_group == q:
                        V_239[j*p+k][3*q+r] = V_233[j*3+k_prime][r]
            for k in range(p):
                k_group = k // 3
                k_prime = k % 3
                if k_group == q:
                    for i in range(n):
                        W_239[k*n+i][3*q+r] = W_233[k_prime*2+i][r]
    
    return {"U": U_239, "V": V_239, "W": W_239}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # First, get schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    
    # Try to find rank-17 decomposition for 2x3x3
    result_233 = _try_rank17_233(deadline - 5.0)
    
    if result_233 is not None:
        U_233, V_233, W_233 = result_233
        result = _build_239_from_233(U_233, V_233, W_233)
        # Verify
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback to schoolbook
    return schoolbook

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
