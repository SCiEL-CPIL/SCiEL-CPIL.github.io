import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition of the multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    rank = U.shape[1]
    # Reshape U to (n, m, rank)
    U_3d = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_3d = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank) -- Note: W index is k*n+i
    W_3d = W.reshape(p, n, rank)
    
    # Compute the tensor product sum
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Using einsum
    # U[i,j,r] V[j,k,r] -> intermediate[i,j,k,r]? No, sum over r.
    # Sum_r U[i,j,r] V[j,k,r] = (U_3d @ V_3d.T?) No.
    # U is (n, m, r), V is (m, p, r).
    # We want sum_r U[i,j,r] V[j,k,r].
    # This is a contraction over r and j? No, j is fixed in the sum?
    # The tensor is T[i,j,k].
    # For fixed i,j,k, sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    
    # Let's compute M[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We can do this in steps.
    # First, compute A[i,j,k] = sum_r U[i,j,r] V[j,k,r]?
    # No, V depends on j and k.
    # So for each j, we have U[:,j,:] and V[j,:,:].
    # This is not a simple matrix multiplication.
    
    # Let's just loop for verification, it's small.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _bclr_23_3x3x3():
    """
    Attempt to return rank-23 decomposition for 3x3x3.
    Returns None if coefficients are not available or fail verification.
    """
    # BCLR 23 coefficients
    # This is a known set. If incorrect, verification will fail.
    # U, V, W are 3x3 matrices for each of the 23 terms.
    # I will use a placeholder structure and fill with known values if possible.
    # Since I don't have the exact 23x3x3 matrices memorized, 
    # I will return None to trigger fallback.
    # However, if I can find them, I would use them.
    # For now, I will try to use a rank-26 or similar if I had it.
    # But I don't.
    # So I will return None.
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_from_3x3x3(U3, V3, W3, n, m, p):
    """
    Construct 3x3x6 decomposition from 3x3x3 decomposition.
    U3, V3, W3 are (9, r), (9, r), (9, r) for 3x3x3.
    Returns U, V, W of shape (9, 2r), (18, 2r), (18, 2r).
    """
    r = U3.shape[1]
    R = 2 * r
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # p is 6, so k ranges 0..5.
    # k' = k % 3
    # q = k // 3 (0 or 1)
    
    for i in range(n):
        for j in range(m):
            for r_idx in range(r):
                # Term 2*r_idx handles k in 0..2
                # Term 2*r_idx+1 handles k in 3..5
                U[i*m + j, 2*r_idx] = U3[i*m + j, r_idx]
                U[i*m + j, 2*r_idx + 1] = U3[i*m + j, r_idx]
                
                # V for k in 0..2
                for k_prime in range(3):
                    V[j*3 + k_prime, 2*r_idx] = V3[j*3 + k_prime, r_idx]
                    V[j*3 + k_prime, 2*r_idx + 1] = 0
                    # V for k in 3..5
                    V[j*3 + (k_prime + 3), 2*r_idx] = 0
                    V[j*3 + (k_prime + 3), 2*r_idx + 1] = V3[j*3 + k_prime, r_idx]
                    
                # W for k in 0..2
                for k_prime in range(3):
                    W[k_prime*n + i, 2*r_idx] = W3[k_prime*n + i, r_idx]
                    W[k_prime*n + i, 2*r_idx + 1] = 0
                    # W for k in 3..5
                    W[(k_prime+3)*n + i, 2*r_idx] = 0
                    W[(k_prime+3)*n + i, 2*r_idx + 1] = W3[k_prime*n + i, r_idx]
                    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try to get rank-23
    res_23 = _bclr_23_3x3x3()
    
    if res_23 is not None:
        U3, V3, W3 = res_23
        # Verify 3x3x3
        if _verify_decomposition(U3, V3, W3, 3, 3, 3):
            # Construct 3x3x6
            res = _construct_from_3x3x3(U3, V3, W3, n, m, p)
            # Verify 3x3x6
            if _verify_decomposition(res["U"], res["V"], res["W"], n, m, p):
                return res
            else:
                print("3x3x6 verification failed")
        else:
            print("3x3x3 verification failed")
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
