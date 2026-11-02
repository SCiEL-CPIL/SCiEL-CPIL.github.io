"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = 2, 2, 13

    def _build_schoolbook():
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb))
        V = np.zeros((m * p, rank_sb))
        W = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1.0
                    V[k * p + j, col] = 1.0
                    W[j * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def _build_strassen_6x7_4():
        # 6 Strassen 2x2x2x2 blocks (rank 7 each) + 1 schoolbook 2x2x2x1 (rank 4)
        rank = 6 * 7 + 4
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        strassen_terms = [
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1])),
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
        ]

        col = 0
        for block in range(6):
            j0 = block * 2
            j1 = block * 2 + 1
            for t in range(7):
                u_loc, v_loc, w_loc = strassen_terms[t]
                U[0, col] = u_loc[0]
                U[1, col] = u_loc[1]
                U[2, col] = u_loc[2]
                U[3, col] = u_loc[3]
                V[0 * p + j0, col] = v_loc[0]
                V[0 * p + j1, col] = v_loc[1]
                V[1 * p + j0, col] = v_loc[2]
                V[1 * p + j1, col] = v_loc[3]
                W[j0 * n + 0, col] = w_loc[0]
                W[j0 * n + 1, col] = w_loc[1]
                W[j1 * n + 0, col] = w_loc[2]
                W[j1 * n + 1, col] = w_loc[3]
                col += 1

        j_last = 12
        for i in range(n):
            for k in range(m):
                U[i * m + k, col] = 1.0
                V[k * p + j_last, col] = 1.0
                W[j_last * n + i, col] = 1.0
                col += 1
        return U, V, W

    def _build_polynomial_shift():
        """
        Exploit the cyclic shift structure of multiplication by x in Z[x]/(x^13).
        
        The key insight: for a 2x2 matrix A and 2x13 matrix B, we can view B as
        two polynomials b0(x), b1(x) in Z[x]/(x^13). The product C = A*B has entries
        C[i][j] = A[i][0]*b0(x) + A[i][1]*b1(x) (mod x^13).
        
        We can decompose this using the fact that multiplication by x is a shift.
        Let's try to find a rank-45 or lower construction.
        
        Approach: Use the minimal polynomial of the companion matrix of x^13.
        The companion matrix C of x^13 has characteristic polynomial x^13.
        Its minimal polynomial is also x^13.
        
        For 2x2 matrix multiplication over Z[x]/(x^13), we can use the fact that
        the ring has dimension 13. The rank of 2x2 matrix multiplication over a
        ring of dimension d is at most 7*d (using Strassen) or 8*d (schoolbook).
        
        But we can do better by exploiting the specific structure of Z[x]/(x^13).
        
        Let's try a direct construction:
        - For each of the 4 entries of A (a00, a01, a10, a11), we need to multiply
          by the corresponding row of B.
        - The multiplication by a scalar in Z[x]/(x^13) is just scalar multiplication,
          which has rank 13 (one for each coefficient).
        - But we can share work between the two rows of B.
        
        Actually, let's try a different approach: use the fact that the tensor
        T[i*m+k, k*p+j, j*n+i] = 1 has a special structure when p is large.
        
        For fixed i and k, the slice T[i*m+k, k*p+j, j*n+i] = 1 for all j.
        This means that for each (i,k) pair, we need a rank-13 decomposition of
        the "all-ones" tensor of shape (1, 13, 13) where the first and third
        dimensions are collapsed.
        
        Wait, that's not right. Let me reconsider.
        
        The tensor is T[a, b, c] where:
        - a = i*m + k (i in 0..1, k in 0..1)
        - b = k*p + j (k in 0..1, j in 0..12)
        - c = j*n + i (j in 0..12, i in 0..1)
        
        For fixed i, k: T[i*m+k, k*p+j, j*n+i] = 1 for all j.
        This is a vector of 13 ones in the middle dimension, with the first and
        third dimensions fixed.
        
        The rank of this "slice" is 13 (it's a rank-13 tensor in the sense that
        it requires 13 rank-1 terms to represent).
        
        But we can share rank-1 terms between different (i,k) pairs!
        
        Let's try to construct a decomposition with rank 45:
        - 4 pairs of (i,k), each requiring rank 13 if independent = 52
        - We can save 7 by using Strassen-like tricks.
        
        Actually, let's just try to find a rank-45 construction by optimization.
        """
        rank = 45
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Initialize with schoolbook-like structure
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    col = i * m * p + k * p + j
                    if col < rank:
                        U[i * m + k, col] = 1.0
                        V[k * p + j, col] = 1.0
                        W[j * n + i, col] = 1.0
        
        # Now try to optimize to reduce rank
        # This is a complex optimization problem, so let's try a heuristic approach
        
        # Use the fact that the tensor has a block structure
        # Group by i (first dimension of output)
        for i in range(n):
            # For fixed i, we need to compute C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j]
            # This is a 2x2 * 2x13 product for each i
            # The rank of 2x2 * 2x13 is at most 7*13 = 91 (using Strassen per column)
            # But we can do better by sharing work
            
            # For fixed i, the tensor slice is:
            # T[i*m+k, k*p+j, j*n+i] for k in 0..1, j in 0..12
            # This is a 2 x 26 x 1 tensor (or 2 x 26 x 1 if we fix the last index)
            
            # Actually, for fixed i, the last dimension is fixed to i, so we have:
            # T[i*m+k, k*p+j, i] = 1 for all k, j
            # This is a 2 x 26 matrix of all ones
            # The rank of this is 1 (it's a rank-1 matrix)
            
            # But we need to represent this as a sum of rank-1 tensors
            # U[i*m+k, r] * V[k*p+j, r] * W[i, r]
            
            # For fixed i, W[i, r] is a scalar, so we need:
            # sum_r U[i*m+k, r] * V[k*p+j, r] * W[i, r] = 1 for all k, j
            
            # This is equivalent to: the matrix M[i] = U_i * diag(W_i) * V_i^T = J (all ones)
            # where U_i is 2 x rank_i, V_i is 26 x rank_i, W_i is rank_i x 1
            
            # The rank of J (2 x 26) is 1, so we can do this with rank 1!
            # But we need to share the same rank-1 terms across different i.
            
            pass
        
        # This approach is getting complex. Let's just return the Strassen construction
        # and try to find a better one through optimization.
        return _build_strassen_6x7_4()

    # Build schoolbook (rank 52)
    U_school, V_school, W_school = _build_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school

    # Build Strassen-based (rank 46)
    try:
        U_46, V_46, W_46 = _build_strassen_6x7_4()
        # Verify
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_target[i * m + k, k * p + j, j * n + i] = 1.0
        T_approx = np.einsum('it,jt,kt->ijk', U_46, V_46, W_46)
        error = np.max(np.abs(T_approx - T_target))
        if error < 1e-10:
            best_rank = 46
            best_U, best_V, best_W = U_46, V_46, W_46
    except Exception:
        pass
    
    # Try polynomial shift approach (rank 45)
    try:
        U_45, V_45, W_45 = _build_polynomial_shift()
        if U_45 is not None:
            T_target = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        T_target[i * m + k, k * p + j, j * n + i] = 1.0
            T_approx = np.einsum('it,jt,kt->ijk', U_45, V_45, W_45)
            error = np.max(np.abs(T_approx - T_target))
            if error < 1e-10 and U_45.shape[1] < best_rank:
                best_rank = U_45.shape[1]
                best_U, best_V, best_W = U_45, V_45, W_45
    except Exception:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
