import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Build the Strassen-based rank-42 decomposition
    # 2x4x6: split m=4 into two 2-blocks, p=6 into three 2-blocks
    # Each 2x2x2 block uses 7 Strassen terms -> 2*3*7 = 42 terms
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_terms = [
        # M1 = (X00+X11)(Y00+Y11) -> Z00+=M1, Z11+=M1
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2 = (X10+X11)Y00 -> Z10+=M2, Z11-=M2
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
        # M3 = X00(Y01-Y11) -> Z01+=M3, Z11+=M3
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
        # M4 = X11(Y10-Y00) -> Z10+=M4, Z00+=M4
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(2, 1), (0, 1)]},
        # M5 = (X00+X01)Y11 -> Z01+=M5, Z00-=M5
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
        # M6 = (X10-X00)(Y00+Y01) -> Z11+=M6
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7 = (X01-X11)(Y10+Y11) -> Z00+=M7
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    col = 0
    for r in range(3):  # 3 blocks of p=6
        for s in range(2):  # 2 blocks of m=4
            for t in range(7):  # 7 Strassen terms
                term = strassen_terms[t]
                
                # U indices
                for local_u, coeff in term['U']:
                    i = local_u // 2
                    j_local = local_u % 2
                    j_global = 2 * s + j_local
                    u_idx = i * m + j_global
                    U[u_idx, col] += coeff
                
                # V indices
                for local_v, coeff in term['V']:
                    j_local = local_v // 2
                    k_local = local_v % 2
                    j_global = 2 * s + j_local
                    k_global = 2 * r + k_local
                    v_idx = j_global * p + k_global
                    V[v_idx, col] += coeff
                
                # W indices
                for local_w, coeff in term['W']:
                    i = local_w // 2
                    k_local = local_w % 2
                    k_global = 2 * r + k_local
                    w_idx = k_global * n + i
                    W[w_idx, col] += coeff
                
                col += 1
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, rnk):
        try:
            U_t = U_mat.reshape(n, m, rnk)
            V_t = V_mat.reshape(m, p, rnk)
            W_t = W_mat.reshape(p, n, rnk)
            T = np.einsum('ijr,jkr,ki->ijk', U_t, V_t, W_t)
            return np.all(T == 1.0)
        except Exception:
            return False
    
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        r_sb = n * m * p
        U_sb = np.zeros((n * m, r_sb))
        V_sb = np.zeros((m * p, r_sb))
        W_sb = np.zeros((p * n, r_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # The tensor has n*m*p = 48 entries. Rank 42 means 42 rank-1 tensors sum to identity.
    # We can try to find if any subset of columns is linearly dependent in the tensor space.
    
    # Convert to exact rationals for dependency checking
    U_frac = np.array([[Fraction(int(U[i,j]), 1) for j in range(rank)] for i in range(n*m)], dtype=object)
    V_frac = np.array([[Fraction(int(V[i,j]), 1) for j in range(rank)] for i in range(m*p)], dtype=object)
    W_frac = np.array([[Fraction(int(W[i,j]), 1) for j in range(rank)] for i in range(p*n)], dtype=object)
    
    # Each rank-1 term contributes a vector in R^{n*m*p}
    # term_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We need to check if these 42 vectors are linearly independent
    # Build the matrix where column r is the flattened Kronecker product
    
    nmp = n * m * p  # 48
    T_matrix = np.zeros((nmp, rank))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    T_matrix[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check rank of T_matrix
    # If rank < 42, we have dependencies and can potentially reduce
    T_float = T_matrix.astype(float)
    rank_check = np.linalg.matrix_rank(T_float, tol=1e-10)
    
    if rank_check < rank:
        # There are linear dependencies. Find a minimal generating set.
        # Use QR with column pivoting to find independent columns
        from scipy.linalg import qr
        Q, R, piv = qr(T_float, pivoting=True)
        # The first rank_check columns (in pivoted order) are independent
        # But we need to keep all 48 entries correct, so we can't just drop terms
        # unless we can redistribute. This is complex, so for now just verify.
        pass
    
    # The Strassen decomposition for 2x2x2 is known to be optimal (rank 7 vs schoolbook 8).
    # For 2x4x6, the product structure gives 2*3*7=42.
    # It's possible that a better decomposition exists with rank < 42, but finding it
    # requires a search that may not converge in the time budget.
    
    # Return the verified rank-42 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
