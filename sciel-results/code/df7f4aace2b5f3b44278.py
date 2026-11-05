"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # Dimensions: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    # Build the target tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Structured rank-1 peeling approach:
    # We exploit the structure of the matrix multiplication tensor.
    # For each output position c = k*n + i, the valid (a,b) pairs are:
    #   a = i*m + j, b = j*p + k for j = 0,...,m-1
    # 
    # Key insight: We can group terms by their "middle index" j.
    # For fixed j, the terms involve all (i,k) pairs.
    # 
    # Strategy: Try to find a decomposition of rank r by iteratively
    # selecting rank-1 components that maximize coverage of remaining tensor entries.
    
    def peel_rank1(residual, U_list, V_list, W_list, max_iter=1000):
        """
        Greedily peel rank-1 terms from the residual tensor.
        Returns (U, V, W) as lists of 1D arrays for each rank-1 term.
        """
        U = []
        V = []
        W = []
        R = residual.copy()
        
        for _ in range(max_iter):
            # Check if R is zero
            if np.allclose(R, 0, atol=1e-10):
                break
            
            # Find the best rank-1 approximation to R
            # Use SVD on matricizations
            # Matricize R as R_{ab, c} (shape (nm*mp, pn))
            R_mat = R.reshape(nm * mp, pn)
            
            # SVD
            try:
                U_mat, S, Vt = np.linalg.svd(R_mat, full_matrices=False)
            except:
                break
            
            # Take the first singular vector
            if len(S) == 0 or S[0] < 1e-10:
                break
            
            u_vec = U_mat[:, 0] * S[0]  # shape (nm*mp,)
            v_vec = Vt[0, :]            # shape (pn,)
            
            # Reshape u_vec into 2D
            u2d = u_vec.reshape(nm, mp)
            
            # Now we need to factorize u2d as outer product of u (nm,) and v (mp,)
            # Use SVD on u2d
            try:
                u_svd, s_svd, v_svd_t = np.linalg.svd(u2d, full_matrices=False)
            except:
                break
            
            if len(s_svd) == 0 or s_svd[0] < 1e-10:
                break
            
            u1 = u_svd[:, 0] * np.sqrt(s_svd[0])  # shape (nm,)
            v1 = v_svd_t[0, :] * np.sqrt(s_svd[0])  # shape (mp,)
            
            # The rank-1 term is u1 ⊗ v1 ⊗ v_vec
            # Subtract from R
            rank1_term = np.einsum('i,j,k->ijk', u1, v1, v_vec)
            R -= rank1_term
            
            U.append(u1)
            V.append(v1)
            W.append(v_vec)
        
        return U, V, W, R
    
    # Try to find a decomposition with rank 20
    target_rank = 20
    U_list, V_list, W_list, residual = peel_rank1(T, None, None, None, max_iter=target_rank)
    
    # Check if we achieved exact decomposition
    if np.allclose(residual, 0, atol=1e-8) and len(U_list) <= target_rank:
        rank = len(U_list)
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for idx in range(rank):
            U[:, idx] = U_list[idx]
            V[:, idx] = V_list[idx]
            W[:, idx] = W_list[idx]
        # Convert to lists for the return interface
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: try with more iterations to see if we can get closer
    U_list2, V_list2, W_list2, residual2 = peel_rank1(T, None, None, None, max_iter=30)
    
    if np.allclose(residual2, 0, atol=1e-8):
        rank = len(U_list2)
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for idx in range(rank):
            U[:, idx] = U_list2[idx]
            V[:, idx] = V_list2[idx]
            W[:, idx] = W_list2[idx]
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook (rank 40)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
