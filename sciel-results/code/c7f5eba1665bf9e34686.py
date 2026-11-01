"""
Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Strategy:
- Start with the known rank-49 Strassen decomposition (tensor product of 2x2 Strassen).
- Attempt to reduce the rank by 1 (to 48) by checking if one of the 49 terms is redundant 
  in the sense that the remaining 48 terms can be adjusted to exactly represent the tensor.
- This is done by fixing the V and W matrices of the remaining 48 terms and solving for U, 
  or by using a more symmetric least-squares approach if direct linear solve fails.
- If a valid rank-48 decomposition is found with acceptable coefficients (integer/half-integer), 
  return it. Otherwise, return the rank-49 decomposition.
- This structural change introduces a search/pruning step that the static seed lacks.
"""

import time
import numpy as np
from scipy.linalg import null_space, svd

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly computes 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
    
    target = np.ones((n, m, p))
    max_err = np.max(np.abs(recon - target))
    return max_err < tol

def _strassen_2x2():
    """
    Return coefficients for 2x2 Strassen decomposition.
    """
    a = np.zeros((7, 2, 2))
    b = np.zeros((7, 2, 2))
    c = np.zeros((2, 2, 7))
    
    a[0] = [[1, 0], [0, 1]]
    b[0] = [[1, 0], [0, 1]]
    a[1] = [[0, 0], [1, 1]]
    b[1] = [[1, 0], [0, 0]]
    a[2] = [[1, 0], [0, 0]]
    b[2] = [[0, 0], [1, -1]]
    a[3] = [[0, 0], [0, 1]]
    b[3] = [[-1, 1], [0, 0]]
    a[4] = [[1, 1], [0, 0]]
    b[4] = [[0, 0], [0, 1]]
    a[5] = [[-1, 0], [1, 0]]
    b[5] = [[1, 1], [0, 0]]
    a[6] = [[0, 1], [0, -1]]
    b[6] = [[0, 0], [1, 1]]
    
    c[0, 0] = [1, 0, 0, 1, -1, 0, 1]
    c[0, 1] = [0, 0, 1, 0, 1, 0, 0]
    c[1, 0] = [0, 1, 0, 1, 0, 0, 0]
    c[1, 1] = [1, -1, 1, 0, 0, 1, 0]
    
    return a, b, c

def _build_strassen_4x4():
    """
    Build rank-49 Strassen decomposition for 4x4 MM.
    """
    a2, b2, c2 = _strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for s1 in range(7):
        for s2 in range(7):
            col = 7 * s1 + s2
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            U[i * 4 + j, col] = a2[s1, i1, j1] * a2[s2, i2, j2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * j1 + k2 # Wait, k index mapping
                            # V index is (j, k) where j is from 2x2 block row, k is from 2x2 block col?
                            # In 4x4, j is 0..3, k is 0..3.
                            # j = 2*j1 + j2, k = 2*k1 + k2
                            k = 2 * k1 + k2
                            V[j * 4 + k, col] = b2[s1, j1, k1] * b2[s2, j2, k2]
            
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            k = 2 * k1 + k2
                            W[k * 4 + i, col] = c2[i1, k1, s1] * c2[i2, k2, s2]
    
    return U, V, W

def _try_reduce_rank(U, V, W, max_time=30.0):
    """
    Attempt to reduce the rank by 1 by removing one term and solving for the rest.
    Returns (new_U, new_V, new_W) if successful, else (None, None, None).
    """
    start_time = time.time()
    n, m, p = 4, 4, 4
    rank = U.shape[1]
    target_rank = rank - 1
    
    # Flatten the target tensor
    target_tensor = np.ones((n, m, p)).reshape(-1)
    
    # Try removing each term
    for r in range(rank):
        if time.time() - start_time > max_time:
            break
            
        # Keep all terms except r
        indices = [i for i in range(rank) if i != r]
        
        # We fix V and W of the remaining terms and solve for U.
        # This is a linear system: sum_{i in indices} U[:, i] * (V[:, i] * W[:, i]) = target
        # Let M be a matrix where column i is the Kronecker product of V[:, i] and W[:, i] (flattened).
        # Actually, the term is U[:, i] * (V[:, i] * W[:, i]) is not quite right because U, V, W are vectors.
        # The contribution of term i to the flattened tensor is u_i * v_i * w_i (element-wise product of vectors? No.)
        # The tensor is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].
        # Flattened T[idx] = sum_r U_flat[idx_U, r] * V_flat[idx_V, r] * W_flat[idx_W, r].
        # This is not a linear system in U if V and W are fixed?
        # Wait, if we fix V and W, then for each output index (i,j,k), we have:
        # T[i,j,k] = sum_{r in indices} U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
        # Let A_r(i,j,k) = V[j*p+k, r] * W[k*n+i, r].
        # Then T[i,j,k] = sum_{r in indices} U[i*m+j, r] * A_r(i,j,k).
        # This is a linear system in U!
        # We have 64 equations (for each i,j,k) and 16*48 = 768 unknowns (U entries).
        # Wait, U has shape (16, 48). So 768 unknowns.
        # The system is underdetermined. We can find a solution if one exists.
        # However, we also need V and W to remain consistent? 
        # We are only adjusting U. V and W are fixed to the values from the original decomposition for the kept terms.
        # This is a valid way to reduce rank IF the removed term's contribution can be absorbed by changing U.
        # But this changes the decomposition structure. The new U will be different.
        # We must check if the new U has "nice" coefficients.
        
        # Construct the linear system for U
        # Unknowns: U_new (16 x 48)
        # We can vectorize U_new to u_new (768 x 1)
        # Equation for each (i,j,k):
        # sum_{r in indices} U_new[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Let's build the matrix M (64 x 768)
        M = np.zeros((64, 16 * target_rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_t = i * m * p + j * p + k
                    for new_r_idx, r in enumerate(indices):
                        # U index: i*m + j, column r
                        u_idx = (i * m + j) * target_rank + new_r_idx
                        # Coefficient: V[j*p+k, r] * W[k*n+i, r]
                        coeff = V[j * p + k, r] * W[k * n + i, r]
                        M[idx_t, u_idx] = coeff
        
        # Solve M * u_new = target_tensor
        # Use least squares or check consistency
        try:
            # Since it's underdetermined, we want a solution with small norm or specific structure.
            # np.linalg.lstsq finds the minimum norm solution.
            u_new, residuals, rank_M, sv = np.linalg.lstsq(M, target_tensor, rcond=None)
            
            # Check if the solution is exact
            if np.max(np.abs(M @ u_new - target_tensor)) < 1e-9:
                # We have an exact solution for U.
                U_new = u_new.reshape(16, target_rank)
                
                # Check if U_new has "nice" coefficients (integer or half-integer)
                # Scale by 2 to check for integers
                U_check = U_new * 2
                U_rounded = np.round(U_check)
                if np.max(np.abs(U_check - U_rounded)) < 1e-6:
                    # Valid integer/half-integer solution
                    V_new = V[:, indices]
                    W_new = W[:, indices]
                    
                    # Final verification
                    if _verify(U_new, V_new, W_new):
                        return U_new, V_new, W_new
        except Exception:
            pass
            
    return None, None, None

def construct():
    start_time = time.time()
    
    # Build the deterministic rank-49 Strassen solution
    U, V, W = _build_strassen_4x4()
    
    # Attempt to reduce rank to 48
    U_red, V_red, W_red = _try_reduce_rank(U, V, W, max_time=60.0)
    
    if U_red is not None:
        return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    # Fallback to rank-49
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
