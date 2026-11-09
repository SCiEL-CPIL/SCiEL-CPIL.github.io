"""
Global Algebraic Rank Reduction for 3x4x4 Matrix Multiplication.

Strategy:
1. Construct a valid rank-44 decomposition (Strassen 2x4x4 + Schoolbook 1x4x4).
2. Represent each of the 44 rank-1 terms as a vector in the tensor space.
3. Compute the SVD of the matrix formed by these vectors to find the effective rank.
4. If the effective rank is less than 44, project the decomposition onto the principal components to obtain a lower-rank exact decomposition.
5. Verify the result.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    result = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.allclose(result, T, atol=tol)

def _build_strassen_2x4x4_clean():
    """
    Clean construction of Strassen for 2x4x4.
    Returns U, V, W for the 2x4x4 sub-problem.
    """
    rank = 28
    U = np.zeros((8, rank))   # A: 2x4, index i*4+j
    V = np.zeros((16, rank))  # B: 4x4, index j*4+k
    W = np.zeros((8, rank))   # C: 4x2, index k*2+i
    
    col = 0
    for s in range(7):
        if s == 0:  # M1 = (A11+A22)(B11+B22)
            X = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1),(2,0,1)], (0,1): [(0,1,1),(2,1,1)],
                 (1,0): [(1,0,1),(3,0,1)], (1,1): [(1,1,1),(3,1,1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,2,1),(1,3,1)]}
        elif s == 1:  # M2 = (A21+A22)B11
            X = {0: [(1,0,1),(1,2,1)], 1: [(1,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1)], (0,1): [(0,1,1)], (1,0): [(1,0,1)], (1,1): [(1,1,1)]}
            out = {0: [(1,0,1),(1,1,1)], 1: [(1,2,-1),(1,3,-1)]}
        elif s == 2:  # M3 = A11(B12-B22)
            X = {0: [(0,0,1)], 1: [(0,1,1)]}
            Y = {(0,0): [(0,2,1),(2,2,-1)], (0,1): [(0,3,1),(2,3,-1)],
                 (1,0): [(1,2,1),(3,2,-1)], (1,1): [(1,3,1),(3,3,-1)]}
            out = {0: [(0,2,1),(0,3,1)], 1: [(1,2,1),(1,3,1)]}
        elif s == 3:  # M4 = A22(B21-B11)
            X = {0: [(1,2,1)], 1: [(1,3,1)]}
            Y = {(0,0): [(2,0,1),(0,0,-1)], (0,1): [(2,1,1),(0,1,-1)],
                 (1,0): [(3,0,1),(1,0,-1)], (1,1): [(3,1,1),(1,1,-1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,0,1),(1,1,1)]}
        elif s == 4:  # M5 = (A11+A12)B22
            X = {0: [(0,0,1),(0,2,1)], 1: [(0,1,1),(0,3,1)]}
            Y = {(0,0): [(2,2,1)], (0,1): [(2,3,1)], (1,0): [(3,2,1)], (1,1): [(3,3,1)]}
            out = {0: [(0,0,-1),(0,1,-1)], 1: [(0,2,1),(0,3,1)]}
        elif s == 5:  # M6 = (A21-A11)(B11+B12)
            X = {0: [(1,0,1),(0,0,-1)], 1: [(1,1,1),(0,1,-1)]}
            Y = {(0,0): [(0,0,1),(0,2,1)], (0,1): [(0,1,1),(0,3,1)],
                 (1,0): [(1,0,1),(1,2,1)], (1,1): [(1,1,1),(1,3,1)]}
            out = {0: [], 1: [(1,2,1),(1,3,1)]}
        elif s == 6:  # M7 = (A12-A22)(B21+B22)
            X = {0: [(0,2,1),(1,2,-1)], 1: [(0,3,1),(1,3,-1)]}
            Y = {(0,0): [(2,0,1),(2,2,1)], (0,1): [(2,1,1),(2,3,1)],
                 (1,0): [(3,0,1),(3,2,1)], (1,1): [(3,1,1),(3,3,1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,2,1),(1,3,1)]}
        
        for j in range(2):
            for k in range(2):
                for (i, a, coeff) in X[j]:
                    U[i * 4 + a, col] += coeff
                for (b, c, coeff) in Y[(j,k)]:
                    V[b * 4 + c, col] += coeff
                for (i, k_out, coeff) in out[k]:
                    W[k_out * 2 + i, col] += coeff
                col += 1
    
    return U, V, W

def _build_schoolbook_1x4x4():
    """Schoolbook for 1x4x4: rank 16."""
    rank = 16
    U = np.zeros((4, rank))
    V = np.zeros((16, rank))
    W = np.zeros((4, rank))
    
    for j in range(4):
        for k in range(4):
            idx = j * 4 + k
            U[j, idx] = 1
            V[idx, idx] = 1
            W[k, idx] = 1
    
    return U, V, W

def _combine_3x4x4():
    """Combine 2x4x4 (Strassen, rank 28) and 1x4x4 (schoolbook, rank 16) for total rank 44."""
    U2, V2, W2 = _build_strassen_2x4x4_clean()
    U1, V1, W1 = _build_schoolbook_1x4x4()
    
    total_rank = 44
    U = np.zeros((12, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((12, total_rank))
    
    # Place 2x4x4 part (columns 0-27)
    for i in range(2):
        for j in range(4):
            U[i*4+j, 0:28] = U2[i*4+j, :]
    V[0:16, 0:28] = V2[:, :]
    for i in range(2):
        for k in range(4):
            W[k*3+i, 0:28] = W2[k*2+i, :]
    
    # Place 1x4x4 part (columns 28-43)
    for j in range(4):
        U[8+j, 28:44] = U1[j, :]
    V[0:16, 28:44] = V1[:, :]
    for k in range(4):
        W[k*3+2, 28:44] = W1[k, :]
    
    return U, V, W

def _reduce_rank_by_svd(U, V, W, n, m, p, target_rank, tol=1e-8):
    """
    Attempt to reduce the rank of the decomposition by finding linear dependencies
    among the rank-1 terms using SVD.
    
    Returns (U_new, V_new, W_new) if reduction to target_rank is possible, else None.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    nm = n * m
    mp = m * p
    pn = p * n
    total_dim = nm * mp * pn
    
    # Build matrix of rank-1 terms
    # Each column c is the flattened tensor T_c = U_c \otimes V_c \otimes W_c
    terms = np.zeros((total_dim, current_rank))
    for c in range(current_rank):
        terms[:, c] = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c]).flatten()
    
    # SVD to find effective rank
    try:
        U_svd, S, Vt = np.linalg.svd(terms, full_matrices=False)
    except Exception:
        return None
    
    # Determine effective rank based on singular values
    # We want to keep the top 'target_rank' components if they span the original space
    # Check if the discarded components are negligible
    if target_rank >= len(S):
        return U, V, W
    
    discarded_energy = np.sum(S[target_rank:]**2)
    total_energy = np.sum(S**2)
    
    # If discarded energy is negligible, we can truncate
    if discarded_energy < tol * total_energy:
        # We can represent the tensor with 'target_rank' terms
        # However, the new terms are linear combinations of the old ones.
        # We need to reconstruct U, V, W from the new basis.
        
        # The new basis vectors in the tensor space are:
        # T_new_c = U_svd[:, c] * S[c]
        # But we need to decompose each T_new_c back into U_c, V_c, W_c.
        # This is the hard part: Tensor Decomposition.
        
        # Since we are in a low-dimensional space, we can try to find a decomposition
        # for each new principal component.
        # However, a simpler approach: The new decomposition is:
        # T = sum_{c=1}^{target_rank} (S_c * U_svd[:, c]) \otimes ... ?
        # No, SVD gives T = U_svd S Vt.
        # T = sum_c S_c * (U_svd[:, c] \otimes Vt[c, :])
        # This is not a sum of rank-1 tensors in the U,V,W sense directly.
        
        # Alternative:
        # The original tensor is T = sum_{k=1}^{current_rank} U_k \otimes V_k \otimes W_k.
        # We found that {U_k \otimes V_k \otimes W_k} are linearly dependent.
        # Let R be the matrix of rank-1 terms.
        # We want to find a new set of rank-1 terms {U'_j \otimes V'_j \otimes W'_j}
        # such that sum_j ... = T and j < current_rank.
        
        # This is equivalent to finding a rank-k tensor decomposition of T.
        # We can use the fact that T lies in the span of the original terms.
        # Let's try to solve for coefficients alpha_j such that:
        # T = sum_{j=1}^{target_rank} alpha_j * (U_svd[:, j] \otimes ... )
        # This path is complex.
        
        # Simpler Heuristic:
        # If the rank is reduced, we can try to project the original U, V, W matrices
        # onto the subspace defined by the top singular vectors of the term matrix.
        
        # Let's try a different approach:
        # Since the terms are linearly dependent, there exist vectors a, b, c not all zero
        # such that sum a_k U_k = 0, sum b_k V_k = 0, etc? No.
        
        # Let's use the property that if the rank is R, we can find a basis of R terms.
        # We can iterate:
        # 1. Pick the first 'target_rank' terms.
        # 2. Check if they span the tensor.
        # 3. If not, add the next term, etc.
        
        # Actually, the most robust way is to use the SVD to find the minimal rank
        # and then use a tensor decomposition algorithm (like HOOI) on the projected tensor.
        
        # For now, let's just return the truncated SVD basis if it's valid.
        # But we need U, V, W matrices.
        
        # Let's try to construct new U, V, W by taking the top 'target_rank'
        # left singular vectors of the term matrix and decomposing them.
        
        # This is computationally expensive and might fail.
        # Let's try a simpler check: if the rank is significantly lower,
        # we can try to find a decomposition of the projected tensor.
        
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with the known rank-44 Strassen-based decomposition
    try:
        U, V, W = _combine_3x4x4()
        if not _validate(U, V, W, n, m, p):
            # Fallback to schoolbook
            U_s, V_s, W_s = _schoolbook(n, m, p)
            return {"U": U_s, "V": V_s, "W": W_s}
        
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try to reduce rank using SVD-based analysis
        # We know the rank is at least 32 (from known bounds) and at most 44.
        # Let's try to reduce to 43, 42, etc.
        
        # The SVD approach for exact reduction is tricky because we need to
        # reconstruct the U, V, W factors.
        
        # Instead, let's use a greedy elimination strategy which is more robust
        # for this specific problem structure.
        
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        current_rank = current_U.shape[1]
        
        while current_rank > 32:
            if time.time() > deadline:
                break
            
            reduced = False
            for c_remove in range(current_rank):
                if time.time() > deadline:
                    break
                
                # Remove column c_remove
                mask = [i for i in range(current_rank) if i != c_remove]
                U_test = current_U[:, mask]
                V_test = current_V[:, mask]
                W_test = current_W[:, mask]
                
                if _validate(U_test, V_test, W_test, n, m, p):
                    current_U, current_V, current_W = U_test, V_test, W_test
                    current_rank -= 1
                    reduced = True
                    break  # Restart with the reduced rank
            
            if not reduced:
                break
        
        if current_rank < best_rank:
            best_rank = current_rank
            best_U, best_V, best_W = current_U, current_V, current_W
            print(f"Reduced rank to {best_rank}")
        
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    except Exception as e:
        print(f"Error: {e}")
        U_s, V_s, W_s = _schoolbook(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
