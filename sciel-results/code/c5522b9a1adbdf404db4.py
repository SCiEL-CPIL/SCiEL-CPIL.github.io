"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication.
Strategy: Start from a known valid 44-rank decomposition (Strassen 2x4x4 + schoolbook 1x4x4),
then attempt rank reduction via alternating linear projection with exactness verification.
Also try direct construction at various ranks using structured ansatz.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    rank = U.shape[1]
    result = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.allclose(result, T, atol=tol)

def _build_strassen_2x4x4():
    """
    Strassen decomposition for 2x4x4 matrix multiplication. Rank 28.
    A is 2x4, B is 4x4, C is 2x4.
    """
    rank = 28
    U = np.zeros((8, rank))
    V = np.zeros((16, rank))
    W = np.zeros((8, rank))
    
    col = 0
    for s in range(7):
        if s == 0:  # M1 = (A11+A22)(B11+B22)
            X = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1),(2,0,1)], (0,1): [(0,1,1),(2,1,1)],
                 (1,0): [(1,0,1),(3,0,1)], (1,1): [(1,1,1),(3,1,1)]}
            out = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}
        elif s == 1:  # M2 = (A21+A22)B11
            X = {0: [(1,0,1),(1,2,1)], 1: [(1,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1)], (0,1): [(0,1,1)], (1,0): [(1,0,1)], (1,1): [(1,1,1)]}
            out = {0: [(1,0,1),(1,2,-1)], 1: [(1,1,1),(1,3,-1)]}
        elif s == 2:  # M3 = A11(B12-B22)
            X = {0: [(0,0,1)], 1: [(0,1,1)]}
            Y = {(0,0): [(0,2,1),(2,2,-1)], (0,1): [(0,3,1),(2,3,-1)],
                 (1,0): [(1,2,1),(3,2,-1)], (1,1): [(1,3,1),(3,3,-1)]}
            out = {0: [(0,2,1),(1,2,1)], 1: [(0,3,1),(1,3,1)]}
        elif s == 3:  # M4 = A22(B21-B11)
            X = {0: [(1,2,1)], 1: [(1,3,1)]}
            Y = {(0,0): [(2,0,1),(0,0,-1)], (0,1): [(2,1,1),(0,1,-1)],
                 (1,0): [(3,0,1),(1,0,-1)], (1,1): [(3,1,1),(1,1,-1)]}
            out = {0: [(0,0,1),(1,0,1)], 1: [(0,1,1),(1,1,1)]}
        elif s == 4:  # M5 = (A11+A12)B22
            X = {0: [(0,0,1),(0,2,1)], 1: [(0,1,1),(0,3,1)]}
            Y = {(0,0): [(2,2,1)], (0,1): [(2,3,1)], (1,0): [(3,2,1)], (1,1): [(3,3,1)]}
            out = {0: [(0,0,-1),(0,2,1)], 1: [(0,1,-1),(0,3,1)]}
        elif s == 5:  # M6 = (A21-A11)(B11+B12)
            X = {0: [(1,0,1),(0,0,-1)], 1: [(1,1,1),(0,1,-1)]}
            Y = {(0,0): [(0,0,1),(0,2,1)], (0,1): [(0,1,1),(0,3,1)],
                 (1,0): [(1,0,1),(1,2,1)], (1,1): [(1,1,1),(1,3,1)]}
            out = {0: [(1,2,1)], 1: [(1,3,1)]}
        elif s == 6:  # M7 = (A12-A22)(B21+B22)
            X = {0: [(0,2,1),(1,2,-1)], 1: [(0,3,1),(1,3,-1)]}
            Y = {(0,0): [(2,0,1),(2,2,1)], (0,1): [(2,1,1),(2,3,1)],
                 (1,0): [(3,0,1),(3,2,1)], (1,1): [(3,1,1),(3,3,1)]}
            out = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}
        
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
    """Schoolbook for 1x4x4. Rank 16."""
    rank = 16
    U = np.zeros((4, rank))
    V = np.zeros((16, rank))
    W = np.zeros((4, rank))
    for j in range(4):
        for k in range(4):
            U[j, j*4+k] = 1
            V[j*4+k, j*4+k] = 1
            W[k, j*4+k] = 1
    return U, V, W

def _combine_3x4x4():
    """Combine 2x4x4 (rank 28) and 1x4x4 (rank 16) for total rank 44."""
    U2, V2, W2 = _build_strassen_2x4x4()
    U1, V1, W1 = _build_schoolbook_1x4x4()
    
    total_rank = 44
    U = np.zeros((12, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((12, total_rank))
    
    for i in range(2):
        for j in range(4):
            U[i*4+j, 0:28] = U2[i*4+j, :]
    V[0:16, 0:28] = V2[:, :]
    for i in range(2):
        for k in range(4):
            W[k*3+i, 0:28] = W2[k*2+i, :]
    
    for j in range(4):
        U[8+j, 28:44] = U1[j, :]
    V[0:16, 28:44] = V1[:, :]
    for k in range(4):
        W[k*3+2, 28:44] = W1[k, :]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition. Rank = n*m*p = 48."""
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
    return U, V, W

def _try_rank_reduction(U, V, W, target_rank, n, m, p, max_iters=200, tol=1e-9):
    """
    Attempt to reduce rank by alternating linear projections.
    Given a valid r-rank decomposition, try to find a target_rank-rank one.
    
    Approach: We have T = sum_c U_c V_c W_c (outer products).
    For a target rank r', we want to find U'(12xr'), V'(16xr'), W'(12xr') such that
    sum_{c=1}^{r'} U'_c V'_c W'_c = T.
    
    Strategy: Fix V and W, solve for U linearly. Then fix U and W, solve for V. Etc.
    Start from the 44-rank solution, project down.
    """
    r0 = U.shape[1]
    if target_rank >= r0:
        return None
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Flatten T
    T_flat = T.flatten()
    dim_a, dim_b, dim_c = n * m, m * p, p * n
    
    # Initialize with truncated SVD-like approach:
    # Take the 44-rank decomposition, form the "matrix" of rank-1 components
    # and try to approximate with fewer terms.
    
    # Alternate: fix V,W solve U; fix U,W solve V; fix U,V solve W
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    # Truncate to target_rank by keeping the most important columns
    # Use the norm of each rank-1 term as importance
    importances = np.zeros(r0)
    for c in range(r0):
        importance = np.linalg.norm(U_curr[:, c]) * np.linalg.norm(V_curr[:, c]) * np.linalg.norm(W_curr[:, c])
        importances[c] = importance
    
    # Keep top target_rank columns
    idx = np.argsort(importances)[::-1][:target_rank]
    U_curr = U_curr[:, idx]
    V_curr = V_curr[:, idx]
    W_curr = W_curr[:, idx]
    
    # Now do alternating minimization
    for it in range(max_iters):
        # Fix V_curr, W_curr; solve for U_curr
        # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # For each (j,k): T[:,j,k] = U * (V[j,:]*W[k,:])
        # This is: for each (j,k), T[:,j,k] = sum_c U[i,c] * (V[j,c]*W[k,c])
        # So: T[:,j,k] = U @ diag(V[j,:]*W[k,:]) -- not quite linear in U directly
        
        # Better: vectorize. T_flat[i*dim_b*dim_c + j*dim_c + k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # For fixed V,W: this is linear in U.
        # T_flat = sum_c (U[:,c] * V[:,c] * W[:,c]).flatten()
        # So T_flat = M_U @ U_flat where M_U is a matrix... 
        
        # Actually: T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
        # For fixed V,W: T is linear in U. 
        # T_flat = sum_c (U[:,c] ⊗ V[:,c] ⊗ W[:,c]).flatten()
        # = sum_c U[:,c] * (V[:,c] ⊗ W[:,c]).flatten()
        # So T_flat = U @ ((V ⊙ W).T) where ⊙ is the outer product per column
        
        # Build the linear system: T_flat = U @ B where B[c, (j,k)] = V[j,c]*W[k,c]
        # T_flat has dim dim_a*dim_b*dim_c entries
        # U has dim_a*target_rank entries
        # B has target_rank * dim_b*dim_c entries
        # So: T_flat.reshape(-1, 1) = U.T @ B.T ... let me think more carefully.
        
        # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # Let's define for each c: term_c[i,j,k] = U[i,c]*V[j,c]*W[k,c]
        # T = sum_c term_c
        
        # For fixed V,W: T is linear in U.
        # T_flat[i*dim_b*dim_c + j*dim_c + k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # = sum_c U[i,c] * (V[j,c]*W[k,c])
        
        # So if we define B[c, idx] = V[j,c]*W[k,c] where idx = j*dim_c+k
        # Then T_flat[i*dim_b*dim_c + idx] = sum_c U[i,c] * B[c, idx]
        # = (U @ B)[i, idx]
        
        # So T_mat = U @ B where T_mat is dim_a x (dim_b*dim_c)
        # T_mat[i, j*dim_c+k] = T[i,j,k]
        
        T_mat = T.reshape(dim_a, dim_b * dim_c)
        
        # B is target_rank x (dim_b*dim_c)
        B = np.zeros((target_rank, dim_b * dim_c))
        for c in range(target_rank):
            B[c, :] = (V_curr[:, c].reshape(-1, 1) * W_curr[:, c].reshape(1, -1)).flatten()
        
        # Solve T_mat = U @ B  =>  U = T_mat @ pinv(B)
        try:
            U_new = T_mat @ np.linalg.pinv(B)
        except:
            U_new = U_curr
        
        # Fix U_new, W_curr; solve for V_curr
        # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # For fixed U,W: T is linear in V.
        # T_flat[i*dim_b*dim_c + j*dim_c + k] = sum_c V[j,c] * U[i,c] * W[k,c]
        # T_mat2[j, i*dim_c+k] = sum_c V[j,c] * (U[i,c]*W[k,c])
        # T_mat2 = V @ C where C[c, i*dim_c+k] = U[i,c]*W[k,c]
        T_mat2 = T.reshape(dim_b, dim_a * dim_c)
        C = np.zeros((target_rank, dim_a * dim_c))
        for c in range(target_rank):
            C[c, :] = (U_new[:, c].reshape(-1, 1) * W_curr[:, c].reshape(1, -1)).flatten()
        try:
            V_new = T_mat2 @ np.linalg.pinv(C)
        except:
            V_new = V_curr
        
        # Fix U_new, V_new; solve for W_curr
        # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # For fixed U,V: T is linear in W.
        # T_flat[i*dim_b*dim_c + j*dim_c + k] = sum_c W[k,c] * U[i,c] * V[j,c]
        # T_mat3[k, i*dim_b+j] = sum_c W[k,c] * (U[i,c]*V[j,c])
        # T_mat3 = W @ D where D[c, i*dim_b+j] = U[i,c]*V[j,c]
        T_mat3 = T.reshape(dim_c, dim_a * dim_b)
        D = np.zeros((target_rank, dim_a * dim_b))
        for c in range(target_rank):
            D[c, :] = (U_new[:, c].reshape(-1, 1) * V_new[:, c].reshape(1, -1)).flatten()
        try:
            W_new = T_mat3 @ np.linalg.pinv(D)
        except:
            W_new = W_curr
        
        U_curr, V_curr, W_curr = U_new, V_new, W_new
        
        # Check convergence
        result = np.zeros((dim_a, dim_b, dim_c))
        for c in range(target_rank):
            result += np.einsum('i,j,k->ijk', U_curr[:, c], V_curr[:, c], W_curr[:, c])
        if np.allclose(result, T, atol=tol):
            return U_curr, V_curr, W_curr
    
    # Final check
    result = np.zeros((dim_a, dim_b, dim_c))
    for c in range(target_rank):
        result += np.einsum('i,j,k->ijk', U_curr[:, c], V_curr[:, c], W_curr[:, c])
    if np.allclose(result, T, atol=tol):
        return U_curr, V_curr, W_curr
    
    return None

def _try_direct_rank(r, n, m, p, max_iters=500, seed=42):
    """
    Try to directly construct a rank-r decomposition using random initialization
    and alternating linear projections.
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    dim_a, dim_b, dim_c = n * m, m * p, p * n
    T_mat = T.reshape(dim_a, dim_b * dim_c)
    T_mat2 = T.reshape(dim_b, dim_a * dim_c)
    T_mat3 = T.reshape(dim_c, dim_a * dim_b)
    
    rng = np.random.RandomState(seed)
    
    for attempt in range(5):
        U = rng.randn(dim_a, r)
        V = rng.randn(dim_b, r)
        W = rng.randn(dim_c, r)
        
        for it in range(max_iters):
            # Solve for U
            B = np.zeros((r, dim_b * dim_c))
            for c in range(r):
                B[c, :] = (V[:, c].reshape(-1, 1) * W[:, c].reshape(1, -1)).flatten()
            try:
                U = T_mat @ np.linalg.pinv(B)
            except:
                pass
            
            # Solve for V
            C = np.zeros((r, dim_a * dim_c))
            for c in range(r):
                C[c, :] = (U[:, c].reshape(-1, 1) * W[:, c].reshape(1, -1)).flatten()
            try:
                V = T_mat2 @ np.linalg.pinv(C)
            except:
                pass
            
            # Solve for W
            D = np.zeros((r, dim_a * dim_b))
            for c in range(r):
                D[c, :] = (U[:, c].reshape(-1, 1) * V[:, c].reshape(1, -1)).flatten()
            try:
                W = T_mat3 @ np.linalg.pinv(D)
            except:
                pass
            
            # Check
            result = np.zeros((dim_a, dim_b, dim_c))
            for c in range(r):
                result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            if np.allclose(result, T, atol=1e-9):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Start with the known valid 44-rank decomposition
    try:
        U, V, W = _combine_3x4x4()
        best_rank = 44
        best_U, best_V, best_W = U, V, W
    except Exception:
        U, V, W = _schoolbook(n, m, p)
        best_rank = n * m * p
        best_U, best_V, best_W = U, V, W
    
    # Try rank reduction from 44 downward
    for target_rank in range(43, 27, -1):
        if time.time() > deadline:
            break
        try:
            result = _try_rank_reduction(best_U, best_V, best_W, target_rank, n, m, p, max_iters=100)
            if result is not None:
                best_rank = target_rank
                best_U, best_V, best_W = result
                break  # Found a lower rank, stop
        except Exception:
            continue
    
    # Also try direct construction at various ranks
    for target_rank in [40, 38, 36, 34, 32, 30]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        try:
            result = _try_direct_rank(target_rank, n, m, p, max_iters=200, seed=target_rank)
            if result is not None:
                best_rank = target_rank
                best_U, best_V, best_W = result
        except Exception:
            continue
    
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
