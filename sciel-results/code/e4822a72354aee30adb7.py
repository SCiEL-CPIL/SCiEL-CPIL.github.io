"""
Search for a rank-minimal exact tensor decomposition of 3x4x6 matrix multiplication.

Strategy:
- Build the matrix multiplication tensor T[i,j,k] = 1 (in the appropriate index convention).
- Start from the schoolbook decomposition (rank = n*m*p = 72).
- Use alternating least squares (ALS) with integer/half-integer rounding to attempt to
  reduce the rank. The ALS step solves for one factor at a time given the other two,
  then rounds to the nearest half-integer. If the residual drops to zero, we have a
  valid decomposition.
- Try ranks from just below schoolbook downward, with a time budget.
"""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k].
    
    The tensor satisfies: for all A (n x m) and B (m x p),
    C = A @ B where C[i,k] = sum_j A[i,j] B[j,k].
    
    The trilinear form is: sum_{i,j,k} A[i,j] B[j,k] C[i,k] = <AB, C>
    So T[i,j,k] = 1 for all valid i, j, k.
    
    But wait - that's the all-ones tensor which has rank 1. That can't be right.
    
    Let me reconsider. The matrix multiplication map is bilinear:
    (A, B) -> AB
    
    As a tensor in R^{n*m} * R^{m*p} * (R^{n*p})^*, the tensor T is:
    T = sum_{i,j,k} e_{i,j} * e_{j,k} * e_{i,k}^*
    
    where e_{i,j} is the basis vector in R^{n*m} with 1 at position (i,j) [row-major: i*m+j],
    e_{j,k} is the basis vector in R^{m*p} with 1 at position (j,k) [row-major: j*p+k],
    e_{i,k}^* is the dual basis vector in (R^{n*p})^* corresponding to position (i,k).
    
    With the convention W[k*n+i, r], the output index is k*n + i.
    
    So the tensor components are:
    T[(i*m+j), (j*p+k), (k*n+i)] = 1
    T[other] = 0
    
    This is NOT the all-ones tensor. It's a sparse tensor where only specific
    (u, v, w) triples have value 1, and u, v, w are linked by the middle index j.
    """
    # T is a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _matmul_tensor_product(U, V, W):
    """Compute the tensor product sum_r U[:,r] * V[:,r] * W[:,r] as a 3D array."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Result: (n*m, m*p, p*n)
    rank = U.shape[1]
    T = np.zeros(U.shape[0:1] + V.shape[0:1] + W.shape[0:1], dtype=float)
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    return T

def _check_decomposition(U, V, W, T, tol=1e-8):
    """Check if the decomposition exactly matches the tensor."""
    T_recon = _matmul_tensor_product(U, V, W)
    return np.max(np.abs(T_recon - T)) < tol

def _round_to_half_integer(x):
    """Round to nearest half-integer."""
    return np.round(2 * x) / 2.0

def _als_step(T, U, V, W, n, m, p, max_iter=500, tol=1e-10, seed=42):
    """Alternating least squares for tensor decomposition.
    
    Given T of shape (n*m, m*p, p*n), find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    such that sum_r U[:,r] V[:,r] W[:,r] ≈ T.
    
    The ALS step:
    - Fix V and W, solve for U: each column u_r of U minimizes ||T - sum_r u_r * (V[:,r] * W[:,r])||
      This is a linear least squares problem.
    - Similarly for V and W.
    """
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        # Solve for U: T = sum_r u_r * outer(V[:,r], W[:,r])
        # Reshape: T_flat = sum_r u_r * vec(outer(V[:,r], W[:,r]))
        # For each r, the contribution to T is u_r * (V[:,r] * W[:,r]) (element-wise product reshaped)
        # Actually, T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For fixed r, the matrix of coefficients for U[:,r] is:
        # M_U[r] where M_U[r][i, (j,k)] = V[j,r] * W[k,r]
        # Then T_flat = M_U @ U_flat where T_flat is T reshaped to (n*m, m*p*p*n)
        
        # More efficiently:
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # For fixed i: sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k] for all j,k
        # This is: U[i,:] * (V * W) = T[i,:,:] where (V*W)[j,k,r] = V[j,r]*W[k,r]
        
        # Let's use a different approach:
        # T = sum_r u_r * v_r * w_r (outer products)
        # Fix V, W: minimize ||T - sum_r u_r * (v_r * w_r)||_F
        # where v_r * w_r is the outer product of V[:,r] and W[:,r]
        
        # Build the design matrix for U:
        # For each i, the equation is: sum_r U[i,r] * (V[:,r] * W[:,r])[j,k] = T[i,j,k]
        # Let's vectorize: for each i, we have m*p*p*n equations in rank unknowns U[i,0:rank]
        
        # Actually, let's just do it directly:
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # For fixed i: T[i,:,:] = sum_r U[i,r] * (V[:,r] * W[:,r])[:,:,0] ... no
        # V[:,r] has shape (m*p,), W[:,r] has shape (p*n,)
        # V[:,r] * W[:,r] element-wise doesn't make sense since they have different lengths.
        
        # I need to be more careful. Let me use einsum.
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # For fixed i: T[i,j,k] = sum_r U[i,r] * (V[j,r] * W[k,r])
        # Define M[i, r, j, k] = V[j,r] * W[k,r]
        # Then T[i,j,k] = sum_r M[i,r,j,k] * U[i,r]
        
        # This is a linear system: for each i, T[i,:,:] = M[i,:,:,:] @ U[i,:]
        # where M[i,:,:,:] has shape (rank, m*p, p*n)
        
        # Build M:
        # M[i, r, j, k] = V[j, r] * W[k, r]
        # But this doesn't depend on i! So M is the same for all i.
        
        # Let's define: for each r, the matrix P_r[j, k] = V[j, r] * W[k, r]
        # Then T[i, j, k] = sum_r U[i, r] * P_r[j, k]
        
        # Vectorize P_r: P_r_flat has length m*p*p*n
        # T_flat[i, :] = sum_r U[i, r] * P_r_flat
        # So T_flat = U @ P_flat where P_flat is (rank, m*p*p*n)
        
        P_flat = np.zeros((rank, m * p * p * n), dtype=float)
        for r in range(rank):
            # P_r[j, k] = V[j, r] * W[k, r]
            P_r = np.outer(V[:, r], W[:, r])  # shape (m*p, p*n)
            P_flat[r, :] = P_r.flatten()
        
        T_flat = T.reshape(n * m, m * p * p * n)
        
        # Solve: T_flat = U @ P_flat.T
        # U = T_flat @ pinv(P_flat.T)
        if rank > 0:
            U_new = T_flat @ np.linalg.pinv(P_flat.T)
        else:
            U_new = np.zeros((n * m, rank), dtype=float)
        
        # Similarly for V:
        # T[i, j, k] = sum_r V[j, r] * (U[i, r] * W[k, r])
        # For fixed j: T[:, j, k] = sum_r V[j, r] * Q_r[i, k]
        # where Q_r[i, k] = U[i, r] * W[k, r]
        Q_flat = np.zeros((rank, n * m * p * n), dtype=float)
        for r in range(rank):
            Q_r = np.outer(U[:, r], W[:, r])  # shape (n*m, p*n)
            Q_flat[r, :] = Q_r.flatten()
        
        T_flat_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
        
        if rank > 0:
            V_new = T_flat_v @ np.linalg.pinv(Q_flat.T)
        else:
            V_new = np.zeros((m * p, rank), dtype=float)
        
        # Similarly for W:
        # T[i, j, k] = sum_r W[k, r] * (U[i, r] * V[j, r])
        # For fixed k: T[i, j, k] = sum_r W[k, r] * R_r[i, j]
        # where R_r[i, j] = U[i, r] * V[j, r]
        R_flat = np.zeros((rank, n * m * m * p), dtype=float)
        for r in range(rank):
            R_r = np.outer(U[:, r], V[:, r])  # shape (n*m, m*p)
            R_flat[r, :] = R_r.flatten()
        
        T_flat_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
        
        if rank > 0:
            W_new = T_flat_w @ np.linalg.pinv(R_flat.T)
        else:
            W_new = np.zeros((p * n, rank), dtype=float)
        
        # Update
        U = U_new
        V = V_new
        W = W_new
        
        # Check convergence
        T_recon = _matmul_tensor_product(U, V, W)
        err = np.max(np.abs(T_recon - T))
        
        if err < tol:
            break
    
    return U, V, W, err

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the tensor
    T = _build_tensor(n, m, p)
    
    # Schoolbook decomposition
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=float)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=float)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Verify schoolbook
    if not _check_decomposition(U_sb, V_sb, W_sb, T):
        # Should not happen, but fallback
        pass
    
    # Try to find lower rank decompositions using ALS
    start_time = time.time()
    time_limit = 100  # seconds
    
    # Try ranks from schoolbook-1 downward
    for rank in range(schoolbook_rank - 1, max(10, schoolbook_rank - 30), -1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
            
            np.random.seed(rank * 1000 + attempt)
            
            # Random initialization with half-integers
            U_try = _round_to_half_integer(np.random.randn(n * m, rank) * 0.5)
            V_try = _round_to_half_integer(np.random.randn(m * p, rank) * 0.5)
            W_try = _round_to_half_integer(np.random.randn(p * n, rank) * 0.5)
            
            # Run ALS
            U_als, V_als, W_als, err = _als_step(
                T, U_try, V_try, W_try, n, m, p, 
                max_iter=300, tol=1e-10, seed=rank * 100 + attempt
            )
            
            # Round to half-integers
            U_rounded = _round_to_half_integer(U_als)
            V_rounded = _round_to_half_integer(V_als)
            W_rounded = _round_to_half_integer(W_als)
            
            # Check if rounded version is exact
            if _check_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-8):
                best_rank = rank
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                found = True
                break
            
            # Also check the unrounded version (in case it's already close to half-integers)
            if _check_decomposition(U_als, V_als, W_als, T, tol=1e-8):
                best_rank = rank
                best_U, best_V, best_W = U_als, V_als, W_als
                found = True
                break
        
        if found:
            continue  # Try to go even lower
        else:
            break  # Couldn't find this rank, stop
    
    # Return the best solution
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
