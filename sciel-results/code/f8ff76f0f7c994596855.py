import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # safety margin before 120s limit

def _time_remaining():
    return _DEADLINE - time.time()

def _build_rank25():
    """
    Build the known-good rank-25 decomposition for 2x2 * 2x7 matrix multiplication.
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
    
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    # Schoolbook for last column (column 6)
    col_start = 21
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_rank28_fallback():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _find_decomposition(n, m, p, rank, max_iters=200, tol=1e-8):
    """
    Try to find a rank decomposition for 2x2xp tensor using ALS.
    Returns (U, V, W) or None.
    """
    if _time_remaining() < 5.0:
        return None
        
    # Target tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if j is the summation index (always true for matrix mult)
    # Actually, for 2x2xp, T[i, j, k] = 1 for all i, j, k?
    # No, T[i, j, k] = 1 if the term A[i, j] B[j, k] exists.
    # In 2x2xp, A is 2x2, B is 2xp.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # So T[i, j, k] = 1 for all i in 0..1, j in 0..1, k in 0..p-1.
    T = np.ones((n, m, p))
    
    best_error = np.inf
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(10):
        if _time_remaining() < 5.0:
            break
            
        # Initialize randomly
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for it in range(max_iters):
            if _time_remaining() < 5.0:
                break
                
            # Update U
            # T = sum_r U_r V_r W_r
            # For fixed V, W, T is linear in U.
            # T.reshape(n*m, m*p*p*n) = U @ (V.reshape(m*p, rank) ... )
            # This is complex to vectorize directly.
            # Use iterative projection.
            
            # Project U
            # T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}
            # For each i, we have a matrix of size (m*p) x (rank) ?
            # Let's solve for U[:, r] by fixing V, W.
            # T.reshape(n, m, p)
            # For a fixed r, U[:, r] is a vector of size n*m.
            # This is a system of linear equations.
            
            # Simpler: Use gradient descent or scipy minimize on the flattened parameters.
            # But ALS is faster.
            
            # Let's use the property:
            # T @ (V_r \otimes W_r) = U_r
            # This is not quite right.
            
            # We can use the following:
            # T = U (V \otimes W)
            # Where V \otimes W is a matrix of size (m*p * p*n) x rank? No.
            
            # Let's just use scipy.optimize.minimize on the error.
            # Flatten U, V, W.
            # This might be slow.
            
            # Alternative: Simple random search with local refinement?
            # Given the small size, let's try a few random initializations and run ALS properly.
            
            # ALS for U:
            # T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}
            # For a fixed i, T_{i, :, :} = sum_r U_{i, r} (V_{:, r} \otimes W_{:, r})
            # Let M_r = V_{:, r} \otimes W_{:, r} (size m*p x p*n? No, V is m*p, W is p*n)
            # V is shape (m*p, rank). W is shape (p*n, rank).
            # For a given r, V[:, r] is a vector of size m*p.
            # W[:, r] is a vector of size p*n.
            # The outer product V[:, r] \otimes W[:, r] is a matrix of size (m*p) x (p*n).
            # T_{i, :, :} is a matrix of size (m*p) x (p*n) if we reshape T[i] to (m, p) -> (m*p)?
            # T[i] is shape (m, p). Reshape to (m*p, 1)? No.
            # T[i, j, k] corresponds to element (j, k) of the matrix T_i.
            # So T_i (size m x p) = sum_r U_{i, r} (V[:, r] \otimes W[:, r])_{j, k}?
            # V[:, r] is size m*p. W[:, r] is size p*n.
            # This doesn't match dimensions directly.
            
            # Let's reshape T to (n, m, p).
            # For fixed i, T[i] is (m, p).
            # We want to express T[i] as sum_r U_{i, r} * (something).
            # The "something" must be a matrix of size (m, p).
            # V[:, r] is size (m*p). W[:, r] is size (p*n).
            # This approach is tricky because V and W indices are coupled with j and k.
            
            # Let's use the explicit loop for ALS update.
            # For each i (0 to n-1):
            #   Build a system for U[i, :].
            #   T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            #   This is a linear system in U[i, :].
            #   Let X be a matrix of size (m*p) x rank.
            #   X[j*p + k, r] = V[j, r] * W[k, r]
            #   Then T[i].flatten() = X @ U[i, :]
            #   Solve U[i, :] = lstsq(X, T[i].flatten())
            
            for i in range(n):
                X = np.zeros((m*p, rank))
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            X[j*p + k, r] = V[j*p + k, r] * W[k*n + i, r]
                # Note: W index is k*n + i.
                # V index is j*p + k? No, V is (m*p, rank).
                # V row index corresponds to (j, k) in B?
                # B is (m, p). Flatten to m*p. Index j*p + k.
                # W is (p*n, rank). Index k*n + i.
                # U is (n*m, rank). Index i*m + j.
                
                # Wait, my index mapping in the problem statement:
                # U: (n*m, rank). Index i*m + j.
                # V: (m*p, rank). Index j*p + k.
                # W: (p*n, rank). Index k*n + i.
                
                # So for T[i, j, k]:
                # Term is U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                
                # For fixed i, we solve for U[i*m + 0 : i*m + m, :]
                # T[i, :, :] is a matrix of size (m, p).
                # Flatten to (m*p, 1).
                # X is (m*p) x rank.
                # X[j*p + k, r] = V[j*p + k, r] * W[k*n + i, r]
                
                t_flat = T[i].flatten()
                U[i*m : (i+1)*m, :] = np.linalg.lstsq(X, t_flat, rcond=None)[0]
            
            # Similarly for V and W.
            # For fixed j, solve for V[j*p : (j+1)*p, :]
            # T[:, j, :] is size (n, p). Flatten to (n*p, 1).
            # X is (n*p) x rank.
            # X[i*n + k, r] = U[i*m + j, r] * W[k*n + i, r]
            
            for j in range(m):
                X = np.zeros((n*p, rank))
                for i in range(n):
                    for k in range(p):
                        for r in range(rank):
                            X[i*n + k, r] = U[i*m + j, r] * W[k*n + i, r]
                t_flat = T[:, j, :].flatten()
                V[j*p : (j+1)*p, :] = np.linalg.lstsq(X, t_flat, rcond=None)[0]
            
            # For fixed k, solve for W[k*n : (k+1)*n, :]
            # T[:, :, k] is size (n, m). Flatten to (n*m, 1).
            # X is (n*m) x rank.
            # X[i*m + j, r] = U[i*m + j, r] * V[j*p + k, r]
            
            for k in range(p):
                X = np.zeros((n*m, rank))
                for i in range(n):
                    for j in range(m):
                        for r in range(rank):
                            X[i*m + j, r] = U[i*m + j, r] * V[j*p + k, r]
                t_flat = T[:, :, k].flatten()
                W[k*n : (k+1)*n, :] = np.linalg.lstsq(X, t_flat, rcond=None)[0]
            
            # Calculate error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(rank):
                            val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                        error += (val - T[i, j, k])**2
                        if _time_remaining() < 2.0:
                            break
                    if _time_remaining() < 2.0: break
                if _time_remaining() < 2.0: break
            if _time_remaining() < 2.0: break
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if error < tol:
                break
                
    if best_error < tol:
        return best_U, best_V, best_W
    return None

def _combine_decompositions(U1, V1, W1, U2, V2, W2, p1, p2):
    """
    Combine two decompositions for 2x2xp1 and 2x2xp2 into 2x2x(p1+p2).
    """
    n, m = 2, 2
    p = p1 + p2
    rank = U1.shape[1] + U2.shape[1]
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    
    # Part 1
    U[:, :r1] = U1
    V[:, :r1] = V1
    W[:, :r1] = W1
    
    # Part 2
    U[:, r1:] = U2
    # V2 is for columns 0..p2-1. We need to shift to p1..p1+p2-1.
    V2_shifted = np.zeros((m*p, r2))
    for j in range(m):
        for k in range(p2):
            V2_shifted[j*p + (k + p1), :] = V2[j*p + k, :]
    V[:, r1:] = V2_shifted
    
    # W2 is for columns 0..p2-1. We need to shift indices.
    # W index is k*n + i.
    W2_shifted = np.zeros((p*n, r2))
    for k in range(p2):
        for i in range(n):
            W2_shifted[(k + p1)*n + i, :] = W2[k*n + i, :]
    W[:, r1:] = W2_shifted
    
    return U, V, W

def construct():
    if _time_remaining() < 5.0:
        return _build_rank28_fallback()
        
    # Try to find Rank 9 for 2x2x3
    U3, V3, W3 = _find_decomposition(2, 2, 3, 9)
    
    # Try to find Rank 12 for 2x2x4
    U4, V4, W4 = _find_decomposition(2, 2, 4, 12)
    
    if U3 is not None and U4 is not None:
        U, V, W = _combine_decompositions(U4, V4, W4, U3, V3, W3, 4, 3)
        rank = U.shape[1]
        # Verify
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # Quick check
        np.random.seed(42)
        A = np.random.rand(2, 2)
        B = np.random.rand(2, 7)
        C_exp = A @ B
        C_comp = np.zeros((2, 7))
        for r in range(rank):
            SA = np.dot(U[:, r], A.flatten())
            SB = np.dot(V[:, r], B.flatten())
            for k in range(7):
                for i in range(2):
                    C_comp[i, k] += W[k*2 + i, r] * SA * SB
        if np.allclose(C_comp, C_exp, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to Rank 25
    return _build_rank25()

# EVOLVE-BLOCK-END
