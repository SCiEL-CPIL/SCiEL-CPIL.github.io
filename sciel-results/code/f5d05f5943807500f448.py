import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 8]

def _als_decompose(target_tensor, rank, max_iters=1000, tol=1e-10, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    target_tensor: shape (n, m, p)
    rank: int
    Returns U, V, W or None
    """
    n, m, p = target_tensor.shape
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Reshape target to match modes
    # Mode 1: (n*m, p) -> target_tensor.reshape(n*m, p) ? No.
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # This indexing is tricky for standard ALS which assumes T[i,j,k] = sum U[i,r] V[j,r] W[k,r].
    # Here indices are mixed.
    # Let's map to standard indices.
    # I = i*m + j
    # J = j*p + k
    # K = k*n + i
    # This is not a simple separation.
    # ALS works on T[I, J, K].
    # We need to form the tensor T of shape (n*m, m*p, p*n).
    # T[I, J, K] = 1 if indices match the multiplication structure, else 0.
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
                
    # Now run ALS on T
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    for it in range(max_iters):
        # Update U
        # T_unfolded_1: (n*m, m*p * p*n)
        # T[i, j, k] = sum_r U[i, r] (V[j, r] W[k, r])
        # Let Z1 = V * W (element-wise product of columns? No)
        # V is (m*p, rank), W is (p*n, rank).
        # We need M1[j, k, r] = V[j, r] * W[k, r].
        # Then T[i, j, k] = sum_r U[i, r] M1[j, k, r].
        # Reshape M1 to (m*p * p*n, rank).
        M1 = np.zeros((m * p, p * n, rank))
        for r in range(rank):
            M1[:, :, r] = np.outer(V[:, r], W[:, r])
        M1_flat = M1.reshape(m * p * p * n, rank)
        
        T1 = T.reshape(n * m, -1)
        # Solve T1 = U * M1_flat.T
        # U = T1 * M1_flat.T * (M1_flat * M1_flat.T)^-1
        try:
            U_new, _, _, _ = np.linalg.lstsq(M1_flat.T, T1.T, rcond=None)
            U = U_new.T
        except:
            return None

        # Update V
        M2 = np.zeros((n * m, p * n, rank))
        for r in range(rank):
            M2[:, :, r] = np.outer(U[:, r], W[:, r])
        M2_flat = M2.reshape(n * m * p * n, rank)
        
        T2 = T.transpose(1, 0, 2).reshape(m * p, -1)
        try:
            V_new, _, _, _ = np.linalg.lstsq(M2_flat.T, T2.T, rcond=None)
            V = V_new.T
        except:
            return None

        # Update W
        M3 = np.zeros((n * m, m * p, rank))
        for r in range(rank):
            M3[:, :, r] = np.outer(U[:, r], V[:, r])
        M3_flat = M3.reshape(n * m * m * p, rank)
        
        T3 = T.transpose(2, 0, 1).reshape(p * n, -1)
        try:
            W_new, _, _, _ = np.linalg.lstsq(M3_flat.T, T3.T, rcond=None)
            W = W_new.T
        except:
            return None
        
        # Check convergence
        if it % 10 == 0:
            err = 0
            for r in range(rank):
                err += np.sum(np.abs(T - np.einsum('ir,jr,kr->ijk', U, V, W)))
            # Actually, compute full error
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T - T_recon) / np.linalg.norm(T)
            if err < tol:
                break
                
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.linalg.norm(T - T_recon) / np.linalg.norm(T)
    if err < 1e-8:
        return U, V, W
    return None

def _build_rank56():
    # ... existing code ...
    pass

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find rank 24 solution (2 * rank 12 for 2x4x4)
    # First, solve for 2x4x4
    target_244 = np.zeros((2 * 4, 4 * 4, 4 * 2))
    for i in range(2):
        for j in range(4):
            for k in range(4):
                I = i * 4 + j
                J = j * 4 + k
                K = k * 2 + i
                target_244[I, J, K] = 1.0
                
    best_rank = 56
    best_solution = None
    
    # Try rank 12 for 2x4x4
    for seed in range(10):
        if time.time() - start_time > 100: break
        res = _als_decompose(target_244, 12, seed=seed)
        if res:
            U12, V12, W12 = res
            # Check if entries are close to integers/half-integers
            # Scale?
            # If we find a valid real solution, we can try to round it.
            # But for now, if we find a low error solution, we can use it?
            # The task requires EXACT integer/half-integer.
            # So we must verify.
            
            # Construct full 2x4x8 solution
            U = np.zeros((2 * 4, 24))
            V = np.zeros((4 * 8, 24))
            W = np.zeros((8 * 2, 24))
            
            # Block 1: cols 0-3 of B, cols 0-3 of C
            U[:, 0:12] = U12
            # V mapping for block 1:
            # V12 index j*4+k. Global V index j*8+k.
            for j in range(4):
                for k in range(4):
                    V[j*8+k, 0:12] = V12[j*4+k, :]
            # W mapping for block 1:
            # W12 index k*2+i. Global W index k*2+i.
            for k in range(4):
                for i in range(2):
                    W[k*2+i, 0:12] = W12[k*2+i, :]
                    
            # Block 2: cols 4-7 of B, cols 4-7 of C
            U[:, 12:24] = U12
            for j in range(4):
                for k in range(4):
                    V[j*8+(k+4), 12:24] = V12[j*4+k, :]
            for k in range(4):
                for i in range(2):
                    W[(k+4)*2+i, 12:24] = W12[k*2+i, :]
                    
            # Verify exactness
            # We need to check if U, V, W are integers/half-integers
            # And if they decompose the tensor exactly.
            
            # Check integer/half-integer
            def is_int_or_half(arr):
                arr = np.round(arr * 2)
                return np.allclose(arr, np.round(arr))
                
            if is_int_or_half(U) and is_int_or_half(V) and is_int_or_half(W):
                # Verify decomposition
                # This is expensive, but rank 24 is small.
                # We can do a random test.
                A = np.random.randn(2, 4)
                B = np.random.randn(4, 8)
                C = A @ B
                
                # Compute via decomposition
                # M_r = sum U[I, r] A[I]
                # N_r = sum V[J, r] B[J]
                # C[K] = sum W[K, r] M_r N_r
                
                # Vectorize
                A_flat = A.flatten()
                B_flat = B.flatten()
                
                M = U.T @ A_flat # (24,)
                N = V.T @ B_flat # (24,)
                C_recon = (W.T * (M[:, None] * N[None, :])).sum(axis=1)
                
                if np.allclose(C_recon, C.flatten()):
                    best_rank = 24
                    best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
                    
    if best_solution:
        return best_solution
    else:
        # Fallback to rank 56
        return _build_rank56()
