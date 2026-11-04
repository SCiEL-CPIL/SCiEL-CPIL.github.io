import numpy as np
import time

def construct():
    start_time = time.time()
    max_time = 140  # Safety margin
    
    n, m, p = 2, 3, 4
    
    # Build the correct sparse target tensor for matrix multiplication
    # U index I = i*m + j
    # V index J = j*p + k
    # W index K = k*n + i (based on task description W[k*n+i])
    
    num_U = n * m
    num_V = m * p
    num_W = p * n
    
    T = np.zeros((num_U, num_V, num_W))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def verify_decomposition(U, V, W, R):
        """Verify that U, V, W with rank R exactly decompose the tensor."""
        # U, V, W are assumed to be 2D arrays of shape (dim, R)
        # Reshape to 3D for einsum or use direct einsum on 2D
        # T_recon[I, J, K] = sum_r U[I, r] * V[J, r] * W[K, r]
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)

    def try_rank(R):
        """Try to find rank R decomposition."""
        # For small R, try random restarts with ALS
        # We search for U, V, W in the space of real numbers, then snap to half-integers?
        # The task asks for integer/half-integer entries.
        
        for restart in range(20):
            # Initialize randomly
            U = np.random.uniform(-1, 1, (num_U, R))
            V = np.random.uniform(-1, 1, (num_V, R))
            W = np.random.uniform(-1, 1, (num_W, R))
            
            # ALS iterations
            for _ in range(50):
                # Update U
                for i in range(num_U):
                    # T[i, :, :] = sum_r U[i, r] * V[:, r] * W[:, r]
                    # We want sum_r U[i, r] * (V[:, r] * W[:, r]) = T[i, :, :]
                    # Let M = V * W (elementwise product, shape num_V x num_W x R)
                    # This is expensive.
                    # Alternative: Fix V, W. Update U[i, :].
                    # T[i, J, K] = sum_r U[i, r] V[J, r] W[K, r]
                    # This is a linear system for U[i, :].
                    # Let's build the matrix for the least squares.
                    # A[r, (J, K)] = V[J, r] * W[K, r]
                    # b[(J, K)] = T[i, J, K]
                    # Solve A x = b for x = U[i, :]
                    
                    # Efficient way:
                    # V_W = V @ W.T ? No.
                    # V is (num_V, R), W is (num_W, R).
                    # Outer product V[:, r] W[:, r] for each r.
                    # We can form a matrix M of shape (R, num_V * num_W)
                    # M[r, J*num_W + K] = V[J, r] * W[K, r]
                    # Then b is T[i].flatten()
                    # U[i, :] = lstsq(M, b)
                    
                    M = (V[:, :, None] * W[:, None, :]).reshape(R, -1)
                    b = T[i].flatten()
                    U[i, :], residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
                
                # Update V
                for j in range(num_V):
                    M = (U[:, :, None] * W[:, None, :]).reshape(R, -1)
                    b = T[:, j, :].flatten()
                    V[j, :], residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
                    
                # Update W
                for k in range(num_W):
                    M = (U[:, :, None] * V[:, None, :]).reshape(R, -1)
                    b = T[:, :, k].flatten()
                    W[k, :], residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
            
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if verify_decomposition(U_snap, V_snap, W_snap, R):
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
        
        return None

    # Try ranks from 1 upwards
    # Note: Rank 1 is impossible for sparse MM tensor unless n=m=p=1
    # Optimal is 19.
    for R in range(1, 25):
        if time.time() - start_time > max_time:
            break
        dec = try_rank(R)
        if dec and verify_decomposition(np.array(dec["U"]), np.array(dec["V"]), np.array(dec["W"]), R):
            return dec
    
    # Fallback to schoolbook rank 24
    rank = 24
    U = [[0] * rank for _ in range(num_U)]
    V = [[0] * rank for _ in range(num_V)]
    W = [[0] * rank for _ in range(num_W)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                U[I][column] = 1
                V[J][column] = 1
                W[K][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
