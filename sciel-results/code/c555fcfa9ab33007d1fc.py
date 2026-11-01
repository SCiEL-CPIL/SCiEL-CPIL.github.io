"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x7 matrix.

    Returns dict with keys 'U', 'V', 'W':
      U: shape (n*m, rank)
      V: shape (m*p, rank)
      W: shape (p*n, rank)

    The decomposition satisfies:
      C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    This implementation uses a numerical search approach to find a lower rank
    decomposition if one exists, falling back to the schoolbook algorithm
    if no lower rank is found within the time limit.
    """
    n, m, p = DIMENSIONS
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # We need to decompose this tensor.
    
    def schoolbook_decomp():
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

    def verify_decomp(U, V, W):
        """
        Verify that the decomposition exactly reconstructs the tensor.
        U: list of lists or numpy array (n*m, rank)
        V: list of lists or numpy array (m*p, rank)
        W: list of lists or numpy array (p*n, rank)
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        rank = U_np.shape[1]
        if V_np.shape[1] != rank or W_np.shape[1] != rank:
            return False
            
        # Reconstruct tensor T[i, j, k]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We check if T[i,j,k] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integers(arr):
        """
        Snap values to the nearest half-integer.
        """
        # Multiply by 2, round, divide by 2
        return np.round(arr * 2) / 2

    def try_find_lower_rank():
        """
        Attempt to find a rank R < n*m*p decomposition using gradient descent.
        """
        max_rank = n * m * p
        # Try ranks from max_rank - 1 down to max_rank - 5
        # to save time.
        for R in range(max_rank - 1, max_rank - 6, -1):
            # Initialize with random values
            np.random.seed(42)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Gradient Descent
            lr = 0.01
            for step in range(500):
                # Compute gradients
                # T_recon[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # dT/dU[i*m+j, r] = sum_k V[j*p+k, r] W[k*n+i, r] * (1 - T_recon[i,j,k])? 
                # Actually, we want to minimize sum (T_recon - 1)^2.
                
                # Efficient computation of gradients is complex, so we use finite differences 
                # or a simple numerical gradient for this small size.
                # Given small size, we can compute the tensor and gradients directly.
                
                # Compute current tensor
                T_recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                
                # Check convergence
                error = np.sum((T_recon - 1.0)**2)
                if error < 1e-12:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = 2 * (T_recon[i,j,k] - 1.0)
                            for r in range(R):
                                grad_U[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                                grad_V[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                                grad_W[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
                                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Optional: Symmetrize or constrain? No, keep it general.
                
            # After optimization, check if we found a solution
            T_final = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_final[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            
            if np.allclose(T_final, 1.0, atol=1e-8):
                # Try to snap to half-integers
                U_snap = snap_to_half_integers(U)
                V_snap = snap_to_half_integers(V)
                W_snap = snap_to_half_integers(W)
                
                # Verify snapped solution
                if verify_decomp(U_snap, V_snap, W_snap):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
        
        return None

    # First, try to find a lower rank
    result = try_find_lower_rank()
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return schoolbook_decomp()
# EVOLVE-BLOCK-END
