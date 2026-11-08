"""Program to find a rank-minimal tensor decomposition for 3x3 by 3x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 115  # Safety margin from 120s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Returns:
        dict: {'U': (9, rank), 'V': (18, rank), 'W': (18, rank)}
              U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
              where n=3, m=3, p=6
    """
    n, m, p = DIMENSIONS
    
    # Construct the target tensor T of shape (n*m, m*p, p*n)
    # T[alpha, beta, gamma] = 1 if:
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    # else 0
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    
    # Initialize schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1

    def compute_error(U, V, W):
        """Compute the Frobenius norm of the difference between T and the decomposition."""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T - recon)

    def try_round_and_verify(U_float, V_float, W_float, target_rank):
        """
        Try to round U, V, W to integer or half-integer values and verify
        that the rounded decomposition exactly reconstructs T.
        Returns (U_rounded, V_rounded, W_rounded, success)
        """
        # Try half-integer rounding
        U_half = np.round(U_float * 2) / 2
        V_half = np.round(V_float * 2) / 2
        W_half = np.round(W_float * 2) / 2
        
        err_half = compute_error(U_half, V_half, W_half)
        if err_half < 1e-9:
            return U_half, V_half, W_half, True
        
        # Try integer rounding
        U_int = np.round(U_float)
        V_int = np.round(V_float)
        W_int = np.round(W_float)
        
        err_int = compute_error(U_int, V_int, W_int)
        if err_int < 1e-9:
            return U_int, V_int, W_int, True
        
        return None, None, None, False

    def als_decompose(target_T, rank, max_iter=200, tol=1e-10):
        """
        Perform ALS decomposition of target_T into U, V, W.
        target_T shape: (I, J, K)
        U shape: (I, rank)
        V shape: (J, rank)
        W shape: (K, rank)
        """
        I, J, K = target_T.shape
        
        # Initialize with small random values
        np.random.seed(42)
        U = np.random.randn(I, rank) * 0.1
        V = np.random.randn(J, rank) * 0.1
        W = np.random.randn(K, rank) * 0.1
        
        # Unfold target_T
        T_1 = target_T.reshape(I, J * K)
        T_2 = np.transpose(target_T, (1, 0, 2)).reshape(J, I * K)
        T_3 = np.transpose(target_T, (2, 0, 1)).reshape(K, I * J)
        
        prev_error = float('inf')
        
        for it in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
                
            # Update U
            G = np.zeros((J * K, rank))
            for j in range(J):
                for k in range(K):
                    G[j * K + k, :] = V[j, :] * W[k, :]
            
            try:
                U_new, _, _, _ = np.linalg.lstsq(G.T, T_1.T, rcond=None)
                U = U_new.T
            except np.linalg.LinAlgError:
                pass

            # Update V
            G2 = np.zeros((I * K, rank))
            for r in range(rank):
                G2[:, r] = np.kron(U[:, r], W[:, r])
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(G2.T, T_2.T, rcond=None)
                V = V_new.T
            except np.linalg.LinAlgError:
                pass

            # Update W
            G3 = np.zeros((I * J, rank))
            for r in range(rank):
                G3[:, r] = np.kron(U[:, r], V[:, r])
            
            try:
                W_new, _, _, _ = np.linalg.lstsq(G3.T, T_3.T, rcond=None)
                W = W_new.T
            except np.linalg.LinAlgError:
                pass

            error = compute_error(U, V, W)
            if abs(prev_error - error) < tol:
                break
            prev_error = error

        return U, V, W, error

    # Try to find a lower rank decomposition
    # Start from rank 50 and go down
    for target_rank in range(50, 40, -1):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        found = False
        for attempt in range(10):
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
            
            # Use different random seeds for diversity
            np.random.seed(attempt * 1000 + target_rank)
            U_curr, V_curr, W_curr, err = als_decompose(T, target_rank, max_iter=100)
            
            if err < 1e-4:
                # Try to round to integer/half-integer
                U_rounded, V_rounded, W_rounded, success = try_round_and_verify(
                    U_curr, V_curr, W_curr, target_rank
                )
                
                if success:
                    best_rank = target_rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    found = True
                    break

        if found:
            break

    # If no better solution found, try a more structured approach
    # based on block decomposition
    if best_rank == n * m * p:
        # Try to exploit the block structure: 3x3 * 3x6 = three 3x3 * 3x3 products
        # with shared A matrix
        # This is equivalent to computing A*B1, A*B2, A*B3 where B = [B1, B2, B3]
        
        # Try rank 53, 52, 51 with more attempts
        for target_rank in [53, 52, 51]:
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            found = False
            for attempt in range(20):
                if time.time() - START_TIME > TIME_LIMIT - 10:
                    break
                
                np.random.seed(attempt * 777 + target_rank * 13)
                U_curr, V_curr, W_curr, err = als_decompose(T, target_rank, max_iter=150)
                
                if err < 1e-3:
                    U_rounded, V_rounded, W_rounded, success = try_round_and_verify(
                        U_curr, V_curr, W_curr, target_rank
                    )
                    
                    if success:
                        best_rank = target_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        found = True
                        break

            if found:
                break

    # Return the best solution found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
