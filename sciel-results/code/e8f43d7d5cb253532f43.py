"""Program to find a rank-minimal tensor decomposition for 3x3 by 3x6 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

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
        # Efficient reconstruction using einsum
        # T[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T - recon)

    def als_decompose(target_T, rank, max_iter=100, tol=1e-9):
        """
        Perform ALS decomposition of target_T into U, V, W.
        target_T shape: (I, J, K)
        U shape: (I, rank)
        V shape: (J, rank)
        W shape: (K, rank)
        """
        I, J, K = target_T.shape
        
        # Initialize
        U = np.random.randn(I, rank)
        V = np.random.randn(J, rank)
        W = np.random.randn(K, rank)
        
        # Unfold target_T
        # T_1: (I, J*K) where T_1[i, j*K+k] = T[i, j, k]
        T_1 = target_T.reshape(I, J * K)
        T_2 = np.transpose(target_T, (1, 0, 2)).reshape(J, I * K)
        T_3 = np.transpose(target_T, (2, 0, 1)).reshape(K, I * J)
        
        prev_error = float('inf')
        
        for it in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - 2:
                break
                
            # Update U
            # T_1 approx U @ (V kron W).T
            # V kron W is (J*K, rank)
            # (V kron W).T is (rank, J*K)
            # U = T_1 @ (V kron W).T @ inv((V kron W) @ (V kron W).T)
            # Or just use least squares: U = lstsq(V_kron_W.T, T_1.T).T
            
            VK = np.kron(V.T, W.T) # (rank, J*K) ? No.
            # We need M such that M.T @ M = (V kron W) @ (V kron W).T
            # Let's form the matrix G = V_kron_W of shape (J*K, rank)
            # G[j*K+k, r] = V[j, r] * W[k, r]
            G = np.zeros((J * K, rank))
            for j in range(J):
                for k in range(K):
                    G[j * K + k, :] = V[j, :] * W[k, :]
            
            # Solve U G = T_1  => U = T_1 G^+
            try:
                U_new, _, _, _ = np.linalg.lstsq(G.T, T_1.T, rcond=None)
                U = U_new.T
            except np.linalg.LinAlgError:
                pass

            # Update V
            # T_2 approx V @ (W kron U).T
            # G2 = W_kron_U of shape (I*K, rank)
            # G2[i*K+k, r] = W[i, r] * U[k, r] ? No.
            # T_2[j, i*K+k] = T[j, i, k]
            # Term: V[j, r] * W[i, r] * U[k, r] ? No.
            # T[j, i, k] = sum_r V[j, r] * U[i, r] * W[k, r]
            # So for fixed j, T_2[j, :] is a vector of length I*K.
            # It equals sum_r V[j, r] * (U[:, r] kron W[:, r]) ?
            # Let's check indices.
            # Vector element at index i*K+k is T[j, i, k].
            # sum_r V[j, r] U[i, r] W[k, r].
            # This is V[j, r] * (U[:, r] * W[:, r] flattened).
            # So G2[:, r] = U[:, r] * W[:, r] flattened?
            # Wait, U is (I, rank), W is (K, rank).
            # For a fixed r, the vector is U[:, r] * W[:, r] (element-wise product of vectors? No, outer product flattened).
            # U[i, r] * W[k, r].
            # So the vector is np.kron(U[:, r], W[:, r])? No, just element-wise product if we align indices?
            # Index in T_2 is i*K + k.
            # Value is U[i, r] * W[k, r].
            # This is exactly the element-wise product of U[:, r] (repeated K times) and W[:, r] (repeated I times)?
            # No, it's the Kronecker product of U[:, r] and W[:, r] if we view them as vectors?
            # np.kron(U[:, r], W[:, r]) has length I*K.
            # Element i*K + k is U[i, r] * W[k, r]. Yes.
            
            G2 = np.zeros((I * K, rank))
            for r in range(rank):
                G2[:, r] = np.kron(U[:, r], W[:, r])
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(G2.T, T_2.T, rcond=None)
                V = V_new.T
            except np.linalg.LinAlgError:
                pass

            # Update W
            # T_3 approx W @ (U kron V).T
            # T_3[k, i*J+j] = T[i, j, k]
            # Term: W[k, r] * U[i, r] * V[j, r]
            # Vector for fixed r: U[:, r] * V[:, r] flattened?
            # Index i*J + j. Value U[i, r] * V[j, r].
            # This is np.kron(U[:, r], V[:, r]).
            
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
    # We know rank 54 is possible.
    # Try ranks 50, 48, 46, 44, 42, 40
    for target_rank in range(50, 39, -2):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        found = False
        for attempt in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            U_curr, V_curr, W_curr, err = als_decompose(T, target_rank, max_iter=50)
            
            if err < 1e-6:
                # Try to round to half-integers
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_curr * 2) / 2
                
                err_rounded = compute_error(U_rounded, V_rounded, W_rounded)
                
                if err_rounded < 1e-9:
                    best_rank = target_rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    found = True
                    break
                else:
                    # Maybe integers?
                    U_int = np.round(U_curr)
                    V_int = np.round(V_curr)
                    W_int = np.round(W_curr)
                    err_int = compute_error(U_int, V_int, W_int)
                    if err_int < 1e-9:
                        best_rank = target_rank
                        best_U = U_int
                        best_V = V_int
                        best_W = W_int
                        found = True
                        break

        if found:
            break

    # If no better integer/half-integer solution found, return schoolbook
    # Note: If we found a float solution with low error but couldn't round it,
    # we might still return it if the evaluator allows floats?
    # The prompt says "integer/half-integer entries".
    # So we must stick to schoolbook if we can't find a valid one.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
