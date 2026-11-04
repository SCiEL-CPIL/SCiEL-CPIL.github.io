"""Rewritten program: rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 115  # safety margin before 120s limit

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k', k*n+i'] = 1 iff j==j' and i==i' and k==k'
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, atol=1e-9):
        """Check if the decomposition exactly reproduces the tensor."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_computed, T, atol=atol)

    def round_and_verify(U, V, W):
        """Try rounding to integers/half-integers and verify."""
        # Try rounding to nearest integer
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if verify(U_r, V_r, W_r, atol=1e-9):
            return U_r, V_r, W_r
        
        # Try rounding to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, atol=1e-9):
            return U_r, V_r, W_r
        
        return None

    # Schoolbook decomposition: rank = n*m*p = 72
    rank_sch = n * m * p
    U_sch = np.zeros((n * m, rank_sch), dtype=np.float64)
    V_sch = np.zeros((m * p, rank_sch), dtype=np.float64)
    W_sch = np.zeros((p * n, rank_sch), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_sch[i * m + j, r] = 1.0
                V_sch[j * p + k, r] = 1.0
                W_sch[k * n + i, r] = 1.0

    # Verify schoolbook is correct
    assert verify(U_sch, V_sch, W_sch), "Schoolbook verification failed"

    best_rank = rank_sch
    best_U = U_sch
    best_V = V_sch
    best_W = W_sch

    # Pre-compute matricizations for ALS
    T_A = T.reshape(n * m, m * p * p * n)
    T_B = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T_C = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)

    def khatri_rao(A, B):
        """Compute Khatri-Rao product of A and B."""
        # A is (m, r), B is (p, r) -> result is (m*p, r)
        return np.einsum('ir,jr->ijr', A, B).reshape(-1, A.shape[1])

    # Try ranks in descending order from 70 down to 45
    # Expanded search range and increased restarts
    target_ranks = list(range(70, 44, -1))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Multiple random restarts for each rank
        found = False
        num_restarts = 15
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart * 7 + target_rank)
            U_init = np.random.randn(n * m, target_rank)
            V_init = np.random.randn(m * p, target_rank)
            W_init = np.random.randn(p * n, target_rank)

            # Refine using alternating least squares with more iterations
            for iteration in range(60):
                if time.time() > deadline:
                    break
                
                # Update U: T_A = U * (V (x) W)^T
                VW_kr = khatri_rao(V_init, W_init)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW_kr, T_A.T, rcond=None)
                    U_new = U_new.T
                except np.linalg.LinAlgError:
                    break

                # Update V: T_B = V * (U (x) W)^T
                UW_kr = khatri_rao(U_new, W_init)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW_kr, T_B.T, rcond=None)
                    V_new = V_new.T
                except np.linalg.LinAlgError:
                    break

                # Update W: T_C = W * (U (x) V)^T
                UV_kr = khatri_rao(U_new, V_new)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV_kr, T_C.T, rcond=None)
                    W_new = W_new.T
                except np.linalg.LinAlgError:
                    break

                U_init, V_init, W_init = U_new, V_new, W_new

            # Check if we found an exact solution
            if verify(U_init, V_init, W_init, atol=1e-7):
                rounded = round_and_verify(U_init, V_init, W_init)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break

        if found:
            break  # Found a valid lower-rank decomposition, stop searching

    # If no better found, use schoolbook
    if best_rank == rank_sch:
        U_out = U_sch
        V_out = V_sch
        W_out = W_sch
    else:
        U_out = best_U
        V_out = best_V
        W_out = best_W

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
