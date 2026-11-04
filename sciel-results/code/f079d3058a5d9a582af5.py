"""Rewritten program: rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 110  # safety margin before 120s limit

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

    def als_step(U, V, W, target_rank, max_iter=60):
        """Alternating least squares refinement with convergence check."""
        U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
        prev_cost = np.inf
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            VW_kr = khatri_rao(V_cur, W_cur)
            try:
                U_new, _, _, _ = np.linalg.lstsq(VW_kr, T_A.T, rcond=None)
                U_new = U_new.T
            except np.linalg.LinAlgError:
                break

            # Update V
            UW_kr = khatri_rao(U_new, W_cur)
            try:
                V_new, _, _, _ = np.linalg.lstsq(UW_kr, T_B.T, rcond=None)
                V_new = V_new.T
            except np.linalg.LinAlgError:
                break

            # Update W
            UV_kr = khatri_rao(U_new, V_new)
            try:
                W_new, _, _, _ = np.linalg.lstsq(UV_kr, T_C.T, rcond=None)
                W_new = W_new.T
            except np.linalg.LinAlgError:
                break

            U_cur, V_cur, W_cur = U_new, V_new, W_new
            
            # Compute cost for convergence check
            T_comp = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
            cost = np.sum((T_comp - T)**2)
            
            if abs(prev_cost - cost) < 1e-12:
                break
            prev_cost = cost
        
        return U_cur, V_cur, W_cur

    # Try ranks in descending order from 71 down to 50
    # We know rank 72 is schoolbook, so start from 71
    target_ranks = list(range(71, 49, -1))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        found = False
        # Multiple random restarts for each rank
        # Use more restarts for higher ranks (closer to 72) as they are more likely to converge
        num_restarts = 40 if target_rank > 65 else 25
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + target_rank * 100 + restart)
            # Initialize with small random values to help convergence
            scale = 0.1
            U_init = np.random.randn(n * m, target_rank) * scale
            V_init = np.random.randn(m * p, target_rank) * scale
            W_init = np.random.randn(p * n, target_rank) * scale

            # Run ALS
            U_ref, V_ref, W_ref = als_step(U_init, V_init, W_init, target_rank)

            # Check if we found an exact solution
            # Use a slightly looser tolerance for the initial check to allow rounding
            if verify(U_ref, V_ref, W_ref, atol=1e-5):
                rounded = round_and_verify(U_ref, V_ref, W_ref)
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
