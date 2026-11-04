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
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
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
        
        # Try a few more rounding strategies with small perturbations
        for scale in [4, 8]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify(U_r, V_r, W_r, atol=1e-8):
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

    def khatri_rao(A, B):
        """Compute Khatri-Rao product of A and B."""
        # A is (m, r), B is (p, r) -> result is (m*p, r)
        return np.einsum('ir,jr->ijr', A, B).reshape(-1, A.shape[1])

    # Pre-compute matricizations for ALS
    T_A = T.reshape(n * m, m * p * p * n)
    T_B = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T_C = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)

    def als_round(U, V, W, target_rank, max_iter=80):
        """Run ALS and try to round to integer/half-integer entries."""
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            VW_kr = khatri_rao(V, W)
            try:
                U_new, _, _, _ = np.linalg.lstsq(VW_kr, T_A.T, rcond=None)
                U_new = U_new.T
            except np.linalg.LinAlgError:
                return None

            # Update V
            UW_kr = khatri_rao(U_new, W)
            try:
                V_new, _, _, _ = np.linalg.lstsq(UW_kr, T_B.T, rcond=None)
                V_new = V_new.T
            except np.linalg.LinAlgError:
                return None

            # Update W
            UV_kr = khatri_rao(U_new, V_new)
            try:
                W_new, _, _, _ = np.linalg.lstsq(UV_kr, T_C.T, rcond=None)
                W_new = W_new.T
            except np.linalg.LinAlgError:
                return None

            U, V, W = U_new, V_new, W_new

        # Check if we found an exact solution
        if verify(U, V, W, atol=1e-6):
            rounded = round_and_verify(U, V, W)
            if rounded is not None:
                return rounded
        
        return None

    # Try to find a lower-rank decomposition using ALS with various initializations
    # Start from rank 60 down to 50, trying different random seeds
    for target_rank in range(65, 45, -1):
        if time.time() > deadline:
            break
        
        found = False
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            seed = 1000 + restart * 13 + target_rank * 7
            np.random.seed(seed)
            
            # Try different initialization strategies
            if restart % 3 == 0:
                # Random Gaussian
                U_init = np.random.randn(n * m, target_rank)
                V_init = np.random.randn(m * p, target_rank)
                W_init = np.random.randn(p * n, target_rank)
            elif restart % 3 == 1:
                # Sparse random initialization
                U_init = np.random.randn(n * m, target_rank)
                V_init = np.random.randn(m * p, target_rank)
                W_init = np.random.randn(p * n, target_rank)
                mask_u = np.random.rand(n * m, target_rank) < 0.3
                mask_v = np.random.rand(m * p, target_rank) < 0.3
                mask_w = np.random.rand(p * n, target_rank) < 0.3
                U_init[mask_u] *= 2
                V_init[mask_v] *= 2
                W_init[mask_w] *= 2
            else:
                # Structured initialization based on schoolbook pattern
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U_init[i * m + j, r] = 1.0
                    V_init[j * p + k, r] = 1.0
                    W_init[k * n + i, r] = 1.0
                    # Add some random perturbation
                    U_init[:, r] += np.random.randn(n * m) * 0.1
                    V_init[:, r] += np.random.randn(m * p) * 0.1
                    W_init[:, r] += np.random.randn(p * n) * 0.1
            
            result = als_round(U_init, V_init, W_init, target_rank, max_iter=60)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = target_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                found = True
                break
        
        if found:
            break

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
