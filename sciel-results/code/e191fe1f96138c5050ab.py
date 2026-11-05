"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 112  # Safety margin before 120s limit

    def verify_fast(U, V, W, rank):
        """Verify that the decomposition is exact using NumPy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        U_t = U_arr.reshape(n, m, rank)
        V_t = V_arr.reshape(m, p, rank)
        W_t = W_arr.reshape(p, n, rank)

        try:
            S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        except Exception:
            return False

        return np.allclose(S, 1.0, atol=1e-9)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    # Start with schoolbook
    U, V, W, rank = schoolbook()
    best_rank = rank
    best_result = (U, V, W, rank)

    def try_rank(target_rank):
        """Try to find a decomposition of the given rank using structured search."""
        if time.time() > deadline:
            return None

        # Use a smarter initialization based on Kronecker products
        # Try different structured patterns
        np.random.seed(int(time.time() * 1000) % (2**31))

        for attempt in range(15):
            if time.time() > deadline:
                break

            # Different initialization strategies
            if attempt % 3 == 0:
                # Random initialization
                U_init = np.random.uniform(-1, 1, (n*m, target_rank))
                V_init = np.random.uniform(-1, 1, (m*p, target_rank))
                W_init = np.random.uniform(-1, 1, (p*n, target_rank))
            elif attempt % 3 == 1:
                # Structured initialization based on DFT-like patterns
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))

                # Use cyclic group algebra approach
                # For each column, assign a pattern based on modular arithmetic
                for c in range(target_rank):
                    # Generate a pattern based on the column index
                    phase = 2 * np.pi * c / target_rank
                    for i in range(n):
                        for j in range(m):
                            U_init[i*m+j, c] = np.exp(1j * phase * (i + j))
                    for j in range(m):
                        for k in range(p):
                            V_init[j*p+k, c] = np.exp(1j * phase * (j - k))
                    for k in range(p):
                        for i in range(n):
                            W_init[k*n+i, c] = np.exp(-1j * phase * (i - k))

                # Take real part
                U_init = np.real(U_init)
                V_init = np.real(V_init)
                W_init = np.real(W_init)
            else:
                # Hybrid initialization
                U_init = np.random.uniform(-0.5, 0.5, (n*m, target_rank))
                V_init = np.random.uniform(-0.5, 0.5, (m*p, target_rank))
                W_init = np.random.uniform(-0.5, 0.5, (p*n, target_rank))

            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()

            # Use adaptive learning rate with momentum
            lr = 0.05
            momentum = 0.9
            vU = np.zeros_like(U_curr)
            vV = np.zeros_like(V_curr)
            vW = np.zeros_like(W_curr)

            for step in range(300):
                if time.time() > deadline:
                    break

                U_t = U_curr.reshape(n, m, target_rank)
                V_t = V_curr.reshape(m, p, target_rank)
                W_t = W_curr.reshape(p, n, target_rank)

                # Compute S
                S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
                E = S - 1.0

                # Compute gradients
                dU_t = np.einsum('ijk,jkc,kic->ijc', E, V_t, W_t)
                dV_t = np.einsum('ijk,ijc,kic->jkc', E, U_t, W_t)
                dW_t = np.einsum('ijk,ijc,jkc->kic', E, U_t, V_t)

                # Update with momentum
                dU = dU_t.reshape(n*m, target_rank)
                dV = dV_t.reshape(m*p, target_rank)
                dW = dW_t.reshape(p*n, target_rank)

                vU = momentum * vU - lr * dU
                vV = momentum * vV - lr * dV
                vW = momentum * vW - lr * dW

                U_curr += vU
                V_curr += vV
                W_curr += vW

                # Check convergence
                if np.max(np.abs(E)) < 1e-12:
                    break

            # Check if valid float solution
            if verify_fast(U_curr, V_curr, W_curr, target_rank):
                # Try to snap to half-integers
                U_snapped = np.round(U_curr * 2) / 2
                V_snapped = np.round(V_curr * 2) / 2
                W_snapped = np.round(W_curr * 2) / 2

                if verify_fast(U_snapped, V_snapped, W_snapped, target_rank):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), target_rank

                # Try snapping to integers
                U_int = np.round(U_curr)
                V_int = np.round(V_curr)
                W_int = np.round(W_curr)

                if verify_fast(U_int, V_int, W_int, target_rank):
                    return U_int.tolist(), V_int.tolist(), W_int.tolist(), target_rank

        return None

    # Try lower ranks
    # We know rank 90 is valid.
    # Try 89, 88, ... down to 80
    for target_rank in range(89, 79, -1):
        if time.time() > deadline:
            break
        result = try_rank(target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Continue trying lower ranks
            continue

    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
