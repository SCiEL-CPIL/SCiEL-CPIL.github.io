import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin

    # Precompute index mappings for efficiency
    # For each (i,j,k), store the row indices in U, V, W
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))

    def compute_tensor(U, V, W):
        """Compute the tensor values from U, V, W (numpy arrays)."""
        T = np.zeros((n, m, p))
        for idx, (ui, vi, wi) in enumerate(indices):
            i, j, k = idx // (m * p), (idx // p) % m, idx % p
            T[i, j, k] = np.dot(U[ui], V[vi] * W[wi])
        return T

    def max_error(U, V, W):
        """Compute max absolute error from 1.0 for non-zero entries, 0 for zero entries."""
        T = compute_tensor(U, V, W)
        # All entries should be 1 (since T[i,j,k] = 1 for all i,j,k in matmul tensor)
        # Wait - actually the tensor has T[i,j,k] = 1 for all valid (i,j,k)
        # because C[i,k] = sum_j A[i,j] * B[j,k]
        # So each term A[i,j]*B[j,k] appears with coefficient 1
        return np.max(np.abs(T - 1.0))

    def try_round_and_verify(U, V, W, max_denom=16):
        """Try to round to rationals with small denominators and verify exactly."""
        U_list = [[0.0] * len(U[0]) for _ in range(len(U))]
        V_list = [[0.0] * len(V[0]) for _ in range(len(V))]
        W_list = [[0.0] * len(W[0]) for _ in range(len(W))]
        
        for d in range(1, max_denom + 1):
            U_rounded = np.zeros_like(U)
            V_rounded = np.zeros_like(V)
            W_rounded = np.zeros_like(W)
            
            for i in range(len(U)):
                for j in range(len(U[0])):
                    U_rounded[i, j] = round(U[i, j] * d) / d
            for i in range(len(V)):
                for j in range(len(V[0])):
                    V_rounded[i, j] = round(V[i, j] * d) / d
            for i in range(len(W)):
                for j in range(len(W[0])):
                    W_rounded[i, j] = round(W[i, j] * d) / d
            
            # Verify exactly using fractions
            valid = True
            for idx, (ui, vi, wi) in enumerate(indices):
                i, j, k = idx // (m * p), (idx // p) % m, idx % p
                s = Fraction(0)
                for r in range(len(U[0])):
                    u_val = Fraction(round(U_rounded[ui, r] * d), d)
                    v_val = Fraction(round(V_rounded[vi, r] * d), d)
                    w_val = Fraction(round(W_rounded[wi, r] * d), d)
                    s += u_val * v_val * w_val
                if s != 1:
                    valid = False
                    break
            
            if valid:
                # Convert to list of lists with float values (or keep as fractions converted to float)
                U_out = [[float(U_rounded[i, j]) for j in range(len(U[0]))] for i in range(len(U))]
                V_out = [[float(V_rounded[i, j]) for j in range(len(V[0]))] for i in range(len(V))]
                W_out = [[float(W_rounded[i, j]) for j in range(len(W[0]))] for i in range(len(W))]
                return U_out, V_out, W_out
        return None

    # Try ranks from 14 to 17, then fall back to 18
    for target_rank in [14, 15, 16, 17]:
        if time.time() > deadline - 10:
            break
        
        n_attempts = 8 if target_rank <= 15 else 5
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(12345 + attempt * 100 + target_rank)
            U = np.random.randn(nm, target_rank) * 0.3
            V = np.random.randn(mp, target_rank) * 0.3
            W = np.random.randn(pn, target_rank) * 0.3
            
            # Gradient descent with adaptive learning rate
            lr = 0.05
            best_err = float('inf')
            
            for step in range(3000):
                if time.time() > deadline - 3:
                    break
                
                # Compute gradients efficiently
                dU = np.zeros((nm, target_rank))
                dV = np.zeros((mp, target_rank))
                dW = np.zeros((pn, target_rank))
                
                for idx, (ui, vi, wi) in enumerate(indices):
                    i, j, k = idx // (m * p), (idx // p) % m, idx % p
                    # T[i,j,k] = sum_r U[ui,r] * V[vi,r] * W[wi,r]
                    VW = V[vi] * W[wi]  # element-wise product, shape (target_rank,)
                    UW = U[ui] * W[wi]
                    UV = U[ui] * V[vi]
                    T_val = np.dot(U[ui], VW)
                    err = T_val - 1.0
                    
                    dU[ui] += err * VW
                    dV[vi] += err * UW
                    dW[wi] += err * UV
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if step % 50 == 0:
                    cur_err = max_error(U, V, W)
                    if cur_err < best_err:
                        best_err = cur_err
                        lr = max(0.001, lr * 0.95)
                    else:
                        lr = min(0.1, lr * 1.1)
                
                if step % 100 == 0 and max_error(U, V, W) < 1e-8:
                    break
            
            final_err = max_error(U, V, W)
            if final_err < 1e-7:
                # Try to round to exact rationals
                result = try_round_and_verify(U, V, W, max_denom=32)
                if result is not None:
                    U_out, V_out, W_out = result
                    return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook decomposition (rank 18)
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
