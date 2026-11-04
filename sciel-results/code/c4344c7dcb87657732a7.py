import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin before 120s limit

    # Precompute index mappings: (U_index, V_index, W_index) for each (i,j,k)
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))

    def max_error(U, V, W, rank):
        """Compute max absolute error across all tensor entries."""
        err = 0.0
        for (ui, vi, wi) in indices:
            s = 0.0
            for r in range(rank):
                s += U[ui][r] * V[vi][r] * W[wi][r]
            e = abs(s - 1.0)
            if e > err:
                err = e
        return err

    def verify_exact(U, V, W, rank):
        """Verify exact decomposition using Fraction arithmetic."""
        for (ui, vi, wi) in indices:
            s = Fraction(0)
            for r in range(rank):
                u_val = U[ui][r]
                v_val = V[vi][r]
                w_val = W[wi][r]
                # Convert to Fraction
                if isinstance(u_val, float):
                    u_val = Fraction(u_val).limit_denominator(10000)
                elif isinstance(u_val, int):
                    u_val = Fraction(u_val)
                else:
                    u_val = Fraction(u_val)
                if isinstance(v_val, float):
                    v_val = Fraction(v_val).limit_denominator(10000)
                elif isinstance(v_val, int):
                    v_val = Fraction(v_val)
                else:
                    v_val = Fraction(v_val)
                if isinstance(w_val, float):
                    w_val = Fraction(w_val).limit_denominator(10000)
                elif isinstance(w_val, int):
                    w_val = Fraction(w_val)
                else:
                    w_val = Fraction(w_val)
                s += u_val * v_val * w_val
            if s != 1:
                return False
        return True

    def try_round_and_verify(U, V, W, rank):
        """Try rounding to rationals with various denominators and verify."""
        for denom in [1, 2, 4, 8, 16, 32, 64, 128]:
            U_r = np.round(U * denom) / denom
            V_r = np.round(V * denom) / denom
            W_r = np.round(W * denom) / denom

            # Quick float check
            valid = True
            for (ui, vi, wi) in indices:
                s = 0.0
                for r in range(rank):
                    s += U_r[ui, r] * V_r[vi, r] * W_r[wi, r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break

            if valid:
                # Convert to lists with exact rational representation
                U_out = [[float(U_r[i, j]) for j in range(rank)] for i in range(nm)]
                V_out = [[float(V_r[i, j]) for j in range(rank)] for i in range(mp)]
                W_out = [[float(W_r[i, j]) for j in range(rank)] for i in range(pn)]

                # Final exact verification
                if verify_exact(U_out, V_out, W_out, rank):
                    return U_out, V_out, W_out

        return None

    def gradient_search(target_rank, max_steps=3000, seed=42):
        """Gradient-based search for a tensor decomposition."""
        if time.time() > deadline - 15:
            return None

        np.random.seed(seed)

        # Initialize with small random values
        U = np.random.randn(nm, target_rank) * 0.3
        V = np.random.randn(mp, target_rank) * 0.3
        W = np.random.randn(pn, target_rank) * 0.3

        lr = 0.02
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        patience = 0
        max_patience = 300

        for step in range(max_steps):
            if time.time() > deadline - 5:
                break

            # Compute gradients
            dU = np.zeros((nm, target_rank))
            dV = np.zeros((mp, target_rank))
            dW = np.zeros((pn, target_rank))

            for (ui, vi, wi) in indices:
                VW = V[vi] * W[wi]
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

            # Decay learning rate
            if step % 100 == 0:
                lr *= 0.95

            # Check error every 20 steps
            if step % 20 == 0:
                max_err = max_error(U, V, W, target_rank)
                if max_err < best_err:
                    best_err = max_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    patience = 0
                else:
                    patience += 20

                if best_err < 1e-7:
                    break
                if patience >= max_patience:
                    break

        # Use best found
        U, V, W = best_U, best_V, best_W
        max_err = max_error(U, V, W, target_rank)

        if max_err > 1e-4:
            return None

        # Try to round to rationals
        result = try_round_and_verify(U, V, W, target_rank)
        return result

    def try_structured_rank16():
        """
        Try a structured rank-16 construction.
        For 2x3x3, we can potentially use algebraic identities to reduce from 18 to 16.
        
        Key idea: For each output C[i,k] = sum_j A[i,j]*B[j,k], we have 3 terms.
        With 6 outputs (2x3), that's 18 terms total.
        
        We can try to share terms between outputs that share the same j-index
        or use bilinear identities.
        
        One approach: For fixed j, A[:,j] * B[j,:] is a 2x3 matrix of rank 1.
        The sum over j of these rank-1 matrices gives C.
        
        To reduce rank, we can try to express some products using shared intermediate
        variables. For example:
        A[i,j]*B[j,k] can be written using (A[i,j] + B[j,k])^2 type identities,
        but this typically increases rank.
        
        A better approach: use the fact that for 2x3x3, there exist non-trivial
        bilinear identities that allow rank reduction.
        
        Let's try a parameterized family and solve for the parameters.
        """
        if time.time() > deadline - 30:
            return None

        # Try a specific structured decomposition
        # Based on the idea of sharing terms across j-indices
        
        rank = 16
        # Initialize with zeros
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        # Strategy: Use 6 terms for "diagonal" contributions and 10 for "off-diagonal"
        # Actually, let's try a different approach:
        # Assign terms to cover the 18 (i,j,k) triples, merging 2 pairs.
        
        # For now, let's just try the gradient search with better initialization
        
        return None

    # Try gradient-based search for rank 16 with multiple seeds and initializations
    best_result = None
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        # Try with different initialization scales
        for scale in [0.2, 0.3, 0.5]:
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed)
            U_init = np.random.randn(nm, 16) * scale
            V_init = np.random.randn(mp, 16) * scale
            W_init = np.random.randn(pn, 16) * scale
            
            # Quick gradient steps
            lr = 0.02
            for step in range(1500):
                if time.time() > deadline - 5:
                    break
                
                dU = np.zeros((nm, 16))
                dV = np.zeros((mp, 16))
                dW = np.zeros((pn, 16))
                
                for (ui, vi, wi) in indices:
                    VW = V_init[vi] * W_init[wi]
                    UW = U_init[ui] * W_init[wi]
                    UV = U_init[ui] * V_init[vi]
                    T_val = np.dot(U_init[ui], VW)
                    err = T_val - 1.0
                    
                    dU[ui] += err * VW
                    dV[vi] += err * UW
                    dW[wi] += err * UV
                
                U_init -= lr * dU
                V_init -= lr * dV
                W_init -= lr * dW
                
                if step % 100 == 0:
                    lr *= 0.95
            
            # Check error
            max_err = max_error(U_init, V_init, W_init, 16)
            if max_err < 1e-4:
                result = try_round_and_verify(U_init, V_init, W_init, 16)
                if result is not None:
                    best_result = result
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U_out, V_out, W_out = best_result
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
