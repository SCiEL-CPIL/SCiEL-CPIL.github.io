"""Program searching for low-rank exact tensor decomposition of 2x3 by 3x8 matrix multiplication.

Strategy:
1. Decompose the 2x3x8 multiplication into 4 independent 2x3x2 multiplications by splitting the 8 columns of B into 4 blocks of 2.
2. Search for a low-rank decomposition of the 2x3x2 tensor using gradient descent.
3. If a low-rank decomposition (e.g., rank 10 or 11) is found, construct the 2x3x8 decomposition by tiling the factors.
4. Verify the final decomposition exactly.
5. Fallback to schoolbook rank 48 if the search fails or time runs out.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 8]
SEED = 42

def construct():
    deadline = time.time() + 140  # safety margin before the 150s wall-clock limit

    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 48

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        """Verify that U, V, W exactly decompose T."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        return bool(np.allclose(computed, T, atol=1e-9, rtol=1e-9))

    # ---------- Schoolbook baseline (guaranteed rank 48) ----------
    def schoolbook(nn, mm, pp):
        r = nn * mm * pp
        U = np.zeros((nn * mm, r))
        V = np.zeros((mm * pp, r))
        W = np.zeros((pp * nn, r))
        c = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, c] = 1
                    V[j * pp + k, c] = 1
                    W[k * nn + i, c] = 1
                    c += 1
        return U, V, W

    U0, V0, W0 = schoolbook(n, m, p)
    best = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist(), "rank": base_rank}

    # ---------- Sub-problem solver for 2x3x2 ----------
    def solve_small(nn, mm, pp, target_rank, max_iters=5000, lr=0.01, restarts=10):
        """Search for a rank-target_rank decomposition of the nn x mm x pp multiplication tensor."""
        T_sub = np.zeros((nn * mm, mm * pp, pp * nn))
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    T_sub[i * mm + j, j * pp + k, k * nn + i] = 1.0
        
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for r in range(target_rank, target_rank + 5):  # Try target and slightly higher
            for restart in range(restarts):
                if time.time() > deadline:
                    break
                random.seed(SEED + restart * 100 + r)
                np.random.seed(SEED + restart * 100 + r)
                
                U = np.random.randn(nn * mm, r) * 0.1
                V = np.random.randn(mm * pp, r) * 0.1
                W = np.random.randn(pp * nn, r) * 0.1
                
                for it in range(max_iters):
                    if time.time() > deadline:
                        break
                    
                    # Compute gradient
                    # E_ijk = sum_r U_ir V_jr W_kr - T_ijk
                    # grad_U_ir = 2 * sum_jk E_ijk V_jr W_kr
                    # This can be computed as:
                    # Temp_jk_r = V_jr W_kr
                    # E_ijk = sum_r U_ir Temp_jk_r - T_ijk
                    # grad_U_ir = 2 * sum_jk E_ijk Temp_jk_r
                    
                    # Efficient computation using einsum
                    # 1. Compute V @ W^T for each r? No, V is (mm*pp, r), W is (pp*nn, r).
                    # We need to reshape V and W to (mm, pp, r) and (pp, nn, r).
                    V_reshaped = V.reshape(mm, pp, r)
                    W_reshaped = W.reshape(pp, nn, r)
                    
                    # Product P_jk_r = sum_r V_jr W_kr? No, we need the full tensor.
                    # Let's compute the residual E directly.
                    # C_ijk = sum_r U_ir V_jr W_kr
                    # U is (nn*mm, r). Reshape to (nn, mm, r).
                    U_reshaped = U.reshape(nn, mm, r)
                    
                    C = np.einsum('ir,jr,kr->ijk', U_reshaped, V_reshaped, W_reshaped)
                    E = C - T_sub.reshape(nn, mm, pp)
                    
                    # Gradients
                    # grad_U_ir = 2 * sum_jk E_ijk V_jr W_kr
                    # = 2 * sum_jk E_ijk (V_jr W_kr)
                    # Let M_jk_r = V_jr W_kr.
                    # grad_U_ir = 2 * sum_jk E_ijk M_jk_r
                    # This is a contraction over j, k.
                    # We can do: 
                    # Temp1_jk_r = V_jr W_kr
                    # grad_U_ir = 2 * np.einsum('ijk,jkr->ir', E, Temp1_jk_r)
                    
                    # To save time, we can compute this step by step.
                    # 1. Compute V @ W^T for each r?
                    # V is (mm, pp, r). W is (pp, nn, r).
                    # V @ W^T is not directly possible without summing over r.
                    # We need the tensor V_jr W_kr.
                    # Let's compute this tensor P_jk_r.
                    P = np.einsum('jr,kr->jkr', V_reshaped, W_reshaped)
                    
                    grad_U = 2 * np.einsum('ijk,jkr->ir', E, P)
                    grad_U = grad_U.reshape(nn * mm, r)
                    
                    # grad_V_jr = 2 * sum_ik E_ijk U_ir W_kr
                    # = 2 * sum_ik E_ijk (U_ir W_kr)
                    # Let Q_ik_r = U_ir W_kr.
                    Q = np.einsum('ir,kr->ikr', U_reshaped, W_reshaped)
                    grad_V = 2 * np.einsum('ijk,ikr->jr', E, Q)
                    grad_V = grad_V.reshape(mm * pp, r)
                    
                    # grad_W_kr = 2 * sum_ij E_ijk U_ir V_jr
                    # = 2 * sum_ij E_ijk (U_ir V_jr)
                    # Let R_ij_r = U_ir V_jr.
                    R = np.einsum('ir,jr->ijr', U_reshaped, V_reshaped)
                    grad_W = 2 * np.einsum('ijk,ijr->kr', E, R)
                    grad_W = grad_W.reshape(pp * nn, r)
                    
                    # Update
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                    
                    # Check loss
                    C_new = np.einsum('ir,jr,kr->ijk', U_reshaped, V_reshaped, W_reshaped)
                    loss = np.sum((C_new - T_sub.reshape(nn, mm, pp))**2)
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        
                    if loss < 1e-10:
                        break
                        
                if best_loss < 1e-10:
                    break
            if best_loss < 1e-10:
                break
        
        if best_loss < 1e-10:
            # Round to half-integers? The task allows integer/half-integer.
            # We can try to round and verify.
            Ur = np.round(best_U * 2) / 2
            Vr = np.round(best_V * 2) / 2
            Wr = np.round(best_W * 2) / 2
            if verify_sub(Ur, Vr, Wr, nn, mm, pp):
                return Ur, Vr, Wr
            else:
                # Try without rounding if it's close enough? No, must be exact.
                # If rounding fails, return None.
                return None
        else:
            return None

    def verify_sub(U, V, W, nn, mm, pp):
        T_sub = np.zeros((nn * mm, mm * pp, pp * nn))
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    T_sub[i * mm + j, j * pp + k, k * nn + i] = 1.0
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        return bool(np.allclose(computed, T_sub, atol=1e-9, rtol=1e-9))

    # Try to find a low-rank decomposition for 2x3x2
    # Schoolbook rank for 2x3x2 is 12. Try 10, 11.
    sub_U, sub_V, sub_W = None, None, None
    for target_rank in [10, 11, 12]:
        if time.time() > deadline:
            break
        sub_U, sub_V, sub_W = solve_small(2, 3, 2, target_rank, max_iters=2000, lr=0.005, restarts=5)
        if sub_U is not None:
            print(f"Found rank {target_rank} decomposition for 2x3x2")
            break

    if sub_U is not None:
        # Construct full 2x3x8 decomposition
        # p=8, split into 4 blocks of 2.
        num_blocks = 4
        r_sub = sub_U.shape[1]
        r_full = num_blocks * r_sub
        
        U_full = np.zeros((n * m, r_full))
        V_full = np.zeros((m * p, r_full))
        W_full = np.zeros((p * n, r_full))
        
        for b in range(num_blocks):
            # U is the same for all blocks (A is shared)
            U_full[:, b*r_sub:(b+1)*r_sub] = sub_U
            
            # V: rows correspond to j*p + k.
            # For block b, k is in {2*b, 2*b+1}.
            # V_sub rows correspond to j*2 + k_local.
            # Map V_sub rows to V_full rows.
            for j in range(m):
                for k_local in range(2):
                    k_global = b * 2 + k_local
                    v_idx = j * p + k_global
                    v_sub_idx = j * 2 + k_local
                    V_full[v_idx, b*r_sub:(b+1)*r_sub] = sub_V[v_sub_idx]
            
            # W: rows correspond to k*n + i.
            # For block b, k is in {2*b, 2*b+1}.
            # W_sub rows correspond to k_local*2 + i.
            for k_local in range(2):
                k_global = b * 2 + k_local
                for i in range(n):
                    w_idx = k_global * n + i
                    w_sub_idx = k_local * 2 + i
                    W_full[w_idx, b*r_sub:(b+1)*r_sub] = sub_W[w_sub_idx]
        
        # Verify full decomposition
        if verify(U_full, V_full, W_full):
            # Remove zero columns
            cols = np.where(np.any(U_full != 0, axis=0) | np.any(V_full != 0, axis=0) | np.any(W_full != 0, axis=0))[0]
            U_full, V_full, W_full = U_full[:, cols], V_full[:, cols], W_full[:, cols]
            r = U_full.shape[1]
            if r < best["rank"]:
                best = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist(), "rank": r}

    return {"U": best["U"], "V": best["V"], "W": best["W"]}
