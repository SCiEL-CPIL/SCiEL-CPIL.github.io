import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard: must finish well before 120s
    deadline = time.time() + 110

    # Fallback: Rank 21 Strassen
    def get_rank_21():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
            ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
            ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
            ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
            ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
        ]
        
        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[t]
                for idx, val in u_coeffs.items():
                    U[idx][r] = val
                for l, val in v_coeffs.items():
                    if l == 0: v_idx = 0 * p + 2 * b
                    elif l == 1: v_idx = 0 * p + 2 * b + 1
                    elif l == 2: v_idx = 1 * p + 2 * b
                    elif l == 3: v_idx = 1 * p + 2 * b + 1
                    V[v_idx][r] = val
                for (i, k_local), val in w_coeffs.items():
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    W[w_idx][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    # Verification helper
    def verify_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid decomposition of the matmul tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Build the matmul tensor T
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # Compute decomposition: T_decomp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        
        diff = np.max(np.abs(T - T_decomp))
        return diff < tol, diff

    # F2 linear algebra check for rank-20 existence
    def check_f2_rank_20():
        """
        Over F2, a rank-r decomposition exists if we can find U,V,W in F2^{nm x r}, etc.
        such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] mod 2.
        
        For the matmul tensor, T[a,b,c] = 1 iff the indices are consistent.
        
        Key insight: Over F2, if we fix U and V, then W is determined linearly.
        We can search over possible U,V patterns (or use structured ansatz).
        
        For 2x2x6, the tensor has 24 non-zero entries.
        A rank-20 decomposition over F2 would need to satisfy 24 equations.
        
        Let's use a structured approach: assume the 6 columns of B are processed
        in 3 blocks of 2, with Strassen-like structure but sharing one factor.
        """
        if time.time() > deadline - 10:
            return None
        
        # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
        # For n=2, m=2, p=6:
        # a = 2i+j, b = 2j+k, c = 2k+i
        # Non-zero when i in {0,1}, j in {0,1}, k in {0,1,2,3,4,5}
        
        # Over F2, we look for rank-r decomposition.
        # The minimum rank over F2 can be less than over Q.
        # But we need integer/half-integer entries, so F2 solutions need lifting.
        
        # Let's try a direct structured search for rank 20.
        # Ansatz: 3 blocks of 2x2 matmul, where blocks 0 and 1 share one factor.
        # Block 0 (k=0,1): 7 factors
        # Block 1 (k=2,3): 7 factors, but factor 6 of block 0 = factor 7 of block 1
        # Block 2 (k=4,5): 6 factors
        # Total: 7 + 6 + 7 = 20? Let me be more careful.
        
        # Actually, let's try: use 7 factors for each of the 3 blocks = 21,
        # then try to merge two factors from different blocks.
        
        # For a merge to work, the merged factor must contribute to both blocks.
        # This requires the U,V,W entries for the merged factor to be compatible.
        
        # Let's set up the linear system over F2 for a specific merge pattern.
        
        # Merge factor 0 (block 0, Strassen term 0) with factor 7 (block 1, Strassen term 0)
        # These have the same structure: U uses {0,3}, V uses {0,3}, W uses {(0,0),(1,1)}
        
        # After merge, we have 20 factors. Let's denote the merged factor as r*.
        # U[a, r*] must satisfy both block 0 and block 1 requirements.
        # For block 0 (k=0,1): U[0, r*] = 1, U[1, r*] = 1, U[2, r*] = 0, U[3, r*] = 1
        # For block 1 (k=2,3): U[0, r*] = 1, U[1, r*] = 1, U[2, r*] = 0, U[3, r*] = 1
        # These are consistent! So the merge is possible in principle.
        
        # But we need to check that the resulting 20-factor decomposition is valid.
        # Let's construct it and verify over the reals.
        
        if time.time() > deadline - 5:
            return None
        
        # Construct the rank-20 ansatz with merged factor
        rank = 20
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen terms for 2x2 matmul
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
            ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
            ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
            ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
            ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
        ]
        
        # Block 0 (k=0,1): factors 0-6 (7 factors)
        # Block 1 (k=2,3): factors 7-13, but factor 7 is merged with factor 0
        # So block 1 uses factors 7-13 where factor 7 = merged factor
        # Block 2 (k=4,5): factors 14-19 (6 factors)
        
        # Wait, this doesn't quite work because the merged factor needs to
        # contribute to both blocks with the right coefficients.
        
        # Let me try a different approach: just use 7+7+6=20 factors,
        # where the third block uses a 6-rank decomposition.
        # But 2x2 matmul has rank 7 over any field (Strassen's lower bound).
        # So this won't work.
        
        # Alternative: try to find if there's a rank-20 decomposition by
        # solving the bilinear system numerically.
        
        # Start from rank-21 and try to drive one factor to zero while
        # adjusting others.
        
        base = get_rank_21()
        U21 = np.array(base["U"], dtype=float)
        V21 = np.array(base["V"], dtype=float)
        W21 = np.array(base["W"], dtype=float)
        
        # Build target tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        best_diff = float('inf')
        best_result = None
        
        for drop_idx in range(21):
            if time.time() > deadline - 5:
                break
            
            U = U21.copy()
            V = V21.copy()
            W = W21.copy()
            U[:, drop_idx] = 0
            V[:, drop_idx] = 0
            W[:, drop_idx] = 0
            
            # Gradient descent to minimize max error
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
                error = T - T_decomp
                max_err = np.max(np.abs(error))
                
                if max_err < 1e-8:
                    valid, diff = verify_decomposition(U, V, W)
                    if valid:
                        idxs = [i for i in range(21) if i != drop_idx]
                        return {
                            "U": U[:, idxs].tolist(),
                            "V": V[:, idxs].tolist(),
                            "W": W[:, idxs].tolist()
                        }
                
                if max_err < best_diff:
                    best_diff = max_err
                    best_result = (U.copy(), V.copy(), W.copy(), drop_idx)
                
                # Compute gradients
                # d/dU[a,r] = sum_{b,c} error[a,b,c] * V[b,r] * W[c,r]
                # d/dV[b,r] = sum_{a,c} error[a,b,c] * U[a,r] * W[c,r]
                # d/dW[c,r] = sum_{a,b} error[a,b,c] * U[a,r] * V[b,r]
                
                dU = np.einsum('abc,br,cr->ar', error, V, W)
                dV = np.einsum('abc,ar,cr->br', error, U, W)
                dW = np.einsum('abc,ar,br->cr', error, U, V)
                
                # Only update non-dropped factors
                mask = np.ones(21, dtype=bool)
                mask[drop_idx] = False
                
                U[:, mask] -= lr * dU[:, mask]
                V[:, mask] -= lr * dV[:, mask]
                W[:, mask] -= lr * dW[:, mask]
                
                # Keep dropped factor at zero
                U[:, drop_idx] = 0
                V[:, drop_idx] = 0
                W[:, drop_idx] = 0
        
        if best_result is not None and best_diff < 1e-6:
            U, V, W, drop_idx = best_result
            valid, diff = verify_decomposition(U, V, W)
            if valid:
                idxs = [i for i in range(21) if i != drop_idx]
                return {
                    "U": U[:, idxs].tolist(),
                    "V": V[:, idxs].tolist(),
                    "W": W[:, idxs].tolist()
                }
        
        return None

    # Try to find rank-20
    result_20 = check_f2_rank_20()
    if result_20 is not None:
        valid, diff = verify_decomposition(result_20["U"], result_20["V"], result_20["W"])
        if valid:
            return result_20
    
    # Fallback to rank 21
    return get_rank_21()
