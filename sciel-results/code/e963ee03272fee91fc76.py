import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Dimensions of factor matrices
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    
    # Build the target tensor T as a dense array for fast access in loop
    # T[a, b, c] = 1 if consistent, else 0
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Flatten T for the linear system
    # We need to map (a,b,c) to a linear index eq_idx
    # And for each eq, we need to know which W entries are involved
    # W is dim_W x R. Flattened W is dim_W * R.
    # Equation for (a,b,c): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    # Let x be flattened W. x[c*R + r] = W[c,r]
    # Coeff of x[c*R + r] in eq (a,b,c) is U[a,r] * V[b,r]
    
    # We will iterate over ranks
    # Known optimal rank is 19. Try 19, then 20.
    
    def check_and_verify(U, V, W, R):
        """
        Verify U, V, W (integers or half-integers) exactly.
        Returns True if valid.
        """
        # Convert to integers by multiplying by 2
        U_int = np.round(2 * U).astype(np.int64)
        V_int = np.round(2 * V).astype(np.int64)
        W_int = np.round(2 * W).astype(np.int64)
        
        # Check if original was close to half-integers
        if not np.allclose(2*U, U_int) or not np.allclose(2*V, V_int) or not np.allclose(2*W, W_int):
            return False
            
        # Compute tensor
        # T_calc[a,b,c] = sum_r (U_int[a,r]/2) * (V_int[b,r]/2) * (W_int[c,r]/2)
        # = 1/8 * sum_r U_int[a,r] V_int[b,r] W_int[c,r]
        # We want T_calc to equal T (which is 0 or 1)
        # So sum_r U_int[a,r] V_int[b,r] W_int[c,r] should be 0 or 8
        
        # Efficient check
        # For each r, compute outer product of U[:,r] and V[:,r]
        # Then multiply by W[:,r] and accumulate
        
        # Actually, just loop over non-zeros and zeros
        # Non-zeros
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(R):
                        s += U_int[a, r] * V_int[b, r] * W_int[c, r]
                    if s != 8:
                        return False
                    
        # Zeros (sample or full? Full is 576-24=552 checks. Fast enough.)
        all_indices = np.indices((dim_U, dim_V, dim_W))
        # Create a mask of non-zeros
        nz_mask = T != 0
        # Iterate over all indices, skip non-zeros
        for a in range(dim_U):
            for b in range(dim_V):
                for c in range(dim_W):
                    if T[a,b,c] != 0:
                        continue
                    s = 0
                    for r in range(R):
                        s += U_int[a, r] * V_int[b, r] * W_int[c, r]
                    if s != 0:
                        return False
        return True

    def solve_for_W(U, V, R):
        """
        Solve linear system for W given U, V.
        Returns W if valid half-integer solution exists, else None.
        """
        # Build A (576, 8*R) and b (576,)
        # A is sparse. We can build it densely for small R.
        # 576 * (8*19) = 576 * 152 = ~87k entries. Fine.
        
        A = np.zeros((dim_U * dim_V * dim_W, dim_W * R), dtype=np.float64)
        b = np.zeros(dim_U * dim_V * dim_W, dtype=np.float64)
        
        idx = 0
        for a in range(dim_U):
            for b_idx in range(dim_V):
                for c in range(dim_W):
                    target_val = T[a, b_idx, c]
                    b[idx] = target_val
                    # Coefficients for W[c, r] are U[a, r] * V[b_idx, r]
                    # W is flattened: W[c, r] is at index c*R + r
                    for r in range(R):
                        A[idx, c * R + r] = U[a, r] * V[b_idx, r]
                    idx += 1
        
        # Solve A x = b
        # Use lstsq for robustness
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
            
        # Check residual
        if len(residuals) > 0 and residuals[0] > 1e-9:
            return None
        if len(residuals) == 0:
            # Underdetermined? Should not happen for R>=19 usually, but check
            # Actually 576 eqs, 152 unknowns. Overdetermined.
            pass

        # x is flattened W
        W = x.reshape(dim_W, R)
        
        # Check if 2*W is close to integer
        W_int = np.round(2 * W)
        if not np.allclose(2 * W, W_int, atol=1e-9):
            return None
            
        # Convert to float for return (half-integers)
        W_out = W_int / 2.0
        
        # Verify exactly
        if check_and_verify(U, V, W_out, R):
            return W_out
        return None

    # Search for R=19
    for R in [19, 20, 21]:
        if time.time() > deadline - 10:
            break
            
        np.random.seed(42 + R)
        
        # Try many random U, V
        for attempt in range(2000):
            if time.time() > deadline - 5:
                break
                
            # Random sparse-ish U, V with entries in {-1, 0, 1}
            # Or {-2, -1, 0, 1, 2}
            U = np.random.randint(-1, 2, size=(dim_U, R))
            V = np.random.randint(-1, 2, size=(dim_V, R))
            
            # Ensure not all zero rows
            # (Not strictly necessary but helps conditioning)
            
            W = solve_for_W(U, V, R)
            if W is not None:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
        # If R=19 failed, try R=20 with different random seeds
        # The loop above handles R=20 if R=19 didn't return.
        # But we should break the R loop if we find one.
        
    # Fallback: Rank 24
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx, (i, j, k) in enumerate(np.ndindex(n, m, p)):
        U[i * m + j][idx] = 1.0
        V[j * p + k][idx] = 1.0
        W[k * n + i][idx] = 1.0
        
    return {"U": U, "V": V, "W": W}
