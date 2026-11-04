import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_size = nm * mp * pn
    
    # Target tensor T flattened
    # T[a, b, c] = 1 if i_a == i_c and j_a == j_b and k_b == k_c
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    T_flat = np.zeros(total_size, dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                b = j * p + k
                c = k * n + i
                T_flat[idx] = 1.0
                idx += 1
                # Note: The iteration order in the seed was i, j, k.
                # The index in the flat vector must match the evaluator's expectation.
                # The evaluator likely iterates i, j, k to form the vector.
                # Let's verify the index mapping.
                # The seed returns U, V, W. The evaluator computes sum_r U[a,r]V[b,r]W[c,r].
                # It checks if this equals the expected multiplication result.
                # The expected result is 1 for valid (i,j,k) triples.
                # The "valid" triples are exactly those where the indices align.
                # In the seed, for each (i,j,k), U, V, W are set.
                # So the target tensor is 1 at (a(i,j), b(j,k), c(k,i)).
                # There are n*m*p such entries.
                # The total size of the tensor is nm * mp * pn.
                # Most entries are 0.
                # We need to construct T_flat correctly.
    
    # Re-construct T_flat properly
    T_flat = np.zeros(total_size, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Index in flattened tensor: a * (mp * pn) + b * pn + c
                T_flat[a * (mp * pn) + b * pn + c] = 1.0

    def objective(params):
        # params: [U_flat, V_flat, W_flat]
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U = params[:size_U].reshape(nm, R)
        V = params[size_U:size_U+size_V].reshape(mp, R)
        W = params[size_U+size_V:].reshape(pn, R)
        
        # Compute sum_r U[a,r] V[b,r] W[c,r]
        # This is a tensor contraction.
        # Efficient way:
        # Temp[a, b, r] = U[a, r] * V[b, r]
        # Result[a, b, c] = sum_r Temp[a, b, r] * W[c, r]
        
        # Vectorized:
        # U is (nm, R), V is (mp, R)
        # U[:, :, None] * V[None, :, :] -> (nm, R, mp)
        # Transpose to (nm, mp, R)
        # Multiply by W (pn, R) -> (nm, mp, pn)
        
        # Memory check: nm=15, mp=30, pn=18, R~80.
        # 15 * 30 * 80 = 36000. Small.
        
        # Actually, we can do:
        # UV = U.T @ V -> (R, mp)? No.
        # We want T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
        # This is equivalent to:
        # M1 = U.T @ V -> (R, mp)? No.
        # Let's use einsum or simple loops if R is small?
        # R is up to 90. nm=15, mp=30, pn=18.
        # Total elements 8100.
        # Loop over R:
        # Res = np.zeros((nm, mp, pn))
        # for r in range(R):
        #    Res += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        
        # This is slow in Python.
        # Use broadcasting:
        # U_ext = U[:, :, None] # (nm, R, 1)
        # V_ext = V[None, :, :] # (1, mp, R)
        # W_ext = W[None, None, :] # (1, 1, pn)
        # Product = U_ext * V_ext * W_ext # (nm, mp, R, pn)
        # Sum over R: np.sum(Product, axis=2) # (nm, mp, pn)
        
        U_ext = U[:, :, None]
        V_ext = V[None, :, :]
        W_ext = W[None, None, :]
        
        # (nm, R, 1) * (1, mp, R) -> (nm, mp, R)
        UV = U_ext * V_ext
        # (nm, mp, R) * (1, 1, pn) -> (nm, mp, R, pn)
        # This creates a large array.
        # Better:
        # Res = np.einsum('ar,br,c r->abc', U, V, W)
        
        Res = np.einsum('ar,br,cr->abc', U, V, W)
        return Res.flatten() - T_flat

    start_time = time.time()
    deadline = 110.0
    
    # Try ranks from low to high
    # Lower bound for 3x5x6 is likely > 60.
    # Start at 70 to save time, or 60?
    # Let's try 70, 75, 80, 85, 90.
    
    best_result = None
    
    for R in [70, 75, 80, 85, 90]:
        if time.time() > deadline - 10.0:
            break
            
        # Multiple restarts for each R
        max_restarts = 5
        for _ in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize
            # Random initialization
            U0 = np.random.randn(nm, R) * 0.1
            V0 = np.random.randn(mp, R) * 0.1
            W0 = np.random.randn(pn, R) * 0.1
            
            # Scale to match the norm of T
            # ||T||_F = sqrt(nmp)
            scale = np.sqrt(n * m * p) / (np.sqrt(nm * R) * np.sqrt(mp * R) * np.sqrt(pn * R))
            # This scaling is heuristic.
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = least_squares(objective, params0, method='lm', max_nfev=1000)
                if res.success and res.cost < 1e-8:
                    # Valid decomposition found
                    size_U = nm * R
                    size_V = mp * R
                    size_W = pn * R
                    U = res.x[:size_U].reshape(nm, R)
                    V = res.x[size_U:size_U+size_V].reshape(mp, R)
                    W = res.x[size_U+size_V:].reshape(pn, R)
                    
                    # Round to nearest integer or half-integer?
                    # The task says "integer/half-integer entries".
                    # The optimizer finds floats. We need to snap to 0.5 grid.
                    # If the error is very small, snapping might introduce error.
                    # But if the true solution is on the grid, the optimizer should be close.
                    
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    # Verify the snapped solution
                    Res_snapped = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                    error = np.linalg.norm(Res_snapped.flatten() - T_flat)
                    
                    if error < 1e-9:
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
            except Exception:
                pass

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
