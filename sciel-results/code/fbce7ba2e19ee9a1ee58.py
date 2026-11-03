import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build target tensor T
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for a in range(dim_u):
        i_a = a // m
        j_a = a % m
        for b in range(dim_v):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(dim_w):
                k_c = c // n
                i_c = c % n
                if i_a == i_c and k_b == k_c:
                    T[a, b, c] = 1.0

    # We aim to find a decomposition with rank R
    # Theoretical lower bound is 16, known best is likely around 17-20.
    # We try ranks from 17 to 29.
    
    def solve_linear_system(A, b):
        """Solve A x = b using least squares."""
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return x
        except np.linalg.LinAlgError:
            return None

    def threshold_matrix(M, k):
        """Keep only the k largest absolute values in M, set others to 0."""
        # Flatten to find global k largest? Or per row?
        # Global sparsity is often better for tensor decomposition rank.
        # But per-row sparsity might be easier to interpret.
        # Let's try global sparsity first: keep k largest entries in the whole matrix.
        flat_M = M.flatten()
        idx = np.argpartition(np.abs(flat_M), -k)[-k:] if k < flat_M.size else np.arange(flat_M.size)
        new_flat = np.zeros_like(flat_M)
        new_flat[idx] = flat_M[idx]
        return new_flat.reshape(M.shape)

    # Try different ranks
    for R in range(17, 30):
        # Multiple trials with different random seeds and sparsity levels
        for trial in range(50):
            np.random.seed(42 + trial * 100 + R)
            
            # Initialize with small integers
            U = np.random.randint(-1, 2, size=(dim_u, R)).astype(float)
            V = np.random.randint(-1, 2, size=(dim_v, R)).astype(float)
            W = np.random.randint(-1, 2, size=(dim_w, R)).astype(float)
            
            # Iterative Hard Thresholding / ALS
            # We alternate solving for one matrix while keeping others fixed.
            # To enforce sparsity and integer-ness, we threshold after each step.
            
            best_residual = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(100):
                # Update W
                # Equation: sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
                # For each c, we solve for W[c, :]
                # Actually, we can vectorize this.
                # Let's reshape U and V to form the coefficient matrix for W.
                # This is tricky to vectorize directly for all c at once without huge memory.
                # But dim_u * dim_v * dim_w is 2400. R is ~20.
                # We can loop over c.
                
                W_new = np.zeros_like(W)
                for c in range(dim_w):
                    # M[eq_idx, r] = U[a, r] * V[b, r]
                    # eq_idx = a * dim_v + b
                    M_c = np.zeros((dim_u * dim_v, R))
                    for a in range(dim_u):
                        for b in range(dim_v):
                            M_c[a * dim_v + b, :] = U[a, :] * V[b, :]
                    
                    T_c_flat = T[:, :, c].flatten()
                    W_c = solve_linear_system(M_c, T_c_flat)
                    if W_c is None:
                        W_c = np.zeros(R)
                    W_new[c, :] = W_c
                
                # Threshold W
                # Keep a certain number of non-zeros. Let's try keeping all initially, 
                # but rounding to nearest 0.5 might be better for "half-integer" constraint?
                # The task allows integer/half-integer.
                # Let's round to nearest 0.5 and zero out small values.
                W_new = np.round(W_new * 2) / 2.0
                W_new[np.abs(W_new) < 1e-6] = 0.0
                
                # Update U
                U_new = np.zeros_like(U)
                for a in range(dim_u):
                    # Equation: sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
                    # For fixed a, we solve for U[a, :].
                    # The variable is U[a, r].
                    # The coefficient for U[a, r] in the equation for (a, b, c) is V[b, r] W[c, r].
                    # We have equations for all b, c.
                    # M_a[eq_idx, r] = V[b, r] W[c, r] where eq_idx = b * dim_w + c
                    M_a = np.zeros((dim_v * dim_w, R))
                    for b in range(dim_v):
                        for c in range(dim_w):
                            M_a[b * dim_w + c, :] = V[b, :] * W[c, :]
                    
                    T_a_flat = T[a, :, :].flatten()
                    U_a = solve_linear_system(M_a, T_a_flat)
                    if U_a is None:
                        U_a = np.zeros(R)
                    U_new[a, :] = U_a
                
                U_new = np.round(U_new * 2) / 2.0
                U_new[np.abs(U_new) < 1e-6] = 0.0
                
                # Update V
                V_new = np.zeros_like(V)
                for b in range(dim_v):
                    # Equation: sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
                    # For fixed b, we solve for V[b, :].
                    # Coefficient for V[b, r] is U[a, r] W[c, r].
                    M_b = np.zeros((dim_u * dim_w, R))
                    for a in range(dim_u):
                        for c in range(dim_w):
                            M_b[a * dim_w + c, :] = U[a, :] * W[c, :]
                    
                    T_b_flat = T[:, b, :].flatten()
                    V_b = solve_linear_system(M_b, T_b_flat)
                    if V_b is None:
                        V_b = np.zeros(R)
                    V_new[b, :] = V_b
                
                V_new = np.round(V_new * 2) / 2.0
                V_new[np.abs(V_new) < 1e-6] = 0.0
                
                # Check convergence / Residual
                # Calculate residual
                residual = 0.0
                for a in range(dim_u):
                    for b in range(dim_v):
                        for c in range(dim_w):
                            val = np.dot(U_new[a, :], V_new[b, :] * W_new[c, :])
                            residual += (val - T[a, b, c]) ** 2
                
                if residual < best_residual:
                    best_residual = residual
                    best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                
                U, V, W = U_new, V_new, W_new
                
                if residual < 1e-9:
                    break
            
            # Verify the best solution found in this trial
            if best_residual < 1e-9:
                # Check if entries are valid integers/half-integers
                # They are already rounded to 0.5 in the loop.
                # Let's verify exactly
                valid = True
                for a in range(dim_u):
                    for b in range(dim_v):
                        for c in range(dim_w):
                            val = np.dot(best_U[a, :], best_V[b, :] * best_W[c, :])
                            if not np.isclose(val, T[a, b, c], atol=1e-9):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to list of numbers
                    # The task asks for integer/half-integer entries.
                    # np.round(x*2)/2.0 ensures this.
                    U_list = best_U.tolist()
                    V_list = best_V.tolist()
                    W_list = best_W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
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
    return {"U": U, "V": U, "W": W}
