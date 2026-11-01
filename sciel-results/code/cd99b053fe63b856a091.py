import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Dimensions
    dim_u = n * m  # 4
    dim_v = m * p  # 6
    dim_w = p * n  # 6
    
    # Target Tensor T[i*m+j, j*p+k, k*n+i] = 1
    T_target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def verify(U, V, W):
        """Check if U, V, W decompose T_target exactly."""
        # T_calc[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_calc, T_target, atol=1e-9)

    def get_rank11_seed():
        """Return the known rank-11 decomposition (Strassen + Schoolbook)."""
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # Strassen Part (Cols 0-6)
        # M1 = (a00+a11)(b00+b11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a10+a11)b00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3 = a00(b01-b11)
        U[0, 2] = 1
        V[1, 2] = 1; V[4, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = a11(b10-b00)
        U[3, 3] = 1
        V[3, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[1, 3] = 1
        
        # M5 = (a00+a01)b11
        U[0, 4] = 1; U[1, 4] = 1
        V[4, 4] = 1
        W[0, 4] = 1
        
        # M6 = (a10-a00)(b00+b01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (a01-a11)(b10+b11)
        U[1, 6] = 1; U[3, 6] = -1
        V[3, 6] = 1; V[4, 6] = 1
        W[0, 6] = -1; W[2, 6] = 1
        
        # Schoolbook Part for C_right (Cols 7-10)
        # C02 = a00*b02 + a01*b12
        U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
        U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
        # C12 = a10*b02 + a11*b12
        U[2, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
        U[3, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
        
        return U, V, W

    # Initial seed
    U_seed, V_seed, W_seed = get_rank11_seed()
    
    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = 11
    
    # Strategy: Try to find rank-10 decomposition.
    # The tensor is T(2,2,3). It is known that rank 10 is achievable.
    # We use a local search approach starting from the rank-11 seed.
    # We fix the structure: 10 columns.
    # We will perturb the last few columns of the seed to try to merge them or adjust to satisfy constraints with 10 cols.
    
    # A more robust approach for small dimensions:
    # Parametrize U, V, W for rank R=10.
    # Since we have a valid rank-11, we can try to remove one column by adjusting the others.
    # However, a direct algebraic search is better.
    
    # Let's try a specific known rank-10 construction if available, or search.
    # Given the time limit, we will attempt a gradient-free local search on the entries
    # of a rank-10 candidate initialized by projecting the rank-11 seed.
    
    def try_rank(R, U_init, V_init, W_init, max_iter=2000):
        """
        Attempt to refine a rank-R decomposition.
        Uses a simple coordinate descent / random perturbation on the factor matrices
        to minimize the Frobenius norm of the error.
        """
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Compute initial error
        def compute_error():
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.linalg.norm(T_calc - T_target)
        
        current_err = compute_error()
        if current_err < 1e-9:
            return U, V, W, True
        
        # Random perturbation search
        # We focus on perturbing the entries.
        # To keep it fast, we use a coarse grid of step sizes.
        
        # We need to be careful with the scaling.
        # Since the problem is homogeneous (scaling U by c, V by 1/c, W by 1 doesn't change product? No.
        # U*c, V/c, W*1 -> product is same.
        # This symmetry makes the landscape flat. We fix a gauge: e.g., norm of each column of U is 1.
        
        def normalize(U, V, W):
            for r in range(R):
                n_u = np.linalg.norm(U[:, r])
                if n_u > 1e-12:
                    U[:, r] /= n_u
                    V[:, r] *= n_u
                # Keep W as is for now, or normalize W too?
                # Let's normalize W as well to avoid overflow/underflow.
                n_w = np.linalg.norm(W[:, r])
                if n_w > 1e-12:
                    W[:, r] /= n_w
                    # We need to balance. If we scale U down and V up, product stays same.
                    # If we scale W down, product changes.
                    # Let's stick to scaling U and V only to preserve the product value.
                    # Actually, if we scale U by 1/n_u and V by n_u, the product U[:,r]*V[:,r]*W[:,r] is unchanged.
                    # So W doesn't need scaling for gauge fixing of the product.
                    # But to keep numbers stable, let's scale W to have norm 1 as well?
                    # If we scale W by 1/n_w, we must scale the "effective" U or V by n_w.
                    # Let's just normalize U to unit norm per column.
            return U, V, W
        
        U, V, W = normalize(U, V, W)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            if current_err < 1e-9:
                break
                
            # Pick a random column and a random entry in U, V, or W to perturb
            r = np.random.randint(R)
            mat_idx = np.random.randint(3)
            
            if mat_idx == 0:
                idx = np.random.randint(dim_u)
                old_val = U[idx, r]
                step = np.random.uniform(-0.5, 0.5)
                U[idx, r] += step
            elif mat_idx == 1:
                idx = np.random.randint(dim_v)
                old_val = V[idx, r]
                step = np.random.uniform(-0.5, 0.5)
                V[idx, r] += step
            else:
                idx = np.random.randint(dim_w)
                old_val = W[idx, r]
                step = np.random.uniform(-0.5, 0.5)
                W[idx, r] += step
            
            # Normalize the column r to maintain gauge
            n_u = np.linalg.norm(U[:, r])
            if n_u > 1e-12:
                U[:, r] /= n_u
                V[:, r] *= n_u
            else:
                # Reset if it collapsed
                U[:, r] = np.random.randn(dim_u)
                V[:, r] = np.random.randn(dim_v)
                n_u = np.linalg.norm(U[:, r])
                U[:, r] /= n_u
                V[:, r] *= n_u
            
            new_err = compute_error()
            
            if new_err < current_err:
                current_err = new_err
            else:
                # Revert
                if mat_idx == 0:
                    U[idx, r] = old_val
                elif mat_idx == 1:
                    V[idx, r] = old_val
                else:
                    W[idx, r] = old_val
                # Re-normalize after revert?
                n_u = np.linalg.norm(U[:, r])
                if n_u > 1e-12:
                    U[:, r] /= n_u
                    V[:, r] *= n_u

        final_err = compute_error()
        if final_err < 1e-9:
            return U, V, W, True
        return U, V, W, False

    # Initialize Rank 10 candidates from Rank 11
    # We can try dropping one column and seeing if the rest can adjust.
    # Or, we can try a "split" where we take the first 10 columns of the seed.
    
    # Attempt 1: Drop the last column (schoolbook part) and see if Strassen part can be tweaked? Unlikely.
    # Attempt 2: Use a known algebraic structure.
    
    # Let's try a different initialization for Rank 10.
    # The tensor T(2,2,3) has a known rank 10 decomposition.
    # We will try to find it via random initialization + local search, using the rank-11 seed as a guide for initialization.
    
    # Since random search in high dimensions is hard, we will use the seed.
    # We'll take the first 10 columns of the seed and try to optimize.
    
    U_10_init = U_seed[:, :10]
    V_10_init = V_seed[:, :10]
    W_10_init = W_seed[:, :10]
    
    # Note: Dropping the 11th column means we lose the contribution of a11*b12 to C12.
    # The optimization might be able to absorb this into other terms if the structure allows.
    
    U_10, V_10, W_10, success = try_rank(10, U_10_init, V_10_init, W_10_init, max_iter=5000)
    
    if success:
        best_U, best_V, best_W = U_10, V_10, W_10
        best_rank = 10
    else:
        # If rank 10 fails, try a few random restarts with rank 10
        # Initialize with random values, but biased towards the seed?
        # Or just try dropping different columns.
        
        for drop_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if time.time() > deadline - 10:
                break
            mask = [i for i in range(11) if i != drop_idx]
            U_init = U_seed[:, mask]
            V_init = V_seed[:, mask]
            W_init = W_seed[:, mask]
            
            U_try, V_try, W_try, success = try_rank(10, U_init, V_init, W_init, max_iter=2000)
            if success:
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = 10
                break
                
    # Final check
    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook rank 12? Or the original rank 11?
        # The task asks for rank-minimal. If we can't prove 10, we return 11.
        # But we must return a VALID decomposition.
        # The seed is valid rank 11.
        pass

    # If we found rank 10, great. If not, we return the best valid one we have.
    # We should verify the best one we are about to return.
    if verify(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # If the "best" is invalid (shouldn't happen if we only update on success),
        # return the seed.
        return {"U": U_seed.tolist(), "V": V_seed.tolist(), "W": W_seed.tolist()}
