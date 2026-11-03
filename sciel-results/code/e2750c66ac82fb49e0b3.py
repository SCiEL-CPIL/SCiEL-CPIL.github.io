import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Precompute constraint indices
    # Equation: sum_r U[i*m + j1, r] V[j2*p + k, r] W[k*n + i, r] = delta[j1, j2]
    # Indices: i in [0, n), j1 in [0, m), j2 in [0, m), k in [0, p)
    cons_indices = []
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                for k in range(p):
                    target = 1.0 if j1 == j2 else 0.0
                    cons_indices.append((i, j1, j2, k, target))
    
    num_cons = len(cons_indices)
    
    def residual_vector(x, R):
        """
        Compute the residual vector for the given parameters x and rank R.
        x is flattened [U, V, W]
        """
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = x[0:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        res = np.zeros(num_cons)
        idx = 0
        for (i, j1, j2, k, target) in cons_indices:
            # sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r]
            # Vectorized dot product
            val = np.dot(U[i*m + j1], np.multiply(V[j2*p + k], W[k*n + i]))
            res[idx] = val - target
            idx += 1
        return res

    def verify_solution(U, V, W):
        """
        Verify if U, V, W satisfy the tensor equations exactly.
        """
        R = U.shape[1]
        for (i, j1, j2, k, target) in cons_indices:
            val = 0.0
            for r in range(R):
                val += U[i*m + j1, r] * V[j2*p + k, r] * W[k*n + i, r]
            if abs(val - target) > 1e-9:
                return False
        return True

    def snap_to_half_integers(U, V, W):
        """
        Snap entries to the nearest half-integer.
        """
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    def get_initial_solution_rank36():
        """
        Construct the known rank-36 solution using Strassen blocks.
        """
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1; W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            # M3
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1; V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1; V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1; W[k0*n + (i0+1)][col] += 1
            col += 1
            # M5
            U[i0*m + j0][col] += 1; U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1; W[(k0+1)*n + i0][col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0][col] += 1; U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1; V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1)][col] += 1; U[(i0+1)*m + (j0+1)][col] -= 1
            V[(j0+1)*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            col += 1

        def add_schoolbook_2x4x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(4):
                    U[(i0 + di)*m + (j0 + dj)][col] = 1
                    V[(j0 + dj)*p + k0][col] = 1
                    W[k0*n + (i0 + di)][col] = 1
                    col += 1

        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x4x1(0, 0, 4)
        
        return U, V, W

    U36, V36, W36 = get_initial_solution_rank36()
    
    start_time = time.time()
    time_limit = 140  # Safety margin
    
    best_rank = 36
    best_U, best_V, best_W = U36, V36, W36
    
    # Try to find lower ranks
    for R in range(35, 25, -1):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        # Try multiple initializations
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            if attempt == 0:
                # Truncate rank 36 solution
                if R < 36:
                    U_init = U36[:, :R]
                    V_init = V36[:, :R]
                    W_init = W36[:, :R]
                else:
                    continue
            elif attempt == 1:
                # Random small integers
                U_init = np.random.randint(-2, 3, size=(n*m, R)).astype(float)
                V_init = np.random.randint(-2, 3, size=(m*p, R)).astype(float)
                W_init = np.random.randint(-2, 3, size=(p*n, R)).astype(float)
            else:
                # Random small floats
                scale = 0.5 * (1 + attempt/10)
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use Levenberg-Marquardt
                result = opt.least_squares(residual_vector, x0, args=(R,), method='lm', max_nfev=10000)
                
                x_opt = result.x
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                U_opt = x_opt[0:size_U].reshape(n*m, R)
                V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                
                # Check residual
                res = residual_vector(x_opt, R)
                if np.max(np.abs(res)) < 1e-6:
                    # Snap to half-integers
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                    if verify_solution(U_snap, V_snap, W_snap):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            except Exception:
                pass

    # If no better rank found, return the rank 36 solution (snapped)
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    if verify_solution(U_snap, V_snap, W_snap):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Fallback
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
