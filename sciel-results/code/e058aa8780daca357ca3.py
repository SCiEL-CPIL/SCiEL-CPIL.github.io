import numpy as np
from itertools import product
import random
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
    # Target tensor dimensions
    # U: (n*m, R) = (6, R)
    # V: (m*p, R) = (24, R)
    # W: (p*n, R) = (16, R)
    
    # The equation for each (i, j, k) is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We will search for a rank R < 48.
    # Lower bound for rank of 2x3x8?
    # For 2x2x2, rank is 7.
    # For 2x3x3, rank is likely around 15-18?
    # Let's try R from 32 to 47.
    
    # To speed up, we can fix one matrix to integers to reduce symmetry.
    # But let's try random search first.
    
    def get_residual(U, V, W):
        # Calculate residual for all (i, j, k)
        # Residual[i, j, k] = 1 - sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can vectorize this.
        # Flatten U, V, W into vectors?
        # Or just loop since 48 points is small.
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    res[i, j, k] = 1.0 - np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
        return res.flatten()

    def solve_rank(R, num_trials=5):
        for _ in range(num_trials):
            # Initialize with random small integers/half-integers
            # Using random integers in {-1, 0, 1} often helps
            U = np.random.choice([-1, 0, 1], size=(n * m, R)) * 1.0
            V = np.random.choice([-1, 0, 1], size=(m * p, R)) * 1.0
            W = np.random.choice([-1, 0, 1], size=(p * n, R)) * 1.0
            
            # We can use scipy.optimize.least_squares to minimize the residual
            # But the variables are 6R + 24R + 16R = 46R.
            # For R=40, that's 1840 variables. This might be slow.
            # Alternative: Use ALS (Alternating Least Squares).
            # Fix V and W, solve for U (linear system).
            # Fix U and W, solve for V.
            # Fix U and V, solve for W.
            
            # ALS is much faster for tensor decomposition.
            # Let's implement a simple ALS step.
            
            # Precompute indices
            idx_u = []
            idx_v = []
            idx_w = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u.append(i * m + j)
                        idx_v.append(j * p + k)
                        idx_w.append(k * n + i)
            
            idx_u = np.array(idx_u)
            idx_v = np.array(idx_v)
            idx_w = np.array(idx_w)
            
            # Run ALS iterations
            for it in range(50):
                # Solve for U
                # For each row u_idx, we have:
                # sum_r U[u_idx, r] * (V[idx_v, r] * W[idx_w, r]) = 1
                # This is a linear system A_u * U_row = 1
                # Where A_u is (48, R)
                
                # Construct A_u
                # A_u[r, c] = V[idx_v[r], c] * W[idx_w[r], c]
                # We can do this for each unique u_idx.
                # There are 6 unique u_idx values.
                
                for u_idx in range(n * m):
                    mask = (idx_u == u_idx)
                    A = np.column_stack([V[idx_v[mask], :] * W[idx_w[mask], :]])
                    b = np.ones(np.sum(mask))
                    # Solve least squares A * U_row = b
                    try:
                        U_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[u_idx, :] = U_row
                    except:
                        pass

                # Solve for V
                for v_idx in range(m * p):
                    mask = (idx_v == v_idx)
                    A = np.column_stack([U[idx_u[mask], :] * W[idx_w[mask], :]])
                    b = np.ones(np.sum(mask))
                    try:
                        V_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[v_idx, :] = V_row
                    except:
                        pass

                # Solve for W
                for w_idx in range(p * n):
                    mask = (idx_w == w_idx)
                    A = np.column_stack([U[idx_u[mask], :] * V[idx_v[mask], :]])
                    b = np.ones(np.sum(mask))
                    try:
                        W_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[w_idx, :] = W_row
                    except:
                        pass

            # After ALS, check residual
            res = get_residual(U, V, W)
            err = np.linalg.norm(res)
            
            if err < 1e-6:
                # Found a real solution. Now round to half-integers.
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check if rounded solution is exact
                res_rounded = get_residual(U_rounded, V_rounded, W_rounded)
                if np.linalg.norm(res_rounded) < 1e-9:
                    # Convert to lists
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        return None

    # Search for minimal rank
    for R in range(32, 48):
        res = solve_rank(R, num_trials=10)
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
