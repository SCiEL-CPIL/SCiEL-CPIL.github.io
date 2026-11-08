import numpy as np
import time
import scipy.optimize as opt

def construct():
    n, m, p = 2, 3, 4
    rank = 22
    
    # Base solution (Rank 22)
    U_base = np.zeros((n * m, rank))
    V_base = np.zeros((m * p, rank))
    W_base = np.zeros((p * n, rank))
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U_base[idx][col] += val
        for idx, val in v_indices:
            V_base[idx][col] += val
        for idx, val in w_indices:
            W_base[idx][col] += val

    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Outer: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    # Convert to lists for return if search fails
    def to_lists(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Target Tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                # C[i, l] += A[i, j] * B[j, l]
                # W index: k*n+i where k=l, i=i -> l*n + i
                T[i * m + j, j * p + l, l * n + i] = 1

    def verify(U, V, W, tol=1e-9):
        recon = np.zeros(T.shape)
        for c in range(U.shape[1]):
            recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(recon, T, atol=tol, rtol=0):
            return False
        for M in (U, V, W):
            for v in M.flat:
                if v != int(v) and 2 * v != int(2 * v):
                    return False
        return True

    # Try to find rank 21 solution
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_solution = to_lists(U_base, V_base, W_base)
    
    try:
        r_target = 21
        # Initialize with rank 22 solution, drop last column
        U_init = U_base[:, :r_target].flatten()
        V_init = V_base[:, :r_target].flatten()
        W_init = W_base[:, :r_target].flatten()
        x0 = np.concatenate([U_init, V_init, W_init])
        
        def residual(x):
            U = x[:n*m*r_target].reshape(n*m, r_target)
            V = x[n*m*r_target : n*m*r_target + m*p*r_target].reshape(m*p, r_target)
            W = x[n*m*r_target + m*p*r_target :].reshape(p*n, r_target)
            recon = np.zeros(T.shape)
            for c in range(r_target):
                recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            return (recon - T).flatten()
        
        # Optimize
        result = opt.least_squares(residual, x0, method='lm', max_nfev=200)
        
        if time.time() < deadline:
            x_opt = result.x
            U_opt = x_opt[:n*m*r_target].reshape(n*m, r_target)
            V_opt = x_opt[n*m*r_target : n*m*r_target + m*p*r_target].reshape(m*p, r_target)
            W_opt = x_opt[n*m*r_target + m*p*r_target :].reshape(p*n, r_target)
            
            # Round to half-integers
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            if verify(U_round, V_round, W_round):
                best_solution = to_lists(U_round, V_round, W_round)
                
    except Exception as e:
        pass

    return best_solution
