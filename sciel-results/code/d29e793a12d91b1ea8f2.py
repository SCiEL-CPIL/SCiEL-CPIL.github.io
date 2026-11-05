import numpy as np
import time
from fractions import Fraction
import scipy.optimize as opt

DIMENSIONS = [2, 3, 4]

def _construct_rank_22():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

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

    # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Try to find a lower rank decomposition (18-21)
    # The tensor T is of size (n*m, m*p, p*n)
    # T[i, j, k] = 1 if i corresponds to A[a,b], j to B[b,c], k to C[a,c]
    # i = a*m + b, j = b*p + c, k = a*p + c
    # Note: A is n x m, B is m x p, C is n x p.
    # Sum over b (inner dim of A and B).
    # C[a,c] = sum_b A[a,b] B[b,c]
    
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = a * p + c
                T[i, j, k] = 1.0
                
    T_flat = T.reshape(n*m, m*p * p*n)
    T_V = T.transpose(0, 2, 1).reshape(n*m*p*n, m*p)
    T_W = T.reshape(n*m*m*p, p*n)

    def check_solution(U, V, W, tol=1e-9):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Reconstruct T_approx
        # T_approx[i, j, k] = sum_l U[i,l] V[j,l] W[k,l]
        # This is a tensor contraction.
        # We can do this efficiently by iterating l or using einsum.
        # np.einsum('il,jl,kl->ijk', U, V, W)
        T_approx = np.einsum('il,jl,kl->ijk', U, V, W)
        if not np.allclose(T, T_approx, atol=tol):
            return False
        
        # Check for half-integer entries
        # The problem states "integer/half-integer entries".
        # This usually means 2*x is an integer.
        for mat in [U, V, W]:
            for val in mat.flatten():
                if not np.isclose(val * 2, np.round(val * 2), atol=1e-7):
                    return False
        return True

    def polish_solution(U, V, W, r):
        # Flatten U, V, W into a single vector x
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_r = x[:n*m*r].reshape(n*m, r)
            V_r = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_r = x[n*m*r + m*p*r :].reshape(p*n, r)
            T_approx = np.einsum('il,jl,kl->ijk', U_r, V_r, W_r)
            return (T_approx - T).flatten()
            
        try:
            # Use least_squares to refine
            result = opt.least_squares(objective, x0, method='trf', max_nfev=500)
            x_opt = result.x
            U_opt = x_opt[:n*m*r].reshape(n*m, r)
            V_opt = x_opt[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_opt = x_opt[n*m*r + m*p*r :].reshape(p*n, r)
            return U_opt, V_opt, W_opt
        except Exception:
            return U, V, W

    def solve_als(r, max_iters=20, tol=1e-12):
        # Initialize U, V, W with small random values
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        prev_err = np.inf
        for it in range(max_iters):
            # Update U
            Z = np.einsum('jl,kl->jkl', V, W).reshape(m*p*p*n, r)
            try:
                # Use lstsq for better numerical stability if possible, or pinv
                U, _, _, _ = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
                U = U.T
            except:
                return None

            # Update V
            Y = np.einsum('il,kl->ikl', U, W).reshape(n*m*p*n, r)
            try:
                V, _, _, _ = np.linalg.lstsq(Y.T, T_V.T, rcond=None)
                V = V.T
            except:
                return None

            # Update W
            X = np.einsum('il,jl->ijl', U, V).reshape(n*m*m*p, r)
            try:
                W, _, _, _ = np.linalg.lstsq(X.T, T_W.T, rcond=None)
                W = W.T
            except:
                return None

            # Check convergence
            err = np.linalg.norm(T - np.einsum('il,jl,kl->ijk', U, V, W))
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
            # Check time
            if time.time() > deadline:
                return None
                
        # Polish the solution
        U_pol, V_pol, W_pol = polish_solution(U, V, W, r)
        
        # Check if polished solution is valid
        if check_solution(U_pol, V_pol, W_pol):
            return U_pol, V_pol, W_pol
        
        # Try snapping to half-integers if close
        # This is a heuristic: if the solution is very close to half-integers, snap and check
        def snap_half_int(A):
            return np.round(A * 2) / 2.0

        U_snap = snap_half_int(U_pol)
        V_snap = snap_half_int(V_pol)
        W_snap = snap_half_int(W_pol)
        
        if check_solution(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
            
        # Also try the unpolished if polishing failed or moved away
        if check_solution(U, V, W):
            return U, V, W
            
        return None

    # Search for ranks 18, 19, 20, 21
    for r in [18, 19, 20, 21]:
        if time.time() > deadline:
            break
        for restart in range(20):
            if time.time() > deadline:
                break
            np.random.seed(restart + r * 1000)
            sol = solve_als(r)
            if sol is not None:
                U, V, W = sol
                # Convert to lists of floats (or ints if possible)
                # The requirement is "integer/half-integer entries"
                # We can just return the numpy arrays or lists.
                # The interface expects {'U', 'V', 'W'} with shapes.
                # The current program returns lists of lists.
                # I will return lists of lists to be safe, or numpy arrays if accepted.
                # The seed program returns lists.
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # Fallback to rank 22
    return _construct_rank_22()
