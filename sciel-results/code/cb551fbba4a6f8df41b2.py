"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Nonzero entries of the tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    nz = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
    nz_set = set(nz)

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact within tolerance."""
        r = U.shape[1]
        # Compute tensor T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # Check all entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = 0.0
                    for t in range(r):
                        val += U[a, t] * V[b, t] * W[c, t]
                    expected = 1.0 if (a, b, c) in nz_set else 0.0
                    if abs(val - expected) > tol:
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using matrix operations."""
        r = U.shape[1]
        # For each t, form outer product u_t outer v_t outer w_t
        # Sum them up and compare with T
        # More efficient: for each (a,b), compute sum_t U[a,t]*V[b,t]*W[:,t] and check against expected
        for a in range(n*m):
            for b in range(m*p):
                # Compute vector w_sum[c] = sum_t U[a,t]*V[b,t]*W[c,t]
                w_sum = np.zeros(p*n)
                for t in range(r):
                    w_sum += U[a, t] * V[b, t] * W[:, t]
                # Check against expected
                for c in range(p*n):
                    expected = 1.0 if (a, b, c) in nz_set else 0.0
                    if abs(w_sum[c] - expected) > tol:
                        return False
        return True

    # Theoretical lower bound
    lb = max(n*m, m*p, p*n)

    np.random.seed(12345)

    # Strategy: Use alternating least squares (ALS) to find a low-rank decomposition
    # T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    
    # For ALS, we fix two matrices and solve for the third using linear least squares.
    # Fix V and W, solve for U:
    #   For each a, we need: sum_t U[a,t] * (V[b,t]*W[c,t]) = T[a,b,c] for all (b,c)
    #   This is a linear system in U[a,:]. The design matrix has rows indexed by (b,c),
    #   with entry (b,c,t) = V[b,t]*W[c,t]. Target is T[a,b,c].
    #
    # Similarly for V and W.

    def als_solve_U(V, W, T_flat):
        """Solve for U given V and W. T_flat is a dict mapping (a,b,c) -> value."""
        r = V.shape[1]
        U = np.zeros((n*m, r))
        for a in range(n*m):
            # Build design matrix A: (m*p * p*n) x r
            # But we only need equations where T[a,b,c] is nonzero, plus zeros elsewhere
            # Actually, we need ALL equations to ensure T[a,b,c]=0 where expected
            # But that's a huge system. Instead, use least squares on all (b,c) pairs.
            
            # For efficiency, note that V[b,t]*W[c,t] can be computed as:
            # Let M = V^T * W  (r x r matrix? No, that's not right)
            # Actually, for each t, the contribution to U[a,t] is coupled with V[b,t]*W[c,t]
            
            # Let's build the full design matrix.
            # Rows: (b,c) pairs, total m*p*p*n = 4*5*5*3 = 300 rows
            # Cols: r
            A = np.zeros((m*p * p*n, r))
            target = np.zeros(m*p * p*n)
            idx = 0
            for b in range(m*p):
                for c in range(p*n):
                    for t in range(r):
                        A[idx, t] = V[b, t] * W[c, t]
                    if (a, b, c) in nz_set:
                        target[idx] = 1.0
                    idx += 1
            
            # Solve least squares: A @ U[a,:] = target
            sol, residuals, rank, sv = np.linalg.lstsq(A, target, rcond=None)
            U[a, :] = sol
        return U

    def als_solve_V(U, W, T_flat):
        """Solve for V given U and W."""
        r = U.shape[1]
        V = np.zeros((m*p, r))
        for b in range(m*p):
            A = np.zeros((n*m * p*n, r))
            target = np.zeros(n*m * p*n)
            idx = 0
            for a in range(n*m):
                for c in range(p*n):
                    for t in range(r):
                        A[idx, t] = U[a, t] * W[c, t]
                    if (a, b, c) in nz_set:
                        target[idx] = 1.0
                    idx += 1
            sol, residuals, rank, sv = np.linalg.lstsq(A, target, rcond=None)
            V[b, :] = sol
        return V

    def als_solve_W(U, V, T_flat):
        """Solve for W given U and V."""
        r = U.shape[1]
        W = np.zeros((p*n, r))
        for c in range(p*n):
            A = np.zeros((n*m * m*p, r))
            target = np.zeros(n*m * m*p)
            idx = 0
            for a in range(n*m):
                for b in range(m*p):
                    for t in range(r):
                        A[idx, t] = U[a, t] * V[b, t]
                    if (a, b, c) in nz_set:
                        target[idx] = 1.0
                    idx += 1
            sol, residuals, rank, sv = np.linalg.lstsq(A, target, rcond=None)
            W[c, :] = sol
        return W

    # Try ranks from lower bound up, using ALS with multiple random restarts
    for r in range(lb, best_rank):
        if time.time() > deadline - 15:
            break
        
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # ALS iterations
            converged = False
            for it in range(100):
                if time.time() > deadline - 15:
                    break
                
                U = als_solve_U(V, W, None)
                V = als_solve_V(U, W, None)
                W = als_solve_W(U, V, None)
                
                # Check convergence
                if it % 10 == 9:
                    if verify_fast(U, V, W, tol=1e-8):
                        converged = True
                        break
            
            if converged:
                # Found a valid decomposition with rank r
                best_rank = r
                U_list = [[float(U[i, j]) for j in range(r)] for i in range(n*m)]
                V_list = [[float(V[i, j]) for j in range(r)] for i in range(m*p)]
                W_list = [[float(W[i, j]) for j in range(r)] for i in range(p*n)]
                best = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found a decomposition with this rank, move to next rank
    
    return best
# EVOLVE-BLOCK-END
