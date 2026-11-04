"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Construct the target tensor T of shape (nm, mp, pn)
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # T[u, v, w] = 0 otherwise
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def residual(params, rank):
        """Vectorized residual computation for the full tensor decomposition."""
        U_flat = params[:nm * rank].reshape(nm, rank)
        V_flat = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W_flat = params[nm * rank + mp * rank:].reshape(pn, rank)

        # Compute the decomposed tensor D[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        # Using einsum for efficiency
        D = np.einsum('ur,vr,wr->uvw', U_flat, V_flat, W_flat)
        
        # Residual is D - T
        res = D - T
        return res.flatten()

    def check_decomposition(U, V, W, rank, tol=1e-8):
        """Verify that the decomposition is exact for all entries."""
        D = np.einsum('ur,vr,wr->uvw', U, V, W)
        max_err = np.max(np.abs(D - T))
        return max_err < tol, max_err

    def to_list(arr):
        """Convert numpy array to nested list with half-integer rounding."""
        lst = []
        for row in arr:
            row_lst = []
            for val in row:
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    row_lst.append(rounded)
                else:
                    row_lst.append(float(val))
            lst.append(row_lst)
        return lst

    def naive_decomposition(rank=None):
        """Return the naive schoolbook decomposition of rank n*m*p."""
        if rank is None:
            rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * (m * p) + j * p + k
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
        return {"U": U, "V": V, "W": W}

    def structured_init(rank, seed):
        """Create a structured initial guess based on naive decomposition structure."""
        np.random.seed(seed)
        # Start with a scaled identity-like structure then add noise
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Fill with small random values
        scale = 1.0 / (rank ** 0.5)
        U += np.random.randn(nm, rank) * scale
        V += np.random.randn(mp, rank) * scale
        W += np.random.randn(pn, rank) * scale
        
        # Add some structure: for each (i,j,k) triple, assign to a random component
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = np.random.randint(rank)
                    U[i * m + j, r] += 1.0
                    V[j * p + k, r] += 1.0
                    W[k * n + i, r] += 1.0
        
        return U, V, W

    # Try ranks from low to high, with multiple random restarts
    # Known lower bound is max(nm, mp, pn) = 12.
    # Known rank for 2x3x4 is 18. Focus more effort on 18.
    rank_order = [18, 17, 19, 16, 20, 15, 21, 14, 22, 13, 23, 12, 24]
    
    for rank in rank_order:
        if time.time() > deadline - 15:
            break

        remaining = deadline - 15 - time.time()
        # Allocate more attempts for rank 18 (known rank)
        if rank == 18:
            max_attempts = max(8, int(remaining / 3))
        else:
            max_attempts = max(3, int(remaining / 6))

        for attempt in range(max_attempts):
            if time.time() > deadline - 8:
                break

            # Use structured initialization for better convergence
            U, V, W = structured_init(rank, rank * 10000 + attempt * 7 + 42)
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                # First pass: coarse optimization with more iterations
                result = least_squares(residual, x0, args=(rank,), method='lm', max_nfev=500)
                if result.cost < 1e-6:
                    U_sol = result.x[:nm * rank].reshape(nm, rank)
                    V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                    W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)

                    valid, max_err = check_decomposition(U_sol, V_sol, W_sol, rank, tol=1e-6)

                    if valid:
                        # Polish pass with tighter tolerance and more iterations
                        x_polish = np.concatenate([U_sol.flatten(), V_sol.flatten(), W_sol.flatten()])
                        result2 = least_squares(residual, x_polish, args=(rank,), method='lm', max_nfev=300)
                        if result2.cost < 1e-10:
                            U_pol = result2.x[:nm * rank].reshape(nm, rank)
                            V_pol = result2.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                            W_pol = result2.x[nm * rank + mp * rank:].reshape(pn, rank)
                            valid2, _ = check_decomposition(U_pol, V_pol, W_pol, rank, tol=1e-10)
                            if valid2:
                                return {"U": to_list(U_pol), "V": to_list(V_pol), "W": to_list(W_pol)}
                        
                        # Second polish if first wasn't tight enough
                        if result2.cost < 1e-8:
                            x_polish2 = np.concatenate([result2.x[:nm * rank].reshape(nm, rank).flatten(),
                                                         result2.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank).flatten(),
                                                         result2.x[nm * rank + mp * rank:].reshape(pn, rank).flatten()])
                            result3 = least_squares(residual, x_polish2, args=(rank,), method='lm', max_nfev=200)
                            if result3.cost < 1e-12:
                                U_pol3 = result3.x[:nm * rank].reshape(nm, rank)
                                V_pol3 = result3.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                                W_pol3 = result3.x[nm * rank + mp * rank:].reshape(pn, rank)
                                valid3, _ = check_decomposition(U_pol3, V_pol3, W_pol3, rank, tol=1e-10)
                                if valid3:
                                    return {"U": to_list(U_pol3), "V": to_list(V_pol3), "W": to_list(W_pol3)}
                        
                        # If polish didn't achieve tight tolerance, still return if close enough
                        return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
            except Exception:
                continue

    # Fallback: naive decomposition
    return naive_decomposition()
# EVOLVE-BLOCK-END
