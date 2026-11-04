"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares, minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Precompute index triples and their corresponding row indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    u_idx_list = np.array([i * m + j for (i, j, k) in indices])
    v_idx_list = np.array([j * p + k for (i, j, k) in indices])
    w_idx_list = np.array([k * n + i for (i, j, k) in indices])
    num_triples = len(indices)

    def compute_residual(U, V, W):
        """Compute the residual for the tensor decomposition."""
        U_rows = U[u_idx_list, :]  # (num_triples, rank)
        V_rows = V[v_idx_list, :]  # (num_triples, rank)
        W_rows = W[w_idx_list, :]  # (num_triples, rank)

        # Compute the product sum for each triple
        products = U_rows * V_rows * W_rows  # (num_triples, rank)
        res = np.sum(products, axis=1) - 1.0
        return res

    def residual(params, rank):
        """Vectorized residual computation for the tensor decomposition."""
        U_flat = params[:nm * rank].reshape(nm, rank)
        V_flat = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W_flat = params[nm * rank + mp * rank:].reshape(pn, rank)
        return compute_residual(U_flat, V_flat, W_flat)

    def check_decomposition(U, V, W, rank, tol=1e-8):
        """Verify that the decomposition is exact."""
        res = compute_residual(U, V, W)
        return np.max(np.abs(res)) < tol

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

    def polish_solution(U, V, W, rank):
        """High-precision polish pass using tighter tolerance."""
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B with tighter convergence criteria
        try:
            result = minimize(
                lambda x: np.sum(np.square(residual(x, rank))),
                x0,
                method='L-BFGS-B',
                options={'ftol': 1e-15, 'gtol': 1e-15, 'maxiter': 500}
            )
            if result.success and result.fun < 1e-12:
                U_sol = result.x[:nm * rank].reshape(nm, rank)
                V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)
                if check_decomposition(U_sol, V_sol, W_sol, rank, tol=1e-10):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
        
        return None

    def structured_init(rank, seed):
        """Generate structured initializations based on tensor structure."""
        np.random.seed(seed)
        
        # Strategy 1: Random with scaling
        scale = 1.0 / (rank ** 0.5)
        U = np.random.randn(nm, rank) * scale
        V = np.random.randn(mp, rank) * scale
        W = np.random.randn(pn, rank) * scale
        
        # Strategy 2: Sparse initialization - each row gets a few non-zero entries
        if seed % 3 == 1:
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            for c in range(rank):
                # Assign random indices to each column
                u_row = np.random.randint(0, nm)
                v_row = np.random.randint(0, mp)
                w_row = np.random.randint(0, pn)
                U[u_row, c] = 1.0
                V[v_row, c] = 1.0
                W[w_row, c] = 1.0
        
        # Strategy 3: Block-structured based on matrix dimensions
        elif seed % 3 == 2:
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            # Try to cover all (i,j,k) triples with minimal factors
            for c in range(rank):
                # Pick a random (i,j,k) and assign
                i = np.random.randint(0, n)
                j = np.random.randint(0, m)
                k = np.random.randint(0, p)
                
                # With some probability, extend to cover related indices
                if np.random.random() < 0.3:
                    # Cover all k for this (i,j)
                    for k2 in range(p):
                        U[i*m+j, c] += 1.0
                        V[j*p+k2, c] += 1.0
                        W[k2*n+i, c] += 1.0
                elif np.random.random() < 0.6:
                    # Cover all j for this (i,k)
                    for j2 in range(m):
                        U[i*m+j2, c] += 1.0
                        V[j2*p+k, c] += 1.0
                        W[k*n+i, c] += 1.0
                else:
                    # Just the single triple
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
        
        return U, V, W

    def try_solve_rank(rank, max_attempts, time_budget):
        """Try to find a decomposition for given rank."""
        for attempt in range(max_attempts):
            if time.time() > start_time + time_budget:
                return None
            
            seed = rank * 10000 + attempt * 7 + 42
            U, V, W = structured_init(rank, seed)
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                # First pass: Levenberg-Marquardt
                result = least_squares(residual, x0, args=(rank,), method='lm', max_nfev=300)
                
                if result.cost < 1e-8:
                    U_sol = result.x[:nm * rank].reshape(nm, rank)
                    V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                    W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)

                    if check_decomposition(U_sol, V_sol, W_sol, rank, tol=1e-6):
                        # Try to polish for higher precision
                        polished = polish_solution(U_sol, V_sol, W_sol, rank)
                        if polished is not None:
                            U_sol, V_sol, W_sol = polished
                        return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
            except Exception:
                continue
                
            # Second pass: Try with different solver
            try:
                result = minimize(
                    lambda x: np.sum(np.square(residual(x, rank))),
                    x0,
                    method='L-BFGS-B',
                    options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 300}
                )
                if result.fun < 1e-10:
                    U_sol = result.x[:nm * rank].reshape(nm, rank)
                    V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                    W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)
                    if check_decomposition(U_sol, V_sol, W_sol, rank, tol=1e-6):
                        polished = polish_solution(U_sol, V_sol, W_sol, rank)
                        if polished is not None:
                            U_sol, V_sol, W_sol = polished
                        return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
            except Exception:
                continue
        
        return None

    # Try ranks from low to high, starting from theoretical lower bound
    # For 2x3 x 3x4, known results suggest rank around 14-16 might work
    lower_bound = max(n*m, m*p, p*n)  # Simple lower bound
    for rank in range(12, 25):
        if time.time() > deadline - 20:
            break

        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
            
        # Allocate time budget based on rank (lower ranks get more attempts)
        if rank <= 14:
            time_budget = min(remaining, 30)
            max_attempts = max(5, int(time_budget / 3))
        elif rank <= 16:
            time_budget = min(remaining, 20)
            max_attempts = max(3, int(time_budget / 4))
        else:
            time_budget = min(remaining, 10)
            max_attempts = max(2, int(time_budget / 5))

        result = try_solve_rank(rank, max_attempts, time_budget)
        if result is not None:
            return result

    # Fallback: naive decomposition
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
# EVOLVE-BLOCK-END
