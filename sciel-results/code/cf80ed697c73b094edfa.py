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

    def snap_to_half_int(arr):
        """Snap array values to nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def verify_snapped(U, V, W, rank):
        """Verify that the snapped matrices form a valid decomposition."""
        U_s = snap_to_half_int(U)
        V_s = snap_to_half_int(V)
        W_s = snap_to_half_int(W)
        
        if check_decomposition(U_s, V_s, W_s, rank, tol=1e-6):
            return U_s, V_s, W_s
        return None

    def try_local_snap_search(U, V, W, rank, max_iters=200):
        """Try local search around continuous solution to find half-integer solution."""
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_cost = np.inf
        
        for it in range(max_iters):
            # Snap current solution
            U_s = snap_to_half_int(best_U)
            V_s = snap_to_half_int(best_V)
            W_s = snap_to_half_int(best_W)
            
            if check_decomposition(U_s, V_s, W_s, rank, tol=1e-6):
                return U_s, V_s, W_s
            
            # Try small perturbations
            found_improvement = False
            for _ in range(10):
                # Random small perturbation
                eps = 0.1 * np.random.randn(nm, rank)
                eps_v = 0.1 * np.random.randn(mp, rank)
                eps_w = 0.1 * np.random.randn(pn, rank)
                
                U_try = best_U + eps
                V_try = best_V + eps_v
                W_try = best_W + eps_w
                
                cost = np.sum(compute_residual(U_try, V_try, W_try) ** 2)
                
                if cost < best_cost:
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_cost = cost
                    found_improvement = True
            
            if not found_improvement:
                break
        
        return None

    def to_list(arr):
        """Convert numpy array to nested list."""
        return arr.tolist()

    # Algebraic construction: try to build decomposition from sum-of-products form
    # For matrix multiplication C = AB, we can use:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # A rank-r decomposition has the form:
    # C[i,k] = sum_s U[i,j_s] * V[j_s,k] * W[i,k]  (simplified view)
    # More precisely, each term s contributes: U[:,s] * V[:,s] * W[:,s] element-wise
    
    # Strategy: use structured initializations based on known algebraic identities
    
    def algebraic_init(rank, seed):
        """Generate structured initializations for the decomposition."""
        np.random.seed(seed)
        
        # Start with a structured pattern
        # For 2x3x4, try to leverage the small dimensions
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Assign each (i,j,k) triple to one or more rank terms
        # Use a hash-like assignment to spread load
        for s in range(rank):
            # Each term s handles a subset of triples
            # Pick random i, j, k patterns
            i_pat = np.random.randint(0, n, size=1)
            j_pat = np.random.randint(0, m, size=1)
            k_pat = np.random.randint(0, p, size=1)
            
            # Assign values based on the pattern
            # For term s, set U[i*m+j, s] for relevant i,j
            # Set V[j*p+k, s] for relevant j,k
            # Set W[k*n+i, s] for relevant k,i
            
            # Use a more systematic approach: each term covers a "slice"
            # Term s: fix i = s % n, j = (s // n) % m, k = (s // (n*m)) % p
            i_s = s % n
            j_s = (s // n) % m
            k_s = (s // (n * m)) % p
            
            # But this only covers n*m*p = 24 terms max
            # For rank < 24, we need shared terms
            
            # Alternative: use random but structured
            # Each term s gets random but correlated indices
            i_s = np.random.randint(0, n)
            j_s = np.random.randint(0, m)
            k_s = np.random.randint(0, p)
            
            # Set the three entries
            U[i_s * m + j_s, s] = 1.0
            V[j_s * p + k_s, s] = 1.0
            W[k_s * n + i_s, s] = 1.0
        
        # Add some noise to break symmetry
        U += 0.1 * np.random.randn(nm, rank)
        V += 0.1 * np.random.randn(mp, rank)
        W += 0.1 * np.random.randn(pn, rank)
        
        return U, V, W

    def structured_init(rank, seed):
        """Another structured initialization using circulant-like patterns."""
        np.random.seed(seed)
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Use a more algebraic structure
        # For each term s, create a "rank-1" contribution
        # that covers multiple (i,j,k) triples
        
        for s in range(rank):
            # Choose a pattern for this term
            # Use modular arithmetic to create structured patterns
            base = s % max(n, m, p)
            
            # Assign to multiple positions
            for i in range(n):
                for j in range(m):
                    # Probability-based assignment
                    if (i + j + s) % max(1, rank // (n * m)) == 0:
                        U[i * m + j, s] = np.random.choice([-1.0, 1.0])
                        
            for j in range(m):
                for k in range(p):
                    if (j + k + s) % max(1, rank // (m * p)) == 0:
                        V[j * p + k, s] = np.random.choice([-1.0, 1.0])
                        
            for k in range(p):
                for i in range(n):
                    if (k + i + s) % max(1, rank // (p * n)) == 0:
                        W[k * n + i, s] = np.random.choice([-1.0, 1.0])
        
        return U, V, W

    # Try ranks from low to high
    for rank in range(15, 25):
        if time.time() > deadline - 10:
            break

        remaining_time = deadline - 10 - time.time()
        if remaining_time <= 0: break
        
        # Allocate more attempts to lower ranks
        attempts = min(30, int(remaining_time / 2.0))
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break

            # Use different initialization strategies
            if attempt % 3 == 0:
                U, V, W = algebraic_init(rank, rank * 1000 + attempt)
            elif attempt % 3 == 1:
                U, V, W = structured_init(rank, rank * 1000 + attempt + 100)
            else:
                # Pure random
                np.random.seed(rank * 1000 + attempt + 200)
                U = np.random.randn(nm, rank)
                V = np.random.randn(mp, rank)
                W = np.random.randn(pn, rank)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                # Use LM for speed
                result = least_squares(residual, x0, args=(rank,), method='lm', max_nfev=300)
                
                if result.cost < 1e-4:
                    U_sol = result.x[:nm * rank].reshape(nm, rank)
                    V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                    W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)

                    # Try direct snap
                    snapped = verify_snapped(U_sol, V_sol, W_sol, rank)
                    if snapped:
                        U_s, V_s, W_s = snapped
                        return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
                    
                    # Try local search around the solution
                    local_result = try_local_snap_search(U_sol, V_sol, W_sol, rank)
                    if local_result:
                        U_s, V_s, W_s = local_result
                        return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
                    
            except Exception:
                continue

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
