"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition (rank = n*m*p = 50)
    rank_school = nm * mp * pn // n  # This is just n*m*p = 50
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.float64)
    V_school = np.zeros((mp, rank_school), dtype=np.float64)
    W_school = np.zeros((pn, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill

    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # Try to find lower rank decompositions
    # Known lower bounds for 2x5x5 matrix multiplication are around 30-35
    # We will try ranks from 30 up to rank_school - 1
    # But given the 150s constraint, we should be strategic.
    # The inspiration tried rank 34. Let's try a few candidate ranks.
    
    candidate_ranks = [30, 32, 34, 35, 36, 38, 40, 42, 45]
    # Filter out ranks >= best_rank
    candidate_ranks = [r for r in candidate_ranks if r < best_rank]
    
    def verify_decomposition(U, V, W, T):
        """Check if U, V, W exactly decompose T."""
        # T_comp[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_comp, T, atol=1e-8)

    def try_rank(R, max_time_per_rank=15.0):
        """Attempt to find a decomposition of rank R."""
        nonlocal best_result, best_rank
        
        size_U = nm
        size_V = mp
        size_W = pn
        total_size = R * (size_U + size_V + size_W)
        
        def objective(params):
            # params is flat array of length R * (size_U + size_V + size_W)
            # We reshape to (R, size_U), (R, size_V), (R, size_W) then transpose
            U = params[:R * size_U].reshape(R, size_U).T
            V = params[R * size_U:R * (size_U + size_V)].reshape(R, size_V).T
            W = params[R * (size_U + size_V):].reshape(R, size_W).T
            
            T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.sum((T_comp - T) ** 2)
            return err

        def get_factors(params, R):
            U = params[:R * size_U].reshape(R, size_U).T
            V = params[R * size_U:R * (size_U + size_V)].reshape(R, size_V).T
            W = params[R * (size_U + size_V):].reshape(R, size_W).T
            return U, V, W

        # Multiple random restarts
        for attempt in range(20):
            if time.time() > start_time + max_time_per_rank + best_rank * 0.5: # Simple heuristic
                break
            if time.time() > deadline:
                return False

            np.random.seed(123 + R * 100 + attempt)
            x0 = np.random.uniform(-1, 1, total_size)
            
            try:
                # Use L-BFGS-B
                res = minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 100, 'maxfun': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-6:
                    U, V, W = get_factors(res.x, R)
                    
                    # Try rounding to integers or half-integers
                    # First try rounding to nearest integer
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    
                    if verify_decomposition(U_int, V_int, T):
                        best_rank = R
                        best_result = {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        return True
                    
                    # Try rounding to nearest half-integer
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    
                    if verify_decomposition(U_half, V_half, T):
                        best_rank = R
                        best_result = {
                            "U": U_half.tolist(),
                            "V": V_half.tolist(),
                            "W": W_half.tolist()
                        }
                        return True
                        
            except Exception as e:
                pass

        return False

    # Try candidate ranks in increasing order
    for R in candidate_ranks:
        if time.time() > deadline - 5.0:
            break
        
        success = try_rank(R)
        if success:
            # Found a valid decomposition with rank R
            # We could stop here, but maybe try lower?
            # Since we are iterating in increasing order, this is the lowest so far.
            # However, finding one doesn't mean it's the absolute minimum.
            # But given time constraints, we return the best found.
            # Let's continue to see if we can find an even lower rank if time permits.
            # Actually, if we found rank R, we should break if we want to be fast, 
            # but the goal is minimal rank. 
            # Let's just record it and continue if time allows, but likely we should stop 
            # to ensure we return a valid result quickly.
            # Let's break to save time for verification/return.
            break

    return best_result
# EVOLVE-BLOCK-END
