"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Pre-compute the list of valid (a, b, c) indices for the tensor
    # a = i*m + j, b = j*p + k, c = k*n + i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    num_triples = len(triples)
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n

    # Strategy: Use Alternating Least Squares (ALS) to find low-rank decompositions.
    # This is more robust than random search for higher ranks and can find exact solutions
    # by polishing with integer/half-integer rounding.
    
    # We try ranks from a lower bound up to the schoolbook rank.
    # Theoretical lower bound is max(n*m, m*p, p*n) = 15.
    # Known rank for 2x3x5 is likely around 18-20 based on literature (e.g., Strassen-like).
    # We start from 15 and go up.
    
    min_rank = 15
    max_rank_to_try = 28  # Don't try too high, schoolbook is 30
    
    # Optimization: Pre-compute the structure for faster ALS steps
    # For each triple index t, we have a_t, b_t, c_t.
    a_idx = np.array([t[0] for t in triples])
    b_idx = np.array([t[1] for t in triples])
    c_idx = np.array([t[2] for t in triples])

    def try_als_decomposition(rank, max_iter=100, restarts=10):
        """
        Attempts to find a rank-r decomposition using ALS.
        Returns (U, V, W) if found and verified, else None.
        """
        if time.time() > deadline - 5:
            return None
            
        np.random.seed(12345 + rank)
        
        for restart in range(restarts):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with random small values
            U = np.random.randn(dim_U, rank) * 0.1
            V = np.random.randn(dim_V, rank) * 0.1
            W = np.random.randn(dim_W, rank) * 0.1
            
            # To avoid trivial zero solutions, we can fix one element or use scaling
            # But ALS usually handles this if we initialize non-zero.
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                # Update U
                # For each triple t: sum_s U[a_t, s] * V[b_t, s] * W[c_t, s] = 1
                # This is linear in U.
                # We want to solve min ||A_U * U_flat - 1||^2
                # A_U has shape (num_triples, dim_U * rank)
                # Row t has non-zeros at columns corresponding to (a_t, s) for s in 0..rank-1
                # Value is V[b_t, s] * W[c_t, s]
                
                A_U = np.zeros((num_triples, dim_U * rank))
                # Efficiently fill A_U
                # For each s in 0..rank-1:
                #   A_U[t, a_t * rank + s] = V[b_t, s] * W[c_t, s]
                for s in range(rank):
                    vals = V[b_idx, s] * W[c_idx, s]
                    cols = a_idx * rank + s
                    A_U[np.arange(num_triples), cols] = vals
                
                rhs = np.ones(num_triples)
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_U, rhs, rcond=None)
                    U_new = U_flat.reshape(dim_U, rank)
                except:
                    continue
                
                # Update V
                A_V = np.zeros((num_triples, dim_V * rank))
                for s in range(rank):
                    vals = U_new[a_idx, s] * W[c_idx, s]
                    cols = b_idx * rank + s
                    A_V[np.arange(num_triples), cols] = vals
                
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_V, rhs, rcond=None)
                    V_new = V_flat.reshape(dim_V, rank)
                except:
                    continue
                
                # Update W
                A_W = np.zeros((num_triples, dim_W * rank))
                for s in range(rank):
                    vals = U_new[a_idx, s] * V_new[b_idx, s]
                    cols = c_idx * rank + s
                    A_W[np.arange(num_triples), cols] = vals
                
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A_W, rhs, rcond=None)
                    W_new = W_flat.reshape(dim_W, rank)
                except:
                    continue
                
                U, V, W = U_new, V_new, W_new
                
                # Check for convergence / exactness
                # Compute error
                # T_calc = sum_s U[a,s] V[b,s] W[c,s]
                # We can check if max abs diff from 1 is small
                err = 0.0
                # Vectorized error check
                # This might be slow if done every iter, but rank is small.
                # Let's do a quick check every few iterations or if it seems close.
                if it % 5 == 0 or it == max_iter - 1:
                    # Compute T for all triples
                    # T_t = sum_s U[a_t, s] * V[b_t, s] * W[c_t, s]
                    # We can use einsum on the full tensors or just loop over s
                    T_vals = np.zeros(num_triples)
                    for s in range(rank):
                        T_vals += U[a_idx, s] * V[b_idx, s] * W[c_idx, s]
                    
                    max_err = np.max(np.abs(T_vals - 1.0))
                    
                    if max_err < 1e-9:
                        # Found a solution!
                        # Try to round to half-integers
                        U_rounded = np.round(U * 2) / 2.0
                        V_rounded = np.round(V * 2) / 2.0
                        W_rounded = np.round(W * 2) / 2.0
                        
                        # Verify rounded solution
                        T_rounded = np.zeros(num_triples)
                        for s in range(rank):
                            T_rounded += U_rounded[a_idx, s] * V_rounded[b_idx, s] * W_rounded[c_idx, s]
                        
                        if np.max(np.abs(T_rounded - 1.0)) < 1e-9:
                            return U_rounded, V_rounded, W_rounded
                        
                        # If rounding failed, return the float solution?
                        # The problem requires integer/half-integer entries.
                        # So we must find a half-integer solution.
                        # If ALS converges to a non-half-integer, we might need to search locally.
                        # For now, let's assume if it's close, rounding works.
                        # If not, we continue searching.
                        pass
                    
                    # If error is very small but not zero, maybe it's stuck.
                    # If error is large, continue.

        return None

    # Try ranks from min_rank to max_rank_to_try
    for target_rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 10:
            break
        
        # Try ALS
        result = try_als_decomposition(target_rank, max_iter=50, restarts=5)
        
        if result is not None:
            U, V, W = result
            # Final verification
            if _verify_decomposition_fast(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break  # Found minimal rank in this range, stop
        
        # If ALS didn't find it, try a quick random search for this rank
        # This helps if ALS gets stuck in local minima.
        if time.time() < deadline - 15:
            np.random.seed(42 + target_rank)
            for attempt in range(20):
                if time.time() > deadline - 10:
                    break
                # Use smaller range for random init to help convergence
                U_rand = np.random.randint(-2, 3, size=(n*m, target_rank)) / 2.0
                V_rand = np.random.randint(-2, 3, size=(m*p, target_rank)) / 2.0
                W_rand = np.random.randint(-2, 3, size=(p*n, target_rank)) / 2.0
                
                if _verify_decomposition_fast(U_rand, V_rand, W_rand, n, m, p):
                    best_rank = target_rank
                    best_result = {
                        "U": U_rand.tolist(),
                        "V": V_rand.tolist(),
                        "W": W_rand.tolist()
                    }
                    break
            if best_rank == target_rank:
                break

    return best_result

def _verify_decomposition_fast(U, V, W, n, m, p):
    """
    Vectorized verification using NumPy.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    
    # Reshape for easier broadcasting
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)
    except Exception:
        return False

# EVOLVE-BLOCK-END
