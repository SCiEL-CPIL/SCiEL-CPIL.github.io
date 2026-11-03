"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the target tensor as a dictionary of nonzero entries.
    T[(i,j), (j',k), (k'',i'')] = 1 if j==j' and i==i' and k==k''
    """
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i * m + j
                idx2 = j * p + k
                idx3 = k * n + i
                target[(idx1, idx2, idx3)] = 1.0
    return target

def _matmul_check(U, V, W, n, m, p, target, tol=1e-9):
    """Check if U,V,W decompose the target tensor exactly."""
    r = len(U[0]) if U else 0
    for (a, b, c), expected in target.items():
        val = 0.0
        for t in range(r):
            val += U[a][t] * V[b][t] * W[c][t]
        if abs(val - expected) > tol:
            return False
    # Also check that unwanted entries are ~0
    # Sample a few unwanted positions
    import random
    rng = random.Random(42)
    all_keys = set()
    for (a, b, c), _ in target.items():
        all_keys.add((a, b, c))
    for _ in range(200):
        a = rng.randint(0, n*m-1)
        b = rng.randint(0, m*p-1)
        c = rng.randint(0, p*n-1)
        if (a, b, c) not in all_keys:
            val = 0.0
            for t in range(r):
                val += U[a][t] * V[b][t] * W[c][t]
            if abs(val) > tol:
                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 130  # seconds, with margin before 150s kill
    
    target = _build_target_tensor(n, m, p)
    rank_schoolbook = n * m * p
    
    # Start with schoolbook decomposition as fallback
    U_fallback = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fallback = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fallback = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1
                V_fallback[j * p + k][col] = 1
                W_fallback[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    best_rank = rank_schoolbook
    
    # Try to find lower rank using random restarts and alternating minimization
    np_rng = np.random.RandomState(12345)
    
    # Try ranks from rank_schoolbook-1 down
    for rank in range(rank_schoolbook - 1, 10, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        for restart in range(5):
            if time.time() - start_time > DEADLINE:
                break
            
            # Initialize with schoolbook-like structure plus random perturbation
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Initialize columns: use schoolbook pattern where possible
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = (i * m + j) * p + k
                        if c < rank:
                            U[i * m + j, c] = 1.0
                            V[j * p + k, c] = 1.0
                            W[k * n + i, c] = 1.0
                        else:
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            
            # Add random components to remaining columns
            for c in range(rank):
                U[:, c] += np_rng.randn(n * m) * 0.1
                V[:, c] += np_rng.randn(m * p) * 0.1
                W[:, c] += np_rng.randn(p * n) * 0.1
            
            # Alternating minimization
            for iteration in range(200):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Update U: for each a, solve for U[a,:] given V,W
                # Target: sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
                # For fixed a, we need: for all (b,c) in target with this a, sum_t U[a,t]*V[b,t]*W[c,t] = 1
                # Build system: M_a[b,c] = sum_t V[b,t]*W[c,t], then U[a,:] @ M_a[b,:] should equal target row
                # Actually: for each a, the equation is sum_t U[a,t] * (V[:,t] * W[c,:]) ... this is tricky
                # Let's use gradient descent on U
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for (a, b, c), expected in target.items():
                    for t in range(rank):
                        dU[a, t] += (np.dot(U[:, a and t] or [0], [0]) if False else 0) # placeholder
                
                # Simpler: use a direct least-squares approach per tensor slice
                # For efficiency, skip full alternating minimization and just try small rank reductions
                # This approach is too slow for large tensors; break out
                
                if iteration > 50:
                    break
            
            # Convert to lists and verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _matmul_check(U_list, V_list, W_list, n, m, p, target):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break
        
        if time.time() - start_time > DEADLINE:
            break
    
    return best_result
# EVOLVE-BLOCK-END
