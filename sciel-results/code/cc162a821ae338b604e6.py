import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that U, V, W form a valid tensor decomposition using vectorized operations."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 115s deadline with margin for 120s limit
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with the trivial decomposition (rank = n*m*p = 24)
    rank_sb = n * m * p
    U0 = np.zeros((nm, rank_sb))
    V0 = np.zeros((mp, rank_sb))
    W0 = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Strategy 1: Try to find lower-rank decompositions using structured random search
    # Try ranks from 23 down to 12
    for target_rank in range(23, 11, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random attempts for this rank
        max_attempts = 20
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random U, V, W with the target rank
            # Use values from a structured set
            rng = np.random.RandomState(attempt * 1000 + target_rank)
            vals = np.array([-1, -0.5, 0, 0.5, 1])
            
            U_rand = np.zeros((nm, target_rank))
            V_rand = np.zeros((mp, target_rank))
            W_rand = np.zeros((pn, target_rank))
            
            for r in range(target_rank):
                U_rand[:, r] = vals[rng.randint(0, len(vals), nm)]
                V_rand[:, r] = vals[rng.randint(0, len(vals), mp)]
                W_rand[:, r] = vals[rng.randint(0, len(vals), pn)]
            
            if _verify(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                found = True
                break
        
        if found:
            break
    
    # Strategy 2: Local search / perturbation from trivial decomposition
    # Try to merge columns by perturbing them
    if best_rank == rank_sb and time.time() < deadline - 10:
        U_curr = U0.copy()
        V_curr = V0.copy()
        W_curr = W0.copy()
        
        # For each pair of columns, try small perturbations to see if we can merge
        for r1 in range(rank_sb):
            if time.time() > deadline - 5:
                break
            for r2 in range(r1 + 1, rank_sb):
                if time.time() > deadline - 5:
                    break
                
                # Try to express column r2 as a linear combination involving column r1
                for delta in [-0.5, 0.5]:
                    U_test = U_curr.copy()
                    V_test = V_curr.copy()
                    W_test = W_curr.copy()
                    
                    # Perturb column r2 by delta * column r1
                    U_test[:, r2] += delta * U_test[:, r1]
                    V_test[:, r2] += delta * V_test[:, r1]
                    W_test[:, r2] += delta * W_test[:, r1]
                    
                    # Check if the perturbed decomposition is still valid
                    if _verify(U_test, V_test, W_test):
                        # Remove the redundant column
                        U_new = np.delete(U_test, r2, axis=1)
                        V_new = np.delete(V_test, r2, axis=1)
                        W_new = np.delete(W_test, r2, axis=1)
                        
                        if _verify(U_new, V_new, W_new):
                            best_rank = rank_sb - 1
                            best_U = U_new.copy()
                            best_V = V_new.copy()
                            best_W = W_new.copy()
                            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
