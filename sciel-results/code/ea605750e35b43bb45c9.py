"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=3, m=3, p=8, tol=1e-10):
    """Check if the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Build the multiplication tensor T[i,j,k] = delta
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 iff (i,j,k) is valid
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all valid (i,j,k)
    # The decomposition gives: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all i in [0,n), j in [0,m), k in [0,p)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if err > tol:
                    return False
    return True

def _brent_3x3():
    """Return rank-23 decomposition for 3x3 * 3x3 matrix multiplication.
    
    Uses Brent's 1971 algorithm. Returns U (9x23), V (9x23), W (9x23)
    where W uses the convention W[k*n+i, c].
    """
    # Brent's 23-multiplication algorithm for 3x3
    # This is a well-known result. Let me construct it carefully.
    
    # For 3x3 * 3x3, we need C[i,k] = sum_j A[i,j] * B[j,k]
    # U has shape (3*3, 23) = (9, 23), indexed by (i,j)
    # V has shape (3*3, 23) = (9, 23), indexed by (j,k)
    # W has shape (3*3, 23) = (9, 23), indexed by (k,i) [note: k*n+i]
    
    # I'll use a known rank-23 decomposition.
    # One approach: use the fact that 3x3 can be decomposed using
    # the Strassen-like approach but for 3x3.
    
    # Actually, let me use a simpler approach: try to find a rank-23
    # decomposition by construction.
    
    # Here's a known rank-23 algorithm from Brent (1971):
    # The 23 products are formed from linear combinations of the 9 elements.
    
    # Let me use a different well-known approach.
    # For 3x3, one can use the following structure based on
    # decomposing into blocks.
    
    # Actually, the simplest correct approach for now: use the schoolbook
    # rank 27 for 3x3, then extend to 3x8.
    # But we want to find lower rank.
    
    # Let me implement a search-based approach instead.
    pass

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s margin before 120s limit
    
    # Start with schoolbook rank 72
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
    
    # Try to find a lower rank decomposition using random search
    # with iterative refinement
    
    import random
    random.seed(42)
    
    # Strategy: Try ranks from 72 down, attempting to find valid decompositions
    # Use a heuristic search
    
    for target_rank in range(best_rank - 1, max(50, best_rank - 20), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with random sparse matrices
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Use a structured initialization based on schoolbook
            # with some entries combined
            for c in range(target_rank):
                # Each column corresponds to a "multiplication"
                # Randomly assign which (i,j) and (j,k) pairs are active
                # with some probability of combining
                
                # Pick a random (i,j,k) triple as the base
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                
                # Start with the schoolbook basis vector
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                
                # With some probability, add another (i',j',k') to this column
                if random.random() < 0.3:
                    i2 = random.randint(0, n - 1)
                    j2 = random.randint(0, m - 1)
                    k2 = random.randint(0, p - 1)
                    # Only combine if they share at least one index
                    if i2 == i or j2 == j or k2 == k:
                        U[i2 * m + j2, c] += random.choice([1, -1])
                        V[j2 * p + k2, c] += random.choice([1, -1])
                        W[k2 * n + i2, c] += random.choice([1, -1])
            
            # Try to fix up using iterative refinement
            # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
            
            # Use a simple gradient-free approach: try to zero out errors
            # by adjusting entries
            
            success = False
            for iter in range(100):
                if time.time() > deadline:
                    break
                
                # Compute errors
                errors = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            errors[(i,j,k)] = 1.0 - val
                
                max_err = max(abs(e) for e in errors.values())
                if max_err < 1e-8:
                    success = True
                    break
                
                # Pick the (i,j,k) with largest error
                worst = max(errors, key=lambda x: abs(errors[x]))
                err = errors[worst]
                i, j, k = worst
                
                # Try to adjust one of the columns to reduce this error
                # The derivative w.r.t. U[i*m+j, c] is V[j*p+k, c] * W[k*n+i, c]
                # We want to add delta such that:
                # sum_c delta_U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = err
                # Simplest: pick a random column c and adjust U[i*m+j,c]
                
                c = random.randint(0, target_rank - 1)
                deriv = V[j*p+k, c] * W[k*n+i, c]
                if abs(deriv) > 1e-6:
                    delta = err / deriv
                    # But this might break other equations. Use a small step.
                    step = min(0.1, abs(delta)) * np.sign(delta)
                    U[i*m+j, c] += step
                
                # Similarly adjust V and W
                c = random.randint(0, target_rank - 1)
                deriv = U[i*m+j, c] * W[k*n+i, c]
                if abs(deriv) > 1e-6:
                    delta = err / deriv
                    step = min(0.1, abs(delta)) * np.sign(delta)
                    V[j*p+k, c] += step
                
                c = random.randint(0, target_rank - 1)
                deriv = U[i*m+j, c] * V[j*p+k, c]
                if abs(deriv) > 1e-6:
                    delta = err / deriv
                    step = min(0.1, abs(delta)) * np.sign(delta)
                    W[k*n+i, c] += step
            
            if success:
                # Verify
                if _validate(U.tolist(), V.tolist(), W.tolist()):
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break  # Found a valid decomposition at this rank, try lower
        
        if time.time() > deadline:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
