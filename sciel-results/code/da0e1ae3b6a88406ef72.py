"""Combined program: structural schoolbook base with structured low-rank search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
_DEADLINE = time.time() + 115.0  # Safety margin before 120s/150s limit

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return U, V, W

def _compute_error(U, V, W, n, m, p, rank):
    """Compute the total absolute error of the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err += abs(s - 1.0)
    return err

def _try_local_search(U, V, W, n, m, p, rank, max_iters=5000, seed=42):
    """Local search to refine a decomposition."""
    rng = np.random.RandomState(seed)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    err = _compute_error(U, V, W, n, m, p, rank)
    if err < 1e-9:
        return U, V, W
    
    best_err = err
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        if err < 1e-9:
            return U, V, W
        
        if err < best_err:
            best_err = err
            if best_err < 0.01:
                # Try local refinement with small steps
                for r in range(rank):
                    for idx in range(n*m):
                        old = U[idx, r]
                        for delta in [-1, -0.5, 0.5, 1]:
                            new_val = old + delta
                            U[idx, r] = new_val
                            new_err = _compute_error(U, V, W, n, m, p, rank)
                            if new_err < err:
                                err = new_err
                            else:
                                U[idx, r] = old
                    for idx in range(m*p):
                        old = V[idx, r]
                        for delta in [-1, -0.5, 0.5, 1]:
                            new_val = old + delta
                            V[idx, r] = new_val
                            new_err = _compute_error(U, V, W, n, m, p, rank)
                            if new_err < err:
                                err = new_err
                            else:
                                V[idx, r] = old
                    for idx in range(p*n):
                        old = W[idx, r]
                        for delta in [-1, -0.5, 0.5, 1]:
                            new_val = old + delta
                            W[idx, r] = new_val
                            new_err = _compute_error(U, V, W, n, m, p, rank)
                            if new_err < err:
                                err = new_err
                            else:
                                W[idx, r] = old
        
        # Random perturbation
        r = rng.randint(rank)
        choice = rng.randint(3)
        if choice == 0:
            idx = rng.randint(n*m)
            U[idx, r] += rng.choice([-1, -0.5, 0.5, 1])
        elif choice == 1:
            idx = rng.randint(m*p)
            V[idx, r] += rng.choice([-1, -0.5, 0.5, 1])
        else:
            idx = rng.randint(p*n)
            W[idx, r] += rng.choice([-1, -0.5, 0.5, 1])
    
    if err < 1e-9:
        return U, V, W
    return None

def _structured_search(n, m, p, start_rank, end_rank):
    """Search for low-rank decompositions using structured initialization."""
    for rank in range(start_rank, min(end_rank, 61)):
        if time.time() > _DEADLINE:
            break
        
        # Try multiple structured initializations
        for seed in range(5):
            if time.time() > _DEADLINE:
                break
            
            rng = np.random.RandomState(seed)
            
            # Initialize with sparse random patterns
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Each rank term gets a random subset of indices
            for r in range(rank):
                # Assign random non-zero entries
                n_u = max(1, int(rng.randint(1, n*m//2 + 1)))
                n_v = max(1, int(rng.randint(1, m*p//2 + 1)))
                n_w = max(1, int(rng.randint(1, p*n//2 + 1)))
                
                u_indices = rng.choice(n*m, size=n_u, replace=False)
                v_indices = rng.choice(m*p, size=n_v, replace=False)
                w_indices = rng.choice(p*n, size=n_w, replace=False)
                
                for idx in u_indices:
                    U[idx, r] = rng.choice([-1, 1])
                for idx in v_indices:
                    V[idx, r] = rng.choice([-1, 1])
                for idx in w_indices:
                    W[idx, r] = rng.choice([-1, 1])
            
            # Try local search
            result = _try_local_search(U, V, W, n, m, p, rank, max_iters=3000, seed=seed)
            if result is not None:
                U_arr, V_arr, W_arr = result
                U_list = [[float(U_arr[i, r]) for r in range(rank)] for i in range(n*m)]
                V_list = [[float(V_arr[i, r]) for r in range(rank)] for i in range(m*p)]
                W_list = [[float(W_arr[i, r]) for r in range(rank)] for i in range(p*n)]
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try structured search for lower rank
    result = _structured_search(n, m, p, 20, 48)
    
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
