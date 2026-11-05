"""Matrix multiplication 3x3x5 rank-minimal exact tensor decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[(i,j),(j,k),(i,k)] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, T):
    """Check if U,V,W exactly decompose T (within tolerance)."""
    rank = U.shape[1]
    approx = np.zeros_like(T)
    for r in range(rank):
        approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(T, approx, atol=1e-10)

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _round_to_halfint(x):
    """Round array to nearest half-integer."""
    return np.round(x * 2) / 2

def _round_to_int(x):
    """Round array to nearest integer."""
    return np.round(x)

def _bini23_init(n, m, p, target_rank, seed=0):
    """
    Initialize using Bini's 23-rank decomposition for the 3x3x3 sub-problem.
    The idea: use the first 3 k-values with Bini 23, then add rank for remaining k-values.
    """
    rng = np.random.RandomState(seed)
    
    # Bini's 23-rank factors for 3x3x3 (simplified - use known structure)
    # For 3x3x3, the tensor is T[i,j,k] = delta_{i,k} * A[i,j] where A is identity-like
    # Actually, for matrix multiplication, T[i,j,k] = 1 when the product C[i,k] = sum_j A[i,j]*B[j,k]
    # The 3x3x3 tensor has entries T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
    
    # Bini 23 factors (integer entries in {-1, 0, 1})
    # These are the standard 23-product decomposition
    U_bini_23 = np.array([
        [1,0,0, 0,0,0, 0,0,0],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 1,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,1],
        [1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 0,0,1],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [1,1,0, 0,0,0, 0,0,0],
        [0,0,0, 1,1,0, 0,0,0],
        [0,0,0, 0,0,0, 1,1,0],
        [1,0,0, 1,0,0, 0,0,0],
        [0,1,0, 0,1,0, 0,0,0],
        [0,0,1, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 1,0,1],
    ], dtype=float)
    
    V_bini_23 = np.array([
        [1,0,0, 0,0,0, 0,0,0],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 1,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,1],
        [0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,0,1],
        [0,0,0, 1,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [1,0,0, 1,0,0, 0,0,0],
        [0,1,0, 0,1,0, 0,0,0],
        [0,0,1, 0,0,1, 0,0,0],
        [1,0,0, 0,1,0, 0,0,0],
        [0,1,0, 0,0,1, 0,0,0],
        [0,0,1, 1,0,0, 0,0,0],
        [0,0,0, 1,1,1, 0,0,0],
    ], dtype=float)
    
    W_bini_23 = np.array([
        [1,0,0, 0,0,0, 0,0,0],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 1,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,1],
        [1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [1,0,0, 1,0,0, 0,0,0],
        [0,1,0, 0,1,0, 0,0,0],
        [0,0,1, 0,0,1, 0,0,0],
        [0,1,0, 1,0,0, 0,0,0],
        [0,0,1, 0,1,0, 0,0,0],
        [1,0,0, 0,0,1, 0,0,0],
        [0,0,0, 1,1,1, 0,0,0],
    ], dtype=float)
    
    # For 3x3x5, we need to extend this to handle 5 k-values
    # Strategy: use Bini 23 for the first 3 k-values, then add 2 more rank terms
    # for k=3 and k=4
    
    U_init = np.zeros((n*m, target_rank))
    V_init = np.zeros((m*p, target_rank))
    W_init = np.zeros((p*n, target_rank))
    
    # Place Bini 23 in the first 23 columns
    bini_rank = min(23, target_rank)
    for r in range(bini_rank):
        # U: shape (9, 23) -> place in (9, target_rank)
        U_init[:, r] = U_bini_23[:, r]
        # V: shape (9, 23) but we need (15, target_rank) for m*p=15
        # V_bini_23 is for 3x3, so it's (9, 23). For 3x5, V should be (15, rank)
        # Map: V[j,k] for j in 0..2, k in 0..4
        # Bini V is for k in 0..2, so place in first 9 rows
        V_init[:9, r] = V_bini_23[:, r]
        # W: shape (15, 23) for p*n=15, Bini W is (9, 23) for 3x3
        # W[k,i] for k in 0..4, i in 0..2
        # Bini W is for k in 0..2, so place in first 9 rows
        W_init[:9, r] = W_bini_23[:, r]
    
    # Add additional rank for k=3,4 if target_rank > 23
    if target_rank > 23:
        for r in range(23, target_rank):
            # Random initialization for the extra rank
            U_init[:, r] = rng.choice([0, 1, -1], size=n*m) * 0.5
            V_init[:, r] = rng.choice([0, 1, -1], size=m*p) * 0.5
            W_init[:, r] = rng.choice([0, 1, -1], size=p*n) * 0.5
    
    return U_init, V_init, W_init

def _try_rank_powell(n, m, p, T, target_rank, deadline, seed=0, init_strategy='random'):
    """
    Attempt to find a rank-target_rank decomposition using Powell's method.
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Generate initial parameters based on strategy
    if init_strategy == 'bini23':
        U_init, V_init, W_init = _bini23_init(n, m, p, target_rank, seed)
    elif init_strategy == 'truncated_schoolbook':
        U_sch, V_sch, W_sch = _schoolbook(n, m, p)
        # Truncate to target_rank
        U_init = U_sch[:, :target_rank].copy()
        V_init = V_sch[:, :target_rank].copy()
        W_init = W_sch[:, :target_rank].copy()
    elif init_strategy == 'sparse_binary':
        U_init = (rng.rand(n*m, target_rank) < 0.3).astype(float) * rng.choice([-1, 1], size=(n*m, target_rank))
        V_init = (rng.rand(m*p, target_rank) < 0.3).astype(float) * rng.choice([-1, 1], size=(m*p, target_rank))
        W_init = (rng.rand(p*n, target_rank) < 0.3).astype(float) * rng.choice([-1, 1], size=(p*n, target_rank))
    else:  # random continuous
        scale = 0.3 + 0.4 * rng.rand()
        U_init = rng.randn(n*m, target_rank) * scale
        V_init = rng.randn(m*p, target_rank) * scale
        W_init = rng.randn(p*n, target_rank) * scale
    
    params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def objective(params):
        U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        approx = np.zeros_like(T)
        for r in range(target_rank):
            approx += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
        diff = T - approx
        return np.sum(diff * diff)
    
    try:
        result = minimize(objective, params_init, method='Powell',
                         options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < 1e-8:
            U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            # Try multiple rounding strategies
            for round_fn in [_round_to_halfint, _round_to_int]:
                U_round = round_fn(U_r)
                V_round = round_fn(V_r)
                W_round = round_fn(W_r)
                
                if _verify(U_round, V_round, W_round, T):
                    return U_round, V_round, W_round
    except Exception:
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    T = _build_tensor(n, m, p)
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as guaranteed fallback
    U_sch, V_sch, W_sch = _schoolbook(n, m, p)
    best_result = (U_sch, V_sch, W_sch)
    best_rank = schoolbook_rank
    
    # Lower bounds
    lower_bound = max(n*m, m*p, p*n, n*p)
    
    # Try ranks from 44 down to lower_bound
    candidate_ranks = list(range(schoolbook_rank - 1, lower_bound - 1, -1))
    
    # Strategy: try each rank with multiple seeds and initialization strategies
    init_strategies = ['bini23', 'truncated_schoolbook', 'sparse_binary', 'random']
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(15):
            if time.time() > deadline - 5:
                break
            
            strategy = init_strategies[seed % len(init_strategies)]
            result = _try_rank_powell(n, m, p, T, target_rank, deadline, 
                                      seed=seed + target_rank, init_strategy=strategy)
            
            if result is not None:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
    
    # Final verification of best result
    U_final, V_final, W_final = best_result
    
    if not _verify(U_final, V_final, W_final, T):
        # Fallback to schoolbook if something went wrong
        U_final, V_final, W_final = _schoolbook(n, m, p)
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

# EVOLVE-BLOCK-END
