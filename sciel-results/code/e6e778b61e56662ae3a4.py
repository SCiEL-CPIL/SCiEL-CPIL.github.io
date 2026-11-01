"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_from_terms(all_terms, n, m, p):
    """Build U, V, W matrices from a list of (u_vec, v_vec, w_vec) terms."""
    rank = len(all_terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    return U, V, W

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]))
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]))
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    terms.append(([1, 0, -1, 0], [1, 1, 0, 0], [0, 1, 0, 0]))
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]))
    return terms

def _make_schoolbook_terms(n, m, p, k_start=0, k_end=None):
    """Make schoolbook terms for columns k_start to k_end-1."""
    if k_end is None:
        k_end = p
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(k_start, k_end):
                u_vec = [0.0] * (n * m)
                u_vec[i * m + j] = 1.0
                v_vec = [0.0] * (m * p)
                v_vec[j * p + k] = 1.0
                w_vec = [0.0] * (p * n)
                w_vec[k * n + i] = 1.0
                terms.append((u_vec, v_vec, w_vec))
    return terms

def _extend_2x2_term_to_2x5(u_vec, v_vec_2x2, w_vec_2x2, n=2, m=2, p=5):
    """Extend a 2x2x2 term to a 2x2x5 term (only uses k=0,1)."""
    u_full = list(u_vec)
    v_full = [0.0] * (m * p)
    for j in range(m):
        for k in range(2):
            v_full[j * p + k] = v_vec_2x2[j * 2 + k]
    w_full = [0.0] * (p * n)
    for k in range(2):
        for i in range(n):
            w_full[k * n + i] = w_vec_2x2[k * 2 + i]
    return u_full, v_full, w_full

def _tensor_error(U, V, W, n, m, p):
    """Compute the squared error of the tensor decomposition."""
    error = 0.0
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                error += (val - 1.0) ** 2
    return error

def _flatten_params(U, V, W):
    """Flatten U, V, W into a single vector."""
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _unflatten_params(params, n, m, p, rank):
    """Unflatten a parameter vector into U, V, W."""
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    
    U = params[:U_size].reshape(n * m, rank)
    V = params[U_size:U_size+V_size].reshape(m * p, rank)
    W = params[U_size+V_size:U_size+V_size+W_size].reshape(p * n, rank)
    return U, V, W

def _polish_solution(U, V, W, n, m, p, max_iter=500):
    """Final polish pass with tighter tolerances and more iterations."""
    def objective(params):
        U_, V_, W_ = _unflatten_params(params, n, m, p, U.shape[1])
        return _tensor_error(U_, V_, W_, n, m, p)
    
    x0 = _flatten_params(U, V, W)
    result = minimize(objective, x0, method='L-BFGS-B',
                     options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14})
    
    if result.fun < 1e-12:
        U_opt, V_opt, W_opt = _unflatten_params(result.x, n, m, p, U.shape[1])
        if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p, tol=1e-10):
            return U_opt, V_opt, W_opt
    return None

def _optimize_rank_from_init(rank, n, m, p, deadline, init_U, init_V, init_W, seed=0, max_iter=300):
    """Try to find a valid decomposition of the given rank starting from an initialization."""
    def objective(params):
        U, V, W = _unflatten_params(params, n, m, p, rank)
        return _tensor_error(U, V, W, n, m, p)
    
    x0 = _flatten_params(init_U, init_V, init_W)
    
    result = minimize(objective, x0, method='L-BFGS-B', 
                     options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
    
    if result.fun < 1e-10:
        U_opt, V_opt, W_opt = _unflatten_params(result.x, n, m, p, rank)
        if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p, tol=1e-8):
            # Try to polish
            polished = _polish_solution(U_opt, V_opt, W_opt, n, m, p)
            if polished is not None:
                return polished
            return U_opt, V_opt, W_opt
    
    return None

def _random_init(rank, n, m, p, seed):
    """Generate a random initialization for rank-r decomposition."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    return U, V, W

def _structured_init_from_higher(U_high, V_high, W_high, n, m, p, target_rank, seed):
    """Create a structured initialization from a higher-rank solution."""
    rng = np.random.RandomState(seed)
    current_rank = U_high.shape[1]
    
    # Select target_rank columns from the higher-rank solution
    # Weight selection by column norms to prefer important terms
    U_norms = np.linalg.norm(U_high, axis=0)
    V_norms = np.linalg.norm(V_high, axis=0)
    W_norms = np.linalg.norm(W_high, axis=0)
    
    weights = U_norms * V_norms * W_norms
    weights = weights / (weights.sum() + 1e-12)
    
    cols = rng.choice(current_rank, size=target_rank, replace=False, p=weights)
    
    U_init = U_high[:, cols].copy()
    V_init = V_high[:, cols].copy()
    W_init = W_high[:, cols].copy()
    
    return U_init, V_init, W_init

def _try_reduce_rank(U, V, W, n, m, p, deadline, target_rank, seed=0):
    """Try to reduce the rank by finding a valid decomposition at target_rank."""
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return None
    
    rng = np.random.RandomState(seed)
    
    # Method 1: Structured initialization from current solution
    for attempt in range(5):
        if time.time() > deadline - 15:
            break
        
        U_init, V_init, W_init = _structured_init_from_higher(U, V, W, n, m, p, target_rank, seed + attempt)
        # Add small perturbation
        U_init = U_init + rng.randn(n * m, target_rank) * 0.01
        V_init = V_init + rng.randn(m * p, target_rank) * 0.01
        W_init = W_init + rng.randn(p * n, target_rank) * 0.01
        
        result = _optimize_rank_from_init(target_rank, n, m, p, deadline, 
                                           U_init, V_init, W_init, seed=seed + attempt)
        if result is not None:
            return result
    
    # Method 2: Random initialization
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
        
        U_init, V_init, W_init = _random_init(target_rank, n, m, p, seed + 100 + attempt)
        result = _optimize_rank_from_init(target_rank, n, m, p, deadline,
                                           U_init, V_init, W_init, seed=seed + 100 + attempt)
        if result is not None:
            return result
    
    # Method 3: Try with direct column selection (no perturbation)
    for attempt in range(2):
        if time.time() > deadline - 5:
            break
        
        cols = rng.choice(current_rank, size=target_rank, replace=False)
        U_init = U[:, cols].copy()
        V_init = V[:, cols].copy()
        W_init = W[:, cols].copy()
        
        result = _optimize_rank_from_init(target_rank, n, m, p, deadline,
                                           U_init, V_init, W_init, seed=seed + 200 + attempt)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Build initial rank-13 decomposition
    strassen_terms = _strassen_2x2_terms()
    all_terms = []
    
    for u_vec, v_vec, w_vec in strassen_terms:
        u_full, v_full, w_full = _extend_2x2_term_to_2x5(u_vec, v_vec, w_vec, n, m, p)
        all_terms.append((u_full, v_full, w_full))
    
    schoolbook_terms = _make_schoolbook_terms(n, m, p, k_start=2, k_end=5)
    all_terms.extend(schoolbook_terms)
    
    U, V, W = _build_from_terms(all_terms, n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try to find lower-rank decompositions
    # Start from lower ranks and work up, then try to reduce from there
    # Known lower bound: at least ceil(n*m*p / something), but let's try from 10 up
    
    # First, try to find the best rank we can achieve
    # Try ranks from 10 to best_rank-1
    for target_rank in range(10, best_rank):
        if time.time() > deadline - 30:
            break
        
        result = _try_reduce_rank(best_U, best_V, best_W, n, m, p, deadline, 
                                   target_rank, seed=target_rank * 13)
        
        if result is not None:
            U_opt, V_opt, W_opt = result
            best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            best_rank = target_rank
            # If we found a solution, try to reduce further
            break
    
    # If we improved, try to reduce further from the new best
    if best_rank < 13:
        for target_rank in range(best_rank - 1, 9, -1):
            if time.time() > deadline - 15:
                break
            
            result = _try_reduce_rank(best_U, best_V, best_W, n, m, p, deadline, 
                                       target_rank, seed=target_rank * 7 + 1)
            
            if result is not None:
                U_opt, V_opt, W_opt = result
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                best_rank = target_rank
    
    # Final polish
    if time.time() < deadline - 5:
        polished = _polish_solution(best_U, best_V, best_W, n, m, p, max_iter=1000)
        if polished is not None:
            best_U, best_V, best_W = polished
    
    # Clean up: convert near-integers to exact integers
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx, val in enumerate(row):
                if abs(val - round(val)) < 1e-10:
                    row[idx] = round(val)
                elif abs(val * 2 - round(val * 2)) < 1e-10:
                    row[idx] = round(val * 2) / 2.0
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
