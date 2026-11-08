import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]

def _build_tensor():
    """Build the matrix multiplication tensor for 3x3 * 3x4."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the tensor."""
    T = _build_tensor()
    rank = U.shape[1]
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=tol)

def _is_valid_integer_solution(U, V, W, tol=1e-9):
    """Check if U, V, W have integer or half-integer entries and verify decomposition."""
    # Check for integer or half-integer entries
    for mat in [U, V, W]:
        doubled = mat * 2
        rounded = np.round(doubled)
        if not np.allclose(doubled, rounded, atol=tol):
            return False
    return _verify_decomposition(U, V, W, tol)

def _get_rank_34_baseline():
    """Return the known rank-34 decomposition."""
    n, m, p = DIMENSIONS
    rank = 34
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Strassen Left (for columns 0,1 of B)
    U[0, col]=1; U[4, col]=1; V[0, col]=1; V[5, col]=1; W[0, col]=1; W[4, col]=1; col+=1
    U[3, col]=1; U[4, col]=1; V[0, col]=1; W[1, col]=1; W[4, col]=-1; col+=1
    U[0, col]=1; V[1, col]=1; V[5, col]=-1; W[3, col]=1; W[4, col]=1; col+=1
    U[4, col]=1; V[4, col]=1; V[0, col]=-1; W[0, col]=1; W[1, col]=1; col+=1
    U[0, col]=1; U[1, col]=1; V[5, col]=1; W[0, col]=-1; W[3, col]=1; col+=1
    U[3, col]=1; U[0, col]=-1; V[0, col]=1; V[1, col]=1; W[4, col]=1; col+=1
    U[1, col]=1; U[4, col]=-1; V[4, col]=1; V[5, col]=1; W[0, col]=1; col+=1
    # A12*B2L
    U[2, col]=1; V[8, col]=1; W[0, col]=1; col+=1
    U[2, col]=1; V[9, col]=1; W[3, col]=1; col+=1
    U[5, col]=1; V[8, col]=1; W[1, col]=1; col+=1
    U[5, col]=1; V[9, col]=1; W[4, col]=1; col+=1
    # A21*B1L
    U[6, col]=1; V[0, col]=1; W[2, col]=1; col+=1
    U[6, col]=1; V[1, col]=1; W[5, col]=1; col+=1
    U[7, col]=1; V[4, col]=1; W[2, col]=1; col+=1
    U[7, col]=1; V[5, col]=1; W[5, col]=1; col+=1
    # A22*B2L
    U[8, col]=1; V[8, col]=1; W[2, col]=1; col+=1
    U[8, col]=1; V[9, col]=1; W[5, col]=1; col+=1
    # Strassen Right (for columns 2,3 of B)
    U[0, col]=1; U[4, col]=1; V[2, col]=1; V[7, col]=1; W[6, col]=1; W[10, col]=1; col+=1
    U[3, col]=1; U[4, col]=1; V[2, col]=1; W[7, col]=1; W[10, col]=-1; col+=1
    U[0, col]=1; V[3, col]=1; V[7, col]=-1; W[9, col]=1; W[10, col]=1; col+=1
    U[4, col]=1; V[6, col]=1; V[2, col]=-1; W[6, col]=1; W[7, col]=1; col+=1
    U[0, col]=1; U[1, col]=1; V[7, col]=1; W[6, col]=-1; W[9, col]=1; col+=1
    U[3, col]=1; U[0, col]=-1; V[2, col]=1; V[3, col]=1; W[10, col]=1; col+=1
    U[1, col]=1; U[4, col]=-1; V[6, col]=1; V[7, col]=1; W[6, col]=1; col+=1
    # A12*B2R
    U[2, col]=1; V[10, col]=1; W[6, col]=1; col+=1
    U[2, col]=1; V[11, col]=1; W[9, col]=1; col+=1
    U[5, col]=1; V[10, col]=1; W[7, col]=1; col+=1
    U[5, col]=1; V[11, col]=1; W[10, col]=1; col+=1
    # A21*B1R
    U[6, col]=1; V[2, col]=1; W[8, col]=1; col+=1
    U[6, col]=1; V[3, col]=1; W[11, col]=1; col+=1
    U[7, col]=1; V[6, col]=1; W[8, col]=1; col+=1
    U[7, col]=1; V[7, col]=1; W[11, col]=1; col+=1
    # A22*B2R
    U[8, col]=1; V[10, col]=1; W[8, col]=1; col+=1
    U[8, col]=1; V[11, col]=1; W[11, col]=1; col+=1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_reduce_rank(U, V, W, target_rank, max_time=5.0):
    """Try to reduce rank from current to target by finding linear dependencies."""
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return None
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = _build_tensor()
    
    start_time = time.time()
    
    # Try to find a subset of columns that spans the same tensor
    # Use a greedy approach: try removing one column at a time
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        
        # Randomly select columns to keep
        cols_to_keep = list(range(current_rank))
        np.random.shuffle(cols_to_keep)
        cols_to_keep = cols_to_keep[:target_rank]
        
        U_sub = U[:, cols_to_keep]
        V_sub = V[:, cols_to_keep]
        W_sub = W[:, cols_to_keep]
        
        # Check if this subset still decomposes the tensor
        if _verify_decomposition(U_sub, V_sub, W_sub):
            # Try to round to integers/half-integers
            U_round = np.round(U_sub * 2) / 2
            V_round = np.round(V_sub * 2) / 2
            W_round = np.round(W_sub * 2) / 2
            
            if _verify_decomposition(U_round, V_round, W_round) and _is_valid_integer_solution(U_round, V_round, W_round):
                return (U_round.tolist(), V_round.tolist(), W_round.tolist())
    
    return None

def _structured_search_for_rank(target_rank, max_time=10.0, seed=42):
    """Search for a decomposition of given rank using structured initialization."""
    if max_time <= 0:
        return None
    
    T = _build_tensor()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    np.random.seed(seed)
    start_time = time.time()
    
    # Start from the rank-34 baseline and try to modify it
    baseline = _get_rank_34_baseline()
    U_base = np.array(baseline['U'])
    V_base = np.array(baseline['V'])
    W_base = np.array(baseline['W'])
    
    for attempt in range(200):
        if time.time() - start_time > max_time:
            break
        
        # Create a modified version by adding small random perturbations
        U = U_base.copy()
        V = V_base.copy()
        W = W_base.copy()
        
        # Add small random perturbations to some entries
        for _ in range(5):
            u_idx = np.random.randint(0, nm)
            v_idx = np.random.randint(0, mp)
            w_idx = np.random.randint(0, pn)
            u_col = np.random.randint(0, U.shape[1])
            v_col = np.random.randint(0, V.shape[1])
            w_col = np.random.randint(0, W.shape[1])
            
            U[u_idx, u_col] += np.random.choice([-0.5, 0.5])
            V[v_idx, v_col] += np.random.choice([-0.5, 0.5])
            W[w_idx, w_col] += np.random.choice([-0.5, 0.5])
        
        # Try to reduce rank
        result = _try_reduce_rank(U, V, W, target_rank, max_time=min(2.0, max_time - (time.time() - start_time)))
        if result is not None:
            return result
    
    return None

def construct():
    deadline = time.time() + 130  # Leave 20s safety margin
    
    # Start with the known rank-34 solution
    baseline = _get_rank_34_baseline()
    best_result = baseline
    best_rank = 34
    
    # Try to find better decompositions using structured search
    for rank in range(30, best_rank):
        if time.time() > deadline - 20:
            break
        
        max_time = min(5.0, max(1.0, deadline - time.time() - 15))
        result = _structured_search_for_rank(rank, max_time=max_time, seed=42+rank)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            # Don't break; keep searching for even better
            continue
    
    return best_result
