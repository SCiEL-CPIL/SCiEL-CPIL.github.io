import numpy as np
from scipy.optimize import least_squares
import time
import random

DIMENSIONS = [2, 3, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized exact verification. U:(n*m,r) V:(m*p,r) W:(p*n,r)."""
    rank = U.shape[1]
    T = np.einsum('uc,vc,wc->uvw', U, V, W)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T[u, v, w] - 1.0) > tol:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok

def _prefilter_counts(U, V, W, n, m, p):
    """Cheap necessary condition for ±1 matrices."""
    rank = U.shape[1]
    if rank % 2 == 0:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                prod = U[u, :] * V[v, :] * W[w, :]
                agree = np.count_nonzero(prod == 1)
                if agree != (rank + 1) // 2:
                    return False
    return True

def _try_random_int(rank, n, m, p, n_attempts, deadline):
    """Random ±1 search with pre-filter."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    for _ in range(n_attempts):
        if time.time() > deadline:
            break
        U = np.random.choice([-1, 1], size=(size_u, rank))
        V = np.random.choice([-1, 1], size=(size_v, rank))
        W = np.random.choice([-1, 1], size=(size_w, rank))
        if not _prefilter_counts(U, V, W, n, m, p):
            continue
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
    return None

def _try_random_mixed(rank, n, m, p, n_attempts, deadline):
    """Random search with entries from {-1, -0.5, 0, 0.5, 1}."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    for _ in range(n_attempts):
        if time.time() > deadline:
            break
        U = np.random.choice(vals, size=(size_u, rank))
        V = np.random.choice(vals, size=(size_v, rank))
        W = np.random.choice(vals, size=(size_w, rank))
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
    return None

def _try_least_squares(rank, n, m, p, n_attempts, deadline):
    """Structured optimization search for half-integer solutions."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    total_size = (size_u + size_v + size_w) * rank

    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    num_triples = len(triples)

    def unpack(x):
        U = x[:size_u * rank].reshape(size_u, rank)
        V = x[size_u * rank:(size_u + size_v) * rank].reshape(size_v, rank)
        W = x[(size_u + size_v) * rank:].reshape(size_w, rank)
        return U, V, W

    def residuals(x):
        U, V, W = unpack(x)
        res = np.empty(num_triples)
        for idx, (u, v, w) in enumerate(triples):
            res[idx] = np.dot(U[u, :], V[v, :] * W[w, :]) - 1.0
        return res

    def jacobian(x):
        U, V, W = unpack(x)
        J = np.zeros((num_triples, total_size))
        for idx, (u, v, w) in enumerate(triples):
            VW = V[v, :] * W[w, :]
            UW = U[u, :] * W[w, :]
            UV = U[u, :] * V[v, :]
            J[idx, u * rank:(u + 1) * rank] = VW
            J[idx, size_u * rank + v * rank:size_u * rank + (v + 1) * rank] = UW
            J[idx, (size_u + size_v) * rank + w * rank:(size_u + size_v) * rank + (w + 1) * rank] = UV
        return J

    best = None
    best_norm = np.inf
    for _ in range(n_attempts):
        if time.time() > deadline:
            break
        # Mix initialization strategies
        r = random.random()
        if r < 0.3:
            x0 = np.random.choice([-1.0, 1.0], size=total_size)
        elif r < 0.6:
            x0 = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=total_size)
        else:
            x0 = np.random.randn(total_size) * 0.5
        try:
            sol = least_squares(residuals, x0, jac=jacobian, method='lm',
                                max_nfev=3000)
            norm = np.linalg.norm(sol.fun)
            if norm < best_norm:
                best_norm = norm
                best = sol.x
            if norm < 1e-7:
                U, V, W = unpack(sol.x)
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_fast(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
        except Exception:
            continue
    
    # Polish the best near-solution if it exists
    if best is not None and best_norm < 1e-3:
        try:
            sol = least_squares(residuals, best, jac=jacobian, method='lm',
                                max_nfev=20000)
            U, V, W = unpack(sol.x)
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
        except Exception:
            pass
    return None

def _try_greedy_column(rank, n, m, p, deadline):
    """
    Greedy column-by-column construction.
    At each step, pick a column (u_col, v_col, w_col) that satisfies the most
    remaining unsatisfied triples exactly, given the current partial solution.
    This is a combinatorial heuristic that exploits the structure.
    """
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Precompute all triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    num_triples = len(triples)
    
    # For each triple, track how many columns have satisfied it
    # and the partial sum
    partial_sum = np.zeros(num_triples)
    satisfied = np.zeros(num_triples, dtype=bool)
    
    U = np.zeros((size_u, rank))
    V = np.zeros((size_v, rank))
    W = np.zeros((size_w, rank))
    
    for c in range(rank):
        if time.time() > deadline:
            break
        
        # Find best column assignment
        # We want to maximize the number of newly satisfied triples
        # A column (u_val, v_val, w_val) contributes u_val[v_val] * w_val to each triple
        
        # For efficiency, try a limited set of candidate columns
        # Candidates: for each row of U, V, W, try values in {-1, -0.5, 0, 0.5, 1}
        # But that's too many. Instead, focus on triples that are not yet satisfied.
        
        # For each unsatisfied triple, we need the sum to reach 1.
        # Let's try to pick columns that help the most unsatisfied triples.
        
        best_score = -1
        best_u_col = None
        best_v_col = None
        best_w_col = None
        
        # Sample a subset of candidate columns
        num_candidates = 2000
        for _ in range(num_candidates):
            if time.time() > deadline:
                break
            
            # Random candidate
            u_idx = random.randint(0, size_u - 1)
            v_idx = random.randint(0, size_v - 1)
            w_idx = random.randint(0, size_w - 1)
            
            u_val = random.choice([-1.0, -0.5, 0.5, 1.0])
            v_val = random.choice([-1.0, -0.5, 0.5, 1.0])
            w_val = random.choice([-1.0, -0.5, 0.5, 1.0])
            
            # Compute how many unsatisfied triples this would help
            score = 0
            for t_idx, (u, v, w) in enumerate(triples):
                if satisfied[t_idx]:
                    continue
                contrib = u_val * v_val * w_val
                new_sum = partial_sum[t_idx] + contrib
                if abs(new_sum - 1.0) < 1e-9:
                    score += 10  # Strong bonus for satisfying
                elif abs(new_sum - 1.0) < abs(partial_sum[t_idx] - 1.0):
                    score += 1  # Small bonus for getting closer
            
            if score > best_score:
                best_score = score
                best_u_col = np.zeros(size_u); best_u_col[u_idx] = u_val
                best_v_col = np.zeros(size_v); best_v_col[v_idx] = v_val
                best_w_col = np.zeros(size_w); best_w_col[w_idx] = w_val
        
        if best_u_col is None:
            break
        
        U[:, c] = best_u_col
        V[:, c] = best_v_col
        W[:, c] = best_w_col
        
        # Update partial sums
        for t_idx, (u, v, w) in enumerate(triples):
            if not satisfied[t_idx]:
                partial_sum[t_idx] += best_u_col[u] * best_v_col[v] * best_w_col[w]
                if abs(partial_sum[t_idx] - 1.0) < 1e-9:
                    satisfied[t_idx] = True
        
        # Check if all satisfied
        if np.all(satisfied):
            return U[:, :c+1], V[:, :c+1], W[:, :c+1]
    
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)

    # Search ranks from low to high, stop at first valid
    for rank in range(15, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Phase 1: random ±1 with pre-filter (fast)
        result = _try_random_int(rank, n, m, p, n_attempts=3000,
                                 deadline=deadline - 10)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break
        
        # Phase 2: random mixed-valued
        if time.time() > deadline - 10:
            break
        result = _try_random_mixed(rank, n, m, p, n_attempts=1000,
                                   deadline=deadline - 10)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break
        
        # Phase 3: structured least-squares
        if time.time() > deadline - 10:
            break
        result = _try_least_squares(rank, n, m, p, n_attempts=20,
                                    deadline=deadline - 10)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break
        
        # Phase 4: greedy column construction
        if time.time() > deadline - 10:
            break
        result = _try_greedy_column(rank, n, m, p, deadline=deadline - 10)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break

    # Convert to lists of integers/half-integers
    U_out = [[(int(x * 2) / 2) for x in row] for row in best_U]
    V_out = [[(int(x * 2) / 2) for x in row] for row in best_V]
    W_out = [[(int(x * 2) / 2) for x in row] for row in best_W]

    return {"U": U_out, "V": V_out, "W": W_out}
