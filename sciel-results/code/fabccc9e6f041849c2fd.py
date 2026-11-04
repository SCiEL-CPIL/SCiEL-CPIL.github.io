import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

def _verify_half_integer(arr, tol=1e-9):
    doubled = arr * 2
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) > tol:
        return False
    return True

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half(arr):
    return np.round(arr * 2) / 2.0

def _make_schoolbook(n, m, p):
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

def _make_strassen_3x4x6():
    """
    Construct a rank-24 decomposition for 3x4x6 matrix multiplication
    using Strassen-like block structure.
    
    We partition:
    - 3x4 into blocks: top 2 rows and bottom 1 row
    - 4x6 into blocks: left 2 cols and right 4 cols, or split 4 into 2+2
    
    Actually, let's use a cleaner approach:
    - View 3x4 as [2x4; 1x4] 
    - View 4x6 as [4x2 | 4x4]
    
    Better: use the fact that 4=2*2 and 6=2*3
    Partition A (3x4) into A1 (2x2), A2 (2x2), A3 (1x2), A4 (1x2)
    Partition B (4x6) into B1 (2x2), B2 (2x2), B3 (2x2), B4 (2x2), B5 (2x2), B6 (2x2)
    
    Actually, let me try a direct construction with rank 24 = 2*2*6 or similar.
    
    For now, let's try a structured random initialization that's more likely to work.
    """
    n, m, p = 3, 4, 6
    r = 24
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Use a Kronecker product structure
    # Split m=4 into 2+2, p=6 into 2+2+2
    # Each column c handles a specific combination
    
    for c in range(r):
        # Determine which (i,j,k) pattern this column handles
        # Use base-3 representation of c
        i_idx = c % 3
        j_idx = (c // 3) % 4
        k_idx = (c // 12) % 6
        
        # But we only have 24 columns for 3*4*6=72 terms
        # So we need to combine multiple terms per column
        
        # Instead, let's use a different approach:
        # Create 24 columns, each with a specific pattern
        
        # Pattern: for column c, assign values based on binary representation
        bits = bin(c).zfill(5)  # 5 bits for 0-23
        
        for i in range(n):
            for j in range(m):
                # Value depends on whether (i,j) matches the pattern
                u_val = 0.0
                if (i == 0 and j < 2) or (i == 1 and j >= 2) or (i == 2):
                    u_val = 1.0 if int(bits[0]) == 0 else -1.0
                U[i*m + j, c] = u_val
        
        for j in range(m):
            for k in range(p):
                v_val = 0.0
                if (j < 2 and k < 3) or (j >= 2 and k >= 3):
                    v_val = 1.0 if int(bits[1]) == 0 else -1.0
                V[j*p + k, c] = v_val
        
        for k in range(p):
            for i in range(n):
                w_val = 0.0
                if (k < 2 and i < 2) or (k >= 2 and i >= 1):
                    w_val = 1.0 if int(bits[2]) == 0 else -1.0
                W[k*n + i, c] = w_val
    
    return U, V, W

def _make_strassen_like_init(n, m, p, r, seed=42):
    """Create a structured initialization inspired by Strassen's algorithm patterns."""
    np.random.seed(seed)
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Create blocks of non-zero entries
    for c in range(r):
        # Each column has a structured pattern
        block_u = np.random.choice([0, 1, -1], size=n*m, p=[0.7, 0.15, 0.15])
        block_v = np.random.choice([0, 1, -1], size=m*p, p=[0.7, 0.15, 0.15])
        block_w = np.random.choice([0, 1, -1], size=p*n, p=[0.7, 0.15, 0.15])
        
        # Scale to encourage half-integers
        scale = np.random.choice([0.5, 1.0, 1.5, 2.0])
        U[:, c] = block_u * scale
        V[:, c] = block_v * scale
        W[:, c] = block_w * scale
    
    return U.flatten(), V.flatten(), W.flatten()

def _make_sparse_block_init(n, m, p, r, seed=42):
    """Create sparse block-structured initialization."""
    np.random.seed(seed)
    total_vars = n*m*r + m*p*r + p*n*r
    x0 = np.zeros(total_vars)
    
    # Divide into blocks and randomly activate some
    block_size = max(1, r // 4)
    for start in range(0, n*m*r, block_size):
        end = min(start + block_size, n*m*r)
        if np.random.random() < 0.5:
            x0[start:end] = np.random.choice([-1, 1], end - start) * np.random.uniform(0.5, 1.5, end - start)
    
    for start in range(n*m*r, n*m*r + m*p*r, block_size):
        end = min(start + block_size, n*m*r + m*p*r)
        if np.random.random() < 0.5:
            x0[start:end] = np.random.choice([-1, 1], end - start) * np.random.uniform(0.5, 1.5, end - start)
    
    for start in range(n*m*r + m*p*r, total_vars, block_size):
        end = min(start + block_size, total_vars)
        if np.random.random() < 0.5:
            x0[start:end] = np.random.choice([-1, 1], end - start) * np.random.uniform(0.5, 1.5, end - start)
    
    return x0

def _make_tensor_product_init(n, m, p, r, seed=42):
    """
    Build initialization using tensor product structure.
    For 3x4x6: exploit 4=2*2, 6=2*3 by building from smaller blocks.
    """
    np.random.seed(seed)
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Strategy: partition columns into groups, each group handles a subset of (i,j,k)
    # Use Kronecker-like structure for better conditioning
    num_groups = min(r, n * m * p)
    per_group = max(1, r // num_groups)
    
    for g in range(num_groups):
        col_start = g * per_group
        col_end = min(col_start + per_group, r)
        
        for c in range(col_start, col_end):
            # Assign each column to focus on certain indices
            # Use structured patterns based on binary representations
            i_pattern = (c >> 0) & 1
            j_pattern = (c >> 1) & 1
            k_pattern = (c >> 2) & 1
            
            for i in range(n):
                for j in range(m):
                    if (i & i_pattern) == i_pattern:
                        U[i*m + j, c] = np.random.choice([-1.0, 1.0]) * np.random.uniform(0.5, 1.5)
            
            for j in range(m):
                for k in range(p):
                    if (j & j_pattern) == j_pattern:
                        V[j*p + k, c] = np.random.choice([-1.0, 1.0]) * np.random.uniform(0.5, 1.5)
            
            for k in range(p):
                for i in range(n):
                    if (k & k_pattern) == k_pattern:
                        W[k*n + i, c] = np.random.choice([-1.0, 1.0]) * np.random.uniform(0.5, 1.5)
    
    return U.flatten(), V.flatten(), W.flatten()

def _optimize_rank(r, n, m, p, time_limit=10, warm_start=None):
    start_time = time.time()
    total_vars = n*m*r + m*p*r + p*n*r

    def objective(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)

        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return err

    def gradient(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)

        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    diff = 2 * (val - 1.0)

                    grad_U[u_idx, :] += diff * (V[v_idx, :] * W[w_idx, :])
                    grad_V[v_idx, :] += diff * (U[u_idx, :] * W[w_idx, :])
                    grad_W[w_idx, :] += diff * (U[u_idx, :] * V[v_idx, :])

        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def _try_polish(x, max_iter=20000, ftol=1e-20, gtol=1e-16):
        """High-precision polish with tighter tolerances."""
        try:
            res = minimize(objective, x, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                   'maxls': 100, 'maxfun': 200000})
            return res.x, res.fun
        except Exception:
            return x, objective(x)

    def _try_polish_with_snap(x, max_iter=10000):
        """Polish then snap to half-integers and verify."""
        polished_x, polished_loss = _try_polish(x, max_iter=max_iter)
        U = polished_x[:n*m*r].reshape(n*m, r)
        V = polished_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = polished_x[n*m*r + m*p*r :].reshape(p*n, r)
        
        U_snap = _snap_to_half(U)
        V_snap = _snap_to_half(V)
        W_snap = _snap_to_half(W)
        
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
            return U_snap, V_snap, W_snap, True
        return None, None, None, False

    def _generate_inits(seed_base):
        inits = []

        # Random Gaussian inits
        for scale in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
            np.random.seed(seed_base + int(scale * 10000))
            inits.append(np.random.randn(total_vars) * scale)

        # Sparse random inits
        for density in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            np.random.seed(seed_base + 1000 + int(density * 100))
            x0 = np.zeros(total_vars)
            num_nonzero = max(1, int(total_vars * density))
            indices = np.random.choice(total_vars, num_nonzero, replace=False)
            x0[indices] = np.random.choice([-1, 1], num_nonzero) * np.random.uniform(0.3, 2.0, num_nonzero)
            inits.append(x0)

        # Structured Strassen-like inits
        for seed in [seed_base + 2000, seed_base + 2100, seed_base + 2200]:
            u_flat, v_flat, w_flat = _make_strassen_like_init(n, m, p, r, seed=seed)
            inits.append(np.concatenate([u_flat, v_flat, w_flat]))

        # Sparse block inits
        for seed in [seed_base + 3000, seed_base + 3100]:
            inits.append(_make_sparse_block_init(n, m, p, r, seed=seed))
        
        # Tensor product structured inits
        for seed in [seed_base + 4000, seed_base + 4100, seed_base + 4200]:
            u_flat, v_flat, w_flat = _make_tensor_product_init(n, m, p, r, seed=seed)
            inits.append(np.concatenate([u_flat, v_flat, w_flat]))

        return inits

    best_loss = float('inf')
    best_x = None

    all_inits = []
    if warm_start is not None:
        all_inits.append(warm_start)

    all_inits.extend(_generate_inits(42))
    all_inits.extend(_generate_inits(123))
    all_inits.extend(_generate_inits(456))

    for idx, x0 in enumerate(all_inits):
        if time.time() - start_time > time_limit - 5:
            break

        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10,
                                   'maxls': 50, 'maxfun': 100000})

            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x.copy()

                # Try immediate snap verification
                U = best_x[:n*m*r].reshape(n*m, r)
                V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = best_x[n*m*r + m*p*r :].reshape(p*n, r)

                U_snap = _snap_to_half(U)
                V_snap = _snap_to_half(V)
                W_snap = _snap_to_half(W)

                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-5):
                    # Try high-precision polish
                    if time.time() - start_time < time_limit - 3:
                        U_p, V_p, W_p, success = _try_polish_with_snap(best_x)
                        if success:
                            return U_p, V_p, W_p, True, None
                    
                    # If polish didn't help, check if current snap works
                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                        return U_snap, V_snap, W_snap, True, None

        except Exception:
            pass

    # Final attempt: polish the best found solution
    if best_x is not None and best_loss < 1e-2:
        if time.time() - start_time < time_limit - 2:
            U_p, V_p, W_p, success = _try_polish_with_snap(best_x)
            if success:
                return U_p, V_p, W_p, True, None

        # Last resort: try direct snap
        U = best_x[:n*m*r].reshape(n*m, r)
        V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = best_x[n*m*r + m*p*r :].reshape(p*n, r)

        U_snap = _snap_to_half(U)
        V_snap = _snap_to_half(V)
        W_snap = _snap_to_half(W)

        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
            return U_snap, V_snap, W_snap, True, None

    return None, None, None, False, best_x

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115

    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    school_rank = n * m * p

    # Strategy 0: Try the structured Strassen-like construction first
    U_strassen, V_strassen, W_strassen = _make_strassen_3x4x6()
    if _verify_decomposition(U_strassen, V_strassen, W_strassen, n, m, p, tol=1e-6):
        U_list = [[float(x) for x in row] for row in U_strassen]
        V_list = [[float(x) for x in row] for row in V_strassen]
        W_list = [[float(x) for x in row] for row in W_strassen]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Strategy 1: Reduce from schoolbook by dropping one term at a time
    # Try rank school_rank-1, school_rank-2, ... with warm starts
    for r in range(school_rank - 1, max(school_rank - 15, 20), -1):
        if time.time() > deadline - 40:
            break

        # Create warm start: use first r columns of schoolbook
        warm = np.zeros(n*m*r + m*p*r + p*n*r)
        warm[:n*m*r] = U_school[:, :r].flatten()
        warm[n*m*r : n*m*r + m*p*r] = V_school[:, :r].flatten()
        warm[n*m*r + m*p*r :] = W_school[:, :r].flatten()

        t_use = min(10, max(4, deadline - time.time() - 35))

        result = _optimize_rank(r, n, m, p, time_limit=t_use, warm_start=warm)

        if len(result) == 5:
            U, V, W, success, warm_out = result
        else:
            U, V, W, success = result
            warm_out = None

        if success:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Strategy 2: Try mid-range ranks with random initializations
    # Focus on ranks that are more likely to be achievable
    for r in [38, 36, 34, 32, 30, 28, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]:
        if time.time() > deadline - 20:
            break

        t_use = min(12, max(6, deadline - time.time() - 15))

        result = _optimize_rank(r, n, m, p, time_limit=t_use)

        if len(result) == 5:
            U, V, W, success, warm_out = result
        else:
            U, V, W, success = result
            warm_out = None

        if success:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Strategy 3: Try very aggressive low ranks with multiple restarts
    for r in [26, 24, 22, 20, 18]:
        if time.time() > deadline - 10:
            break

        t_use = min(8, max(3, deadline - time.time() - 8))

        result = _optimize_rank(r, n, m, p, time_limit=t_use)

        if len(result) == 5:
            U, V, W, success, warm_out = result
        else:
            U, V, W, success = result
            warm_out = None

        if success:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
