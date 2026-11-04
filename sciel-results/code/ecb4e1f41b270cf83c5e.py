"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a low-rank exact tensor decomposition for 4x4 x 4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Start with schoolbook rank 112
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Try ranks from 100 down to 70
    for target_rank in range(100, 69, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random seeds for each rank
        found = False
        for seed in range(20):
            if time.time() > deadline - 25:
                break
            result = _try_decomposition(n, m, p, target_rank, seed, deadline)
            if result is not None:
                best_rank = target_rank
                best_result = result
                found = True
                break
        
        if found:
            # Continue trying lower ranks
            continue
    
    return best_result


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_decomposition(n, m, p, target_rank, seed, deadline):
    """
    Attempt to find a rank-`target_rank` decomposition using ALS with random initialization.
    """
    np.random.seed(seed)
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 (it's the matmul tensor)
    # We need: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    
    # Random initialization
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Build the target tensor as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # ALS iterations with proper linear solves
    max_iters = 100
    prev_loss = np.inf
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Update U: for each a, solve linear system
        for a in range(n * m):
            # For fixed a, we need: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all b,c
            # This is: U[a,:] @ M_a = t_a where M_a[b_idx, r] = V[b_idx, r] * W[c_idx, r]
            # But this is a 2D constraint. Let's use the fact that T[a,b,c] is sparse.
            
            # For a = i*m+j, T[a, j*p+k, k*n+i] = 1 for k=0..p-1, else 0
            # So we have p nonzero constraints and many zero constraints
            
            # Build design matrix from nonzero positions
            i_a, j_a = divmod(a, m)
            rows = []
            targets = []
            for k in range(p):
                b_idx = j_a * p + k
                c_idx = k * n + i_a
                row = V[b_idx, :] * W[c_idx, :]
                rows.append(row)
                targets.append(1.0)
            
            if len(rows) > 0:
                M = np.array(rows)  # (p, target_rank)
                t = np.array(targets)  # (p,)
                # Solve M^T @ U[a,:] = t
                # This is: (M @ M^T) @ U[a,:] = M @ t
                # Use lstsq for robustness
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
                except np.linalg.LinAlgError:
                    pass
        
        # Update V: for each b, solve linear system
        for b in range(m * p):
            # For fixed b = j*p+k, T[i*m+j, b, k*n+i] = 1 for i=0..n-1, else 0
            j_b, k_b = divmod(b, p)
            rows = []
            targets = []
            for i in range(n):
                a_idx = i * m + j_b
                c_idx = k_b * n + i
                row = U[a_idx, :] * W[c_idx, :]
                rows.append(row)
                targets.append(1.0)
            
            if len(rows) > 0:
                M = np.array(rows)  # (n, target_rank)
                t = np.array(targets)  # (n,)
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
                except np.linalg.LinAlgError:
                    pass
        
        # Update W: for each c, solve linear system
        for c in range(p * n):
            # For fixed c = k*n+i, T[i*m+j, j*p+k, c] = 1 for j=0..m-1, else 0
            k_c, i_c = divmod(c, n)
            rows = []
            targets = []
            for j in range(m):
                a_idx = i_c * m + j
                b_idx = j * p + k_c
                row = U[a_idx, :] * V[b_idx, :]
                rows.append(row)
                targets.append(1.0)
            
            if len(rows) > 0:
                M = np.array(rows)  # (m, target_rank)
                t = np.array(targets)  # (m,)
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
                except np.linalg.LinAlgError:
                    pass
        
        # Check convergence and validity
        if it % 10 == 9:
            if _check_validity(U, V, W, n, m, p, tol=1e-4):
                # Try to snap to half-integers
                U_snap, V_snap, W_snap = _snap_to_half_integers(U, V, W)
                if _check_validity(U_snap, V_snap, W_snap, n, m, p, tol=1e-8):
                    return _to_lists(U_snap, V_snap, W_snap)
                elif _check_validity(U, V, W, n, m, p, tol=1e-8):
                    return _to_lists(U, V, W)
    
    return None


def _check_validity(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is valid."""
    # Check all valid positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check a sample of invalid positions
    np.random.seed(12345)
    for _ in range(200):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        
        # Check if this corresponds to a valid triple
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue  # This is a valid position
        
        val = np.sum(U[a] * V[b] * W[c])
        if abs(val) > tol:
            return False
    
    return True


def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer if close."""
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    return U_snap, V_snap, W_snap


def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
