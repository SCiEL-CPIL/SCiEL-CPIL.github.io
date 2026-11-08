"""Matrix multiplication 2x4 by 4x7 rank-minimal decomposition search."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reconstruct tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    T_true = _build_tensor(n, m, p)
    return np.allclose(T, T_true, atol=tol)

def _check_half_integer(arr, tol=1e-9):
    """Check if all entries are integer or half-integer."""
    for val in np.array(arr).flat:
        v = float(val)
        if abs(2 * v - round(2 * v)) > tol:
            return False
    return True

def _to_list(arr):
    """Convert numpy array to list of lists with rounded half-integer values."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = float(val)
            rv = round(2 * v) / 2.0
            r.append(rv)
        result.append(r)
    return result

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _try_polynomial_system_solve(n, m, p, time_limit=120.0):
    """
    Try to find a low-rank decomposition by solving polynomial systems.
    
    For a rank-r decomposition, we have variables U (n*m*r), V (m*p*r), W (p*n*r).
    The constraints are:
      For each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    This is a system of polynomial equations. We use a structured approach:
    1. Fix W (or use a parameterization)
    2. Solve the resulting linear/quadratic system for U and V
    3. Iterate with different W choices
    
    Key insight: If we fix W, the problem becomes finding rank-1 matrices
    X_r = U[:,r] V[:,r]^T such that sum_r X_r[a,b] * W[c,r] = T[a,b,c].
    
    For fixed W with full column rank, X_vec[a,b,:] is uniquely determined:
      X_vec[a,b,:] = T[a,b,:] @ W @ (W^T W)^{-1}
    Then we need each X_vec[:,:,r] to be rank 1.
    """
    start_time = time.time()
    T = _build_tensor(n, m, p)
    support = np.argwhere(T != 0)
    support_set = set(map(tuple, support))
    
    # Compute matricization ranks for lower bounds
    T_mat1 = T.reshape(n * m, m * p * p * n)
    T_mat2 = T.reshape(n * m * m * p, p * n)
    T_mat3 = T.reshape(n * m * p * n, m * p)
    
    rank_lb1 = np.linalg.matrix_rank(T_mat1)
    rank_lb2 = np.linalg.matrix_rank(T_mat2)
    rank_lb3 = np.linalg.matrix_rank(T_mat3)
    rank_lower = max(rank_lb1, rank_lb2, rank_lb3)
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Strategy: Try different W matrices and solve for U, V
    # Use random W with structure to encourage rank-1 solutions
    
    np.random.seed(42)
    
    # Try ranks from lower bound + 1 up to schoolbook - 1
    # (schoolbook is always available as fallback)
    for r in range(max(rank_lower, 20), n * m * p):
        if time.time() - start_time > time_limit:
            break
        
        # Multiple random attempts for this rank
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random W with values in {-1, 0, 1} for half-integer compatibility
            W_rand = np.random.choice([-1, 0, 1], size=(p * n, r))
            
            # Check if W has full column rank
            if np.linalg.matrix_rank(W_rand) < r:
                continue
            
            try:
                WtW = W_rand.T @ W_rand
                WtW_inv = np.linalg.inv(WtW)
            except np.linalg.LinAlgError:
                continue
            
            # Compute X_vec: X_vec[a,b,r_idx] = sum_c T[a,b,c] * W[c,r_idx] * (WtW_inv)[r_idx,s]
            # Actually: X_vec[a,b,:] @ W.T = T[a,b,:]
            # So: X_vec[a,b,:] = T[a,b,:] @ W @ (W.T @ W)^{-1}
            
            # Efficient computation using einsum
            TW = np.einsum('abc,cr->abr', T, W_rand)
            X_vec = np.einsum('abr,rs->abs', TW, WtW_inv)
            
            # Check if each slice X_vec[:,:,r_idx] is rank 1
            U_sol = np.zeros((n * m, r))
            V_sol = np.zeros((m * p, r))
            all_rank1 = True
            
            for r_idx in range(r):
                C_r = X_vec[:, :, r_idx]
                # SVD to check rank
                svals = np.linalg.svd(C_r, compute_uv=False)
                if len(svals) > 1:
                    if svals[1] > 1e-10 * max(svals[0], 1e-12):
                        all_rank1 = False
                        break
                if svals[0] < 1e-12:
                    U_sol[:, r_idx] = 0
                    V_sol[:, r_idx] = 0
                else:
                    U_sv, S_sv, Vt_sv = np.linalg.svd(C_r)
                    U_sol[:, r_idx] = U_sv[:, 0] * np.sqrt(S_sv[0])
                    V_sol[:, r_idx] = Vt_sv[0, :] * np.sqrt(S_sv[0])
            
            if not all_rank1:
                continue
            
            # Verify the decomposition on support
            T_recon_support = np.zeros(len(support))
            for idx, (a, b, c) in enumerate(support):
                val = 0.0
                for r_idx in range(r):
                    val += U_sol[a, r_idx] * V_sol[b, r_idx] * W_rand[c, r_idx]
                T_recon_support[idx] = val
            
            if not np.allclose(T_recon_support, 1.0, atol=1e-8):
                continue
            
            # Verify on zero entries (sample)
            all_abcs = np.array([(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)])
            zero_mask = np.array([tuple(abc) not in support_set for abc in all_abcs])
            zero_entries = all_abcs[zero_mask]
            
            if len(zero_entries) > 0:
                sample_size = min(500, len(zero_entries))
                sample_idx = np.random.choice(len(zero_entries), size=sample_size, replace=False)
                ok_zero = True
                for s in sample_idx:
                    a, b, c = zero_entries[s]
                    val = 0.0
                    for r_idx in range(r):
                        val += U_sol[a, r_idx] * V_sol[b, r_idx] * W_rand[c, r_idx]
                    if abs(val) > 1e-8:
                        ok_zero = False
                        break
                if not ok_zero:
                    continue
            
            # Check half-integer constraint
            if _check_half_integer(U_sol) and _check_half_integer(V_sol) and _check_half_integer(W_rand):
                best_rank = r
                best_U = U_sol.copy()
                best_V = V_sol.copy()
                best_W = W_rand.copy()
                break
            else:
                # Try rounding to half-integers
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_rand * 2) / 2
                
                # Verify rounded version
                T_recon = np.zeros((n * m, m * p, p * n), dtype=np.float64)
                for r_idx in range(r):
                    T_recon += np.einsum('i,j,k->ijk', U_round[:, r_idx], V_round[:, r_idx], W_round[:, r_idx])
                
                if np.allclose(T_recon, T, atol=1e-8):
                    best_rank = r
                    best_U = U_round
                    best_V = V_round
                    best_W = W_round
                    break
        
        if best_rank == r:
            # Found a valid decomposition at this rank, continue searching for better
            continue
    
    if best_U is not None:
        return _to_list(best_U), _to_list(best_V), _to_list(best_W)
    
    return None

def _try_alternative_parameterization(n, m, p, time_limit=120.0):
    """
    Alternative approach: parameterize U and V with sparse structure.
    
    For the 2x4x4x7 case, exploit the structure:
    - n=2, m=4, p=7
    - The tensor has a specific pattern based on the indices
    
    Try to find a decomposition where each rank-1 term covers multiple
    non-zero entries by having overlapping supports.
    """
    start_time = time.time()
    T = _build_tensor(n, m, p)
    
    # The non-zero entries are at (i*m+j, j*p+k, k*n+i) for i in [0,1], j in [0,3], k in [0,6]
    # That's 2*4*7 = 56 entries
    
    # For a rank-r decomposition, we need r rank-1 tensors that sum to T.
    # Each rank-1 tensor U_r ⊗ V_r ⊗ W_r has support on a "grid" pattern.
    
    # Key: For the decomposition to work, the W vectors must be such that
    # the resulting X_r matrices are rank 1.
    
    # Try a structured W based on the problem dimensions
    # W has shape (p*n, r) = (14, r)
    
    np.random.seed(123)
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try different structures for W
    for structure in range(5):
        if time.time() - start_time > time_limit:
            break
        
        for r in range(40, n * m * p):
            if time.time() - start_time > time_limit:
                break
            
            for attempt in range(20):
                if time.time() - start_time > time_limit:
                    break
                
                if structure == 0:
                    # Random {-1, 0, 1}
                    W_rand = np.random.choice([-1, 0, 1], size=(p * n, r))
                elif structure == 1:
                    # Sparse: only 2 non-zeros per row
                    W_rand = np.zeros((p * n, r))
                    for i in range(p * n):
                        cols = np.random.choice(r, size=2, replace=False)
                        W_rand[i, cols] = np.random.choice([-1, 1], size=2)
                elif structure == 2:
                    # Block structure based on k index
                    W_rand = np.zeros((p * n, r))
                    for k in range(p):
                        for i in range(n):
                            row = k * n + i
                            col = (k * n + i) % r
                            W_rand[row, col] = 1
                elif structure == 3:
                    # Random with more zeros
                    W_rand = np.random.choice([-1, 0, 0, 0, 1], size=(p * n, r))
                else:
                    # Gaussian, then threshold
                    W_rand = np.random.randn(p * n, r)
                    W_rand[W_rand < -0.5] = -1
                    W_rand[W_rand > 0.5] = 1
                    W_rand[(W_rand >= -0.5) & (W_rand <= 0.5)] = 0
                
                if np.linalg.matrix_rank(W_rand) < r:
                    continue
                
                try:
                    WtW = W_rand.T @ W_rand
                    WtW_inv = np.linalg.inv(WtW)
                except np.linalg.LinAlgError:
                    continue
                
                TW = np.einsum('abc,cr->abr', T, W_rand)
                X_vec = np.einsum('abr,rs->abs', TW, WtW_inv)
                
                U_sol = np.zeros((n * m, r))
                V_sol = np.zeros((m * p, r))
                all_rank1 = True
                
                for r_idx in range(r):
                    C_r = X_vec[:, :, r_idx]
                    svals = np.linalg.svd(C_r, compute_uv=False)
                    if len(svals) > 1 and svals[1] > 1e-10 * max(svals[0], 1e-12):
                        all_rank1 = False
                        break
                    if svals[0] < 1e-12:
                        U_sol[:, r_idx] = 0
                        V_sol[:, r_idx] = 0
                    else:
                        U_sv, S_sv, Vt_sv = np.linalg.svd(C_r)
                        U_sol[:, r_idx] = U_sv[:, 0] * np.sqrt(S_sv[0])
                        V_sol[:, r_idx] = Vt_sv[0, :] * np.sqrt(S_sv[0])
                
                if not all_rank1:
                    continue
                
                # Quick verification on support
                support = np.argwhere(T != 0)
                ok = True
                for a, b, c in support:
                    val = sum(U_sol[a, ri] * V_sol[b, ri] * W_rand[c, ri] for ri in range(r))
                    if abs(val - 1.0) > 1e-8:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Check half-integer
                if _check_half_integer(U_sol) and _check_half_integer(V_sol) and _check_half_integer(W_rand):
                    best_rank = r
                    best_U = U_sol.copy()
                    best_V = V_sol.copy()
                    best_W = W_rand.copy()
                    break
                else:
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_rand * 2) / 2
                    
                    T_recon = np.zeros((n * m, m * p, p * n), dtype=np.float64)
                    for ri in range(r):
                        T_recon += np.einsum('i,j,k->ijk', U_round[:, ri], V_round[:, ri], W_round[:, ri])
                    
                    if np.allclose(T_recon, T, atol=1e-8):
                        best_rank = r
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        break
            
            if best_rank == r:
                break
    
    if best_U is not None:
        return _to_list(best_U), _to_list(best_V), _to_list(best_W)
    
    return None

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Try polynomial system approach
    try:
        remaining = deadline - time.time()
        if remaining > 5.0:
            result = _try_polynomial_system_solve(n, m, p, time_limit=min(remaining, 60.0))
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Try alternative parameterization
    try:
        remaining = deadline - time.time()
        if remaining > 5.0:
            result = _try_alternative_parameterization(n, m, p, time_limit=min(remaining, 60.0))
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
