import time
import numpy as np
import scipy.optimize as opt
from numpy.linalg import matrix_rank, null_space, solve

DIMENSIONS = [2, 4, 4]

def generate_strassen_28(n, m, p):
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    def apply_strassen_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, base_rank):
        a_idx = [
            a_r0 * m + a_c0, a_r0 * m + a_c0 + 1,
            (a_r0 + 1) * m + a_c0, (a_r0 + 1) * m + a_c0 + 1
        ]
        b_idx = [
            b_r0 * p + b_c0, b_r0 * p + b_c0 + 1,
            (b_r0 + 1) * p + b_c0, (b_r0 + 1) * p + b_c0 + 1
        ]
        c_idx = [
            c_c0 * n + c_r0, (c_c0 + 1) * n + c_r0,
            c_c0 * n + (c_r0 + 1), (c_c0 + 1) * n + (c_r0 + 1)
        ]
        
        r = base_rank
        # M1
        U[a_idx[0]][r] += 1; U[a_idx[3]][r] += 1
        V[b_idx[0]][r] += 1; V[b_idx[3]][r] += 1
        W[c_idx[0]][r] += 1; W[c_idx[3]][r] += 1
        r += 1
        # M2
        U[a_idx[2]][r] += 1; U[a_idx[3]][r] += 1
        V[b_idx[0]][r] += 1
        W[c_idx[2]][r] += 1; W[c_idx[3]][r] += -1
        r += 1
        # M3
        U[a_idx[0]][r] += 1
        V[b_idx[1]][r] += 1; V[b_idx[3]][r] += -1
        W[c_idx[1]][r] += 1; W[c_idx[3]][r] += 1
        r += 1
        # M4
        U[a_idx[3]][r] += 1
        V[b_idx[2]][r] += 1; V[b_idx[0]][r] += -1
        W[c_idx[0]][r] += 1; W[c_idx[2]][r] += 1
        r += 1
        # M5
        U[a_idx[0]][r] += 1; U[a_idx[1]][r] += 1
        V[b_idx[3]][r] += 1
        W[c_idx[0]][r] += -1; W[c_idx[1]][r] += 1
        r += 1
        # M6
        U[a_idx[2]][r] += 1; U[a_idx[0]][r] += -1
        V[b_idx[0]][r] += 1; V[b_idx[1]][r] += 1
        W[c_idx[3]][r] += 1
        r += 1
        # M7
        U[a_idx[1]][r] += 1; U[a_idx[3]][r] -= 1
        V[b_idx[2]][r] += 1; V[b_idx[3]][r] += 1
        W[c_idx[0]][r] += 1
        r += 1
        return r

    r = apply_strassen_block(0, 0, 0, 0, 0, 0, 0)
    r = apply_strassen_block(0, 2, 2, 0, 0, 0, r)
    r = apply_strassen_block(0, 0, 0, 2, 0, 2, r)
    r = apply_strassen_block(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28
    return U.tolist(), V.tolist(), W.tolist()

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W (lists of lists of floats/ints) form a valid decomposition.
    Assumes entries are close to half-integers.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check if entries are close to half-integers
    if np.max(np.abs(U_arr * 2 - np.round(U_arr * 2))) > 1e-6: return False
    if np.max(np.abs(V_arr * 2 - np.round(V_arr * 2))) > 1e-6: return False
    if np.max(np.abs(W_arr * 2 - np.round(W_arr * 2))) > 1e-6: return False
    
    # Convert to integers by multiplying by 2
    U_int = np.round(U_arr * 2).astype(np.int64)
    V_int = np.round(V_arr * 2).astype(np.int64)
    W_int = np.round(W_arr * 2).astype(np.int64)
    
    # Compute tensor T_recon * 8
    T_recon = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int, optimize=True)
    
    # Target is all 8s
    target = np.full((n, m, p), 8, dtype=np.int64)
    
    return np.all(T_recon == target)

def find_linear_dependencies(U, V, W, n, m, p):
    """
    Find linear dependencies among the rank-1 tensors in the decomposition.
    A linear dependency is a vector c such that sum_r c_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0.
    This is equivalent to: for each (i,j,k), sum_r c_r * U[i,r] * V[j,r] * W[k,r] = 0.
    
    We can reformulate: for each pair (i,j), define vector v_ij where v_ij[r] = U[i,r]*V[j,r].
    Then the condition is: for each k, sum_r c_r * v_ij[r] * W[k,r] = 0.
    
    Alternatively, think of it as: the 28 vectors (flattened rank-1 tensors) span a subspace,
    and we want to find if any are linearly dependent.
    
    A more direct approach: construct the 32x4x4 tensor from the decomposition and check
    if we can express it with fewer than 28 rank-1 terms.
    
    Actually, the cleanest approach: for each subset of columns, check if removing one column
    still allows us to represent the tensor. But that's exponential.
    
    Better: Use the fact that if there's a linear dependency, then for some non-zero c,
    sum_r c_r * U[:,r] * V[:,r]^T * W[:,r]^T = 0 as a tensor.
    
    Let's use a different approach: the tensor T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r].
    If we have a dependency sum_r c_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0, then
    we can remove one term and adjust the others.
    
    Practical approach: try removing each term one at a time and see if the remaining 27
    can still represent the tensor (by solving for adjusted coefficients).
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # Build the full tensor
    T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr, optimize=True)
    
    # For each term r, check if we can express T without it using the other rank-1 tensors
    # This is a linear system: for each (i,j,k), T[i,j,k] = sum_{r' != r} alpha_{r'} * U[i,r']*V[j,r']*W[k,r']
    # Unknowns: alpha_{r'} for r' != r
    # This is a linear system in the alpha's
    
    # But this is 32 equations in 27 unknowns, so generally overdetermined.
    # Let's use least squares and check if the residual is zero.
    
    for r_remove in range(rank):
        if time.time() > 110:
            break
            
        # Build the matrix A where A[flat_idx, r'] = U[i,r']*V[j,r']*W[k,r'] for (i,j,k) -> flat_idx
        # Only for r' != r_remove
        indices = [r for r in range(rank) if r != r_remove]
        A = np.zeros((n * m * p, len(indices)), dtype=np.float64)
        
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    flat = (ii * m + jj) * p + kk
                    for idx, r_prime in enumerate(indices):
                        A[flat, idx] = U_arr[ii, r_prime] * V_arr[jj, r_prime] * W_arr[kk, r_prime]
        
        b = T.flatten()
        
        # Solve least squares
        try:
            alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            residual = np.linalg.norm(A @ alpha - b)
            
            if residual < 1e-8:
                # We found a dependency! We can remove term r_remove.
                # Construct new decomposition with rank-1 terms
                U_new = np.zeros((n, m, rank - 1), dtype=np.float64)
                V_new = np.zeros((m, p, rank - 1), dtype=np.float64)
                W_new = np.zeros((p, n, rank - 1), dtype=np.float64)
                
                for idx, r_prime in enumerate(indices):
                    U_new[:, :, idx] = U_arr[:, r_prime] * alpha[idx]
                    V_new[:, :, idx] = V_arr[:, r_prime]
                    W_new[:, :, idx] = W_arr[:, r_prime]
                
                # Verify
                T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new, optimize=True)
                if np.allclose(T_check, T, atol=1e-8):
                    # Snap to half-integers
                    U_snap = np.round(U_new * 2) / 2.0
                    V_snap = np.round(V_new * 2) / 2.0
                    W_snap = np.round(W_new * 2) / 2.0
                    
                    if verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        
        except Exception:
            continue
    
    return None, None, None

def residuals(x, n, m, p, rank):
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
    
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
    return T_recon.flatten() - 1.0

def try_find_low_rank(n, m, p, target_rank, deadline, max_restarts=100):
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    np.random.seed(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with various strategies
        if restart % 3 == 0:
            # Random initialization
            x0 = np.random.randn(total_size) * 0.5
        elif restart % 3 == 1:
            # Sparse random initialization
            x0 = np.random.choice([-1, 0, 1], size=total_size, p=[0.3, 0.4, 0.3])
        else:
            # Small random initialization
            x0 = np.random.randn(total_size) * 0.1
        
        try:
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), 
                                       method='lm', max_nfev=2000, ftol=1e-12, xtol=1e-12)
            
            if np.linalg.norm(result.fun) < 1e-8:
                U = result.x[:size_U].reshape(n*m, target_rank)
                V = result.x[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = result.x[size_U + size_V : ].reshape(p*n, target_rank)
                
                # Try to snap to half-integers
                U_snapped = np.round(U * 2) / 2.0
                V_snapped = np.round(V * 2) / 2.0
                W_snapped = np.round(W * 2) / 2.0
                
                if verify_decomposition(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
                    
        except Exception:
            pass
            
    return None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    # Generate fallback Rank 28
    U28, V28, W28 = generate_strassen_28(n, m, p)
    
    # Try to find linear dependencies in the rank-28 decomposition (peeling)
    U27, V27, W27 = find_linear_dependencies(U28, V28, W28, n, m, p)
    
    if U27 is not None:
        return {"U": U27, "V": V27, "W": W27}
    
    # Try to find Rank 24 via optimization
    U24, V24, W24 = try_find_low_rank(n, m, p, 24, deadline, max_restarts=50)
    
    if U24 is not None:
        return {"U": U24, "V": V24, "W": W24}
        
    return {"U": U28, "V": V28, "W": W28}
