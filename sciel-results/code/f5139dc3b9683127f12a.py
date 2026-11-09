import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 5]
DEADLINE_MARGIN = 15.0

def _build_target(n, m, p):
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    target = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
    return target

def _schoolbook(n, m, p):
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    R = n * m * p
    U = np.zeros((dim_u, R), dtype=int)
    V = np.zeros((dim_v, R), dtype=int)
    W = np.zeros((dim_w, R), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    return U, V, W

def _verify_rational(U_int, V_int, W_int, target_int):
    """Verify that U @ V @ W == target using rational arithmetic.
    U_int, V_int, W_int are integer matrices (already scaled by common denominator).
    Returns True if the decomposition is exact.
    """
    dim_u, R = U_int.shape
    dim_v, _ = V_int.shape
    dim_w, _ = W_int.shape
    
    # We need to check: sum_r U[a,r]*V[b,r]*W[c,r] == target[a,b,c] for all a,b,c
    # Using Fraction for exact arithmetic
    target_frac = {}
    for a in range(dim_u):
        for b in range(dim_v):
            if target_int[a, b] != 0:
                # target is a 3D array stored as (dim_u, dim_v) with implicit dim_w=1?
                # No, target_int is actually flattened. Let me re-examine.
                pass
    
    # Actually target_int is 2D: (dim_u * dim_w, dim_v) or similar.
    # Let me re-check the original _build_target. It returns (dim_u, dim_v, dim_w).
    # So target_int is 3D.
    dim_w2 = target_int.shape[2]
    
    for a in range(dim_u):
        for b in range(dim_v):
            for c in range(dim_w2):
                expected = Fraction(target_int[a, b, c])
                actual = Fraction(0)
                for r in range(R):
                    actual += Fraction(U_int[a, r]) * Fraction(V_int[b, r]) * Fraction(W_int[c, r])
                if actual != expected:
                    return False
    return True

def _verify_float(U, V, W, target):
    """Fast float verification."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, target, atol=1e-9)

def _search_integer_rank(target, R, n, m, p, deadline):
    """Search for integer decomposition of rank R.
    
    Uses the structure: each rank-1 term corresponds to a choice of
    (i, j, k) indices, but we allow linear combinations.
    
    For 3x3x5, the tensor has 45 non-zero entries.
    We parameterize by choosing which "patterns" to use and their integer coefficients.
    
    Strategy: 
    - The support of the tensor is {(i*m+j, j*p+k, k*n+i) : i in 0..2, j in 0..2, k in 0..4}
    - A rank-1 tensor is u ⊗ v ⊗ w
    - We try to find R vectors u_r, v_r, w_r with small integer entries such that
      sum_r u_r ⊗ v_r ⊗ w_r = target
    
    Approach: Fix V and W to have a specific sparse structure, then solve for U,
    or use a combinatorial search over small integer patterns.
    """
    dim_u = n * m  # 9
    dim_v = m * p  # 15
    dim_w = p * n  # 15
    
    # The key insight: for matrix multiplication, the "natural" decomposition uses
    # rank-1 terms where each term corresponds to one (i,j,k) triple.
    # To get rank < 45, we need to combine multiple terms.
    #
    # A known approach: for multiplying n×m by m×p, if we can find a 
    # "fast matrix multiplication" scheme, the rank can be lower.
    # For 3×3 × 3×5, this is not a square case, so standard Strassen doesn't apply directly.
    #
    # Let's try a more direct search: assume entries are in {-2,-1,0,1,2} and
    # each factor matrix is sparse.
    
    # Generate candidate patterns for U, V, W
    # Each column of U is a vector in Z^9, etc.
    
    # Given time constraints, try a focused search:
    # Assume the rank-1 terms have a structure related to the original (i,j,k) indices
    # but allow for linear combinations.
    
    # For now, try random integer search with verification
    np.random.seed(42)
    
    max_attempts = 5000
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Generate random sparse integer matrices
        # Each entry is 0 with prob 0.6, otherwise random in {-1, 0, 1, 2}
        U = np.zeros((dim_u, R), dtype=int)
        V = np.zeros((dim_v, R), dtype=int)
        W = np.zeros((dim_w, R), dtype=int)
        
        for r in range(R):
            # U column: sparse
            nz_u = np.random.choice(dim_u, size=min(3, dim_u), replace=False)
            U[nz_u, r] = np.random.randint(-1, 3, size=len(nz_u))
            
            nz_v = np.random.choice(dim_v, size=min(4, dim_v), replace=False)
            V[nz_v, r] = np.random.randint(-1, 3, size=len(nz_v))
            
            nz_w = np.random.choice(dim_w, size=min(4, dim_w), replace=False)
            W[nz_w, r] = np.random.randint(-1, 3, size=len(nz_w))
        
        if _verify_float(U, V, W, target):
            if _verify_rational(U, V, W, target):
                return U, V, W
    
    return None

def _search_with_fixed_structure(target, R, n, m, p, deadline):
    """Try structured search where we fix some aspects of the decomposition.
    
    For matrix multiplication M (n×m) × A (m×p) = C (n×p),
    the tensor T[i*m+j, j*p+k, k*n+i] = 1.
    
    Key structure: the middle index j appears in both input and output.
    We can try to exploit this by having V columns that are "basis-like" 
    for the middle dimension.
    """
    dim_u = n * m  # 9
    dim_v = m * p  # 15
    dim_w = p * n  # 15
    
    # Strategy: For each j (middle index 0,1,2), we can have terms that
    # "cover" all i and k simultaneously.
    # If we use terms where V has support only on indices with fixed j,
    # then we need fewer terms.
    
    # Try: group by j. For each j, we need to multiply a 3×1 vector by 1×5 vector
    # which has rank 1 (outer product). But that gives rank 3*1*5 = 15 for the "diagonal" part.
    # That's not better than 45.
    
    # Alternative: use the fact that we can combine i and k indices.
    # For a fixed j, the sub-tensor is u_i ⊗ e_j ⊗ w_k where u is 3-dim, w is 5-dim.
    # The rank of u⊗e⊗w as a rank-1 tensor in the full space is 1, but we need to
    # cover all (i,k) pairs for each j.
    
    # Let me try a different approach: use the fact that 3×5 multiplication
    # can potentially use rank < 15 via fast algorithms.
    # The rank of 3×5 matrix multiplication is at most 3*5=15 (schoolbook)
    # and at least... well, for 2×2 it's 7 (Strassen), for 3×3 it's at most 23.
    # For 3×5, the rank is between 15 and 3*5=15... actually the min rank for m×n
    # matrix multiplication is not simply bounded.
    
    # Given the complexity, let's try a targeted numerical search with integer snap.
    from scipy.optimize import least_squares
    
    def res_func(params):
        U = params[:dim_u*R].reshape(dim_u, R)
        V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
        W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_recon - target).flatten()
    
    best_result = None
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(1000 + restart * 7919)
        
        # Initialize with scaled schoolbook or random
        if restart < 5:
            U_sb, V_sb, W_sb = _schoolbook(n, m, p)
            # Take a random subset of schoolbook columns
            indices = np.random.choice(n*m*p, R, replace=False)
            U_init = U_sb[:, indices].astype(float).copy()
            V_init = V_sb[:, indices].astype(float).copy()
            W_init = W_sb[:, indices].astype(float).copy()
        else:
            U_init = np.random.randn(dim_u, R) * 0.5
            V_init = np.random.randn(dim_v, R) * 0.5
            W_init = np.random.randn(dim_w, R) * 0.5
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(
                res_func,
                params_init,
                method='lm',
                max_nfev=100,
                ftol=1e-14,
                xtol=1e-14,
                gtol=1e-14
            )
            
            if result.cost < 1e-10:
                params = result.x
                U = params[:dim_u*R].reshape(dim_u, R)
                V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
                W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
                
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if _verify_float(U_snap, V_snap, W_snap, target):
                    # Convert to integers by scaling by 2
                    U_int = (U_snap * 2).astype(int)
                    V_int = (V_snap * 2).astype(int)
                    W_int = (W_snap * 2).astype(int)
                    if _verify_rational(U_int, V_int, W_int, target):
                        return U_int, V_int, W_int
                
                # Try snapping to integers
                U_snap2 = np.round(U)
                V_snap2 = np.round(V)
                W_snap2 = np.round(W)
                if _verify_float(U_snap2, V_snap2, W_snap2, target):
                    U_int = U_snap2.astype(int)
                    V_int = V_snap2.astype(int)
                    W_int = W_snap2.astype(int)
                    if _verify_rational(U_int, V_int, W_int, target):
                        return U_int, V_int, W_int
                        
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    target = _build_target(n, m, p)
    sb_R = n * m * p  # 45
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = sb_R
    
    start_time = time.time()
    deadline = start_time + 150 - DEADLINE_MARGIN
    
    # Try structured search first for promising ranks
    for R in range(sb_R - 1, 20, -1):
        if time.time() > deadline:
            break
        
        result = _search_with_fixed_structure(target, R, n, m, p, deadline)
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_rank = R
            best_U = U_sol
            best_V = V_sol
            best_W = W_sol
        
        if time.time() > deadline:
            break
    
    # If time remains, try random integer search
    if time.time() < deadline - 2:
        for R in range(min(best_rank - 1, 40), 15, -1):
            if time.time() > deadline:
                break
            result = _search_integer_rank(target, R, n, m, p, deadline)
            if result is not None:
                U_sol, V_sol, W_sol = result
                best_rank = R
                best_U = U_sol
                best_V = V_sol
                best_W = W_sol
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
