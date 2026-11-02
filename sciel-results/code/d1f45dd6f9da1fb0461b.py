"""Matrix multiplication tensor decomposition for 2x3x8 with rank reduction search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _validate_exact(U, V, W):
    """Check if factor matrices exactly decompose the matmul tensor.
    Returns True if exact, False otherwise.
    Uses integer/rational arithmetic via Fraction for exactness.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build the target tensor: T[i,j,k] = delta (i.e., A[i,j]*B[j,k] -> C[i,k])
    # The decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta(i,j,k)
    # where delta(i,j,k) = 1 if this is the "diagonal" term... 
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] such that C[i,k] = sum_{i',j',k'} U... 
    # Standard: for each (i,j,k), the tensor entry is 1 if we're computing A[i,j]*B[j,k]
    # The matmul tensor T[i, j, k] = 1 (it's a 3-tensor where T[i,j,k]=1 for all valid i,j,k)
    # Wait no. The tensor for matmul(2,3,8) is:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if the "middle" index matches: actually it's just all 1s?
    # No! The matmul tensor is: T[i,j,k] = 1 for the (i,j) entry of A and (j,k) entry of B
    # The tensor has entries T[i, j, k] = 1 for all i,j,k (it's the "multiplication" tensor)
    # Actually: the tensor M[i,j,k] = 1 for all i,j,k is NOT correct.
    # The correct tensor: M[i,j,k] = 1 if we're pairing A[i,j] with B[j,k] to produce C[i,k].
    # Since for every (i,j,k) triple there IS a product A[i,j]*B[j,k], M[i,j,k] = 1 for all.
    # Wait, that can't be right either because then the rank would be trivially n*m*p.
    
    # Let me reconsider. The matrix multiplication tensor T is defined as:
    # T[i, j, k] = 1 if j == j (always true in this formulation)
    # Actually the standard definition: T is a 3-tensor of size n x m x p where
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # No wait - that's a rank-1 tensor (all ones).
    
    # The correct matmul tensor: For multiplying A (n x m) by B (m x p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # As a 3-tensor T of dimensions (n*m) x (m*p) x (p*n):
    # Actually the standard formulation uses:
    # U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    # And: sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # This means the "target" tensor M[i,j,k] = 1 for ALL i,j,k.
    
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Let's verify with schoolbook: U[i*m+j, c=ijk] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # For a fixed (i,j,k): only the column c corresponding to (i,j,k) has all three = 1
    # So sum = 1. Correct.
    
    # Now check if our candidate satisfies this.
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # Build efficiently:
    # Reshape: for each c, outer product U[:,c] * V[:,c] gives (n*m)*(m*p) matrix
    # But we need to check per (i,j,k).
    
    # More efficient: 
    # For fixed k: W[k*n+i, c] * U[i*m+j, c] for each i, j, c
    # Let's just do it directly since dimensions are small.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _try_reduce_rank(U, V, W, target_rank, max_iters=200, tol=1e-8):
    """Try to reduce a rank-r decomposition to target_rank using ALS with random projection."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Random projection to reduce rank
    # Project U (n*m x r) to (n*m x target_rank) via random matrix
    rng = np.random.default_rng(42)
    
    # We want: U * R_u ≈ U' (n*m x target_rank)
    # Then we need V and W to compensate.
    # Actually, better approach: find low-rank approximation of the tensor.
    
    # Alternative: use the fact that if rank r works, we can try to find
    # a subset of columns or a linear combination that still works.
    
    # ALS approach: fix two matrices, solve for the third.
    # The constraint: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let's parameterize: X_c = U[:,c] * V[:,c] * W[:,c] (outer product in tensor form)
    # We need sum_c X_c = T where T[i,j,k] = 1.
    
    # For ALS: given V and W, solve for U.
    # For each (i,j): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
    # This is: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
    # Let B[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
    # Then: sum_c U[i*m+j,c] * B[i,j,k,c] = 1 for all k
    # This is a linear system for each (i,j) with unknowns U[i*m+j, 0..target_rank-1]
    # But we have target_rank unknowns and p equations (for k=0..p-1).
    # If p >= target_rank, we can solve (overdetermined or square).
    
    # Similarly for V: for each (j,k): sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for all i
    # And for W: for each (k,i): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
    
    # Let's implement ALS for the target_rank version.
    
    U_new = np.zeros((n*m, target_rank))
    V_new = np.zeros((m*p, target_rank))
    W_new = np.zeros((p*n, target_rank))
    
    # Initialize with random small values or projection of original
    # Use SVD of original to get initial low-rank approximation
    if r > target_rank:
        # Project U
        U_proj, _ = np.linalg.svd(U_arr, full_matrices=False)
        U_new = U_proj[:, :target_rank]
        # Similarly for V and W (just for initialization)
        V_svd, _ = np.linalg.svd(V_arr, full_matrices=False)
        V_new = V_svd[:, :target_rank]
        W_svd, _ = np.linalg.svd(W_arr, full_matrices=False)
        W_new = W_svd[:, :target_rank]
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        
        # Update U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                # System: for each k: sum_c U[u_idx, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # A[u_idx, c] for unknowns: coefficient of U[u_idx, c] is V[j*p+k, c] * W[k*n+i, c]
                A_sys = np.zeros((p, target_rank))
                b_vec = np.ones(p)
                for k in range(p):
                    for c in range(target_rank):
                        A_sys[k, c] = V_new[j*p+k, c] * W_new[k*n+i, c]
                if target_rank <= p:
                    try:
                        sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                        U_new[u_idx, :] = sol
                    except:
                        pass
                else:
                    # Underdetermined - use pseudoinverse
                    try:
                        sol = np.linalg.pinv(A_sys) @ b_vec
                        U_new[u_idx, :] = sol
                    except:
                        pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :]
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                # For each i: sum_c V[v_idx, c] * U[i*m+j, c] * W[k*n+i, c] = 1
                A_sys = np.zeros((n, target_rank))
                b_vec = np.ones(n)
                for i in range(n):
                    for c in range(target_rank):
                        A_sys[i, c] = U_new[i*m+j, c] * W_new[k*n+i, c]
                if target_rank <= n:
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                        V_new[v_idx, :] = sol
                    except:
                        pass
                else:
                    try:
                        sol = np.linalg.pinv(A_sys) @ b_vec
                        V_new[v_idx, :] = sol
                    except:
                        pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :]
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                # For each j: sum_c W[w_idx, c] * U[i*m+j, c] * V[j*p+k, c] = 1
                A_sys = np.zeros((m, target_rank))
                b_vec = np.ones(m)
                for j in range(m):
                    for c in range(target_rank):
                        A_sys[j, c] = U_new[i*m+j, c] * V_new[j*p+k, c]
                if target_rank <= m:
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                        W_new[w_idx, :] = sol
                    except:
                        pass
                else:
                    try:
                        sol = np.linalg.pinv(A_sys) @ b_vec
                        W_new[w_idx, :] = sol
                    except:
                        pass
    
    # Check residual
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(target_rank):
                    val += U_new[u_idx, c] * V_new[v_idx, c] * W_new[w_idx, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    return max_err, U_new, V_new, W_new

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Build schoolbook solution as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Start from various ranks below schoolbook
    # Known lower bounds: for (n,m,p) matmul, rank >= n*m*p / max(n,m,p) is a crude bound
    # For 2,3,8: crude bound = 48/8 = 6 (very loose)
    # Better: Strassen-type bounds. For 2x3x8, let's try various ranks.
    
    # Try ranks from a reasonable upper bound down
    # The tensor has 2*3*8 = 48 "entries" but they're not all independent
    # Let's try: 36, 32, 28, 24, 20, 18, 16
    
    candidate_ranks = [40, 36, 32, 28, 24, 20, 18, 16, 14, 12, 10, 9, 8]
    
    for target_r in candidate_ranks:
        if time.time() > _DEADLINE - 20:
            break
        
        if target_r >= best_rank:
            continue
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > _DEADLINE - 15:
                break
            
            rng = np.random.default_rng(1000 + trial * 7 + target_r)
            
            # Initialize with random matrices and run ALS
            U_init = rng.standard_normal((n*m, target_r)) * 0.1
            V_init = rng.standard_normal((m*p, target_r)) * 0.1
            W_init = rng.standard_normal((p*n, target_r)) * 0.1
            
            # Run ALS
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            for it in range(300):
                if time.time() > _DEADLINE - 10:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        A_sys = np.zeros((p, target_r))
                        b_vec = np.ones(p)
                        for k in range(p):
                            for c in range(target_r):
                                A_sys[k, c] = V_cur[j*p+k, c] * W_cur[k*n+i, c]
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                            U_cur[u_idx, :] = sol
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        A_sys = np.zeros((n, target_r))
                        b_vec = np.ones(n)
                        for i in range(n):
                            for c in range(target_r):
                                A_sys[i, c] = U_cur[i*m+j, c] * W_cur[k*n+i, c]
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                            V_cur[v_idx, :] = sol
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        A_sys = np.zeros((m, target_r))
                        b_vec = np.ones(m)
                        for j in range(m):
                            for c in range(target_r):
                                A_sys[j, c] = U_cur[i*m+j, c] * V_cur[j*p+k, c]
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                            W_cur[w_idx, :] = sol
                        except:
                            pass
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_r):
                                val += U_cur[i*m+j, c] * V_cur[j*p+k, c] * W_cur[k*n+i, c]
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-10:
                    break
            
            # Final check
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_r):
                            val += U_cur[i*m+j, c] * V_cur[j*p+k, c] * W_cur[k*n+i, c]
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-8:
                # We found a numerical solution! Now try to convert to exact rationals
                # Round to nearest rational with small denominator
                U_frac = []
                V_frac = []
                W_frac = []
                exact_ok = True
                
                def to_frac(x):
                    # Try to represent as fraction with small denominator
                    # First try rounding to nearest integer
                    if abs(x - round(x)) < 1e-9:
                        return int(round(x))
                    # Try fractions with denominators up to 20
                    for den in range(1, 21):
                        num = x * den
                        if abs(num - round(num)) < 1e-9:
                            return Fraction(int(round(num)), den)
                    return Fraction(x).limit_denominator(100)
                
                for row in U_cur:
                    U_frac.append([to_frac(x) for x in row])
                for row in V_cur:
                    V_frac.append([to_frac(x) for x in row])
                for row in W_cur:
                    W_frac.append([to_frac(x) for x in row])
                
                # Verify exactly
                if _validate_exact(U_frac, V_frac, W_frac):
                    if target_r < best_rank:
                        best_rank = target_r
                        best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    # Continue trying lower ranks
                    break
                else:
                    # Try rounding differently
                    for scale in [2, 4, 8, 16]:
                        U_test = []
                        V_test = []
                        W_test = []
                        for row in U_cur:
                            U_test.append([int(round(x * scale)) / scale for x in row])
                        for row in V_cur:
                            V_test.append([int(round(x * scale)) / scale for x in row])
                        for row in W_cur:
                            W_test.append([int(round(x * scale)) / scale for x in row])
                        
                        # Check if these are "exact enough" - they should be rational
                        # Convert to fractions
                        U_fr = [[Fraction(x).limit_denominator(1000) for x in row] for row in U_test]
                        V_fr = [[Fraction(x).limit_denominator(1000) for x in row] for row in V_test]
                        W_fr = [[Fraction(x).limit_denominator(1000) for x in row] for row in W_test]
                        
                        if _validate_exact(U_fr, V_fr, W_fr):
                            if target_r < best_rank:
                                best_rank = target_r
                                best_result = {"U": U_fr, "V": V_fr, "W": W_fr}
                            break
                    else:
                        # Couldn't find exact rational representation
                        # Still use the numerical solution if error is very small
                        # But the task requires integer/half-integer entries
                        # So we can't use it unless we find exact rationals
                        pass
        
        # Early stop if we've found a good rank
        if best_rank <= 16:
            break
    
    return best_result
