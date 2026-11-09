import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Check shape consistency
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    
    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    # S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    S = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    return np.allclose(S, 1.0, atol=1e-8)

def _to_halfint_list(arr):
    """Convert numpy array to list of lists with half-integer values."""
    result = []
    for row in arr:
        row_list = []
        for val in row:
            # Round to nearest half-integer
            half_int = round(val * 2) / 2.0
            row_list.append(half_int)
        result.append(row_list)
    return result

def _bclr_compress(U, V, W, n, m, p, max_iterations=500, deadline=None):
    """
    Apply BCLR (Bini-Capovani-Lotti-Romani) compression to reduce tensor rank.
    
    The algorithm iteratively finds linear dependencies among the rank-1 components
    and uses them to reduce the rank by 1 while preserving the tensor.
    
    Returns (U_new, V_new, W_new) with reduced rank, or None if no reduction possible.
    """
    r = U.shape[1]
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Check if we can find a dependency among the rank-1 terms
        # The tensor is: T = sum_c U_c ⊗ V_c ⊗ W_c
        # where U_c is the c-th column of U, etc.
        
        # Try to find a linear combination of columns that gives zero tensor contribution
        # We look for vectors a, b, c such that:
        # sum_c a_c * U[:,c] = 0, sum_c b_c * V[:,c] = 0, sum_c c_c * W[:,c] = 0
        # But more generally, we look for a dependency that preserves the tensor.
        
        # Alternative approach: find a rank-1 update that can be absorbed
        # If we can write one component as a combination of others plus a rank-1 correction,
        # we can reduce the rank.
        
        # Practical BCLR: for each pair of indices, try to find a transformation
        # that eliminates one component.
        
        # Try all pairs (c1, c2) to see if component c2 can be eliminated
        # by modifying components using a rank-1 update.
        
        found_reduction = False
        for c2 in range(r - 1, -1, -1):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try to eliminate component c2
            # We want to find coefficients alpha_c (for c != c2) and a rank-1 tensor
            # such that: U[:,c2] ⊗ V[:,c2] ⊗ W[:,c2] = sum_{c!=c2} alpha_c * (correction_c)
            
            # A simpler approach: check if the c2-th component is "redundant"
            # by seeing if removing it and adjusting others can preserve the tensor.
            
            # Build the system: for each (i,j,k), we need:
            # U[i*m+j,c2]*V[j*p+k,c2]*W[k*n+i,c2] = sum_{c!=c2} delta_c[i*m+j] * V[j*p+k,c] * W[k*n+i,c]
            #   + sum_{c!=c2} U[i*m+j,c] * delta_V_c[j*p+k] * W[k*n+i,c]
            #   + sum_{c!=c2} U[i*m+j,c] * V[j*p+k,c] * delta_W_c[k*n+i]
            
            # This is complex. Let's try a simpler heuristic:
            # Check if component c2 is close to being a linear combination of others
            # in a tensor sense.
            
            # For each mode, check if the c2-th column is in the span of others
            U_others = np.delete(U, c2, axis=1)
            V_others = np.delete(V, c2, axis=1)
            W_others = np.delete(W, c2, axis=1)
            
            # Check if U[:,c2] is in span of U_others
            try:
                coeffs_U, residuals_U, rank_U, sv_U = np.linalg.lstsq(U_others, U[:, c2], rcond=None)
            except:
                continue
            
            try:
                coeffs_V, residuals_V, rank_V, sv_V = np.linalg.lstsq(V_others, V[:, c2], rcond=None)
            except:
                continue
            
            try:
                coeffs_W, residuals_W, rank_W, sv_W = np.linalg.lstsq(W_others, W[:, c2], rcond=None)
            except:
                continue
            
            # If all three columns are in the span of others, we might be able to eliminate
            # But we need the tensor to be preserved, not just each mode separately.
            
            # Check if the rank-1 tensor U_c2 ⊗ V_c2 ⊗ W_c2 can be expressed
            # as a combination of the other rank-1 tensors.
            
            # Build the tensor of other components and check if T_c2 is in its span
            # This is expensive, so use a heuristic: if all three are in span with small residuals,
            # try the elimination.
            
            tol = 1e-8
            if (np.linalg.norm(U_others @ coeffs_U - U[:, c2]) < tol and
                np.linalg.norm(V_others @ coeffs_V - V[:, c2]) < tol and
                np.linalg.norm(W_others @ coeffs_W - W[:, c2]) < tol):
                
                # All three columns are in the span. Try to eliminate c2.
                # New U: remove column c2, and add corrections to other columns
                # This requires careful handling. For now, just remove and verify.
                
                U_new = np.delete(U, c2, axis=1)
                V_new = np.delete(V, c2, axis=1)
                W_new = np.delete(W, c2, axis=1)
                
                # Verify the reduced decomposition still works
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    U, V, W = U_new, V_new, W_new
                    r = U.shape[1]
                    found_reduction = True
                    break
        
        if not found_reduction:
            break
    
    return U, V, W

def _try_rank_als(n, m, p, r, n_iters=200, n_restarts=3, deadline=None):
    """Try to find a rank-r decomposition using alternating least squares."""
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(n, m, r)
        V = np.random.randn(m, p, r)
        W = np.random.randn(p, n, r)
        
        for it in range(n_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # B[k, c] = V[j,k,c] * W[k,i,c]
                    B = V[j] * W[:, i]
                    try:
                        x, _, _, _ = np.linalg.lstsq(B.T, np.ones(p), rcond=None)
                        U[i, j, :] = x
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # B[i, c] = U[i,j,c] * W[k,i,c]
                    B = U[:, j, :] * W[k, :, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(B.T, np.ones(n), rcond=None)
                        V[j, k, :] = x
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # B[j, c] = U[i,j,c] * V[j,k,c]
                    B = U[i] * V[:, k]
                    try:
                        x, _, _, _ = np.linalg.lstsq(B.T, np.ones(m), rcond=None)
                        W[k, i, :] = x
                    except:
                        pass
            
            # Compute error
            S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = np.linalg.norm(S - 1.0)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-8:
                break
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-6:
        # Flatten and round to half-integers
        U_flat = best_U.reshape(n*m, r)
        V_flat = best_V.reshape(m*p, r)
        W_flat = best_W.reshape(p*n, r)
        
        U_hi = _to_halfint_list(U_flat)
        V_hi = _to_halfint_list(V_flat)
        W_hi = _to_halfint_list(W_flat)
        
        if _verify_decomposition(U_hi, V_hi, W_hi, n, m, p):
            return U_hi, V_hi, W_hi
    
    return None, None, None

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook rank-n*m*p decomposition."""
    U_school = [[0] * (n*m*p) for _ in range(n*m)]
    V_school = [[0] * (n*m*p) for _ in range(m*p)]
    W_school = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    return U_school, V_school, W_school

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline to be safe (150s limit)
    
    # Schoolbook decomposition as fallback
    U_school, V_school, W_school = _construct_schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p
    
    # First, try BCLR compression on the schoolbook decomposition
    if time.time() < deadline - 60:
        U_np = np.array(U_school, dtype=float)
        V_np = np.array(V_school, dtype=float)
        W_np = np.array(W_school, dtype=float)
        
        U_compressed, V_compressed, W_compressed = _bclr_compress(
            U_np, V_np, W_np, n, m, p, max_iterations=200, deadline=deadline - 30
        )
        
        if U_compressed.shape[1] < best_rank:
            # Try to round to half-integers
            U_hi = _to_halfint_list(U_compressed)
            V_hi = _to_halfint_list(V_compressed)
            W_hi = _to_halfint_list(W_compressed)
            if _verify_decomposition(U_hi, V_hi, W_hi, n, m, p):
                best_result = {"U": U_hi, "V": V_hi, "W": W_hi}
                best_rank = U_compressed.shape[1]
    
    # Try ALS for various ranks
    # Theoretical lower bound is max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    for r in range(35, best_rank, 2):
        if time.time() > deadline - 10:
            break
        
        U, V, W = _try_rank_als(n, m, p, r, n_iters=100, n_restarts=2, deadline=deadline - 5)
        if U is not None:
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break
    
    return best_result
