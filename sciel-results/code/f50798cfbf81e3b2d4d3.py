"""Matrix multiplication 2x3x6: rank-minimal tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n=2, m=3, p=6, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    # Build the full tensor: T[i*m+j, j*p+k, k*n+i] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Compute the decomposed tensor
    # T_decomp[a, b, c] = sum_d U[a,d] * V[b,d] * W[c,d]
    # We need T_decomp[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), 0 otherwise
    
    # Build U * outer with V first
    # Actually, let's be more careful. The tensor has dimensions (n*m) x (m*p) x (p*n)
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # For valid entries: i'=i, j'=j, k'=k gives T[i*m+j, j*p+k, k*n+i] = 1
    
    # Compute sum over d of U[a,d] * V[b,d] * W[c,d]
    # This is a 3-tensor contraction
    # Use einsum or manual
    # Let's use: result[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
    # = sum_d (U[a,d] * V[b,d]) * W[c,d]
    
    # For efficiency with small sizes, just do it directly
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for d in range(r):
        u_col = U[:, d]
        v_col = V[:, d]
        w_col = W[:, d]
        # outer product u_col * v_col gives (n*m, m*p), multiply by w_col broadcast
        # Actually: result += np.einsum('a,b,c->abc', u_col, v_col, w_col)
        # For small sizes this is fine
        for a in range(n*m):
            if abs(u_col[a]) > 1e-15:
                for b in range(m*p):
                    if abs(v_col[b]) > 1e-15:
                        val = u_col[a] * v_col[b]
                        for c in range(p*n):
                            result[a, b, c] += val * w_col[c]
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(result[a, b, c] - 1.0) > tol:
                    return False
    
    # Check invalid entries are close to 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
    # More thorough check: all (a,b,c) combinations
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                # Is this a valid (i,j,k)?
                i = a // m
                j = a % m
                # b should correspond to same j: b = j'*p + k', so j' = b // p
                j2 = b // p
                k = b % p
                # c = k'*n + i', so i' = c % n, k' = c // n
                i2 = c % n
                k2 = c // n
                if i == i2 and j == j2 and k == k2:
                    if abs(result[a, b, c] - 1.0) > tol:
                        return False
                else:
                    if abs(result[a, b, c]) > tol:
                        return False
    return True

def _schoolbook():
    """Return the standard schoolbook decomposition of rank n*m*p = 36."""
    n, m, p = 2, 3, 6
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x3x6():
    """
    Construct a rank-33 decomposition for 2x3x6 matrix multiplication.
    
    2x3x6 = 2x3x3 x 3x3x6 (two 2x3x3 blocks along the 6 dimension)
    Actually, let's think differently.
    
    A 2x3 matrix A times a 3x6 matrix B gives a 2x6 result C.
    C = A * B where A is 2x3, B is 3x6.
    
    We can split B into two 3x3 blocks: B = [B1 | B2]
    Then C = [A*B1 | A*B2]
    
    So we need to compute two 2x3*3x3 products.
    
    The known rank for 2x3x3 is 16.5 (with half-integers) or 17.
    Actually, the minimum rank for 2x3x3 is 16 (with integers) or 16.5 with half-integers.
    
    Wait, let me recall. The tensor [2,3,3] has rank 16 (over integers, no, over reals it's 16.5?).
    
    Actually, I recall that M(2,3,3) = 16.5 (with half-integer coefficients).
    So two such multiplications would need rank 33.
    
    Let me try to construct this. For the 2x3x3 case:
    A is 2x3: [[a11, a12, a13], [a21, a22, a23]]
    B is 3x3: [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    C is 2x3: C[i,j] = sum_k A[i,k] * B[k,j]
    
    The 16.5 rank decomposition... Let me think about whether I can derive it.
    
    Actually, I'm not confident I can produce the exact 16.5 rank decomposition from memory.
    Let me instead try a different approach: use the fact that 2x3x6 can be seen as
    a single tensor and try to find a low-rank decomposition through optimization.
    
    But first, let me try a simpler idea: can we save rank by exploiting structure?
    
    The schoolbook has 36 terms. Can we do better?
    
    Let me try a random search approach to find a valid lower-rank decomposition.
    """
    # This function will be a placeholder; the actual search happens in construct()
    pass

def _search_low_rank(deadline):
    """
    Search for a low-rank decomposition using random initialization and refinement.
    """
    n, m, p = 2, 3, 6
    n_rows = n * m  # 6
    m_rows = m * p  # 18
    p_rows = p * n  # 12
    
    # The target tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k)
    # We want rank r such that T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    
    # Start with rank 33 and try to verify; if that fails, try 32, 31, ...
    
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_rank = 36  # schoolbook
    
    # Try ranks from 33 down to 20
    for target_rank in range(33, 19, -1):
        if time.time() > deadline:
            break
        
        # Try several random initializations
        max_iters = 2000
        for init_iter in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n_rows, target_rank)
            V = np.random.randn(m_rows, target_rank)
            W = np.random.randn(p_rows, target_rank)
            
            # Scale to have roughly unit norm columns
            for c in range(target_rank):
                norm_u = np.linalg.norm(U[:, c])
                norm_v = np.linalg.norm(V[:, c])
                norm_w = np.linalg.norm(W[:, c])
                if norm_u > 1e-15:
                    U[:, c] /= norm_u
                if norm_v > 1e-15:
                    V[:, c] /= norm_v
                if norm_w > 1e-15:
                    W[:, c] /= norm_w
            
            # Alternating minimization
            converged = False
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U: for each a, U[a,:] should satisfy
                # sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
                # Fix V and W, solve for U
                # T[a,b,c_idx] = sum_c U[a,c] * (V[b,c] * W[c_idx,c])
                # This is a linear system: for fixed b, c_idx:
                # sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
                # Let M[a, b, c_idx, c] = V[b,c] * W[c_idx,c]
                # Then U[a,:] * M[a, b, c_idx, :] = T[a,b,c_idx]
                # This is underdetermined in general, but we can use least squares
                
                # Build the design matrix for U
                # For each (a, b, c_idx) triplet where T[a,b,c_idx] != 0:
                # We have: sum_c U[a,c] * V[b,c] * W[c_idx,c] = 1
                
                # Actually, let's use a different approach: 
                # The tensor T can be viewed as a matrix T_{(a,(b,c_idx))}
                # And we want T_{(a,(b,c_idx))} = sum_c U[a,c] * (V[b,c] * W[c_idx,c])
                # Let's define Z[b, c_idx, c] = V[b,c] * W[c_idx,c]
                # Then T_{(a,(b,c_idx))} = sum_c U[a,c] * Z[b, c_idx, c]
                
                # Reshape: T_mat[a, b*p_rows + c_idx] = sum_c U[a,c] * Z[b, c_idx, c]
                # Let Z_mat[b, c_idx, c] -> Z_flat[b*p_rows + c_idx, c]
                # Then T_mat[a, :] = U[a, :] * Z_flat[:, :]^T
                # So T_mat = U * Z_flat^T
                # This is a matrix factorization!
                
                # Build T_mat: shape (n_rows, m_rows * p_rows) = (6, 18*12) = (6, 216)
                # T_mat[a, b*p_rows + c_idx] = 1 if (a,b,c_idx) is a valid (i,j,k), else 0
                
                T_mat = np.zeros((n_rows, m_rows * p_rows), dtype=np.float64)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c_idx = k * n + i
                            T_mat[a, b * p_rows + c_idx] = 1.0
                
                # Z_flat: shape (m_rows * p_rows, target_rank)
                # Z_flat[b*p_rows + c_idx, c] = V[b,c] * W[c_idx,c]
                Z_flat = np.zeros((m_rows * p_rows, target_rank), dtype=np.float64)
                for b in range(m_rows):
                    for c_idx in range(p_rows):
                        Z_flat[b * p_rows + c_idx, :] = V[b, :] * W[c_idx, :]
                
                # Solve U from T_mat = U * Z_flat^T
                # U = T_mat * (Z_flat^T)^+  (pseudo-inverse)
                # Or use least squares
                if target_rank <= m_rows * p_rows:
                    U_new, _, _, _ = np.linalg.lstsq(Z_flat.T, T_mat.T, rcond=None)
                    U_new = U_new.T
                else:
                    break
                
                # Update V: similarly
                # T_mat[a, b*p_rows + c_idx] = sum_c U[a,c] * V[b,c] * W[c_idx,c]
                # Fix U and W: for fixed a, c_idx:
                # sum_c V[b,c] * (U[a,c] * W[c_idx,c]) = T_mat[a, b*p_rows + c_idx]
                # Let P[a, c_idx, c] = U[a,c] * W[c_idx,c]
                # Then for each a, c_idx: sum_c V[b,c] * P[a, c_idx, c] = T_mat[a, b*p_rows + c_idx]
                
                # Reshape: for fixed a, c_idx: V[:, c] * P[a, c_idx, c] = T_mat[a, b*p_rows + c_idx] for all b
                # This is: for each (a, c_idx), V * P[a, c_idx, :].T = T_mat[a, c_idx*p_rows : ... ]
                # Hmm, let me think again.
                
                # T_mat has shape (n_rows, m_rows * p_rows)
                # T_mat[a, b*p_rows + c_idx] = sum_c U[a,c] * V[b,c] * W[c_idx,c]
                
                # Fix U, W. Define Q[a, b, c_idx, c] = U[a,c] * W[c_idx,c]
                # Then T_mat[a, b*p_rows + c_idx] = sum_c Q[a, b, c_idx, c] * V[b,c]
                
                # For fixed (a, c_idx): T_mat[a, b*p_rows + c_idx] for all b
                # = sum_c V[b,c] * (U[a,c] * W[c_idx,c])
                # Let R[a, c_idx, c] = U[a,c] * W[c_idx,c]
                # Then for all b: T_mat[a, b*p_rows + c_idx] = sum_c V[b,c] * R[a, c_idx, c]
                
                # Reshape T_mat to T_4d[a, b, c_idx]
                T_4d = T_mat.reshape(n_rows, m_rows, p_rows)
                
                # For each (a, c_idx): V * R[a, c_idx, :].T = T_4d[a, :, c_idx]
                # This is a system of n_rows * p_rows equations in m_rows * target_rank unknowns
                # Let's vectorize: 
                # Stack all (a, c_idx) pairs: 
                # For each (a, c_idx): V[b,c] * R[a,c_idx,c] = T_4d[a,b,c_idx]
                # Let's build a big linear system for V
                # Unknowns: V[b,c] for b in range(m_rows), c in range(target_rank) -> m_rows * target_rank unknowns
                # Equations: for each (a, b, c_idx): sum_c V[b,c] * R[a,c_idx,c] = T_4d[a,b,c_idx]
                # Number of equations: n_rows * m_rows * p_rows = 6 * 18 * 12 = 1296
                
                # Build the system:
                # x = V.reshape(-1) of size m_rows * target_rank
                # Equation (a, b, c_idx): sum_c V[b,c] * R[a,c_idx,c] = T_4d[a,b,c_idx]
                # This involves only V[b, :] (target_rank variables)
                
                # For each (a, c_idx), we have m_rows equations (one per b):
                # sum_c V[b,c] * R[a,c_idx,c] = T_4d[a,b,c_idx]
                # This is: V * R[a,c_idx,:].T = T_4d[a,:,c_idx]
                # As a linear system in V: for each b, the row is R[a,c_idx,:] (length target_rank)
                # and the RHS is T_4d[a,b,c_idx]
                # So: R[a,c_idx,:] * V[b,:].T = T_4d[a,b,c_idx]
                # i.e., V[b,:] @ R[a,c_idx,:] = T_4d[a,b,c_idx]
                
                # Stack all (a, c_idx): 
                # A_V is of size (n_rows * p_rows * m_rows, m_rows * target_rank)? No...
                # Actually each equation involves only one row of V.
                # Let's just solve for each b separately... no, that's not right either.
                
                # Alternative: use the fact that this is separable.
                # Let's just use lstsq on the full system.
                
                # Variables: V.reshape(-1), size m_rows * target_rank
                # For equation (a, b, c_idx): coefficient of V[b', c] is R[a, c_idx, c] if b'==b else 0
                # So the row for (a, b, c_idx) has non-zeros only in the block for row b of V.
                
                # This is a sparse system. Let's build it densely since sizes are small.
                num_eqs = n_rows * m_rows * p_rows  # 1296
                num_vars = m_rows * target_rank
                A_V = np.zeros((num_eqs, num_vars), dtype=np.float64)
                rhs_V = np.zeros(num_eqs, dtype=np.float64)
                
                eq_idx = 0
                for a in range(n_rows):
                    for b in range(m_rows):
                        for c_idx in range(p_rows):
                            for c in range(target_rank):
                                A_V[eq_idx, b * target_rank + c] = U[a, c] * W[c_idx, c]
                            rhs_V[eq_idx] = T_4d[a, b, c_idx]
                            eq_idx += 1
                
                V_new, _, _, _ = np.linalg.lstsq(A_V, rhs_V, rcond=None)
                V_new = V_new.reshape(m_rows, target_rank)
                
                # Update W: similar
                # T_4d[a, b, c_idx] = sum_c U[a,c] * V[b,c] * W[c_idx,c]
                # Fix U, V: for fixed (a, b):
                # sum_c W[c_idx,c] * (U[a,c] * V[b,c]) = T_4d[a,b,c_idx] for all c_idx
                # Let S[a, b, c] = U[a,c] * V[b,c]
                # Then: W[c_idx,:] @ S[a,b,:].T = T_4d[a,b,c_idx]
                
                num_eqs_w = n_rows * m_rows * p_rows  # 1296
                num_vars_w = p_rows * target_rank
                A_W = np.zeros((num_eqs_w, num_vars_w), dtype=np.float64)
                rhs_W = np.zeros(num_eqs_w, dtype=np.float64)
                
                eq_idx = 0
                for a in range(n_rows):
                    for b in range(m_rows):
                        for c_idx in range(p_rows):
                            for c in range(target_rank):
                                A_W[eq_idx, c_idx * target_rank + c] = U[a, c] * V[b, c]
                            rhs_W[eq_idx] = T_4d[a, b, c_idx]
                            eq_idx += 1
                
                W_new, _, _, _ = np.linalg.lstsq(A_W, rhs_W, rcond=None)
                W_new = W_new.reshape(p_rows, target_rank)
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence / validity
                if _verify(U, V, W, tol=1e-8):
                    converged = True
                    break
                
                if time.time() > deadline:
                    break
            
            if converged:
                best_rank = target_rank
                # Convert to lists
                U_list = [[float(U[a, c]) for c in range(target_rank)] for a in range(n_rows)]
                V_list = [[float(V[b, c]) for c in range(target_rank)] for b in range(m_rows)]
                W_list = [[float(W[c_idx, c]) for c in range(target_rank)] for c_idx in range(p_rows)]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # First, get the schoolbook solution as a fallback
    schoolbook = _schoolbook()
    
    # Try to find a lower-rank decomposition
    if time.time() < deadline - 5:
        result = _search_low_rank(deadline)
        if result is not None:
            # Verify it's correct
            if _verify(result["U"], result["V"], result["W"], tol=1e-8):
                return result
    
    # Fallback to schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
