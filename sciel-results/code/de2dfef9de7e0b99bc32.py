"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    # The multiplication tensor T has T[a,b,c] = 1 when a=(i,m+j), b=(j,p+k), c=(k,n+i)
    # for some i in [0,n), j in [0,m), k in [0,p).
    # We use a randomized numerical optimization to find low-rank decompositions,
    # then verify exactness with rational reconstruction (sympy) and snap to integers/half-integers.
    
    n, m, p = DIMENSIONS
    # Tensor dimensions: U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    dimU = n * m  # 6
    dimV = m * p  # 18
    dimW = p * n  # 12
    
    # Build the sparse tensor T as a 3D array
    # T[a, b, c] = 1 if there exist i,j,k with a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((dimU, dimV, dimW), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Known bounds: rank at least max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # Schoolbook rank = n*m*p = 36
    # For 2x3x6, we try ranks from 18 up to 36.
    # Actually, the known lower bound is the tensor rank >= max dim = 18.
    # Let's try from 18 upward.
    
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_result = None
    best_rank = None
    
    # Try increasing ranks
    for rank in range(18, 37):
        if time.time() > deadline:
            break
        result = _try_rank(rank, T, dimU, dimV, dimW, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    if best_result is None:
        # Fallback to schoolbook
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
    
    return best_result


def _try_rank(rank, T, dimU, dimV, dimW, deadline):
    """Try to find an exact rank-`rank` decomposition of tensor T."""
    # Randomized iterative approach:
    # Initialize U, V, W randomly with the sparsity pattern enforced
    # Then use ALS (alternating least squares) or gradient-based optimization
    # to drive the residual to zero.
    
    # For a sparse tensor with small dimensions, we can set up a system of equations.
    # T[a,b,c] = sum_l U[a,l] * V[b,l] * W[c,l]
    # This is a system of dimU*dimV*dimW equations with rank*3 unknowns.
    # But most equations are 0 = 0 (for zero entries of T), and only non-zero entries give constraints.
    # Plus we need to handle the sparsity: for entries where T[a,b,c]=0, we need sum_l ... = 0.
    
    # Strategy: Use the fact that for small tensors, we can use numerical optimization
    # with random restarts, then verify with exact rational arithmetic.
    
    np.random.seed(rank * 7919)  # Deterministic seed based on rank
    
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None
        
        # Initialize with random values, but enforce that the decomposition
        # can produce the correct sparsity pattern.
        # Simple initialization: random normal
        U = np.random.randn(dimU, rank)
        V = np.random.randn(dimV, rank)
        W = np.random.randn(dimW, rank)
        
        # Scale to reasonable magnitudes
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        # ALS-style optimization: minimize ||T - U ⊗ V ⊗ W||_F
        # For each mode, solve a least-squares problem.
        converged = False
        max_iter = 100
        tol = 1e-12
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U: for each a, U[a,:] should satisfy
            # T[a,:,:] = sum_l U[a,l] * V[b,l] * W[c,l]
            # Flatten: T[a,:,:].reshape(-1) = U[a,:] @ (V * W).reshape(dimV*dimW)
            # Actually: T[a,b,c] = sum_l U[a,l] * (V[b,l] * W[c,l])
            # Let M_U[a] = V * W reshaped? No.
            # For fixed a: T[a,:,:] = sum_l U[a,l] * (V[:,l] * W[:,l])
            # So T[a].reshape(dimV*dimW) = U[a,:] @ (V * W) where V*W is outer product per column?
            # More precisely: let Z[l] = V[:,l] * W[:,l] (element-wise, shape dimV x dimW)
            # Then T[a] = sum_l U[a,l] * Z[l]
            # So T[a].reshape(-1) = U[a,:] @ Z.reshape(rank, dimV*dimW)
            
            # Build Z matrix: Z is (rank, dimV, dimW)
            # Z[l, b, c] = V[b, l] * W[c, l]
            Z = np.einsum('bl,cl->bcl', V, W)  # (dimV, dimW, rank)
            Z_mat = Z.transpose(2, 0, 1).reshape(rank, dimV * dimW)  # (rank, dimV*dimW)
            
            # For each a: T[a].reshape(-1) = U[a,:] @ Z_mat.T
            # So U[a,:] = T[a].reshape(-1) @ pinv(Z_mat.T)
            # Better: solve least squares for all a at once
            # T_flat[a, idx] = sum_l U[a,l] * Z_mat[l, idx]
            # T_flat @ Z_mat.T = U @ (Z_mat @ Z_mat.T)? No.
            # T_flat = U @ Z_mat
            # So U = T_flat @ pinv(Z_mat)
            
            T_flat = T.reshape(dimU, dimV * dimW)
            # Solve U @ Z_mat = T_flat
            U, _, _, _ = np.linalg.lstsq(Z_mat.T, T_flat.T, rcond=None)
            U = U.T  # Now U is (dimU, rank)
            
            # Update V: for each b, T[:,b,:].reshape(dimU*dimW) = sum_l V[b,l] * (U[:,l] * W[:,l])
            # Z_V[l] = U[:,l] * W[:,l] but U[:,l] has length dimU, W[:,l] has length dimW
            # Z_V[l, a, c] = U[a,l] * W[c,l]
            Z_V = np.einsum('al,cl->acl', U, W)  # (dimU, dimW, rank)
            Z_V_mat = Z_V.transpose(2, 0, 1).reshape(rank, dimU * dimW)
            
            T_flat_V = T.transpose(1, 0, 2).reshape(dimV, dimU * dimW)
            V, _, _, _ = np.linalg.lstsq(Z_V_mat.T, T_flat_V.T, rcond=None)
            V = V.T
            
            # Update W: for each c, T[:,:,c].reshape(dimU*dimW) = sum_l W[c,l] * (U[:,l] * V[:,l])
            Z_W = np.einsum('al,bl->abl', U, V)  # (dimU, dimV, rank)
            Z_W_mat = Z_W.transpose(2, 0, 1).reshape(rank, dimU * dimV)
            
            T_flat_W = T.transpose(2, 0, 1).reshape(dimW, dimU * dimV)
            W, _, _, _ = np.linalg.lstsq(Z_W_mat.T, T_flat_W.T, rcond=None)
            W = W.T
        
        # Check residual
        # Compute T_decomp = sum_l U[:,l] ⊗ V[:,l] ⊗ W[:,l]
        T_decomp = np.einsum('al,bl,cl->abc', U, V, W)
        residual = np.linalg.norm(T - T_decomp)
        
        if residual < 1e-10:
            # Now try to snap to integers or half-integers
            result = _snap_to_rationals(U, V, W, T)
            if result is not None:
                return result
    
    return None


def _snap_to_rationals(U, V, W, T):
    """Try to convert the floating-point decomposition to exact integer/half-integer entries."""
    # The problem states entries are integer or half-integer.
    # This means values are in 0.5 * Z.
    # We check if U, V, W are close to such values.
    
    def check_and_snap(mat, tol=1e-6):
        # Multiply by 2, round, divide by 2
        snapped = np.round(mat * 2) / 2
        # Check if the difference is small
        if np.allclose(mat, snapped, atol=tol):
            return snapped
        return None

    U_s = check_and_snap(U)
    V_s = check_and_snap(V)
    W_s = check_and_snap(W)
    
    if U_s is not None and V_s is not None and W_s is not None:
        # Verify the decomposition with snapped values
        T_check = np.einsum('al,bl,cl->abc', U_s, V_s, W_s)
        if np.allclose(T_check, T, atol=1e-6):
            def to_list(arr):
                return [[float(x) for x in row] for row in arr]
            return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
            
    return None
# EVOLVE-BLOCK-END
