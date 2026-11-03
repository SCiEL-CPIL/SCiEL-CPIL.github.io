"""Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication
via rank-revealing QR with column pivoting on the schoolbook coefficient matrix."""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 130.0

def construct():
    n, m, p = 3, 4, 8
    dim_U = n * m   # 12
    dim_V = m * p   # 32
    dim_W = p * n   # 24
    schoolbook_rank = n * m * p  # 96

    # Build the 3-mode tensor T as a 3D array indexed by (U_idx, V_idx, W_idx)
    # T[u, v, w] = 1 iff the corresponding (i,j,k) triple is valid for matrix mult
    # U index: u = i*m + j  (i in [0,n), j in [0,m))
    # V index: v = j*p + k  (j in [0,m), k in [0,p))
    # W index: w = k*n + i  (k in [0,p), i in [0,n))
    # For valid (i,j,k): u = i*m+j, v = j*p+k, w = k*n+i
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Construct coefficient matrix M of shape (dim_U * dim_W, schoolbook_rank)
    # Column c corresponds to schoolbook term (i_c, j_c, k_c) where
    # c = i*m*p + j*p + k (standard enumeration)
    # M[u*dim_W + w, c] = 1 iff T[u, v_c, w] = 1 for the v_c associated with column c
    # For column c = i*m*p + j*p + k: v_c = j*p + k
    # M[u*dim_W + w, c] = T[u, v_c, w]
    M = np.zeros((dim_U * dim_W, schoolbook_rank), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                v_c = j * p + k
                # For this column, M[u*dim_W + w, c] = T[u, v_c, w]
                # T[u, v_c, w] = 1 only when u = i*m+j and w = k*n+i
                # But we need to check all (u,w) pairs
                # Actually T[u, v_c, w] is nonzero only at specific (u,w)
                # For v_c = j*p+k, T[u, v_c, w] = 1 iff u corresponds to (i',j) with j'=j
                # and w corresponds to (k,i') with k'=k, and i'=i
                # So T[u, v_c, w] = 1 iff u = i*m+j and w = k*n+i
                # This means for column c, only one entry per (u,w) grid is 1
                # Specifically at (u, w) = (i*m+j, k*n+i)
                u_star = i * m + j
                w_star = k * n + i
                M[u_star * dim_W + w_star, c] = 1.0

    # Now M has exactly one 1 per column (at position determined by (i,j,k))
    # The tensor decomposition requires:
    #   T[u, v, w] = sum_c M_uw[c] * U[u,c] * V[v,c] * W[w,c]
    # For schoolbook: U[u,c] = 1 if u = i*m+j else 0, V[v,c] = 1 if v=j*p+k else 0, W[w,c] = 1 if w=k*n+i else 0
    # So T[u,v,w] = sum_c delta(u, i_c*m+j_c) * delta(v, j_c*p+k_c) * delta(w, k_c*n+i_c) * M_uw[c]
    # This works because for each (u,v,w), exactly one schoolbook term has all three deltas = 1.
    #
    # For a general decomposition with U, V, W matrices:
    #   T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # We need to find U, V, W such that this holds.
    #
    # Key insight: If we fix U and W, then for each (u,w) pair,
    #   T[u,v,w] as a function of v must equal sum_r U[u,r]*W[w,r]*V[v,r]
    # which is a linear combination of the rows of V (weighted by U[u,r]*W[w,r]).
    #
    # Alternative approach: Use the fact that the tensor can be sliced.
    # For fixed v (i.e., fixed (j,k)), T[:,:,v] is a matrix of shape (dim_U, dim_W).
    # The tensor rank equals the minimum number of rank-1 matrices whose sum gives T.
    #
    # Let's use a different formulation. The tensor T can be viewed as a set of
    # "slices". For each v in [0, dim_V), the slice T[:,:,v] is a dim_U x dim_W matrix.
    # T[:,:,v] = sum_r (U[:,r] * W[:,r]^T) * V[v,r]
    # So T[:,:,v] is a linear combination of the rank-1 matrices U[:,r]*W[:,r]^T,
    # with coefficients V[v,r].
    #
    # We can set up: let P = U * W^T (outer product, shape dim_U x dim_W).
    # Then T[:,:,v] = P * diag(V[v,:])... no, that's not right either.
    #
    # Actually: T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # Let's fix v. Then T[u,w,v] = sum_r (U[u,r] * W[w,r]) * V[v,r]
    # = (U * diag(V[v,:]) * W^T)[u,w]... 
    # Actually T[:,:,v] (as a matrix with rows indexed by u, cols by w) = 
    #   sum_r U[:,r] * V[v,r] * W[:,r]^T
    # = U * diag(V[v,:]) * W^T
    #
    # So the slice T[:,:,v] must factor as U * diag(V[v,:]) * W^T.
    #
    # Let A_v = T[:,:,v] (dim_U x dim_W matrix for each v).
    # We need A_v = U * diag(V[v,:]) * W^T for all v.
    #
    # This is equivalent to: A_v * W = U * V[v,:]^T (as a vector in R^{dim_U})... 
    # Hmm, let me think differently.
    #
    # A_v = U * D_v * W^T where D_v = diag(V[v,:]).
    # This means A_v = sum_r V[v,r] * (U[:,r] * W[:,r]^T)
    #
    # So all slices A_v lie in the span of the rank-1 matrices R_r = U[:,r] * W[:,r]^T.
    # The minimum number of such rank-1 matrices needed to span all A_v gives the rank.
    #
    # Equivalently, vectorize: vec(A_v) = sum_r V[v,r] * vec(R_r) = sum_r V[v,r] * (W[:,r] ⊗ U[:,r])
    # So vec(A_v) lies in the column space of the matrix [vec(R_1), ..., vec(R_rank)].
    #
    # The minimum rank is the dimension of the span of {vec(A_v) : v = 0, ..., dim_V-1}.
    #
    # So: construct matrix S of shape (dim_U * dim_W, dim_V) where S[:,v] = vec(T[:,:,v])
    # The rank of S gives the minimum number of rank-1 tensors needed!
    #
    # Wait, that's the "slice rank" or the rank of the tensor when viewed as a matrix.
    # This is actually the correct rank for this type of decomposition!
    # Because: T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    # If we fix v, the map (u,w) -> T[u,v,w] is a bilinear form.
    # The collection of all such bilinear forms (one for each v) must be a linear combination
    # of rank-1 bilinear forms.
    # The number of rank-1 bilinear forms needed = rank of the "slice matrix".
    
    # Construct S: S[:, v] = vec(T[:,:,v]) for v = 0..dim_V-1
    S = np.zeros((dim_U * dim_W, dim_V), dtype=np.float64)
    for v in range(dim_V):
        # v = j*p + k
        j = v // p
        k = v % p
        # T[:,:,v] is a dim_U x dim_W matrix
        # T[u, v, w] = 1 iff u = i*m+j, w = k*n+i for some i
        # So for this v, T[u,v,w] = 1 at (u,w) = (i*m+j, k*n+i) for i = 0..n-1
        for i in range(n):
            u = i * m + j
            w = k * n + i
            S[u * dim_W + w, v] = 1.0
    
    # Now find the rank of S using rank-revealing QR with column pivoting
    # S is (dim_U*dim_W) x dim_V = 288 x 32
    # The rank of S tells us the minimum number of rank-1 tensors needed.
    
    # Perform QR with column pivoting on S
    # scipy.linalg.qr with pivoting, or use numpy's lstsq approach
    try:
        from scipy.linalg import qr as scipy_qr
        Q, R, piv = scipy_qr(S, pivoting=True)
        # The rank is the number of significant diagonal entries of R
        # R is upper triangular (after permuting columns by piv)
        # Check diagonal of R
        diag_R = np.abs(np.diag(R))
        # Determine rank: number of diagonal entries above tolerance
        tol = np.max(diag_R) * S.shape[0] * np.finfo(np.float64).eps * 100
        if len(diag_R) == 0:
            rank = 0
        else:
            rank = int(np.sum(diag_R > tol))
    except ImportError:
        # Fallback: use SVD
        U_svd, s_vals, Vt = np.linalg.svd(S, full_matrices=False)
        tol = np.max(s_vals) * S.shape[0] * np.finfo(np.float64).eps * 100
        rank = int(np.sum(s_vals > tol))
        piv = np.arange(S.shape[1])
    
    # rank is the minimum rank of the tensor decomposition
    # Now we need to construct actual U, V, W matrices.
    #
    # From the QR decomposition: S = Q * R (with column pivoting: S[:,piv] = Q * R)
    # S[:,v] = sum_r Q[:,r] * R[r,v] for r < rank (approximately, if rank < dim_V)
    # Actually S[:,piv] = Q*R, so S[:,v] = sum_r Q[:,piv_r] * R[r,v] where piv_r is the r-th pivot column.
    # 
    # We have: vec(T[:,:,v]) = sum_r V[v,r] * vec(R_r)
    # where R_r = U[:,r] * W[:,r]^T
    # So we need to identify U and W from the structure.
    #
    # From S = Q * R (with pivoting), we have S[:, v] = Q[:, piv[v]] * 1... no.
    # S[:, piv] = Q * R means S[:, piv[j]] = sum_i Q[:,i] * R[i, j] for all j.
    # 
    # Let's reindex: for the pivoted columns, let's say the j-th pivot column is column piv[j].
    # Then S[:, piv[j]] = sum_{i=0}^{rank-1} Q[:, i] * R[i, j]
    # So for v = piv[j]: vec(T[:,:,v]) = sum_i Q[:,i] * R[i,j]
    # This means T[:,:,v] (reshaped) = sum_i (Q_reshaped[:,i]) * R[i,j]
    # where Q_reshaped is Q reshaped to (dim_U, dim_W, rank).
    #
    # So T[:,:,v] = sum_i R[i,j] * (Q_reshaped[:,:,i])
    # And T[:,:,v] = sum_r U[:,r] * V[v,r] * W[:,r]^T
    #
    # Matching: we can set W[:,r] and U[:,r] from Q_reshaped[:,:,r], and V from R.
    # Specifically: Q_reshaped[:,:,r] = U[:,r] * W[:,r]^T (rank-1 matrix)
    # And V[v,r] = R[r, j] where v = piv[j]... but we need V for ALL v, not just pivoted ones.
    #
    # Actually, since S has rank `rank`, we can use the first `rank` columns of Q as the basis.
    # S[:,v] = sum_{r=0}^{rank-1} Q[:,r] * (Q^T S)[:,v,r]... 
    # Let's just use: S ≈ Q_rank * R_rank where Q_rank = Q[:,:rank], R_rank = R[:rank,:]*something
    # 
    # With pivoting: S[:,piv] = Q*R, so S[:,piv[j]] = Q*R[:,j]
    # For non-pivot columns, S[:,v] = Q*R[:,j] where j is such that piv[j] = v (if v is in piv).
    # Since piv is a permutation, every v appears exactly once in piv.
    # So: let inv_piv[v] = j such that piv[j] = v.
    # Then S[:,v] = Q*R[:,inv_piv[v]]
    #
    # So: vec(T[:,:,v]) = sum_r Q[:,r] * R[r, inv_piv[v]]
    # This means: T[:,:,v] = sum_r (Q_reshaped[:,:,r]) * R[r, inv_piv[v]]
    #
    # Now I need to factor Q_reshaped[:,:,r] = U[:,r] * W[:,r]^T.
    # Q_reshaped[:,:,r] is a dim_U x dim_W matrix. I can take its SVD or just pick
    # U[:,r] = Q_reshaped[:,:,r][:,0] and W[:,r] = Q_reshaped[:,:,r][0,:] / Q_reshaped[:,:,r][0,0]
    # but that only works if the matrix is truly rank 1.
    #
    # Actually, Q_reshaped[:,:,r] may not be rank 1! The QR gives us a basis for the column
    # space of S, but each basis vector vec(Q_reshaped[:,:,r]) doesn't have to be a Kronecker
    # product (rank-1 matrix).
    #
    # I need a different approach to extract U and W. Let me use the fact that
    # T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r].
    #
    # From the slice perspective: T[:,:,v] = U * diag(V[v,:]) * W^T
    # Let's pick specific v values to determine U and W.
    #
    # Alternatively, I can set up a linear system. Let's choose V first (from R), then solve for U and W.
    #
    # Let me use a cleaner approach:
    # 1. Compute the rank-r basis of the slice space.
    # 2. Express each slice as a linear combination.
    # 3. Factor each basis matrix into U[:,r] and W[:,r].
    
    # Let's use the SVD of S for cleaner factorization
    U_svd, s_vals, Vt = np.linalg.svd(S, full_matrices=False)
    tol = np.max(s_vals) * max(S.shape) * np.finfo(np.float64).eps * 100
    rank = int(np.sum(s_vals > tol))
    
    if rank == 0:
        # Degenerate case, shouldn't happen
        rank = schoolbook_rank
        U = [[0]*rank for _ in range(dim_U)]
        V = [[0]*rank for _ in range(dim_V)]
        W = [[0]*rank for _ in range(dim_W)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i*m*p + j*p + k
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
        return {"U": U, "V": V, "W": W}
    
    # S = U_svd @ diag(s_vals) @ Vt
    # S[:,v] = sum_r U_svd[:,r] * s_vals[r] * Vt[r,v]
    # vec(T[:,:,v]) = sum_r (U_svd[:,r] * s_vals[r]) * Vt[r,v]
    # Let's define: B_r = U_svd[:,r] * s_vals[r] (a vector in R^{dim_U*dim_W})
    # Then vec(T[:,:,v]) = sum_r B_r * Vt[r,v]
    # T[:,:,v] = sum_r (B_r reshaped to dim_U x dim_W) * Vt[r,v]
    
    # Now I need to factor each B_r reshaped as U[:,r] * W[:,r]^T.
    # B_r reshaped is a dim_U x dim_W matrix. It may not be exactly rank 1,
    # but since it comes from the SVD of a structured matrix, it should be close.
    # Let me project it to the nearest rank-1 matrix via SVD.
    
    U_mat = np.zeros((dim_U, rank), dtype=np.float64)
    W_mat = np.zeros((dim_W, rank), dtype=np.float64)
    V_mat = np.zeros((dim_V, rank), dtype=np.float64)
    
    for r in range(rank):
        B_r = (U_svd[:, r] * s_vals[r]).reshape(dim_U, dim_W)
        # Project to nearest rank-1: SVD of B_r
        U_br, s_br, Vt_br = np.linalg.svd(B_r, full_matrices=False)
        if len(s_br) > 0 and s_br[0] > 1e-15:
            U_mat[:, r] = U_br[:, 0] * np.sqrt(s_br[0])
            W_mat[:, r] = Vt_br[0, :] * np.sqrt(s_br[0])
        else:
            U_mat[:, r] = 0
            W_mat[:, r] = 0
        V_mat[:, r] = Vt[r, :]
    
    # Verify: reconstruct T and check
    # T_recon[u,v,w] = sum_r U_mat[u,r] * V_mat[v,r] * W_mat[w,r]
    # Let's check a few entries
    # Actually, let's check the whole thing
    T_recon = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for r in range(rank):
        T_recon += np.einsum('u,v,w->uvw', U_mat[:, r], V_mat[:, r], W_mat[:, r])
    
    max_err = np.max(np.abs(T_recon - T))
    
    # If reconstruction is not accurate enough, increase rank
    if max_err > 1e-10:
        # Try increasing rank
        for try_rank in range(rank+1, min(dim_V, rank+10)+1):
            tol2 = np.max(s_vals) * max(S.shape) * np.finfo(np.float64).eps * 10
            if try_rank <= len(s_vals) and s_vals[try_rank-1] > 1e-12:
                pass
            # Use exactly try_rank components
            U_mat2 = np.zeros((dim_U, try_rank), dtype=np.float64)
            W_mat2 = np.zeros((dim_W, try_rank), dtype=np.float64)
            V_mat2 = np.zeros((dim_V, try_rank), dtype=np.float64)
            for r2 in range(try_rank):
                B_r2 = (U_svd[:, r2] * s_vals[r2]).reshape(dim_U, dim_W)
                U_br2, s_br2, Vt_br2 = np.linalg.svd(B_r2, full_matrices=False)
                if len(s_br2) > 0 and s_br2[0] > 1e-15:
                    U_mat2[:, r2] = U_br2[:, 0] * np.sqrt(s_br2[0])
                    W_mat2[:, r2] = Vt_br2[0, :] * np.sqrt(s_br2[0])
                V_mat2[:, r2] = Vt[r2, :]
            T_recon2 = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
            for r2 in range(try_rank):
                T_recon2 += np.einsum('u,v,w->uvw', U_mat2[:, r2], V_mat2[:, r2], W_mat2[:, r2])
            max_err2 = np.max(np.abs(T_recon2 - T))
            if max_err2 < 1e-10:
                rank = try_rank
                U_mat, W_mat, V_mat = U_mat2, W_mat2, V_mat2
                max_err = max_err2
                break
    
    # If still not accurate, fall back to schoolbook
    if max_err > 1e-8:
        rank = schoolbook_rank
        U_mat = np.zeros((dim_U, rank), dtype=np.float64)
        V_mat = np.zeros((dim_V, rank), dtype=np.float64)
        W_mat = np.zeros((dim_W, rank), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U_mat[i * m + j, c] = 1.0
                    V_mat[j * p + k, c] = 1.0
                    W_mat[k * n + i, c] = 1.0
    
    # Convert to lists of integers or half-integers
    # Check if entries are close to integers or half-integers
    def to_fraction_list(mat):
        """Convert numpy matrix to list of lists with integer/half-integer values."""
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                half_int = round(val * 2) / 2.0
                if abs(val - half_int) < 1e-9:
                    row_list.append(half_int)
                else:
                    # Try to find a simple fraction
                    # For now, round to nearest integer
                    nearest_int = round(val)
                    if abs(val - nearest_int) < 1e-6:
                        row_list.append(nearest_int)
                    else:
                        # Use the value as-is (shouldn't happen for exact decomposition)
                        row_list.append(val)
            result.append(row_list)
        return result
    
    U_list = to_fraction_list(U_mat)
    V_list = to_fraction_list(V_mat)
    W_list = to_fraction_list(W_mat)
    
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback to schoolbook
        rank = schoolbook_rank
        U_fb = [[0]*rank for _ in range(dim_U)]
        V_fb = [[0]*rank for _ in range(dim_V)]
        W_fb = [[0]*rank for _ in range(dim_W)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i*m*p + j*p + k
                    U_fb[i*m+j][c] = 1
                    V_fb[j*p+k][c] = 1
                    W_fb[k*n+i][c] = 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_list, "V": V_list, "W": W_list}
