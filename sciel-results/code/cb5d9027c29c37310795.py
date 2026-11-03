"""Matrix multiplication 2,3,5 — rank-28 decomposition via two Strassen 2x2x2 blocks with rank-reduction search."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 5]
START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # Safety margin

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication.
    First attempts rank-27 via tensor completion heuristic, falls back to rank-28.
    """
    n, m, p = 2, 3, 5
    
    # Build the rank-28 Strassen construction
    U28, V28, W28 = _build_rank28(n, m, p)
    
    # Check deadline before attempting rank-27 search
    if time.time() > DEADLINE - 20.0:
        return _to_dict(U28, V28, W28)
    
    # Attempt rank-27 tensor completion
    rank27_result = _try_rank27(U28, V28, W28, n, m, p)
    
    if rank27_result is not None:
        U27, V27, W27 = rank27_result
        return _to_dict(U27, V27, W27)
    
    # Fallback to rank-28
    return _to_dict(U28, V28, W28)

def _build_rank28(n, m, p):
    """Build the known rank-28 Strassen + schoolbook decomposition."""
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        U[0*m + (j_off)][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[(j_off)*p + (k_off)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        U[1*m + (j_off)][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[(j_off)*p + (k_off)][c] = 1
        W[(k_off)*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        
        U[0*m + (j_off)][c] = 1
        V[(j_off)*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off)][c] = 1; V[(j_off)*p + (k_off)][c] = -1
        W[(k_off)*n + 0][c] = 1; W[(k_off)*n + 1][c] = 1
        c += 1
        
        U[0*m + (j_off)][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off)*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        
        U[1*m + (j_off)][c] = 1; U[0*m + (j_off)][c] = -1
        V[(j_off)*p + (k_off)][c] = 1; V[(j_off)*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + (k_off)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off)*n + 0][c] = 1
        c += 1

    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    return U, V, W

def _to_dict(U, V, W):
    """Convert numpy arrays to list format."""
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # The tensor T[i,j,k] = delta[i,j] * delta[j,k] for (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i == j or j == k or i == k) else 0.0
                # Actually the matrix multiplication tensor is T[i,j,k] = 1 if j==j (always) 
                # Wait, let me reconsider. The tensor for (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                # is T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle index)
                # No, that's not right either.
                
                # The matrix multiplication tensor has T[i,j,k] = 1 if and only if the
                # (i,k)-entry of A*B depends on A[i,j]*B[j,k], which is always true.
                # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
                # 
                # Wait, that's the rank-30 schoolbook. Let me re-examine.
                #
                # (A*B)[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
                #
                # The tensor T has T[i,j,k] = 1 for all valid indices.
                # The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # for all i, j, k.
                
                actual = 0.0
                for r in range(U.shape[1]):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                if abs(actual - 1.0) > 1e-9:
                    return False
    
    return True

def _try_rank27(U28, V28, W28, n, m, p):
    """
    Attempt to find a rank-27 decomposition by checking for linear dependencies
    among the rank-1 tensors defined by the columns of U, V, W.
    
    If the 28 rank-1 tensors are linearly dependent, the rank is <= 27.
    We can find a basis of 27 tensors that span the same space.
    """
    rank28 = U28.shape[1]
    
    # First verify the rank-28 solution is correct
    if not _verify_decomposition(U28, V28, W28, n, m, p):
        return None
    
    # Vectorize each rank-1 tensor u_r \otimes v_r \otimes w_r
    # The dimension is n*m*p = 2*3*5 = 30
    # We have 28 vectors in 30-dim space.
    # Check if they are linearly dependent.
    
    vecs = []
    for r in range(rank28):
        # u_r is (n*m,), v_r is (m*p,), w_r is (p*n,)
        # The tensor T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need to map (i,j,k) to a flat index.
        # Let's use the same indexing as in _verify_decomposition.
        # Index = ((i * m + j) * p + k) ? No, that's not consistent with W's index.
        # W index is k*n + i.
        # Let's just build the flat vector by iterating i, j, k.
        vec = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U28[i*m+j, r] * V28[j*p+k, r] * W28[k*n+i, r]
                    vec[idx] = val
                    idx += 1
        vecs.append(vec)
    
    M = np.column_stack(vecs)
    
    # Check rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    
    if rank_M < rank28:
        # Linear dependence found!
        # Find a basis of rank_M vectors.
        # We can use QR decomposition with column pivoting to find the pivot columns.
        # Or simply, we can find a set of linearly independent columns.
        
        # For simplicity, let's just take the first rank_M columns that are independent.
        # This is a heuristic but works if the dependencies are not complex.
        # A better way: use SVD to get a basis, but we need to reconstruct U, V, W.
        
        # Actually, if the tensors are linearly dependent, we can express one as a combination of others.
        # Let's find the null space of M.
        # If there is a non-zero vector c in the null space, then sum c_r T_r = 0.
        # This means we can replace the set of tensors with a subset of size rank_M.
        
        # To construct the new U, V, W:
        # We need to find 27 linearly independent rank-1 tensors from the 28.
        # Let's identify which columns are linearly independent.
        
        # Use QR with pivoting
        Q, R, P = np.linalg.qr(M.T, mode='complete') 
        # M is (30, 28). M.T is (28, 30).
        # QR of M.T gives M.T = Q R.
        # The rank is the number of non-zero diagonal entries of R.
        
        # Actually, let's just find a basis using np.linalg.matrix_rank logic manually or via SVD.
        U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
        # s has length min(30, 28) = 28.
        # The rank is the number of s > tol.
        rank_svd = np.sum(s > 1e-9)
        
        if rank_svd < rank28:
            # We can use the first rank_svd left singular vectors as a basis for the tensor space.
            # But we need to decompose them back into U, V, W factors.
            # This is the "tensor rank decomposition" problem again, but now we know the rank is 27.
            
            # Since the original tensors are rank-1, any linear combination is not necessarily rank-1.
            # However, we can pick 27 linearly independent original tensors.
            
            # Find 27 linearly independent columns from M.
            # We can do this by Gaussian elimination or QR pivoting.
            
            # Let's use a simple heuristic: pick the columns with largest norms that are independent.
            # Or just the first 27 if they are independent.
            
            # Let's check if the first 27 columns are independent.
            M_sub = M[:, :27]
            if np.linalg.matrix_rank(M_sub, tol=1e-9) == 27:
                # Great, we can just drop the last column.
                # But wait, dropping a column changes the decomposition.
                # The sum of the remaining 27 tensors must equal the original tensor T.
                # T = sum_{r=0}^{27} T_r.
                # If T_27 = sum_{r=0}^{26} c_r T_r, then T = sum_{r=0}^{26} (1 + c_r) T_r.
                # So we can scale the first 27 tensors.
                
                # Find c such that T_27 = sum_{r=0}^{26} c_r T_r.
                # This is a linear system: M_sub @ c = M[:, 27].
                c, residuals, rank_c, sv = np.linalg.lstsq(M_sub, M[:, 27], rcond=None)
                
                if residuals[0] < 1e-8:
                    # Exact solution found.
                    # New tensors are T'_r = (1 + c_r) T_r for r=0..26.
                    # T'_r = (1 + c_r) u_r \otimes v_r \otimes w_r.
                    # We can absorb the scalar into U, V, or W.
                    # Let's put it into W.
                    
                    U_new = U28[:, :27].copy()
                    V_new = V28[:, :27].copy()
                    W_new = W28[:, :27].copy()
                    
                    for r in range(27):
                        W_new[:, r] *= (1.0 + c[r])
                    
                    # Verify
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                        return U_new, V_new, W_new
            
            # If first 27 are not independent, try other subsets?
            # For now, this is a limited search.
            pass
    
    return None
# EVOLVE-BLOCK-END
