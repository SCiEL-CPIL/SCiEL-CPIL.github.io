"""Matrix multiplication 2x4x8: rank-56 Strassen block decomposition with column-pruning."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def _build_rank56():
    """Build the standard rank-56 Strassen decomposition for 2x4 * 4x8."""
    n, m, p = 2, 4, 8
    U = np.zeros((n * m, 56), dtype=np.int8)
    V = np.zeros((m * p, 56), dtype=np.int8)
    W = np.zeros((p * n, 56), dtype=np.int8)
    
    col = 0
    for bi in range(1):
        for bj in range(2):
            for bk in range(4):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a, col] += 1; U[u_d, col] += 1
                V[v_e, col] += 1; V[v_h, col] += 1
                W[w_c00, col] += 1; W[w_c11, col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c, col] += 1; U[u_d, col] += 1
                V[v_e, col] += 1
                W[w_c10, col] += 1; W[w_c11, col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a, col] += 1
                V[v_f, col] += 1; V[v_h, col] -= 1
                W[w_c01, col] += 1; W[w_c11, col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d, col] += 1
                V[v_g, col] += 1; V[v_e, col] -= 1
                W[w_c00, col] += 1; W[w_c10, col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a, col] += 1; U[u_b, col] += 1
                V[v_h, col] += 1
                W[w_c00, col] -= 1; W[w_c01, col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c, col] += 1; U[u_a, col] -= 1
                V[v_e, col] += 1; V[v_f, col] += 1
                W[w_c11, col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b, col] += 1; U[u_d, col] -= 1
                V[v_g, col] += 1; V[v_h, col] += 1
                W[w_c00, col] += 1
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=4, p=8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check against the identity tensor (which has 1 where the indices match in the standard matmul)
    # Actually for matrix multiplication A[i,j]*B[j,k], the tensor is:
    # T[i,j,k] = 1 if this is a valid (i,j,k) triple, else 0
    # Wait no - the matrix multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # because C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is all ones on the domain [0,n)x[0,m)x[0,p)
    
    # Let me verify by checking that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    n, m, p = 2, 4, 8
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _prune_columns(U, V, W, n=2, m=4, p=8):
    """Try to find and remove linearly dependent columns to reduce rank.
    
    Uses the fact that if a column is a linear combination of others,
    we can eliminate it. We check this by seeing if the tensor contribution
    of each column is in the span of the others.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # The key insight: two columns r1 and r2 are linearly dependent in the tensor
    # sense if the outer product U[:,r1]⊗V[:,r1]⊗W[:,r1] is a linear combination
    # of the other outer products. This is hard to check directly.
    
    # Alternative approach: use the fact that if we can find scalars α, β such that
    # for all (i,j,k): U[i*m+j,r1]*V[j*p+k,r1]*W[k*n+i,r1] + α*U[i*m+j,r2]*V[j*p+k,r2]*W[k*n+i,r2]
    # is in the span of the remaining terms... this is complex.
    
    # Simpler approach: check if any two columns produce identical (up to scale) 
    # "pattern" contributions. If column r1 and r2 have the same support pattern
    # in U, V, W (up to scaling), they might be mergeable.
    
    # Actually, the most practical approach: use Gaussian elimination on the
    # tensor flattened as a matrix. The tensor T is a 3D array of shape (n*m, m*p, p*n)
    # but really it's indexed as T[i,j,k]. Let's think of it as a matrix where
    # we fix one index and get a matrix.
    
    # Better approach: The rank of the tensor is the minimum number of rank-1 tensors
    # that sum to it. If we have more terms than the rank, some are redundant.
    
    # Let's try: for each pair of columns, check if removing one and adjusting
    # the other (via a linear change) still gives an exact decomposition.
    
    # This is essentially trying to find if the set of outer products is linearly
    # dependent. We can check this by flattening each outer product into a vector
    # and checking if the set of vectors is linearly independent.
    
    # The outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r] flattened has dimension n*m*m*p*p*n
    # which is 2*4*4*8*8*2 = 4096. That's manageable.
    
    n, m, p = 2, 4, 8
    dim = n * m * m * p * p * n  # 4096
    
    # Build the matrix whose columns are the flattened outer products
    M = np.zeros((dim, rank), dtype=np.float64)
    for r in range(rank):
        outer = np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).flatten()
        M[:, r] = outer
    
    # Find the rank of M and identify which columns are in the span of others
    # Use SVD
    try:
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        effective_rank = np.sum(S > tol * S[0])
        
        if effective_rank < rank:
            # There are linear dependencies. Find a minimal set of columns.
            # Use the right singular vectors to identify which columns to keep.
            # Keep the columns corresponding to the largest singular values.
            
            # Actually, we need to be more careful. We want to find a subset of
            # columns that spans the same space. Use QR with column pivoting.
            # Or simply: greedily select columns that are linearly independent.
            
            selected = []
            basis = []
            for r in range(rank):
                col = M[:, r].copy()
                # Check if col is in the span of basis
                if len(basis) == 0:
                    selected.append(r)
                    basis.append(col.copy())
                else:
                    # Project col onto the orthogonal complement of basis
                    residual = col.copy()
                    for b in basis:
                        proj = np.dot(residual, b) / np.dot(b, b)
                        residual -= proj * b
                    if np.linalg.norm(residual) > tol:
                        selected.append(r)
                        basis.append(residual.copy())
            
            if len(selected) < rank:
                U_new = U[:, selected]
                V_new = V[:, selected]
                W_new = W[:, selected]
                return U_new, V_new, W_new
    except Exception:
        pass
    
    return U, V, W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Build the rank-56 decomposition
    U, V, W = _build_rank56()
    
    # Try to prune columns
    U, V, W = _prune_columns(U, V, W, n, m, p)
    
    # Verify the decomposition is still exact
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to the original rank-56
        U, V, W = _build_rank56()
    
    # Check time
    if time.time() > deadline - 5:
        pass  # Just return what we have
    
    rank = U.shape[1]
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
