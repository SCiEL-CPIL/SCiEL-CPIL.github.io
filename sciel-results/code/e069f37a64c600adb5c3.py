import time
import numpy as np
from scipy.linalg import svd

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        # m1 = (a+d)(e+h)
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h)
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e)
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f)
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h)
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    def build_decomposition():
        U_t = []
        V_t = []
        W_t = []
        
        # Strassen blocks for A[0:2, 0:2] * B[0:2, c:c+2] for c in 0, 2, 4, 6
        for c in range(0, 8, 2):
            strassen_decomp(0, 0, 0, c, U_t, V_t, W_t)
            
        # Direct terms for remaining A indices
        # A[0:2, 2] * B[2, 0:8]
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k, U_t, V_t, W_t)
                
        # A[2, 0:2] * B[0:2, 0:8]
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k, U_t, V_t, W_t)
                
        # A[2, 2] * B[2, 0:8]
        for k in range(8):
            direct_term(2, 2, k, U_t, V_t, W_t)
            
        rank = len(W_t)
        
        # Build matrices
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i, idx] = coeff
                
        return U, V, W

    def to_nice_list(mat):
        res = []
        for row in mat:
            nrow = []
            for v in row:
                if abs(v - round(v*2)/2) < 1e-9:
                    nv = round(v*2)/2
                    if abs(nv - round(nv)) < 1e-9:
                        nrow.append(int(round(nv)))
                    else:
                        nrow.append(nv)
                else:
                    nrow.append(v)
            res.append(nrow)
        return res

    # Initial decomposition
    U, V, W = build_decomposition()
    rank = W.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U, V, W

    # Attempt to reduce rank using SVD on the tensor matricization
    # The tensor T is in R^(n*m) x R^(m*p) x R^(p*n)
    # We can flatten it into a matrix of size (n*m) x (m*p * p*n) or similar.
    # However, the standard way to find linear dependencies in a decomposition is to
    # view the decomposition terms as columns in a high-dimensional space.
    # Let's try flattening the tensor into a vector of size (n*m * m*p * p*n) is too big (5184).
    # Instead, we can use the fact that the decomposition is T = sum U_i V_i W_i.
    # We can form a matrix M where each column is the "coefficient vector" of a term in a specific basis?
    # No, the terms are rank-1 tensors.
    # A better approach:
    # We want to find if the columns of U, V, W are linearly dependent in a way that allows rank reduction.
    # This is equivalent to finding the rank of the tensor.
    # We can perform SVD on the matricization of the tensor.
    # Let's reshape the tensor T (which we can reconstruct from U, V, W) into a matrix.
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Note: indices in U are (i, j), in V are (j, k), in W are (k, i) ?
    # Wait, the problem statement says:
    # U: (n*m, rank) -> indices (i, j)
    # V: (m*p, rank) -> indices (j, k)
    # W: (p*n, rank) -> indices (k, i) -- wait, "W[k*n+i,r]" means row index is k*n+i.
    # So W index is (k, i).
    # T_{i, j, k} = A_{i, j} B_{j, k} = C_{i, k}.
    # The decomposition is sum_r U[i, j, r] V[j, k, r] W[k, i, r] ?
    # No, the standard tensor decomposition for matmul is:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}
    # The tensor is T_{i, j, k} where T_{i, j, k} = 1 if we are computing C_{i, k} from A_{i, j} and B_{j, k}.
    # Actually, the "multiplication tensor" is defined such that <T, A, B> = <C, I>.
    # T_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in C_{i, k}.
    # So T_{i, j, k} = 1 if the indices match, 0 otherwise?
    # No, T is a 3-tensor of size (n*m) x (m*p) x (p*n)?
    # Let's check the shapes.
    # U: (n*m, rank). Row index (i, j).
    # V: (m*p, rank). Row index (j, k).
    # W: (p*n, rank). Row index (k, i).
    # The term r contributes U[i, j, r] * V[j, k, r] * W[k, i, r] to the "computation".
    # This sum must equal 1 if the indices are valid for the multiplication, 0 otherwise?
    # Actually, the condition is:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all valid (i, j, k) ?
    # No, the multiplication is C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # This can be written as C_{i, k} = sum_{j} A_{i, j} B_{j, k} * 1.
    # So the tensor T_{i, j, k} has T_{i, j, k} = 1 for all i, j, k?
    # No, that would mean C_{i, k} = sum_{i, j, k} A_{i, j} B_{j, k} which is wrong.
    # The tensor T is defined on indices (i, j) for A, (j, k) for B, and (i, k) for C.
    # T_{(i, j), (j, k), (i, k)} = 1.
    # And 0 otherwise? But (i, j) and (j, k) are linked by j.
    # In the tensor space R^{(n*m) x (m*p) x (p*n)}, the basis elements are e_{(i,j)} \otimes e_{(j',k)} \otimes e_{(k',i')}.
    # The multiplication tensor is sum_{i, j, k} e_{(i,j)} \otimes e_{(j,k)} \otimes e_{(k,i)}.
    # So T_{(i,j), (j',k), (k',i')} = 1 if j=j' and k=k' and i=i', else 0.
    # This is a sparse tensor.
    
    # We can check for linear dependencies in the decomposition by forming a matrix.
    # Let's flatten the tensor T into a vector of length (n*m * m*p * p*n) = 9 * 24 * 24 = 5184.
    # The current decomposition has `rank` terms.
    # We can form a matrix M of size (5184, rank) where each column is the vectorized rank-1 tensor.
    # Then we can compute the SVD of M.
    # The rank of M is the number of linearly independent terms.
    # If the rank of M is less than `rank`, we can reduce the number of terms.
    
    # However, 5184 is small enough to handle.
    # Let's try it.
    
    if time.time() < DEADLINE:
        dim = n * m * m * p * p * n
        M = np.zeros((dim, rank))
        
        for r in range(rank):
            # Vectorize the rank-1 tensor U[:, r] \otimes V[:, r] \otimes W[:, r]
            # U[:, r] is size n*m
            # V[:, r] is size m*p
            # W[:, r] is size p*n
            # The Kronecker product size is (n*m) * (m*p) * (p*n)
            vec = np.kron(np.kron(U[:, r], V[:, r]), W[:, r])
            M[:, r] = vec
            
        # Compute SVD
        # M is 5184 x 68.
        # We want to find the rank of M.
        # We can use SVD to find the smallest singular value.
        # If there are small singular values, we can remove those columns?
        # No, we need to find a basis for the column space.
        
        # Let's compute the rank using SVD.
        U_svd, s, Vt = svd(M, full_matrices=False)
        
        # Determine the numerical rank
        # Since the entries are integers/half-integers, the singular values should be well-separated if independent.
        # But we might have exact dependencies.
        # Let's check the singular values.
        
        # If the rank is less than `rank`, we can project the columns onto the subspace spanned by the first `new_rank` left singular vectors.
        # Actually, we can just take the first `new_rank` columns of Vt (which are the right singular vectors) ?
        # No, the columns of M are the terms.
        # If M = U_svd s Vt, then M = sum s_i u_i v_i^T.
        # The columns of M are linear combinations of u_i.
        # We can represent the original columns as M = U_svd[:, :new_rank] * (s[:new_rank] * Vt[:new_rank, :]) ?
        # This is getting complicated.
        
        # Simpler approach:
        # If the rank is r < rank, we can find a set of r linearly independent columns.
        # But we need to express the other columns as linear combinations.
        # This requires solving a linear system.
        
        # Let's just check if the rank is actually lower.
        # If it is, we can try to find a better decomposition by using the null space.
        
        # For now, let's just return the initial decomposition if no improvement is found.
        # The SVD might be expensive or not yield an integer decomposition.
        
        # Let's check the singular values to see if there's a clear gap.
        # If s[new_rank-1] is very small, we can drop it.
        
        # But we must ensure the resulting decomposition is exact.
        # Dropping a term with a small singular value will introduce error.
        # We need an exact reduction.
        
        # If there is an exact linear dependency, one of the singular values will be 0.
        # Due to floating point, it might be ~1e-15.
        
        # Let's assume the initial decomposition is already minimal or close to it.
        # The Strassen-based decomposition is known to be good.
        
        pass

    return {"U": to_nice_list(best_U), "V": to_nice_list(best_V), "W": to_nice_list(best_W)}
# EVOLVE-BLOCK-END
