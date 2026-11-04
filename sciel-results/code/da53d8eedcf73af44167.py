"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_computed = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_arr[:, c].reshape(n, m)
        v_c = V_arr[:, c].reshape(m, p)
        w_c = W_arr[:, c].reshape(p, n)
        T_computed += u_c[:, :, None] * v_c[:, None, :] * w_c[None, :, :]
    
    T_expected = np.ones((n, m, p))
    return np.allclose(T_computed, T_expected, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    t0 = time.time()
    deadline = t0 + 130  # Safety margin
    
    # Build a structured decomposition using Strassen for 2x2x2 blocks
    # and schoolbook for 1x2x2 blocks.
    
    # Initialize with schoolbook as fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now try the structured Strassen-based approach
    # Block structure:
    # i-blocks: [0,1] (size 2), [2] (size 1)
    # j-blocks: [0,1] (size 2), [2,3] (size 2)
    # k-blocks: [0,1] (size 2), [2,3] (size 2), [4,5] (size 2)
    
    U_terms = []  # list of (index, coeff) for each term
    V_terms = []
    W_terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term to the decomposition."""
        U_terms.append(u_dict)
        V_terms.append(v_dict)
        W_terms.append(w_dict)
    
    # Strassen decomposition for 2x2x2 block at (i0, j0, k0)
    def strassen_2x2x2(i0, j0, k0):
        """Add 7 Strassen terms for the 2x2x2 sub-block."""
        # M1 = (A[i0,j0]+A[i0+1,j0]) * (B[j0,k0]+B[j0+1,k0])
        # M2 = (A[i0,j0+1]+A[i0+1,j0+1]) * B[j0,k0]
        # M3 = A[i0,j0] * (B[j0+1,k0]-B[j0,k0])
        # M4 = A[i0+1,j0] * (B[j0+1,k0+1]-B[j0,k0])
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        # M6 = (A[i0+1,j0+1]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        # M7 = (A[i0+1,j0+1]-A[i0+1,j0]) * (B[j0+1,k0+1]-B[j0+1,k0])
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M2 + M4
        # C10 = M3 + M5
        # C11 = M1 - M3 + M5 - M6
        
        # Each M_r is a product of linear forms in A and B.
        # In tensor decomposition, each M_r corresponds to one term where:
        # U has the A-side coefficients, V has the B-side coefficients,
        # W has the C-side coefficients (which determine how it contributes to C).
        
        # But wait - in the tensor decomposition, we need:
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
        
        # The Strassen algorithm computes C = AB using 7 products.
        # Each product M_r = (sum a_ij A[i,j]) * (sum b_jk B[j,k])
        # contributes to C[i,k] = sum_j A[i,j] B[j,k]
        
        # In the tensor decomposition, for a single M_r:
        # U[i,j] = a_ij (coefficient of A[i,j] in the first factor)
        # V[j,k] = b_jk (coefficient of B[j,k] in the second factor)  
        # W[k,i] = c_ki (coefficient of C[i,k] in the output)
        
        # The contribution to the tensor from this term is:
        # sum_{i,j,k} U[i,j] V[j,k] W[k,i] x_i y_j z_k
        # where x, y, z are the "input" vectors.
        
        # For the tensor to equal the matmul tensor, we need:
        # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k] = 1
        
        # So W[k,i,c] is NOT the coefficient of C[i,k] in the output.
        # Instead, the decomposition is:
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        
        # This is a different formulation. Let me reconsider.
        
        # The matrix multiplication tensor is:
        # T[i,j,k] = 1 (for all valid i, j, k)
        # because (AB)[i,k] = sum_j A[i,j] B[j,k], and the coefficient of A[i,j]B[j,k] is 1.
        
        # So we need: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
        
        # This is equivalent to finding a rank-r decomposition of the all-ones tensor.
        # The all-ones tensor of size 3x4x6 has a specific rank.
        
        # Actually, I think I've been confusing two different formulations.
        # Let me re-read the task.
        
        # "exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 6 matrix"
        # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # "W[k*n+i,r]"
        
        # So the decomposition is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        # This is indeed the all-ones tensor decomposition.
        
        # For the all-ones tensor, a simple rank-1 decomposition is:
        # U[i,j,c] = 1, V[j,k,c] = 1, W[k,i,c] = 1 for c=0
        # But that gives sum = 1 only if rank=1, which works!
        
        # Wait, that can't be right. Let me check:
        # If U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1, then
        # sum_c U[i,j,c]V[j,k,c]W[k,i,c] = 1*1*1 = 1. Yes!
        
        # So the all-ones tensor has rank 1?! That seems too good.
        
        # But wait - the matrix multiplication tensor is NOT the all-ones tensor.
        # The matmul tensor T[i,j,k] = 1 means that the bilinear form is:
        # <A,B>_T = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
        # = sum_i,k (sum_j A[i,j] B[j,k]) = sum_i,k (AB)[i,k]
        
        # But that's not quite right either. The standard matmul tensor has:
        # T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is indexed such that T[i,j,k] gives the coefficient of A[i,j]*B[j,k]
        # in the computation of (AB)[i,k].
        
        # So T[i,j,k] = 1 for all i,j,k is correct - every pair (A[i,j], B[j,k])
        # contributes to (AB)[i,k] with coefficient 1.
        
        # And the tensor decomposition sum_c U[i,j,c]V[j,k,c]W[k,i,c] = T[i,j,k] = 1
        # is indeed just decomposing the all-ones tensor.
        
        # The all-ones tensor of any size has rank 1! Because it's the outer product
        # of three all-ones vectors.
        
        # So U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1 gives rank 1?
        
        # Let me verify: for n=2,m=2,p=2:
        # U = [[1,1],[1,1]], V = [[1,1],[1,1]], W = [[1,1],[1,1]]
        # sum_c U[i,j,c]V[j,k,c]W[k,i,c] = 1*1*1 = 1 for all i,j,k. Yes!
        
        # But that contradicts the known result that matmul has rank 7 for 2x2x2.
        
        # I think I'm misunderstanding the tensor structure. Let me reconsider.
        
        # The issue is that the tensor for matrix multiplication is NOT the all-ones tensor
        # in the naive sense. The correct tensor is:
        # T[i,j,k] = 1 where the indices are such that j is the "contracted" index.
        # But in the formulation given, U has shape (n*m, rank), V has shape (m*p, rank),
        # W has shape (p*n, rank).
        
        # The bilinear form is:
        # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_c (sum_{i,j} U[i,j,c] A[i,j]) 
        #                                              (sum_{j,k} V[j,k,c] B[j,k])
        #                                              (sum_{k,i} W[k,i,c] ???)
        
        # Wait, W has shape (p*n, rank) with index k*n+i. What does W represent?
        # It should be related to the output C[i,k].
        
        # I think the correct interpretation is:
        # (AB)[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k]) * W[k,i,c]
        
        # No, that doesn't make sense dimensionally.
        
        # Let me re-read: "W[k*n+i,r]" - so W is indexed by (k,i) pairs.
        # The decomposition should be:
        # (AB)[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # summed over j as well? No, the sum is only over c.
        
        # Actually, I think the correct formula is:
        # (AB)[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # No wait, that would double-count. Let me think again.
        
        # The standard tensor decomposition for bilinear maps:
        # A bilinear map f: X × Y → Z has a tensor T ∈ X* ⊗ Y* ⊗ Z*
        # such that f(x,y) = T(x,y,·) as a linear map.
        
        # For matmul: f(A,B) = AB where A ∈ R^{n×m}, B ∈ R^{m×p}, AB ∈ R^{n×p}
        # The tensor T ∈ (R^{n×m})* ⊗ (R^{m×p})* ⊗ (R^{n×p})*
        # T(i,j; j',k; i',k') = δ_{i,i'} δ_{k,k'} δ_{j,j'}
        
        # So T[i,j, j',k, i',k'] = 1 if i=i', k=k', j=j'
        
        # In the flattened form:
        # U has index (i*m+j, c), V has index (j'*p+k, c), W has index (i'*n+k', c)
        # But the task says W has shape (p*n, rank) with index k*n+i.
        
        # So W[k*n+i, c] corresponds to the output entry C[i,k] (with indices swapped).
        
        # The decomposition is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
        
        # where T[i,j,k] is the coefficient such that:
        # (AB)[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
        
        # For standard matmul, T[i,j,k] = 1 for all i,j,k (since each A[i,j]*B[j,k]
        # contributes to (AB)[i,k] with coefficient 1, and j is summed over).
        
        # Wait no! j is the SAME index in A[i,j] and B[j,k]. So:
        # (AB)[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T[i,j,k] = 1 means that the term A[i,j]*B[j,k] appears with coeff 1.
        
        # So yes, T is the all-ones tensor, and it has rank 1.
        
        # But this contradicts the known result that matmul(2,2,2) has rank 7.
        
        # I must be misunderstanding something fundamental. Let me look at this differently.
        
        # Ah, I think the issue is that in the tensor decomposition, the three factors
        # are NOT independent. The index j in U[i*m+j, c] and V[j*p+k, c] must match.
        # So we can't freely choose U, V, W independently.
        
        # No wait, in a tensor decomposition, U, V, W ARE independent. The constraint
        # is that the sum over c reproduces the tensor.
        
        # Let me just try the rank-1 solution and see if it works:
        # U[i*m+j, 0] = 1 for all i,j
        # V[j*p+k, 0] = 1 for all j,k  
        # W[k*n+i, 0] = 1 for all k,i
        # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1*1*1 = 1. ✓
        
        # So the rank is 1? That can't be right for matmul.
        
        # I think the problem is that I'm not accounting for the fact that the same
        # intermediate value should be shared. In actual matmul, we compute:
        # M_j = A[:,j] * B[j,:] (outer product, but actually we sum over j)
        # C = sum_j A[:,j] B[j,:]
        
        # The "bilinear" structure means that for each j, we have a rank-1 update
        # to C. The total computation is a sum of m rank-1 updates.
        
        # In the tensor decomposition framework, the rank of the matmul tensor
        # MM_{n,m,p} is the minimum number of scalar multiplications needed.
        # For MM_{2,2,2}, this is 7 (Strassen), not 1.
        
        # The discrepancy is because the tensor T[i,j,k]=1 is not the correct tensor
        # for matmul. The correct tensor has a more complex structure.
        
        # Let me look at this from the bilinear form perspective:
        # <A,B> = trace(A^T B) for the Frobenius inner product.
        # For matmul, we want: (AB)_{ik} = sum_j A_{ij} B_{jk}
        
        # The tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p} such that:
        # (AB)_{ik} = sum_{j,j'} T_{ij, j'k, ik} A_{ij} B_{j'k}
        
        # For this to equal sum_j A_{ij} B_{jk}, we need:
        # T_{ij, j'k, ik} = δ_{jj'}
        
        # So T is NOT the all-ones tensor! It's δ_{jj'}, which is 1 when j=j' and 0 otherwise.
        
        # In the flattened indexing:
        # U[i*m+j, c], V[j'*p+k, c], W[i*n+k, c] (or W[k*n+i, c] per the task)
        # sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] = δ_{jj'}
        
        # So the tensor is: T[i*m+j, j'*p+k, k*n+i] = δ_{j,j'}
        
        # This is a much more complex tensor than all-ones!
        
        # For the rank-1 solution to work, we'd need:
        # U[i*m+j, 0] * V[j'*p+k, 0] * W[k*n+i, 0] = δ_{j,j'}
        
        # This is impossible with rank 1 because δ_{j,j'} depends on j and j' separately.
        
        # OK so now I understand. The tensor is δ_{j,j'}, not all-ones.
        
        # The schoolbook decomposition: for each j, set U[i*m+j, j]=1, V[j*p+k, j]=1,
        # W[k*n+i, j]=1. Then sum_c gives δ_{j,j'}.
        
        # Now, can we do better than rank m=4?
        
        # The rank of the matmul tensor MM_{n,m,p} is at least max(n*m, m*p, n*p)
        # by the "slice rank" lower bound. For n=3,m=4,p=6:
        # max(12, 24, 18) = 24.
        
        # So rank >= 24, and schoolbook gives rank 72. We need to find something between.
        
        pass
    
    # Given the complexity, let me try a different approach.
    # Use the known result that we can combine terms.
    
    # For the tensor T[i,j,j',k] = δ_{j,j'}, we can write:
    # δ_{j,j'} = sum_c U[i,j,c] V[j',k,c] W[i,k,c]
    
    # One approach: use a low-rank factorization of the identity matrix on the j-index.
    # If we can write I_m = sum_c u_c v_c^T (outer products), then:
    # δ_{j,j'} = sum_c u_c[j] v_c[j']
    
    # But we also have the i and k indices to handle.
    
    # Actually, the tensor is:
    # T[i,j,j',k] = δ_{j,j'} (independent of i and k)
    
    # So we can write:
    # T[i,j,j',k] = sum_c U[i,j,c] V[j',k,c] W[i,k,c]
    
    # If we set U[i,j,c] = a_c[j], V[j',k,c] = b_c[j'], W[i,k,c] = 1,
    # then we need sum_c a_c[j] b_c[j'] = δ_{j,j'}
    
    # This is just a rank decomposition of the identity, which has rank m=4.
    
    # But we can also use the i and k indices to help.
    # For example, if we set U[i,j,c] = u_c[i,j], V[j',k,c] = v_c[j',k], W[i,k,c] = w_c[i,k],
    # we have more freedom.
    
    # The key insight: the tensor factors as (identity on j) ⊗ (all-ones on i,k).
    # So the rank is at least the rank of the identity on j, which is m=4.
    
    # But we can potentially do better by exploiting the i and k structure.
    
    # For now, let me just return the schoolbook decomposition.
    # A proper optimization would require more sophisticated techniques.
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
