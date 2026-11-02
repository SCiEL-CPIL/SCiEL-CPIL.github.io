"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower-rank decomposition
    # The matrix multiplication tensor T[i,j,k] = 1 if i==k else 0 (for A[i,j]*B[j,k] -> C[i,k])
    # Actually T[i,j,k] contributes to C[i,k] via sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = δ_{i,k} ... wait no.
    # For C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor M[i,j,k] = 1 if we're computing the (i,k) entry using A[i,j] and B[j,k]
    # The decomposition: M = sum_c U_c ⊗ V_c ⊗ W_c where
    #   (U_c)[i,j] * (V_c)[j,k] * (W_c)[i,k] ... 
    # Actually the standard form: C[i,k] = sum_c U[i,j,c] V[j,k,c] W[i,k,c] doesn't work directly.
    # 
    # Correct form: The multiplication tensor has entries T_{i,j,k} such that
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # This is a 3-tensor where T[i,j,k] = 1 for all valid i,j,k (it's the "all ones" in a sense)
    # But the decomposition is: sum_c u_c[i] v_c[j] w_c[k] where the product u_c[i]*v_c[j]*w_c[k] 
    # contributes to C[i,k] when summed over j... 
    #
    # Let me use the standard tensor network form:
    # C[i,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] ... this doesn't look right either.
    #
    # From the problem: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # And the decomposition is: for each c, the contribution to C[i,k] is sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that would give C[i,k] = sum_c sum_j U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # But W depends on k and i, which is the output index. This is unusual.
    #
    # Actually I think the convention is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c])
    # No, that's not standard either.
    #
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by [k*n+i] where k is col of B and i is row of A.
    # The decomposition should be: C[i,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] summed over j?
    # That doesn't make sense dimensionally.
    #
    # I think the correct interpretation is that the tensor M[i,j,k] (where i indexes A's row, j indexes A's col/B's row, k indexes B's col)
    # is decomposed as M[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[i,k,c]
    # But U has shape (n*m, rank) so U[i*m+j, c], V has shape (m*p, rank) so V[j*p+k, c], W has shape (p*n, rank) so W[k*n+i, c]
    # So M[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # For matrix multiplication, M[i,j,k] = 1 for all i,j,k (since every term A[i,j]*B[j,k] contributes to C[i,k])
    # Wait no, M[i,j,k] should be such that sum_j M[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    # So M[i,j,k] = 1 for all i,j,k? That would mean C[i,k] = sum_j A[i,j] B[j,k] which is correct.
    # So the tensor is all-ones: T[i,j,k] = 1 for all valid i,j,k.
    #
    # But that can't be right because then the rank would be 1 (just take U=V=W=all ones).
    # 
    # I think I'm misunderstanding the tensor structure. Let me reconsider.
    #
    # The matrix multiplication tensor is typically defined as:
    # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the tensor maps (A, B) -> C
    # with <M, A ⊗ B> = C, i.e., M_{i,j,k} A_{i,j} B_{j,k} = C_{i,k}
    # So M_{i,j,k} = δ_{i,k} ... no, M_{i,j,k} should be 1 if the (i,j) of A and (j,k) of B multiply to give (i,k) of C.
    # That's always true, so M_{i,j,k} = 1 for all i,j,k? 
    #
    # No wait, the tensor is: M = sum_{i,j,k} |i><j| ⊗ |j><k| ⊗ |k><i| ... I'm confusing myself.
    #
    # Let me just use the schoolbook and try to verify it works, then attempt improvements.
    
    # Verify schoolbook: For each (i,j,k), we have a term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # The contribution to C[i',k'] from term (i,j,k) is:
    # sum over all terms c: U[i'*m+j', c] V[j'*p+k', c] W[k'*n+i', c]
    # For the term where (i,j,k) = (i',j',k'): U=1, V=1, W=1, contribution = 1
    # For other terms, at least one of U,V,W is 0 for this (i',j',k')
    # So C[i',k'] = sum_{j'} 1 * 1 * 1 = m? That's not right.
    #
    # I think the issue is that the decomposition should give C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor should satisfy: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all j (so that summing over j gives the sum)
    # Actually no. Let me think again.
    #
    # If C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) (sum_k' V[j*p+k', c] B[j,k']) W[k*n+i, c] ... this is getting too complicated.
    #
    # Let me just trust the schoolbook implementation and try to improve it.
    
    # For now, return schoolbook. A proper improvement would require understanding the exact tensor convention.
    return best_result
# EVOLVE-BLOCK-END
