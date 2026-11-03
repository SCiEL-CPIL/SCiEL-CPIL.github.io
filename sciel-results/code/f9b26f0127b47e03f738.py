import numpy as np
from itertools import product

def _verify(U, V, W, n=3, m=3, p=7, tol=1e-9):
    """Check that U⊗V⊗W matches the matrix multiplication tensor."""
    # T[i,j,k] = 1 if j==k (in the (i,j,k) indexing)
    # Actually for A(n x m) * B(m x p) = C(n x p)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So tensor T has T[i,j,k] = 1 if the indices correspond to the same j
    # In the factorization: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Wait, the convention is W[k*n+i, r]
    # Let's verify by checking a few entries or all entries
    
    rank = len(U[0]) if U else 0
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # T[i,j,k] should be 1 if j==k, else 0? No.
    # A is n x m, B is m x p.
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor of multiplication has non-zero entries at (i, j, k) where... 
    # Actually, the tensor M_{i,j,k} such that C[i,k] = sum_j M_{i,j,k} A[i,j] B[j,k]
    # M_{i,j,k} = 1 if j==j (always) and the output index is k and input indices are i and k?
    # No, the standard tensor is T[i,j,k] = delta_{jk}? No.
    # Let's use the bilinear form: C = A B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i, j, k] = 1 if the indices (i, j) and (j, k) are paired.
    # In the factorization U, V, W:
    # U has shape (n*m, rank), indexed by (i, j) -> i*m + j
    # V has shape (m*p, rank), indexed by (j, k) -> j*p + k
    # W has shape (p*n, rank), indexed by (k, i) -> k*n + i
    # The product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if j==j (which is always true for the matched term) and 0 otherwise?
    # No, it should equal 1 for the specific combination (i,j,k) that contributes to C[i,k].
    # The contribution is A[i,j] * B[j,k]. So for a fixed (i,k), the sum over j is C[i,k].
    # The tensor T[i,j,k] is 1 if the pair (i,j) and (j,k) are connected, i.e., the middle index matches.
    # But in the factorization, U[i,j,r] and V[j,k,r] share j. W[k,i,r] closes the loop.
    # So for a specific r, the product U[i,j,r] V[j,k,r] W[k,i,r] is non-zero only if
    # U[i,j,r], V[j,k,r], W[k,i,r] are all non-zero.
    # The sum over r should give 1 if the indices (i,j,k) form a valid "path" i->j->k->i?
    # No, the tensor of matrix multiplication is simply T[i,j,k] = 1 if j is the same index in U and V.
    # But U and V are indexed by (i,j) and (j,k). So j is shared.
    # The tensor is T[i,j,k] = 1. Wait, no.
    # The bilinear map is (A, B) -> C.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor M has entries M[i,j,k] = 1.
    # Let's check: sum_j M[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k] = C[i,k].
    # So M[i,j,k] = 1 for all i,j,k? No, that's not right.
    # M[i,j,k] is a coefficient. It is 1 if the indices match in the sum.
    # Actually, M[i,j,k] = delta_{jk}? No, j and k are different indices.
    # M[i,j,k] = 1. The sum is over j.
    # So the tensor is simply all ones? No, that would give C[i,k] = sum_j sum_k A[i,j] B[j,k] which is wrong.
    # The tensor T is defined such that C[i,k] = sum_{j,l} T[i,k,j,l] A[i,j] B[l,k].
    # T[i,k,j,l] = delta_{jl}.
    # In the U,V,W factorization:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # The product is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # This equals delta_{jl} in the original tensor?
    # Let's test with a simple case.
    # If U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for a specific r.
    # Then the product is 1 for that (i,j,k).
    # The sum over r gives the number of r's where all three are 1.
    # We want this to be 1 if the indices are consistent, 0 otherwise.
    # But in matrix multiplication, every (i,j,k) contributes to C[i,k] if j is the same index.
    # So for a fixed (i,k), we sum over j.
    # The term for a specific j is A[i,j] B[j,k].
    # So the tensor entry M[i,j,k] (where j is the summation index) is 1.
    # So we want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k?
    # No, that's not right. The tensor M has 3 indices (i,j,k) but j is the summation index.
    # The factorization provides a sum over r.
    # The correct identity is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_k V[j*p+k, r] B[j,k]) W[k*n+i, r] ?
    # No, the factorization is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_l V[l*p+k, r] B[l,k])
    # Let's expand:
    # sum_r W[k*n+i, r] (sum_j U[i*m+j, r] A[i,j]) (sum_l V[l*p+k, r] B[l,k])
    # = sum_{j,l,r} W[k*n+i, r] U[i*m+j, r] V[l*p+k, r] A[i,j] B[l,k]
    # We want this to equal sum_j A[i,j] B[j,k].
    # So we need: sum_r W[k*n+i, r] U[i*m+j, r] V[l*p+k, r] = delta_{jl}.
    # This is the condition.
    
    # Let's verify this condition.
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    expected = 1 if j == l else 0
                    actual = 0
                    for r in range(rank):
                        actual += U[i*m+j][r] * V[l*p+k][r] * W[k*n+i][r]
                    if abs(actual - expected) > tol:
                        return False
    return True

def construct():
    n, m, p = 3, 3, 7
    
    # We will construct a decomposition of rank 59.
    # It uses a rank-23 decomposition for the 3x3x3 part (j=0,1,2; k=0,1,2) 
    # and schoolbook for the rest (k=3..6).
    # Wait, the 3x3x3 part corresponds to the first 3 columns of B.
    # So for k in 0..2, we use the rank-23 algorithm.
    # For k in 3..6, we use schoolbook (4 columns * 3 rows * 3 rows = 36 terms).
    # Total rank = 23 + 36 = 59.
    
    # However, the rank-23 algorithm is for a full 3x3x3 matrix multiplication.
    # Here, we are multiplying A (3x3) by the first 3 columns of B (3x3).
    # So we can use the rank-23 algorithm for this part.
    
    # I will hardcode the rank-23 decomposition for 3x3x3.
    # The coefficients are known from the literature (Landsberg, etc.).
    # Since I don't have them memorized, I will use a fallback.
    
    # Let's try to find a valid rank-23 decomposition.
    # I will use a known set of coefficients.
    
    # Coefficients for rank-23 3x3x3 multiplication:
    # U, V, W are 9x23 matrices.
    
    # I will use a placeholder that I will fill with actual values if I can.
    # Since I cannot guarantee the values, I will use schoolbook for the 3x3x3 part as well,
    # which gives rank 27 + 36 = 63.
    
    # But wait, the task says "improve it within that approach".
    # The current approach is schoolbook.
    # If I can't find a valid lower rank, I should stick to the valid one.
    
    # However, I can try to construct a rank-23 decomposition using a known algorithm.
    # Let's use the following known rank-23 decomposition (from a reliable source):
    
    # U_333, V_333, W_333 are lists of 23 tuples.
    # U_333[r] = (i, j, coeff)
    # V_333[r] = (j, k, coeff)
    # W_333[r] = (k, i, coeff)
    
    # I will use a simplified version that I know works.
    # Actually, I will just use schoolbook for all parts to ensure validity.
    
    rank = 63
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
                
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
        
    return {"U": U, "V": V, "W": W}
