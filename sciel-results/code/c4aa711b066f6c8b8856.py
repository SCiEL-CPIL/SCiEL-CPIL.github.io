"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]
TIME_LIMIT = 150
SAFETY_MARGIN = 5

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # For 2x2 * 2x14, the tensor is T_{(i,j),(j,k),(k,i)}
    # Indices: A has indices (i,j) with i in [0,2), j in [0,2)
    #          B has indices (j,k) with j in [0,2), k in [0,14)
    #          C has indices (k,i) with k in [0,14), i in [0,2)
    # W index is k*n + i
    
    # The standard schoolbook gives rank = 2*2*14 = 56
    # We can try to find a lower rank decomposition.
    
    # For matrix multiplication of (n x m) * (m x p), the rank is at least max(n*m, m*p, n*p)
    # Here: max(4, 28, 28) = 28
    # Also, for fixed m=2, the problem decomposes into two independent multiplications
    # for each column j of A. But since j is shared, we need to be careful.
    
    # Actually, for A (n x m) * B (m x p), we can write:
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # This is a bilinear form. The tensor has rank at most n*m*p.
    
    # Let's try to find a better decomposition using the structure.
    # For m=2, we have:
    # C[k,0] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[k,1] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # We can use the fact that for each k, we have a 2x2 matrix multiplication
    # between the column of A (indexed by i) and the row of B (indexed by k).
    
    # Actually, let's think of this differently. The tensor is:
    # T[i,j,j',k,k',i'] = delta_{j,j'} * delta_{k,k'} * ... no wait.
    # T[A_idx, B_idx, C_idx] where A_idx=(i,j), B_idx=(j',k), C_idx=(k',i')
    # T = 1 if j=j' and k=k', else 0.
    
    # So the tensor is a "block diagonal" type structure.
    # For each j in [0,2), and each k in [0,14), we have a rank-1 component:
    # U[i*m+j, c] = delta_{i, i_c}
    # V[j'*p+k, c] = delta_{j', j_c} * delta_{k, k_c}
    # W[k'*n+i', c] = delta_{k', k_c} * delta_{i', i_c}
    
    # This is exactly the schoolbook decomposition with rank = n*m*p = 56.
    
    # Can we do better? The lower bound is max(n*m, m*p, n*p) = max(4, 28, 28) = 28.
    # For matrix multiplication, the rank is known to be n*m*p for small cases,
    # but for larger cases, there may be improvements.
    
    # For 2x2 * 2x14, let's see if we can achieve rank 28.
    # One approach: for each k, we need to compute a 2x2 * 2x1 multiplication.
    # The 2x2 * 2x1 multiplication has rank at most 2*2*1 = 4, but the lower bound
    # is max(2*2, 2*1, 2*1) = max(4, 2, 2) = 4.
    
    # So for each k, we need rank 4. With 14 values of k, that's 14*4 = 56.
    # But we can share some structure across different k values.
    
    # Actually, the multiplication is:
    # [A00 A01] * [B0k] = [A00*B0k + A01*B1k]
    # [A10 A11]   [B1k]   [A10*B0k + A11*B1k]
    
    # This is two independent dot products of length 2.
    # The rank of a single dot product of length m is m (for m>=2, using Karatsuba-like tricks
    # for m=3, but for m=2, the rank is 2).
    
    # So for each k, we need rank 2 for the first output and rank 2 for the second output,
    # but they share the inputs A and B.
    
    # Let's try a direct construction. For each k, we have 4 products:
    # A00*B0k, A01*B1k, A10*B0k, A11*B1k
    # And we need:
    # C0k = A00*B0k + A01*B1k
    # C1k = A10*B0k + A11*B1k
    
    # We can use 2 rank-1 terms for C0k and 2 for C1k, total 4 per k.
    # With 14 k's, that's 56.
    
    # But can we share? The B values B0k and B1k are different for different k,
    # so we can't directly share.
    
    # However, we might be able to use a different decomposition.
    # Let's try to see if rank 28 is achievable.
    
    # For now, let's return the schoolbook decomposition.
    # We could try to search for a better one, but given the time constraint,
    # let's be safe.
    
    # Actually, let's try a simple improvement: for each k, we only need 2 rank-1 terms
    # if we can combine the two outputs. But we can't because they have different i indices.
    
    # Let's just return the schoolbook for now.
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
