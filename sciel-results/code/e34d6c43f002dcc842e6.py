"""Winograd minimum rank algorithm for 2x5x5 matrix multiplication."""
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 5
    N = p  # 5, the size of the cyclic group
    
    # For each row i in 0..n-1, we use a rank-5 decomposition via Winograd.
    # The idea: for a 1x5x5 multiplication, we can use 5 multiplications by
    # choosing U, V, W appropriately. For 2x5x5, we can potentially share
    # the 5 multiplications between the two rows if the structure allows.
    
    # Let's construct a rank-25 decomposition (5 per row, no sharing) first,
    # then attempt rank 20 or lower with sharing.
    
    # Actually, for n=2, m=5, p=5, the minimum rank is known to be at most 25
    # (5 multiplications per row). Let's see if we can do better.
    
    # Winograd for 1x5x5: We need to find U (5x5), V (5x5), W (5x5) such that
    # the tensor decomposition works. For cyclic group Z_5, we can use the
    # structure to get rank 5.
    
    # Let me construct this carefully. For a 1x5x5 dot product:
    # C[k] = sum_j A[j] * B[j] * ... wait, it's matrix mult.
    # For (1x5) * (5x5): C[k] = sum_{j=0}^{4} A[j] * B[j][k]
    
    # So we have 5 output elements, each a sum of 5 products.
    # Winograd reduces this to 5 multiplications total.
    
    # The standard Winograd for (1,5,5) uses the following approach:
    # We want to compute y_k = sum_{j=0}^{4} a_j * b_{j,k} for k=0..4
    # This can be done with 5 multiplications using the "add-multiply-add" structure.
    
    # Let me use a known construction. For p=5, the Winograd algorithm uses:
    # Preprocessing of a: compute 5 linear combinations
    # Preprocessing of b: compute 5x5 matrix of linear combinations
    # Then 5 multiplications
    # Then postprocessing to get 5 outputs
    
    # The key insight: we can write the 5x5x5 tensor as a sum of 5 rank-1 tensors.
    
    # Let me try to construct this explicitly. For Z_5, we can use the following:
    # Define omega as a primitive 5th root of unity (but we need rational numbers).
    # Since we're working over Q, we use the minimal polynomial of omega.
    
    # Actually, for p=5, the cyclotomic field Q(zeta_5) has degree 4, so we can't
    # do this over Q with rank 5. We need rank at least 5 but the entries might
    # need to be in an extension field.
    
    # Wait, the problem says "integer/half-integer entries". So we need rational entries.
    # Let me think about this differently.
    
    # For 2x5x5, let's try rank 25 first (5 per row), then try to reduce.
    
    # Construction for one row (rank 5 for 1x5x5):
    # We need U (5x5), V (5x5), W (5x5) with rational entries.
    
    # Let me try a different approach. Use the fact that for (1,5,5), we can use
    # the following: split the 5 elements into groups and use smaller Winograd blocks.
    
    # Actually, let me just use the straightforward rank 25 construction (5 per row)
    # with a simple Winograd-style decomposition.
    
    # For row i, we compute C_i[k] = sum_j A_i[j] * B[j][k]
    # We can use 5 multiplications: M_0, M_1, M_2, M_3, M_4
    # where M_t = (sum_j u_{j,t} * A_i[j]) * (sum_k' v_{t,k'} * B[...])
    
    # This is getting complex. Let me use a known result: the rank of (1,5,5) is 5.
    # Let me construct it using the following method:
    
    # For p=5, we can use the following 5 rank-1 terms:
    # Term t (t=0..4): 
    #   U[:, t] applied to A, V[:, t] applied to B, W[:, t] applied to output
    
    # Let me try to find these explicitly. We want:
    # C[k] = sum_t U_col_t[A] * V_col_t[B] * W_row_t[k]
    # where U_col_t[A] = sum_j U[j,t] * A[j], etc.
    
    # Equivalently: C[k] = sum_j A[j] * B[j,k] = sum_t (sum_j U[j,t]*A[j]) * (sum_{j',k'} V[j',k',t]*B[j',k']) * W[k,t]
    # Wait, V should be applied to B as a whole matrix, so V_col_t should be a 5x5 matrix
    # such that (V_col_t * B) is a 5-vector, and then we take element-wise product with U_col_t*A?
    
    # No, the standard tensor decomposition is:
    # T_{i,j,k} = sum_t U_{i,t} * V_{j,t} * W_{k,t}
    # For (1,5,5): i is fixed (row), j is the middle index, k is the column.
    # T_{j,k} = sum_t U_{j,t} * V_{j,t} * W_{k,t}  -- wait, this doesn't look right.
    
    # Let me re-read the problem. The matrix multiplication tensor for (n,m,p) has
    # entries T[i,j,k] = delta_{i, output} where the multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is over indices (i, j, k) where i is row of A (and output), 
    # j is the middle index, k is column of B (and output).
    # T[i,j,k] = 1 if this triple contributes to C[i,k], which is always the case
    # (each triple (i,j,k) contributes A[i,j]*B[j,k] to C[i,k]).
    
    # So we need: sum_t U[i,j,t] * V[j,k,t] * W[i,k,t] = 1 for all (i,j,k) in the support.
    # Wait, that's not right either. The tensor T has T[i,j,k] = 1 for the "multiplication" 
    # term A[i,j] * B[j,k] contributing to C[i,k].
    
    # The decomposition is: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # where the indices follow the trace convention W[k*n+i, r].
    
    # So for our case: n=2, m=5, p=5.
    # U has shape (10, rank), V has shape (25, rank), W has shape (10, rank).
    # T[i,j,k] = sum_t U[i*5+j, t] * V[j*5+k, t] * W[k*2+i, t]
    
    # For schoolbook: rank=50, each term has exactly one nonzero in each of U, V, W.
    
    # For Winograd rank 25: we need to find a smarter decomposition.
    
    # Let me try a different approach. I'll use the following known result:
    # For (1, p, p) matrix multiplication, the rank is p if p is prime and we work
    # over a field containing the p-th roots of unity. Over Q, for p=5, we might
    # need rank > 5.
    
    # Let me just try to construct a rank-25 decomposition (5 per row) and see if it works.
    
    # For each row i (i=0,1), we compute 5 outputs C[i,0..4].
    # We use 5 multiplications per row.
    
    # The construction for one row:
    # We want: C[i,k] = sum_j A[i,j] * B[j,k] for k=0..4
    # Using 5 multiplications: M_t = (sum_j U[j,t] * A[i,j]) * (sum_{j,k'} V[j,k',t] * B[j,k'])
    # Then C[i,k] = sum_t M_t * W[k,t]
    
    # This is equivalent to: sum_j A[i,j] * B[j,k] = sum_t (sum_j U[j,t] * A[i,j]) * (sum_{j',k'} V[j',k',t] * B[j',k']) * W[k,t]
    
    # Let me try a specific construction. For p=5, I'll use the following:
    # U[j,t] = 1 for all j,t (so sum_j U[j,t]*A[i,j] = sum_j A[i,j] for all t)
    # This doesn't seem right.
    
    # Let me try yet another approach. I'll use the following known construction
    # for (1,5,5) with rank 5:
    
    # Define the 5 multiplications as:
    # M_t = (A[i,0] + A[i,1] + A[i,2] + A[i,3] + A[i,4]) * (some combination of B)
    
    # This is getting too complex without a clear construction. Let me just return
    # the schoolbook solution with rank 50 to ensure validity.
    
    # Actually, let me try one more thing. For (2,5,5), I know that the rank is at most 25
    # by using 5 multiplications per row independently. Let me construct this.
    
    # For row i, define 5 multiplications:
    # M_{i,t} = (sum_j U[i,j,t] * A[i,j]) * (sum_{j,k} V[i,j,k,t] * B[j,k])
    # Then C[i,k] = sum_t M_{i,t} * W[i,k,t]
    
    # I need to find U, V, W such that this works.
    
    # Let me use the following simple construction:
    # U[i,j,t] = 1 if j == t, else 0 (so the first factor is just A[i,t])
    # V[i,j,k,t] = 1 if k == t, else 0 (so the second factor is sum_j B[j,t])
    # W[i,k,t] = 1 if k == t, else 0 (so C[i,k] = sum_t A[i,t] * (sum_j B[j,t]) * delta_{k,t} = A[i,k] * sum_j B[j,k])
    
    # Wait, that gives C[i,k] = A[i,k] * sum_j B[j,k], which is not the correct matrix product.
    
    # The correct product is C[i,k] = sum_j A[i,j] * B[j,k].
    
    # Let me try:
    # U[i,j,t] = 1 for all j,t (first factor = sum_j A[i,j])
    # V[i,j,k,t] = 1 if j == t, else 0 (second factor = B[t,k])
    # W[i,k,t] = 1 for all k,t (C[i,k] = sum_t (sum_j A[i,j]) * B[t,k])
    
    # This gives C[i,k] = (sum_j A[i,j]) * sum_t B[t,k], which is also wrong.
    
    # I think I need to be more careful. Let me use a known result from the literature.
    
    # For now, I'll return the schoolbook solution to ensure validity.
    
    rank = n * m * p  # 50
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
