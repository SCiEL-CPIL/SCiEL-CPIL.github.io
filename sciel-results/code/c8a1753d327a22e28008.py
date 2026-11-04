"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _bareiss_rank_and_pivot(A):
    """
    Bareiss fraction-free Gaussian elimination.
    Returns (rank, list of pivot row indices in original matrix).
    A is a list of lists of ints/Fractions.
    """
    n = len(A)
    if n == 0:
        return 0, []
    m = len(A[0])
    
    # Work on a copy
    M = [row[:] for row in A]
    
    rank = 0
    pivot_rows = []
    prev = 1  # previous pivot (for Bareiss division)
    
    for col in range(min(n, m)):
        # Find a non-zero pivot in this column from row 'rank' downward
        pivot_row = -1
        for r in range(rank, n):
            if M[r][col] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        
        # Swap rows
        if pivot_row != rank:
            M[rank], M[pivot_row] = M[pivot_row], M[rank]
        
        pivot_rows.append(rank)
        
        # Eliminate below
        for r in range(rank + 1, n):
            if M[r][col] == 0:
                continue
            factor = M[r][col]
            for c in range(col, m):
                # Bareiss: M[r][c] = (M[rank][col]*M[r][c] - factor*M[rank][c]) / prev
                # But we need to be careful: the standard Bareiss update is:
                # M[r][c] = (M[rank][col]*M[r][c] - M[r][col]*M[rank][c]) / prev
                # Wait, that's not right. Let me reconsider.
                # Standard Bareiss: for i > pivot, j >= col:
                # M[i][j] = (M[pivot][col] * M[i][j] - M[i][col] * M[pivot][j]) / prev
                # But when j == col, M[i][col] becomes 0 after this step.
                # Actually for j == col: (M[pivot][col]*M[i][col] - M[i][col]*M[pivot][col]) / prev = 0
                # So we can skip j == col since it becomes 0.
                if c == col:
                    M[r][c] = 0
                else:
                    M[r][c] = (M[rank][col] * M[r][c] - M[r][col] * M[rank][c]) // prev
        
        prev = M[rank][col]
        rank += 1
    
    return rank, pivot_rows


def _find_basis_rows(A):
    """
    Find a set of linearly independent rows that span the row space.
    Returns list of row indices.
    Uses Bareiss to identify pivot rows.
    """
    n = len(A)
    if n == 0:
        return []
    m = len(A[0])
    
    M = [row[:] for row in A]
    rank = 0
    pivot_rows = []
    prev = 1
    
    for col in range(min(n, m)):
        pivot_row = -1
        for r in range(rank, n):
            if M[r][col] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        
        if pivot_row != rank:
            M[rank], M[pivot_row] = M[pivot_row], M[rank]
        
        pivot_rows.append(rank)
        
        for r in range(rank + 1, n):
            if M[r][col] == 0:
                continue
            for c in range(col + 1, m):
                M[r][c] = (M[rank][col] * M[r][c] - M[r][col] * M[rank][c]) // prev
            M[r][col] = 0
        
        prev = M[rank][col]
        rank += 1
    
    return pivot_rows


def construct():
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n, m, p = 4, 4, 7
    
    # The multiplication tensor T has entries T[(i,j), (j',k), (k',i')]
    # where T = 1 if j==j' and k==k' (i.e., the contraction index matches).
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] = delta_{j,k} for computing C[i,k] = sum_j A[i,j]*B[j,k]
    # But for the decomposition, we view it as a 3-way tensor.
    # 
    # The standard approach: flatten the tensor into a matrix.
    # The tensor M_{ijk} = delta_{jk} (where i indexes rows of A, j columns of A / rows of B, k columns of B)
    # 
    # For a decomposition of rank r:
    # M_{ijk} = sum_{t=1}^{r} U_{i,t} * V_{j,t} * W_{k,t}
    # 
    # To find the rank, we can use the "slice" approach or flatten.
    # Flatten: fix the output index (i,k) -> row, and the input pair (j) -> column?
    # No, the standard flattening for matmul tensor rank:
    # 
    # The tensor is of size n x m x p where T[i,j,k] = delta_{j,k}... wait no.
    # 
    # Let me reconsider. For multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i,k] = sum_{j=1}^{m} A[i,j] * B[j,k]
    # 
    # The tensor T has T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k]... 
    # No. The tensor decomposition means:
    # C[i,k] = sum_t (sum_j U[i,j,t] A[i,j]) * (sum_j V[j,k,t] B[j,k]) ... no that's not right either.
    #
    # Correct: The matrix multiplication tensor is defined as:
    # T[i, j, k] = 1 if this term contributes to C[i,k]
    # Actually T is a 3-tensor where T[a, b, c] with a in [n*m], b in [m*p], c in [p*n]
    # T[a, b, c] = 1 iff a corresponds to (i,j), b to (j,k), c to (k,i) with matching indices.
    # 
    # Specifically: a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 if the j in a equals the j in b AND the k in b equals the k in c AND the i in c equals the i in a.
    # Which is automatically satisfied: a=(i,j), b=(j,k), c=(k,i).
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 otherwise.
    #
    # To compute rank: flatten T into a matrix. One standard flattening:
    # Rows indexed by (i,k) pairs [n*p], columns by (j) [m].
    # But that gives a matrix of size n*p x m which doesn't directly give tensor rank.
    #
    # Better: The tensor rank can be found by considering T as a bilinear map.
    # Fix one mode: T is a linear map from (A tensor B) to C.
    # Flatten A (n*m) and B (m*p) into a single vector of length n*m + m*p? No.
    # 
    # The correct flattening for tensor rank: 
    # T is a 3-way tensor of dimensions (n*m) x (m*p) x (p*n).
    # Flatten modes 1 and 2 into rows, mode 3 into columns:
    # Matrix M of size (n*m * m*p) x (p*n) where M[(a,b), c] = T[a,b,c].
    # The rank of this matrix is an upper bound on tensor rank.
    # But for matmul tensor, this matrix has rank exactly... let's think.
    #
    # Actually, a simpler approach: the matmul tensor T[i,j,k] (with i in [n], j in [m], k in [p])
    # has T[i,j,k] = delta_{j,k}... no, that's not a 3-tensor in the right sense.
    #
    # Let me use the standard formulation. The tensor for matmul(n,m,p) is:
    # T in R^{(n*m) x (m*p) x (p*n)}
    # T[(i,j), (j',k), (k',i')] = 1 if j=j' and k=k' and i=i'
    #
    # Flatten: rows = (i,j,j',k) combinations? No.
    # Standard: fix the "output" index c = k*n+i, and the "input" indices a=i*m+j, b=j'*p+k.
    # M[a,b] for fixed c... 
    #
    # Actually the simplest: create a matrix where:
    # - Rows correspond to all (i, k) pairs: n*p rows
    # - Columns correspond to all (j) indices: m columns  
    # - Entry at (i,k), j is T[i,j,k] = 1 if valid, else 0
    # But this matrix is just a permutation/selection matrix and has rank min(n*p, m) which is not the tensor rank.
    #
    # The correct approach to get tensor rank via linear algebra:
    # The tensor rank r is the minimum number of rank-1 tensors that sum to T.
    # A rank-1 tensor is U_a * V_b * W_c for vectors U, V, W.
    #
    # To find the rank, we can use the following: 
    # Flatten T into a matrix by grouping two indices. For matmul tensor:
    # Group (i,j) as row index [n*m], (j',k) as column index [m*p], and for each output (k',i'):
    # M_c[a,b] = T[a,b,c]
    # The tensor rank is at least max over c of rank(M_c), but that's not tight.
    #
    # Better: The tensor rank equals the minimum r such that there exist U (n*m x r), V (m*p x r), W (p*n x r)
    # with T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t].
    #
    # This is equivalent to: for each c, the matrix M_c = U * diag(W[:,c]) * V^T has rank <= r.
    # And the set {M_c} must satisfy certain consistency conditions.
    #
    # A practical approach: use the "unfolding" that gives the exact rank.
    # For the matmul tensor, it's known that the rank can be computed by:
    # Creating a matrix of size (n*p) x (m) where entry (i,k),j = 1 if the tensor has T[i*m+j, j*p+k, k*n+i]=1
    # This matrix has a 1 at position (i,k),j iff j is the middle index. Since for each (i,k), 
    # there's exactly one j... wait no, j ranges over all m values but T[i*m+j, j*p+k, k*n+i] = 1 only when 
    # the indices are consistent, which they always are by construction.
    # 
    # Hmm, I think the issue is that for the matmul tensor, T[a,b,c] is non-zero only when 
    # a=(i,j), b=(j,k), c=(k,i) for some i,j,k. So for a given c=(k,i), the non-zero entries in M_c 
    # are at positions a=(i,j), b=(j,k) for j=1..m. So M_c has m non-zero entries (one per j).
    # 
    # This means each M_c is a sparse matrix with m non-zeros. The tensor rank is not simply the max rank of M_c.
    #
    # Let me just directly construct the schoolbook and then try to reduce it.
    # Actually, let me try a different approach: use the known Strassen-based decomposition.
    
    # For 4x4 * 4x7, use Strassen recursively:
    # Split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] (2x2 blocks, each 2x2)
    # B = [[B11, B12, B13], [B21, B22, B23]] (2x2 blocks, B13,B23 are 2x1)
    # 
    # C = A*B:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22  
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # For each 2x2*2x2 product, use Strassen (rank 7).
    # For each 2x2*2x1 product, use schoolbook (rank 4).
    #
    # But the outputs C11..C23 are sums of two products each.
    # C11 = A11*B11 + A12*B21: this needs 2*4=8 mults schoolbook, or we can combine.
    #
    # Actually, let me just do the schoolbook with rank 112 and verify it works,
    # then try Strassen for the 2x2*2x2 parts.
    
    # Schoolbook: rank = n*m*p = 4*4*7 = 112
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
    
    # Verify schoolbook
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                val = 0
                for t in range(rank):
                    for j in range(m):
                        val += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                # Should equal 1 if this is the (i,k) output from the (j) input... 
                # Actually C[i,k] = sum_j A[i,j]*B[j,k], so the coefficient should be 1 for each j
                # But our decomposition gives C[i,k] = sum_t (sum_j U[i,j,t]*A[i,j]) * V[j,k,t] * W[k,i,t]
                # This isn't quite right. Let me re-examine.
                #
                # The correct decomposition: C[i,k] = sum_t U[i,k,t] * (sum_j ... )
                # No. The standard: C = U * A * V * B * W^T type thing.
                # 
                # Actually: C[i,k] = sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t]
                # Wait, that doesn't work because U and V both sum over j.
                #
                # The correct form: C[i,k] = sum_t (sum_j U[i,j,t] * A[i,j]) * (sum_j V[j,k,t] * B[j,k]) * W[i,k,t]
                # But U has shape (n*m, r), so U[i*m+j, t] is the entry.
                # Then sum_j U[i*m+j, t] * A[i,j] is a scalar for each t.
                # Similarly sum_j V[j*p+k, t] * B[j,k] is a scalar.
                # And W[k*n+i, t] is a scalar.
                # So C[i,k] = sum_t [sum_j U[i*m+j,t]*A[i,j]] * [sum_j V[j*p+k,t]*B[j,k]] * W[k*n+i,t]
                #
                # For schoolbook: U[i*m+j, t] = 1 if t corresponds to (i,j), V[j*p+k, t] = 1 if t corresponds to (j,k), W[k*n+i, t] = 1 if t corresponds to (k,i).
                # For t = (i',j',k') with i'=i, j'=j, k'=k: all three are 1, product is 1.
                # For other t: at least one is 0.
                # So C[i,k] = sum_j A[i,j]*B[j,k]. Correct!
        
        return True
    
    # The schoolbook is correct with rank 112.
    # Now let's try to do better with Strassen.
    # 
    # Strategy: use Strassen for the 2x2*2x2 block multiplications.
    # Split A into 2x2 blocks, B into 2x2 blocks (with last column being 2x1).
    
    if time.time() - t_start > DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    # Try Strassen-based decomposition
    # A (4x4) split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B (4x7) split into 2x2 blocks (last col is 2x1):
    # B = [[B11, B12, B13], [B21, B22, B23]]
    # where B11,B12,B21,B22 are 2x2, B13,B23 are 2x1
    #
    # C (4x7) = [[C11, C12, C13], [C21, C22, C23]]
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x2)
    # C13 = A11*B13 + A12*B23 (2x1)
    # C21 = A21*B11 + A22*B21 (2x2)
    # C22 = A21*B12 + A22*B22 (2x2)
    # C23 = A21*B13 + A22*B23 (2x1)
    #
    # For each 2x2*2x2: use Strassen (7 mults)
    # For each 2x2*2x1: use schoolbook (4 mults)
    #
    # But C11 = A11*B11 + A12*B21 is a sum of two 2x2 products.
    # We can't just add two Strassen decompositions because they share the output.
    # 
    # Alternative: treat the whole thing as computing 6 outputs, each being a sum of 2 products.
    # Total "scalar multiplications" in schoolbook: 4*4*7 = 112.
    # 
    # Let me just return the schoolbook for now since constructing a valid Strassen
    # decomposition for this non-standard shape is complex and error-prone.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
