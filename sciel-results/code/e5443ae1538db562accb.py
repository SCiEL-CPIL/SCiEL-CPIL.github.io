"""Find a rank-minimal exact tensor decomposition for multiplying a 3x4 by 4x6 matrix."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) factor matrices."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _mat_to_frac_mat(M):
    """Convert a numpy object matrix of integers to a list of Fractions."""
    return [[Fraction(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]


def _rank_over_Q(mat):
    """Compute the rank of a matrix over the rationals via exact Gaussian elimination."""
    mat = [row[:] for row in mat]  # deep copy
    m = len(mat)
    if m == 0:
        return 0
    n = len(mat[0])
    if n == 0:
        return 0
    r = 0
    for col in range(n):
        if r >= m:
            break
        # Find pivot
        pivot_row = None
        for row in range(r, m):
            if mat[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        # Swap
        mat[r], mat[pivot_row] = mat[pivot_row], mat[r]
        pivot_val = mat[r][col]
        # Normalize pivot row
        for c in range(col, n):
            mat[r][c] = mat[r][c] / pivot_val
        # Eliminate
        for row in range(m):
            if row != r and mat[row][col] != 0:
                factor = mat[row][col]
                for c in range(col, n):
                    mat[row][c] = mat[row][c] - factor * mat[r][c]
        r += 1
    return r


def _nullspace_over_Q(mat, max_dim=200):
    """Compute a basis for the nullspace of mat over Q. mat is list of lists of Fractions.
    Returns list of lists of Fractions, each representing a null vector."""
    mat_copy = [row[:] for row in mat]
    m = len(mat_copy)
    if m == 0:
        n = len(mat[0]) if mat and len(mat) > 0 else 0
        return [[Fraction(1) if j == i else Fraction(0) for j in range(n)] for i in range(n)]
    n = len(mat_copy[0])
    r = 0
    pivot_cols = []
    for col in range(n):
        if r >= m:
            break
        pivot_row = None
        for row in range(r, m):
            if mat_copy[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        mat_copy[r], mat_copy[pivot_row] = mat_copy[pivot_row], mat_copy[r]
        pivot_val = mat_copy[r][col]
        for c in range(col, n):
            mat_copy[r][c] = mat_copy[r][c] / pivot_val
        for row in range(m):
            if row != r and mat_copy[row][col] != 0:
                factor = mat_copy[row][col]
                for c in range(col, n):
                    mat_copy[row][c] = mat_copy[row][c] - factor * mat_copy[r][c]
        pivot_cols.append(col)
        r += 1
    # Free variables are columns not in pivot_cols
    free_cols = [c for c in range(n) if c not in pivot_cols]
    basis = []
    for fc in free_cols:
        vec = [Fraction(0)] * n
        vec[fc] = Fraction(1)
        for idx, pc in enumerate(pivot_cols):
            vec[pc] = -mat_copy[idx][fc]
        basis.append(vec)
    return basis


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 120s limit (leave margin)
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    U_f = _mat_to_frac_mat(U)
    V_f = _mat_to_frac_mat(V)
    W_f = _mat_to_frac_mat(W)
    rank = n * m * p
    
    # Try to reduce rank by finding linear dependencies
    # The tensor T is decomposed as sum_c U[:,c] \otimes V[:,c] \otimes W[:,c]
    # We need T to be recovered exactly.
    # A rank-r decomposition means we can find U,V,W of width r.
    # 
    # Strategy: try to find a subset of columns of (U,V,W) that spans the same tensor.
    # We'll iteratively try to remove columns that are linearly dependent.
    
    # For each column c, the contribution is U[:,c] * V[:,c] * W[:,c] (outer product tensor)
    # We want to find the minimal set of columns whose outer-product tensors span T.
    #
    # This is equivalent to: the tensor T lives in R^{n*m*p}. Each column c gives a vector
    # in R^{n*m*p} (the vec of U[:,c] * V[:,c] * W[:,c]). We need these to span T.
    #
    # Actually, the tensor decomposition is: T = sum_c u_c * v_c * w_c
    # where u_c = U[:,c], v_c = V[:,c], w_c = W[:,c]
    # The tensor T is the matrix multiplication tensor for (n,m,p).
    #
    # Let's use a different approach: try random linear combinations of columns.
    # If we can express column c as a linear combination of other columns in a way
    # that preserves the tensor, we can eliminate it.
    
    # A more practical approach: use the fact that the matrix multiplication tensor
    # for (3,4,6) might have rank less than 72.
    # Known: rank(2x2x2) <= 7 (Strassen), rank(2x2x2) >= 7
    # rank(3x3x3) <= 23 (best known upper), >= 19
    # For (3,4,6): we can use Kronecker products.
    # (3,4,6) = (3,2,3) * (1,2,2) via tensor product?
    # Actually, (n,m,p) tensor product with (n',m',p') gives (n*n', m*m', p*p')
    # 3*1=3, 4=2*2, 6=3*2... not quite.
    # 3 = 3*1, 4 = 2*2, 6 = 3*2: so (3,4,6) = (3,2,3) x (1,2,2)
    # rank(3,2,3) <= ? and rank(1,2,2) <= ?
    # (1,2,2): multiply 1x2 by 2x2 -> 1x2. Schoolbook rank = 1*2*2=4.
    # But actually (1,2,2) is just taking 2 linear combinations, rank is 2? No, the tensor is
    # T_{i,j,k} = delta_{i,0} * delta_{j,k}... wait, for (1,2,2):
    # A is 1x2, B is 2x2, C = AB is 1x2.
    # C_{0,k} = sum_j A_{0,j} B_{j,k}
    # Tensor: T_{i,j,k} = 1 if i=0 and... it's just the identity in some sense.
    # Actually the tensor has rank 2: use columns (j=0,k=0) and (j=1,k=1) with appropriate...
    # No, let me think more carefully.
    # For (1,m,p): C_{0,k} = sum_j A_{0,j} B_{j,k}
    # This is just: the output is A * B where A is a row vector and B is m x p.
    # The tensor is sum_j (e_j in A-space) * (e_j * e_k^T in B-space)...
    # Actually it's sum over j of (vector in A-space) * (vector in B-space) for each k.
    # For each k: C_k = A * B[:,k], which is a linear functional of A and B[:,k].
    # The tensor is: T = sum_{j,k} A_j B_{j,k} C_k = sum_k (sum_j A_j B_{j,k}) C_k
    # = sum_{j,k} (e_j) (e_j e_k^T) (e_k) ... this is rank m*p = 2*2 = 4? 
    # No wait, we can do better. T = sum_k (A * B[:,k]) C_k
    # Each term (A * B[:,k]) is a scalar = dot(A, B[:,k])
    # So T = sum_k (dot(A, B[:,k])) C_k
    # This is already a sum of m*p = 4 terms, but can we do less?
    # The tensor T_{i,j,k} for (1,m,p) has T_{0,j,k} = 1 (for all j,k), else 0.
    # This tensor is actually the all-ones matrix in the (j,k) plane, tensored with e_0.
    # The all-ones m x p matrix has rank 1, so the tensor rank is 1!
    # T = e_0 * (sum_j e_j) * (sum_k e_k)... no.
    # T_{0,j,k} = 1 means T = (sum_j e_j) in B-space * (sum_k e_k) in C-space * e_0 in A-space
    # That's rank 1! Indeed: A * B = (A_0 * sum_j B_{j,k})... 
    # Wait: C_k = sum_j A_j B_{j,k}. If we set u = (1,1,...,1) in B-space (length m),
    # v_k = 1 for all k in C-space, then u * B gives (sum_j B_{j,k})_k which is not right.
    # The tensor is T(A,B) = A * B (matrix product). For (1,m,p):
    # C_k = sum_j A_j B_{jk}
    # This is bilinear in A and B. The tensor is sum_{j,k} A_j B_{jk} C_k
    # = sum_j (A_j) * (sum_k B_{jk} C_k)
    # The "B_{jk} C_k" part: for fixed j, sum_k B_{jk} C_k is the j-th component of B^T * C
    # So T = sum_j A_j * (B^T C)_j = A^T (B^T C) = (A^T B^T) C = (BA)^T C
    # Hmm, this is getting complicated. Let me just use the schoolbook and try to reduce.
    
    # Let me try a different structural approach.
    # Use the known Strassen decomposition for 2x2x2 (rank 7) and build up.
    
    # (3,4,6) can be decomposed as:
    # We can write 3 = 2+1, 4 = 2+2, 6 = 2+2+2
    # Block decomposition with blocks of size 2 and 1.
    # But this gets complex. Let me try a more direct approach.
    
    # Direct approach: try to find a rank-r decomposition by solving a system.
    # The matrix multiplication tensor T has T[i,j,k] = 1 if i=k else 0 (for appropriate indices).
    # Actually for (n,m,p): T[i,j,k] = 1 if the corresponding product term exists, i.e.,
    # C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor
    # in the sense that every triple (i,j,k) contributes).
    # Wait no: the tensor is T = sum_{i,j,k} e_i^A * e_j^B * e_k^C where the coefficient is 1
    # and the product is A[i,j] * B[j,k] * C[k]... 
    # The tensor decomposition T = sum_c u_c \otimes v_c \otimes w_c means:
    # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # For matrix multiplication: T[i,j,k] = 1 for all valid i,j,k (since every triple contributes
    # to the sum C[i,k] = sum_j A[i,j]B[j,k]).
    # So we need: sum_c U[i,c] V[j,c] W[k,c] = 1 for all i,j,k.
    # This is a constraint. The schoolbook has U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 for
    # c = (i,j,k), giving exactly one nonzero term per (i,j,k).
    
    # To reduce rank, we need to find U,V,W with fewer columns such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # This is a system of n*m*p = 72 equations. We have r*(n*m + m*p + p*n) = r*(12+24+18) = 54r unknowns.
    # For r < 72, we have 54r < 3888 unknowns but only 72 equations, so there's a lot of freedom.
    # But the constraints are multiplicative (trilinear), making it hard.
    
    # Let me try a structured approach using known decompositions.
    # (3,4,6) = (2,2,2) x (2,2,3)? 2*2=4, 2*2=4, 2*3=6. Yes!
    # (3,4,6) can be embedded in (4,4,6) or we can use (2,2,2) x (2,2,3) = (4,4,6)
    # and then restrict to (3,4,6).
    # rank(2,2,2) = 7 (Strassen)
    # rank(2,2,3) <= ? Schoolbook is 12. Maybe we can do better.
    # The tensor rank of (2,2,3): T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
    # C[i,k] = sum_j A[i,j] B[j,k]
    # T[i,j,k] = 1 for all i,j,k (6*... no, 2*2*3=12 entries, all 1).
    # We need sum_c U[i,c]V[j,c]W[k,c] = 1 for all i,j,k.
    # This is a 2x2x3 tensor that's all ones.
    # The all-ones tensor of shape (2,2,3) has rank... 
    # T = (sum_i e_i) * (sum_j e_j) * (sum_k e_k) -- that's rank 1!
    # Check: U[i,c] = 1, V[j,c] = 1, W[k,c] = 1 for c=0.
    # sum_c U[i,c]V[j,c]W[k,c] = 1*1*1 = 1. Yes!
    # So rank(2,2,3) = 1? That seems too good. Let me verify.
    # The matrix multiplication tensor for (2,2,3) is NOT all ones.
    # T[i,j,k] should be 1 if the (i,j) entry of A and (j,k) entry of B multiply to contribute to (i,k) entry of C.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    # The coefficient of A[i,j] B[j,k] C[i,k] is 1 for each valid (i,j,k).
    # So yes, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # For (2,2,3): T is a 2x2x3 all-ones tensor.
    # And indeed, the all-ones tensor has rank 1: u = (1,1), v = (1,1), w = (1,1,1).
    # U[i,0]=1, V[j,0]=1, W[k,0]=1 gives sum = 1 for all (i,j,k).
    # 
    # WAIT. This means rank(n,m,p) = 1 for ALL n,m,p?! That can't be right.
    # Let me re-examine. The tensor decomposition for matrix multiplication:
    # T = sum_c u_c * v_c * w_c
    # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # For matrix multiplication, we need:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
    # where T[i,j,k] = 1 for all (i,j,k).
    # So the bilinear form is: (A, B) -> sum_{i,j,k} A[i,j] B[j,k] = C
    # The tensor T has T[i,j,k] = 1 for all i,j,k.
    # And the all-ones tensor DOES have rank 1!
    # 
    # But this contradicts known results (Strassen proved rank(2,2,2) >= 7).
    # 
    # I'm making an error. Let me look at this more carefully.
    # The matrix multiplication tensor is NOT simply the all-ones tensor.
    # The correct tensor is: T[i,j,k] = delta_{condition} where the condition ensures
    # that A[i,j] * B[j,k] contributes to C[i,k].
    # For standard matrix multiplication, EVERY (i,j,k) triple is valid:
    # A is n x m, B is m x p, C is n x p.
    # A[i,j] * B[j,k] contributes to C[i,k] for all i in [0,n), j in [0,m), k in [0,p).
    # So T[i,j,k] = 1 for all valid (i,j,k), and the tensor shape is (n, m, p).
    # 
    # The claim that rank(all-ones tensor) = 1 would imply that matrix multiplication
    # can be done with 1 multiplication, which is absurd.
    #
    # The error: the tensor decomposition T = sum_c u_c * v_c * w_c means:
    # T[i,j,k] = sum_c U[c,i] V[c,j] W[c,k] (or similar indexing)
    # The bilinear form is: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # = sum_c (sum_i U[c,i] A[i,?]) * ... 
    # I need to be more careful with the tensor convention.
    #
    # The task says: W[k*n+i, r] -- this is the trace convention.
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # 
    # So the decomposition is: C = A * B where
    # C[i,k] = sum_j A[i,j] B[j,k]
    # And the tensor decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k'} ... no.
    # 
    # Actually, looking at the seed code:
    # U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # The sum over c should give 1 for each (i,j,k) to make:
    # C[i,k] = sum_j sum_c A[i,j] U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] B[j,k]
    # = sum_j (sum_c A[i,j] U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]) B[j,k]
    # For this to equal C[i,k] = sum_j A[i,j] B[j,k], we need:
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all (i,j,k)
    # 
    # So yes, the constraint IS that the trilinear form equals 1 for all (i,j,k).
    # And the all-ones tensor in (n*m, m*p, p*n) space... 
    # 
    # But wait, the indices are coupled! U is indexed by (i,j), V by (j,k), W by (k,i).
    # The constraint is: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all (i,j,k).
    # This is NOT the same as an arbitrary all-ones tensor because the indices are
    # shared/coupled between the three factors.
    #
    # If we set U[i,j,c] = 1, V[j,k,c] = 1, W[k,i,c] = 1 for c=0, then
    # sum_c = 1*1*1 = 1. So rank 1 works?!
    #
    # C[i,k] = sum_j A[i,j] B[j,k] * 1 = (A*B)[i,k]. Yes!
    # 
    # But this contradicts all of tensor rank theory! Let me find my error.
    #
    # AH. I see the issue. The tensor decomposition for matrix multiplication is:
    # T = sum_c u_c * v_c * w_c
    # where u_c is in the A-space (n*m dimensional), v_c in B-space (m*p), w_c in C-space (n*p).
    # And T[i,j,k] is the tensor in the space A*space * B*space * C*space.
    # The bilinear form is: f(A,B) = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # For matrix multiplication, f(A,B) = C = A*B, which as a vector in C-space is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] should be such that sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] e_{i,k}
    # This means T[i,j,k] = delta_{i, i'} delta_{k, k'} ... 
    # 
    # I think the issue is that the tensor T is not a scalar tensor but a vector-valued tensor.
    # Let me use the standard formulation:
    # The matrix multiplication tensor T_{n,m,p} is in F^{n*m} * F^{m*p} * F^{n*p}
    # and is defined by: T = sum_{i,j,k} E_{i,j} * E_{j,k} * E_{i,k}
    # where E_{i,j} is the basis vector in F^{n*m}, etc.
    # 
    # So T[i*m+j, j*p+k, i*p+k] = 1 (in the flattened indexing).
    # Or in the convention of the task: U has shape (n*m, rank), V (m*p, rank), W (p*n, rank).
    # The decomposition is: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # T[a, b, d] = sum_c U[a,c] V[b,c] W[d,c]
    # where a = i*m+j, b = j*p+k, d = k*n+i.
    # T[a,b,d] = 1 if (a,b,d) corresponds to a valid (i,j,k), i.e., 
    # a = i*m+j, b = j*p+k, d = k*n+i for some i,j,k.
    # And T[a,b,d] = 0 otherwise.
    # 
    # So it's NOT all ones! It's a sparse tensor where T[i*m+j, j*p+k, k*n+i] = 1.
    # For (3,4,6): there are 3*4*6 = 72 nonzero entries, one for each (i,j,k).
    # The total tensor space is 12 * 24 * 18 = 5184 dimensional.
    # 
    # Now, the schoolbook has rank 72 with one column per (i,j,k).
    # Can we do better? Yes, that's the whole point of Strassen etc.
    
    # Let me try a concrete approach: use a known decomposition.
    # For (3,4,6), one approach is to use the fact that:
    # (3,4,6) can be built from smaller blocks.
    # 3 = 2+1, 4 = 2+2, 6 = 2+2+2
    # Use block matrix multiplication with 2x2 and 1x1 blocks.
    # This gives a block decomposition.
    
    # Alternatively, let me try to use a computational approach.
    # We'll set up the constraint system and try to find a solution with r < 72.
    
    # For a given r, we need:
    # sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    # This is a system of 72 trilinear equations in 54r variables.
    
    # Let me try a specific structured decomposition.
    # (3,4,6) = (3,2,3) tensor product (1,2,2)? 
    # 3*1=3, 2*2=4, 3*2=6. Yes!
    # rank(3,2,3) * rank(1,2,2) is an upper bound on rank(3,4,6).
    # 
    # (1,2,2): C[0,k] = sum_j A[0,j] B[j,k] for j=0,1; k=0,1.
    # T[a,b,d] where a in [0,2), b in [0,2), d in [0,2).
    # T[0*2+j, j*2+k, k*1+0] = T[j, j*2+k, k] = 1.
    # So T[0, 0, 0] = 1, T[0, 1, 1] = 1, T[1, 2, 0] = 1, T[1, 3, 1] = 1.
    # That's 4 nonzeros. Rank at most 4 (schoolbook).
    # Can we do better? 
    # Constraint: sum_c U[j,c] V[j*2+k,c] W[k,c] = 1 for (j,k) in {(0,0),(0,1),(1,0),(1,1)}.
    # 4 equations, 6r variables. For r=1: U[j,0]V[j*2+k,0]W[k,0] = 1.
    # U[0]V[0]W[0]=1, U[0]V[1]W[1]=1, U[1]V[2]W[0]=1, U[1]V[3]W[1]=1.
    # Let U[0]=U[1]=1, W[0]=W[1]=1, then V[0]=V[1]=V[2]=V[3]=1. Works!
    # So rank(1,2,2) = 1.
    
    # (3,2,3): C[i,k] = sum_j A[i,j] B[j,k] for i=0,1,2; j=0,1; k=0,1,2.
    # T[i*2+j, j*3+k, k*3+i] = 1.
    # 12 nonzeros. Schoolbook rank 12.
    # Can we do better?
    # 
    # This is getting complex. Let me just try to implement a general solver.
    
    # Actually, let me try a different approach: use a random/structured initialization
    # and try to verify if a lower rank is possible.
    
    # For now, let me try the block decomposition (3,2,3) x (1,2,2).
    # rank(3,2,3) <= 12, rank(1,2,2) <= 1, so rank(3,4,6) <= 12.
    # But we need to construct the actual factors.
    
    # Let me implement the Kronecker product of decompositions.
    # If T1 = sum_{c1} u1_{c1} * v1_{c1} * w1_{c1} for (n1,m1,p1)
    # and T2 = sum_{c2} u2_{c2} * v2_{c2} * w2_{c2} for (n2,m2,p2)
    # then T1 x T2 = sum_{c1,c2} (u1_{c1} * u2_{c2}) * (v1_{c1} * v2_{c2}) * (w1_{c1} * w2_{c2})
    # for (n1*n2, m1*m2, p1*p2).
    
    def _kronecker_product(T1, T2):
        """Kronecker product of two tensor decompositions.
        T1: (U1, V1, W1) for (n1,m1,p1), shapes (n1*m1, r1), (m1*p1, r1), (p1*n1, r1)
        T2: (U2, V2, W2) for (n2,m2,p2), shapes (n2*m2, r2), (m2*p2, r2), (p2*n2, r2)
        Returns (U, V, W) for (n1*n2, m1*m2, p1*p2).
        """
        U1, V1, W1 = T1
        U2, V2, W2 = T2
        n1, m1, p1 = U1.shape[0] // V1.shape[0] * 0 + 1, 0, 0  # need to extract dimensions
        # Actually, let me pass dimensions separately.
        pass
    
    # This is getting too complex. Let me try a more direct computational approach.
    
    # Try to find a rank-r decomposition by solving the constraint system.
    # For small r, we can try random initializations and use iterative refinement.
    
    # For now, let me just return the schoolbook decomposition.
    # The score will be 1.0 (same as seed), but at least it's valid.
    # 
    # Actually, let me try one more thing: a simple structured decomposition.
    
    # For (3,4,6), try rank 36 by using a "half" decomposition.
    # Split the 4 middle dimension into two groups of 2.
    # C[i,k] = sum_{j=0}^{1} A[i,j]B[j,k] + sum_{j=2}^{3} A[i,j]B[j,k]
    # = (A1 * B1) + (A2 * B2) where A1 = A[:,0:2], B1 = B[0:2,:], etc.
    # Each term is (3,2,6) multiplication.
    # rank(3,2,6) <= 36 (schoolbook).
    # Total rank <= 72. No improvement.
    
    # Let me try (3,4,6) = (3,4,2) x (1,1,3)? No, 2*3=6 but 1*3=3 not 6.
    # (3,4,6) = (3,2,6) x (1,2,1)? 3*1=3, 2*2=4, 6*1=6. Yes!
    # rank(3,2,6) * rank(1,2,1).
    # (1,2,1): C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]. Rank 2 (schoolbook) or 1?
    # T[j, j*1+0, 0*1+0] = T[j, j, 0] = 1 for j=0,1.
    # sum_c U[j,c]V[j,c]W[0,c] = 1 for j=0,1.
    # r=1: U[0]V[0]W[0]=1, U[1]V[1]W[0]=1. Set W[0]=1, U[0]=U[1]=1, V[0]=V[1]=1. Works!
    # rank(1,2,1) = 1.
    # (3,2,6): rank <= 36. Total rank <= 36.
    
    # But I need to actually construct it. Let me try to build (3,2,6) with a lower rank.
    
    # For (3,2,6): C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    # T[i*2+j, j*6+k, k*3+i] = 1 for i=0,1,2; j=0,1; k=0,1,2,3,4,5.
    # 12 nonzeros.
    # 
    # Let me try to see if rank(3,2,6) < 36.
    # This is the same as multiplying a 3x2 matrix by a 2x6 matrix.
    # C = A * B where A is 3x2, B is 2x6.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # The tensor has 12 nonzero entries. The schoolbook rank is 12 (one per (i,j,k) triple... 
    # wait, 3*2*6 = 36 triples, but only 12 are "active"? No, all 36 are active.
    # T[i*2+j, j*6+k, k*3+i] = 1 for all i in [0,3), j in [0,2), k in [0,6).
    # That's 3*2*6 = 36 nonzero entries. Schoolbook rank = 36.
    
    # I don't think I can easily find a lower rank decomposition without a sophisticated search.
    # Let me just return the schoolbook for now and ensure it's valid.
    
    U_list = [[int(U[i, j]) for j in range(rank)] for i in range(U.shape[0])]
    V_list = [[int(V[i, j]) for j in range(rank)] for i in range(V.shape[0])]
    W_list = [[int(W[i, j]) for j in range(rank)] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
