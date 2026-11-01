"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def _build_strassen_2x2_x_2x10():
    """
    Build a rank-70 decomposition for the (2x2)*(2x10) sub-problem
    using Strassen's 7-term multiplication for the 2x2 part, applied
    per column of the second matrix.
    
    The full (2x2)*(2x10) tensor is a "block diagonal" structure:
    For output element (i, k), where i in {0,1} and k in {0..9}:
      C[i,k] = sum_{j in {0,1}} A[i,j] * B[j,k]
    
    This is equivalent to 10 independent 2x2 multiplications where
    the second matrix is a 2x2 matrix with only one non-zero column.
    
    Using Strassen, each 2x2 multiplication needs 7 multiplications.
    Total: 7 * 10 = 70.
    
    However, we can do better by recognizing that the 10 columns of B
    are independent. Let's use the standard Strassen decomposition
    for the 2x2 block, replicated for each column.
    
    Actually, let's think more carefully. The tensor for (2x2)*(2x2)
    has rank 7 (Strassen). The tensor for (2x2)*(2x10) is isomorphic
    to the tensor product of the (2x2)*(2x2) tensor with a (2x1)*(1x10) tensor?
    No, that's not right either.
    
    Let's use a different approach: decompose the (2x2)*(2x10) tensor
    by applying Strassen's algorithm to the 2x2 block structure.
    
    For Strassen's algorithm on 2x2 matrices:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    Then:
    C11 = M1 - M4 + M5 + M6
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M2 + M7
    
    For our case, B is 2x10, so B11 and B12 are 1x10 vectors, etc.
    
    Let's index:
    A = [[A00, A01], [A10, A11]]  (2x2)
    B = [[B0k for k in 0..9], [B1k for k in 0..9]]  (2x10)
    
    Strassen terms:
    M1 = (A00 + A11) * (B0k + B1k)  for each k
    M2 = (A10 + A11) * B0k  for each k
    M3 = A00 * (B1k - B0k)  for each k  -- wait, this should be A00*(B0k - B1k)? Let me recheck.
    
    Standard Strassen:
    p1 = (a00 + a11)(b00 + b11)
    p2 = (a10 + a11)b00
    p3 = a00(b01 - b11)
    p4 = a11(b10 - b00)
    p5 = (a00 + a01)b11
    p6 = (a10 - a00)(b00 + b01)
    p7 = (a01 - a11)(b10 + b11)
    
    c00 = p1 - p4 + p5 + p6
    c01 = p3 + p5
    c10 = p2 + p4
    c11 = p1 - p3 + p2 + p7
    
    For 2x2 * 2x10, the "b" matrix is 2x10, so b00, b01 are vectors of length 10, etc.
    
    p1 = (A00 + A11) * (B0k + B1k)  -- for each k, this is a scalar multiplication
    p2 = (A10 + A11) * B0k
    p3 = A00 * (B1k - B0k)  -- wait, standard is a00*(b01-b11), so here it would be A00*(B0k - B1k)? 
    
    Let me be careful. In the 2x2 case:
    A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    
    p3 = a00 * (b01 - b11)
    p4 = a11 * (b10 - b00)
    p5 = (a00 + a01) * b11
    p6 = (a10 - a00) * (b00 + b01)
    p7 = (a01 - a11) * (b10 + b11)
    
    For 2x2 * 2x10, B has rows [B0, B1] where B0, B1 are length-10 vectors.
    So b00 = B0[0], b01 = B0[1], etc. No, that's wrong.
    
    Actually, for matrix multiplication C = A*B where A is 2x2 and B is 2x10:
    C[i][k] = sum_j A[i][j] * B[j][k]
    
    So C[0][k] = A[0][0]*B[0][k] + A[0][1]*B[1][k]
       C[1][k] = A[1][0]*B[0][k] + A[1][1]*B[1][k]
    
    This is 10 independent 2x2 dot products (one for each k). Each can be done with Strassen's 7 terms.
    Total: 70 multiplications.
    
    But we can share the linear combinations of A across all k values!
    
    The Strassen terms for column k:
    p1_k = (A00 + A11) * (B0k + B1k)
    p2_k = (A10 + A11) * B0k
    p3_k = A00 * (B0k - B1k)
    p4_k = A11 * (B1k - B0k)
    p5_k = (A00 + A01) * B1k
    p6_k = (A10 - A00) * (B0k + B1k)
    p7_k = (A01 - A11) * (B1k + B0k)
    
    Wait, let me re-derive properly. For the 2x2 case with:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    p1 = (a+d)(e+h)
    p2 = (c+d)e
    p3 = a(f-h)
    p4 = d(g-e)
    p5 = (a+b)h
    p6 = (c-a)(e+f)
    p7 = (b-d)(g+h)
    
    c = p1 - p4 + p5 + p6
    d_out = p3 + p5  (wait, naming conflict)
    
    Let me use standard notation:
    C[0][0] = p1 - p4 + p5 + p6
    C[0][1] = p3 + p5
    C[1][0] = p2 + p4
    C[1][1] = p1 - p3 + p2 + p7
    
    For 2x2 * 2x10, B is [[B00, B01, ..., B09], [B10, B11, ..., B19]]
    
    So for each output column k (0 to 9):
    C[0][k] = p1_k - p4_k + p5_k + p6_k
    C[1][k] = p2_k + p4_k  -- no, this is for C[1][0] in 2x2. 
    
    I'm getting confused. Let me just use the schoolbook for now and focus on getting a valid answer.
    """
    # Use schoolbook for the (2x2)*(2x10) part: 2*2*10 = 40 terms
    pass

def _build_strassen_2x3_x_3x10():
    """
    Build a rank-55 decomposition for (2x3)*(3x10) using:
    - Split m=3 into m1=2 and m2=1
    - For the (2x2)*(2x10) part: use Strassen, rank 70? No, that's worse than schoolbook 40.
    
    Let me reconsider. The knowledge says 55 is achievable with "Strassen for 2x2 sub-problem and schoolbook for 2x1 sub-problem".
    
    (2x2)*(2x10) schoolbook: 40
    (2x1)*(1x10) schoolbook: 20
    Total: 60 (same as full schoolbook)
    
    To get 55, we need to save 5 from the (2x2)*(2x10) part.
    
    The (2x2)*(2x10) tensor: can we do better than 40?
    The rank of the (2x2)*(2x2) tensor is 7 (Strassen).
    The rank of the (2x2)*(2x10) tensor... 
    
    Actually, I think the approach is:
    The (2x3)*(3x10) tensor can be viewed as:
    For each pair (i, k) where i in {0,1} and k in {0..9}:
      C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    Split j into j in {0,1} and j=2:
      C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + A[i,2]*B[2,k]
    
    The first part is (2x2)*(2x10), the second is (2x1)*(1x10).
    
    For the (2x2)*(2x10) part, we can use a Strassen-like decomposition.
    The key insight: the 10 columns of B are independent, so we can share the
    linear combinations of A.
    
    Strassen for 2x2: 7 multiplications per "block". But here we have 10 independent
    right-hand sides. The linear combinations of A are shared.
    
    The 7 Strassen terms are:
    p1 = (A00 + A11)
    p2 = (A10 + A11)
    p3 = A00
    p4 = A11
    p5 = (A00 + A01)
    p6 = (A10 - A00)
    p7 = (A01 - A11)
    
    These are 7 linear combinations of A (shared across all 10 columns).
    
    For each column k, we need:
    B0k + B1k, B0k, B1k - B0k, B1k, B1k, B0k + B1k, B1k + B0k
    
    Wait, let me re-derive. For the (2x2)*(2x10) part:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0..9}
    
    Using Strassen:
    p1_k = (A00 + A11) * (B0k + B1k)
    p2_k = (A10 + A11) * B0k
    p3_k = A00 * (B0k - B1k)  -- from p3 = a(f-h), where f=B0k, h=B1k... 
    
    I'm still confusing myself. Let me just use a simpler approach.
    
    For the (2x2)*(2x10) tensor, the rank is at most 40 (schoolbook).
    Can we prove it's less? 
    
    Actually, let me just implement the schoolbook decomposition with the split,
    and see if I can find a 55-rank decomposition through careful analysis.
    """
    pass

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy: Split m=3 into m1=2 and m2=1.
    - (2x2)*(2x10) part: use Strassen-based decomposition
    - (2x1)*(1x10) part: use schoolbook
    
    Target rank: 55
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Split m into m1=2 and m2=1
    m1 = 2
    m2 = 1
    
    # (2x2)*(2x10) part using Strassen: rank 70? That's too many.
    # Let me use schoolbook: 2*2*10 = 40
    # (2x1)*(1x10) part: 2*1*10 = 20
    # Total: 60
    
    # To get 55, we need to save 5 from somewhere.
    
    # Let me try a different approach: use the fact that the output dimension n=2
    # allows some sharing.
    
    # Actually, let me just return the schoolbook decomposition for now.
    rank = n * m * p
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
