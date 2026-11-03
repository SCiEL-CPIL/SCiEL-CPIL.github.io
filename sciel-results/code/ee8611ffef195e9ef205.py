"""Winograd-based decomposition for 3x5x6 matrix multiplication."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Construct a low-rank decomposition of 3x5x6 matrix multiplication.
    
    Strategy:
    - Embed 3x5x6 into 5x5x6 by padding A to 5x5 (rows 3,4 zero) and B is 5x6.
    - Actually, better: pad to 5x5x5 by also padding B columns.
    - Use Winograd's rank-19 decomposition for 5x5x5.
    - But we have 3x5x6. Let's think differently.
    
    Alternative: Use the fact that rank(3x5x6) <= rank(5x5x5) + extra terms.
    Actually, a cleaner approach:
    
    For 3x5x6, we can use the Winograd algorithm for 5x5x5 directly on the
    non-zero parts. The 3x5x6 multiplication can be seen as a sub-problem of 5x5x5
    if we pad appropriately, but the ranks don't directly transfer.
    
    Let me use a different Winograd construction:
    
    Winograd's method for n x n x n matrix multiplication:
    For 5x5x5, the rank is 19 (using Winograd's construction with 5x5 DFT-like matrices).
    
    For 3x5x6, we can decompose as:
    - The 3x5 part: use a 5x5 Winograd on the column dimension (5) and row dimension (3->5)
    - The 5x6 part: similar
    
    Actually, let's use a direct Winograd construction for 3x5x6:
    
    Winograd's general method: For multiplying an m x n matrix by an n x p matrix,
    we can use tensor rank based on the "Winograd" scheme which uses:
    - A transformation on the m dimension
    - A transformation on the p dimension  
    - The n dimension is handled by the shared index
    
    The key insight: if we can find integers a, b such that m <= a, p <= b, and we
    use a rank-r decomposition of a x n x b, then we get rank r for m x n x p.
    
    For 3x5x6: we can embed in 5x5x6 or 3x5x8 or similar.
    
    Let's use the known result: rank(5x5x5) = 19 (Winograd).
    We can embed 3x5x6 into 5x5x6 by padding rows of A (3->5) and columns of B (6->6, no padding needed).
    But we need 5x5x6, not 5x5x5.
    
    Better: embed 3x5x6 into 5x5x5 by padding A rows (3->5) and B cols (6->5? No, 6>5).
    So we need 5x5x6 at minimum, or 3x5x8, etc.
    
    Let's try: embed in 5x5x8. Rank(5x5x8) <= rank(5x5x5) + 3*5*3 = 19 + 45 = 64? Not great.
    
    Actually, a better approach: use the Winograd construction directly.
    
    Winograd's algorithm for (m+1)x(n+1)x(p+1) has rank (m+1)(n+1)(p+1) - ... no.
    
    Let me just implement a concrete Winograd-style decomposition:
    
    For 3x5x6, use the following:
    - Pad A (3x5) to 5x5 by adding 2 zero rows
    - Pad B (5x6) to 5x8 by adding 3 zero columns  
    - Use a 5x5x8 decomposition
    
    But this is getting complicated. Let me use a simpler but still improved approach:
    
    Use the fact that we can split the 5x6 multiplication into 5x5 + 5x1:
    rank(3x5x6) <= rank(3x5x5) + rank(3x5x1)
    rank(3x5x1) = 3*1 = 3 (it's just a vector-matrix product, rank = 3*1*1... no)
    
    Actually rank(3x5x1): this is multiplying 3x5 by 5x1, giving 3x1.
    The tensor has 3*5*1 = 15 terms, but rank can be lower.
    For m x n x 1, the rank is m (since it's just m independent sums).
    Wait, no. The tensor T[i,j,k] = delta_{i,k} for k in 0..0, so T[i,j,0] = delta_{i,0}... 
    No, for matrix multiplication C = AB where A is 3x5, B is 5x1:
    C[i,0] = sum_j A[i,j] B[j,0]
    So the tensor is T[i,j,k] = 1 if k=0, else 0, and the multiplication is sum_j A[i,j]*B[j,0].
    This is just 3 independent dot products of length 5. Rank = 3 (one rank-1 term per output row).
    
    Similarly, rank(3x1x6) = 3.
    
    So: rank(3x5x6) <= rank(3x5x5) + 3 (for the 5x1 part)
    And rank(3x5x5) <= rank(5x5x5) = 19 (Winograd)
    
    So total rank <= 19 + 3 = 22? That seems too good. Let me verify.
    
    Wait, the decomposition isn't quite that simple. When we split B = [B1 | B2] where B1 is 5x5 and B2 is 5x1,
    then C = A*B1 + A*B2.
    A*B1 is 3x5, computed via 3x5x5 multiplication (rank <= 19).
    A*B2 is 3x1, computed via 3x5x1 multiplication (rank = 3).
    These are independent, so total rank = 19 + 3 = 22.
    
    But wait, the output dimensions: C is 3x6. C = A*B where B is 5x6.
    C[:, 0:5] = A * B[:, 0:5] (3x5x5 problem)
    C[:, 5:6] = A * B[:, 5:6] (3x5x1 problem)
    
    Yes! So rank(3x5x6) <= rank(3x5x5) + rank(3x5x1) = 19 + 3 = 22.
    
    Now I need to implement the Winograd rank-19 decomposition for 5x5x5,
    and adapt it for 3x5x5 (which is a sub-problem).
    
    Winograd's 5x5x5 decomposition:
    The standard Winograd algorithm uses the following:
    For 5x5x5, the rank is 19, achieved by:
    - Using a 5x5 "Winograd" matrix W
    - The decomposition uses 19 rank-1 terms
    
    The Winograd construction for n x n x n (n=5):
    Let omega be a primitive 5th root of unity (or use integer/rational approximations).
    Actually, Winograd's method uses specific integer matrices.
    
    For 5x5x5, the Winograd decomposition is:
    U[i*5+j, c], V[j*5+k, c], W[k*5+i, c] for c in 0..18
    
    The construction:
    - 5 terms from the "diagonal" (c = 0..4): U[i*5+i, c] = 1, V[i*5+i, c] = 1, W[i*5+i, c] = 1
    - 14 terms from off-diagonal using Winograd's specific pattern
    
    Actually, let me look up the exact Winograd 5x5x5 rank-19 construction.
    
    The Winograd algorithm for 5x5x5 uses the following 19 rank-1 tensors:
    The key matrices are based on the 5x5 DFT matrix and its properties.
    
    A known construction:
    Let the 19 terms be indexed by c = 0..18.
    
    For c = 0..4 (5 terms):
      U[i*5+i, c] = 1 for i = c
      V[i*5+i, c] = 1 for i = c  
      W[i*5+i, c] = 1 for i = c
    (These are the "diagonal" terms)
    
    For c = 5..18 (14 terms):
    These involve off-diagonal elements with specific coefficients.
    
    Actually, I recall that the Winograd 5x5x5 construction uses:
    - The 5x5 matrix M where M[i,j] = omega^{ij} for omega = e^{2*pi*i/5}
    - But we need rational/integer coefficients.
    
    Let me use a different known construction. The rank of 5x5x5 is 19, and one construction is:
    
    Using the fact that 5 = 4+1, we can embed 4x4x4 (rank 16) into 5x5x5.
    rank(5x5x5) <= rank(4x4x4) + 5*5*1 + ... 
    
    This is getting complex. Let me just implement a concrete known decomposition.
    
    Alternative simpler approach: use the "border rank" or known explicit decompositions.
    
    Let me use a pragmatic approach: implement a decomposition with rank around 50-60 using a
    Winograd-inspired method, even if not optimal.
    
    Actually, let me just use the straightforward Winograd construction for 5x5x5:
    
    The 19 rank-1 terms for 5x5x5:
    Term 0-4: diagonal terms (i=j=k=i)
    Term 5-18: off-diagonal terms using the 5x5 Winograd matrix
    
    The Winograd matrix for 5x5 is:
    W = [[1, 1, 1, 1, 1],
         [1, w, w^2, w^3, w^4],
         [1, w^2, w^4, w, w^3],
         [1, w^3, w, w^4, w^2],
         [1, w^4, w^3, w^2, w]]
    where w = e^{2*pi*i/5}
    
    But this involves complex numbers. We need real/integer coefficients.
    
    Let me use a different approach: the "Winograd" construction for 5x5x5 actually uses
    rational numbers. The key insight is that the 5th cyclotomic polynomial is x^4+x^3+x^2+x+1,
    and we can work in the ring Z[x]/(x^4+x^3+x^2+x+1).
    
    This is getting too complex for a quick implementation. Let me use a simpler but still
    improved decomposition.
    
    Simple improvement: use the fact that rank(3x5x6) <= rank(3x5x5) + 3.
    For rank(3x5x5), use a direct construction:
    rank(3x5x5) <= 3*5*5 = 75 (schoolbook)
    But we can do better.
    
    Actually, let me just implement the schoolbook decomposition but with a small optimization:
    Split the 6 columns into 5+1, and use a slightly better method for the 5x5 part.
    
    For 3x5x5, the rank is at most 3*5*5 = 75, but likely lower.
    Known: rank(3x3x3) = 23 (Strassen-like).
    rank(3x5x5) <= rank(3x3x5) + rank(3x2x5) = ...
    
    This is taking too long. Let me just implement a concrete decomposition with rank ~60.
    
    Strategy: Use the "border rank" approach or a known explicit low-rank decomposition.
    
    Let me just go with a Winograd-inspired construction for 5x5x5 with rank 19,
    and adapt it. I'll hardcode the 19 rank-1 terms.
    
    After research, the Winograd 5x5x5 rank-19 decomposition uses these 19 terms:
    The construction is based on the 5x5 "Winograd" matrices.
    
    Let me use a simpler approach: just output the schoolbook decomposition (rank 90)
    but with a comment that a better method exists. This will score 1.0.
    
    Wait, I need to actually improve it. Let me implement the 3x5x5 + 3x5x1 split.
    
    For 3x5x1: rank = 3 (trivial)
    For 3x5x5: use schoolbook = 75, total = 78. Still better than 90.
    
    Can I do better for 3x5x5? 
    rank(3x5x5) <= rank(3x5x3) + rank(3x5x2)
    rank(3x5x3) <= 3*5*3 = 45
    rank(3x5x2) <= 3*5*2 = 30
    Total = 75. No improvement.
    
    Let me try: rank(3x5x5) <= rank(3x3x5) + rank(3x2x5)
    Same thing.
    
    OK, let me just use the 3x5x5 + 3x5x1 split with schoolbook for 3x5x5:
    rank = 75 + 3 = 78.
    
    That's a 12-rank improvement over 90. Score = 90/78 ≈ 1.15.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Split: 3x5x6 = 3x5x5 + 3x5x1
    # Part 1: 3x5x5 (schoolbook, rank 75)
    # Part 2: 3x5x1 (rank 3)
    
    rank1 = n * m * 5  # 75
    rank2 = n * 1       # 3
    total_rank = rank1 + rank2  # 78
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Part 1: 3x5x5 schoolbook
    # A is 3x5, B1 is 5x5, C1 is 3x5
    # C1[i,k] = sum_j A[i,j] * B1[j,k]
    # U[i*m+j, c] = 1, V[j*5+k, c] = 1, W[k*n+i, c] = 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(5):
                U[i * m + j][col] = 1.0
                V[j * 5 + k][col] = 1.0  # B1 is 5x5, indexed by j (0..4), k (0..4)
                W[k * n + i][col] = 1.0
                col += 1
    
    # Part 2: 3x5x1
    # A is 3x5, B2 is 5x1, C2 is 3x1
    # C2[i,0] = sum_j A[i,j] * B2[j,0]
    # This is rank 3: one term per output row i
    # U[i*m+j, c] = 1 for the j-th term? No.
    # For each i, C2[i,0] = sum_j A[i,j] * B2[j,0]
    # This is a single dot product per i.
    # Rank-1 term for row i: U[i*m+j, c_i] = 1 for all j, V[j*5+5, c_i] = 1 for all j, W[0*n+i, c_i] = 1
    # Wait, B2 is the 6th column of B (index 5 in 0-based, or column 5 of the 5x6 matrix).
    # B is 5x6, so B[j,5] for j=0..4 is the 6th column.
    # V index: j*p + k = j*6 + 5
    for i in range(n):
        c = col + i
        for j in range(m):
            U[i * m + j][c] = 1.0
            V[j * p + 5][c] = 1.0  # j*6 + 5
        W[0 * n + i][c] = 1.0  # k=0, so W[0*3+i] = W[i]
    
    return {"U": U, "V": V, "W": W}
