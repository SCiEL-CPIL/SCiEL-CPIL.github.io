"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    # Build Strassen-based decomposition
    result = _build_strassen_3x5x6(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    U: (4, 7) indexed as U[i*2+j, c] for A[i,j]
    V: (4, 7) indexed as V[j*2+k, c] for B[j,k]
    W: (4, 7) indexed as W[k*2+i, c] for C[k,i]... 
    Wait, the convention is W[k*n+i, c] where n=2, so W[k*2+i, c].
    
    Actually let me re-check. The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    For 2x2x2: n=2, m=2, p=2
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    T[i,j,k] = C[i,k] (output matrix element)
    
    So W[k*2+i, c] contributes to C[i,k].
    
    Strassen's 7 products:
    m0 = (a+d)(e+h)
    m1 = (c+d)(e)
    m2 = (a)(f-h)
    m3 = (d)(g-e)
    m4 = (a+b)(h)
    m5 = (c-a)(e+f)
    m6 = (b-d)(g+h)
    
    C[0,0] = m0 + m3 - m4 + m6
    C[0,1] = m2 + m4
    C[1,0] = m1 + m3
    C[1,1] = m0 - m1 + m2 + m5
    
    Where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    
    U rows (i*2+j):
    a -> row 0: U[0,:]
    b -> row 1: U[1,:]
    c -> row 2: U[2,:]
    d -> row 3: U[3,:]
    
    m0 = (a+d)(e+h): U[0,0]=1, U[3,0]=1
    m1 = (c+d)(e): U[2,1]=1, U[3,1]=1
    m2 = (a)(f-h): U[0,2]=1
    m3 = (d)(g-e): U[3,3]=1
    m4 = (a+b)(h): U[0,4]=1, U[1,4]=1
    m5 = (c-a)(e+f): U[2,5]=1, U[0,5]=-1
    m6 = (b-d)(g+h): U[1,6]=1, U[3,6]=-1
    
    V rows (j*2+k):
    e -> row 0: V[0,:]
    f -> row 1: V[1,:]
    g -> row 2: V[2,:]
    h -> row 3: V[3,:]
    
    m0 = (a+d)(e+h): V[0,0]=1, V[3,0]=1
    m1 = (c+d)(e): V[0,1]=1
    m2 = (a)(f-h): V[1,2]=1, V[3,2]=-1
    m3 = (d)(g-e): V[2,3]=1, V[0,3]=-1
    m4 = (a+b)(h): V[3,4]=1
    m5 = (c-a)(e+f): V[0,5]=1, V[1,5]=1
    m6 = (b-d)(g+h): V[2,6]=1, V[3,6]=1
    
    W rows (k*2+i):
    C[0,0] -> k=0,i=0: row 0: W[0,:]
    C[0,1] -> k=1,i=0: row 2: W[2,:]
    C[1,0] -> k=0,i=1: row 1: W[1,:]
    C[1,1] -> k=1,i=1: row 3: W[3,:]
    
    C[0,0] = m0 + m3 - m4 + m6: W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1
    C[0,1] = m2 + m4: W[2,2]=1, W[2,4]=1
    C[1,0] = m1 + m3: W[1,1]=1, W[1,3]=1
    C[1,1] = m0 - m1 + m2 + m5: W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U: A factors
    U[0, 0] = 1   # a in m0
    U[3, 0] = 1   # d in m0
    U[2, 1] = 1   # c in m1
    U[3, 1] = 1   # d in m1
    U[0, 2] = 1   # a in m2
    U[3, 3] = 1   # d in m3
    U[0, 4] = 1   # a in m4
    U[1, 4] = 1   # b in m4
    U[2, 5] = 1   # c in m5
    U[0, 5] = -1  # -a in m5
    U[1, 6] = 1   # b in m6
    U[3, 6] = -1  # -d in m6
    
    # V: B factors
    V[0, 0] = 1   # e in m0
    V[3, 0] = 1   # h in m0
    V[0, 1] = 1   # e in m1
    V[1, 2] = 1   # f in m2
    V[3, 2] = -1  # -h in m2
    V[2, 3] = 1   # g in m3
    V[0, 3] = -1  # -e in m3
    V[3, 4] = 1   # h in m4
    V[0, 5] = 1   # e in m5
    V[1, 5] = 1   # f in m5
    V[2, 6] = 1   # g in m6
    V[3, 6] = 1   # h in m6
    
    # W: C factors (W[k*2+i, c])
    W[0, 0] = 1   # m0 in C[0,0]
    W[0, 3] = 1   # m3 in C[0,0]
    W[0, 4] = -1  # -m4 in C[0,0]
    W[0, 6] = 1   # m6 in C[0,0]
    W[2, 2] = 1   # m2 in C[0,1]
    W[2, 4] = 1   # m4 in C[0,1]
    W[1, 1] = 1   # m1 in C[1,0]
    W[1, 3] = 1   # m3 in C[1,0]
    W[3, 0] = 1   # m0 in C[1,1]
    W[3, 1] = -1  # -m1 in C[1,1]
    W[3, 2] = 1   # m2 in C[1,1]
    W[3, 5] = 1   # m5 in C[1,1]
    
    return U, V, W


def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using Strassen block structure.
    
    Split A (3x5) into:
      - Top 2x5 block: A_top (2x5)
      - Bottom 1x5 block: A_bot (1x5)
    
    Split B (5x6) into:
      - Left 5x2: B_left
      - Middle 5x2: B_mid  
      - Right 5x2: B_right
    
    For the top 2x5 @ 5x6 part, split B into three 5x2 blocks.
    For each 2x5 @ 5x2 block, we can further split A into 2x2 + 2x3 or use
    a more structured approach.
    
    Actually, let's think more carefully.
    
    A is 3x5, B is 5x6.
    
    Approach: Split the computation into:
    1. Top 2 rows of A (2x5) times B (5x6) -> 2x6 output
    2. Bottom 1 row of A (1x5) times B (5x6) -> 1x6 output
    
    For part 1: 2x5 @ 5x6
      Split B into three 5x2 blocks: B1, B2, B3
      For each 2x5 @ 5x2:
        Split A into 2x2 + 2x3... this gets complicated.
    
    Alternative: Split A into 2x2 + 2x3 + 2x0... no.
    
    Let me try a different split:
    A (3x5) = [A1; A2] where A1 is 2x5 and A2 is 1x5
    B (5x6) = [B1 B2 B3] where each Bi is 5x2
    
    Part 1: A1 (2x5) @ B (5x6)
      = A1 @ [B1 B2 B3] = [A1@B1, A1@B2, A1@B3]
      Each A1@Bi is 2x2. Use Strassen for each: rank 7 each, total 21.
    
    Part 2: A2 (1x5) @ B (5x6)
      = [A2@B1, A2@B2, A2@B3]
      Each A2@Bi is 1x2. This is just a dot product with 5 elements producing 2 outputs.
      Rank for 1x5 @ 5x2: need to compute sum_j A2[j]*B[j,k] for k=0,1.
      This can be done with rank 5 (one product per j, then combine).
      Actually, rank of 1x5x2 tensor: T[j,k] for j=0..4, k=0..1
      T = sum_c U[j,c] V[c,k] ... no, it's a 3-way tensor.
      T[i,j,k] where i is fixed (only i=0 for the 1-row case in the full problem).
      Wait, in the full tensor, for the bottom row:
      T[2, j, k] = sum_c U[2*5+j, c] V[j*6+k, c] W[k*3+2, c]
      
      For a 1x5 @ 5x2 sub-problem, the tensor has shape (1, 5, 2).
      The minimum rank of a 1x5x2 tensor is min(5, 2*1, ...) = 2? No.
      A 1x5x2 tensor T[0,j,k] = sum over j of a[j]*b[j,k].
      This is an outer product of a vector a (5,) with a matrix b (5,2) viewed as...
      Actually T[0,j,k] = a[j] * b[j,k]. This is not a simple outer product.
      
      The tensor T with T[j,k] = a[j]*b[j,k] for fixed a,b is a rank-1 tensor in the
      "flattened" sense? No. T is a 5x2 matrix that equals diag(a) @ b.
      The rank of this as a tensor (1,5,2) is at most 5 (one per j).
      Can we do better? If a has rank 1 (always, it's a vector) and b is 5x2...
      T[j,k] = a[j] b[j,k]. This is a "diagonal times matrix" structure.
      The tensor rank of diag(a) @ B where B is 5x2... 
      Each column k of T is a[k] * B[:,k]... no, T[:,k] = a * B[:,k] (element-wise).
      So T is the Hadamard product of the 5x1 vector a (broadcast) and B.
      The rank of this tensor: we can write T[j,k] = sum_c U[j,c] V[c,k] where
      we need V to be 5x2... but we also have the W factor.
      
      Actually for the full problem, the bottom row computation:
      T[2,j,k] = sum_c U[10+j, c] V[6j+k, c] W[3k+2, c]
      
      For schoolbook, each (j,k) pair uses its own c. That's 5*2=10 products.
      Can we share? T[2,j,k] = a[j] * B[j,k] where a is the bottom row of A and B is the full B.
      
      Let's just use rank 10 for this part (schoolbook for the 1x5x2 sub-tensor).
      
    So total rank = 21 (for 2x5@5x6 via three Strassen 2x2x2) + 10 (for 1x5@5x6) = 31.
    
    Wait, but I need to be more careful. The 2x5 @ 5x2 problem:
    A1 is 2x5, B1 is 5x2.
    T[i,j,k] = sum over the 5 middle indices... no wait.
    
    In matrix multiplication C = A @ B, C[i,k] = sum_j A[i,j] B[j,k].
    The tensor is T[i,j,k] = A[i,j] * B[j,k], and C[i,k] = sum_j T[i,j,k].
    
    But in our tensor decomposition, we're decomposing the multiplication tensor itself:
    T[i,j,k] = 1 for all i,j,k (the structure tensor).
    And we want sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k.
    
    So the tensor is the structure tensor of matrix multiplication, which is 1 for all (i,j,k).
    
    OK so my approach is correct. Let me just implement it.
    
    For 2x5 @ 5x2:
    Split A1 (2x5) into two 2x2 blocks + one 2x1? No, 5 = 2+2+1.
    Or just use the fact that 2x5 @ 5x2 can be done by splitting B into columns.
    
    Actually, the cleanest approach:
    For the top 2x2 sub-block: use Strassen (rank 7).
    For 2x5 @ 5x2: split the 5 middle dimension.
    
    Hmm, let me think about this differently.
    
    The structure tensor for n x m @ m x p multiplication has T[i,j,k] = 1.
    We need rank-r factors such that sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
    
    For n=3, m=5, p=6: we need this for i in {0,1,2}, j in {0..4}, k in {0..5}.
    
    Let me try a block decomposition:
    Split i into {0,1} and {2}.
    Split k into {0,1}, {2,3}, {4,5}.
    
    For block (i in {0,1}, k in {0,1}): 2x5x2 sub-tensor.
    For block (i in {0,1}, k in {2,3}): 2x5x2 sub-tensor.
    For block (i in {0,1}, k in {4,5}): 2x5x2 sub-tensor.
    For block (i in {2}, k in {0,1}): 1x5x2 sub-tensor.
    For block (i in {2}, k in {2,3}): 1x5x2 sub-tensor.
    For block (i in {2}, k in {4,5}): 1x5x2 sub-tensor.
    
    The j dimension (0..4) is shared across all blocks.
    
    For each 2x5x2 block: T[i,j,k] = 1 for i in {0,1}, j in {0..4}, k in the 2-column block.
    This is the structure tensor for 2x5 @ 5x2 multiplication.
    Minimum rank of 2x5x2 tensor: at least max(2*2, 5*2, 2*5) = max(4,10,10) = 10?
    No, that's a lower bound for the tensor rank. Actually the trivial lower bound is max(n*p, m*p, n*m) but that's not tight.
    For 2x2x2, rank is 7 (Strassen). For 2x5x2... 
    
    Actually, I realize I can split the j dimension too.
    For 2x5x2: split j into {0,1}, {2,3}, {4}.
    Block (i in {0,1}, j in {0,1}, k in {0,1}): 2x2x2 -> Strassen, rank 7.
    Block (i in {0,1}, j in {2,3}, k in {0,1}): 2x2x2 -> Strassen, rank 7.
    Block (i in {0,1}, j in {4}, k in {0,1}): 2x1x2 -> rank 2 (schoolbook: 2*1*2=4, but can do with 2).
    
    Wait, 2x1x2 tensor: T[i,j,k] = 1 for i in {0,1}, j=0, k in {0,1}.
    This is just 4 ones. We can decompose with rank 2:
    U[0,0]=1, U[1,0]=1, U[0,1]=1, U[1,1]=1 (for the two i values, 2 columns)
    V[0,0]=1, V[0,1]=1
    W[0,0]=1, W[1,0]=1, W[0,1]=1, W[1,1]=1
    Check: sum_c U[i,0,c]*V[0,c]*W[k,c] for i in {0,1}, k in {0,1}
    = sum_c U[i,0,c] * 1 * W[k,c]
    = U[i,0,0]*W[k,0] + U[i,0,1]*W[k,1]
    = 1*1 + 1*1 = 2. Not 1.
    
    Let me redo. For 2x1x2:
    Need T[i,0,k] = 1 for all i,k.
    sum_c U[i*1+0, c] V[0*2+k, c] W[k*2+i, c] = 1
    
    With rank 2:
    c=0: U[i,0,0]=1 for both i, V[0,0]=1, V[1,0]=0, W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1
    Hmm this is getting complicated with the indexing.
    
    Let me just use schoolbook for the small blocks and Strassen for 2x2x2 blocks.
    
    Total rank estimate:
    3 blocks of 2x2x2 (Strassen): 3 * 7 = 21
    For the j=4 part (2x1x2): 4 (schoolbook)
    For the bottom row (i=2): 1x5x6, which splits into 3 blocks of 1x2x2 (j in pairs) + 1x1x2.
    1x2x2: rank 4 (schoolbook) each, 3 of them = 12
    1x1x2: rank 2 (schoolbook)
    Total for bottom: 12 + 2 = 14
    Total: 21 + 4 + 14 = 39
    
    That's better than 90 but not great. Let me see if I can do better for the 1x5x2 blocks.
    
    1x5x2 tensor: T[0,j,k] = 1 for j in {0..4}, k in {0,1}.
    Need sum_c U[j, c] V[j*2+k, c] W[2*0+k, c]... wait, n=1 for this sub-problem.
    W[k*1+0, c] = W[k, c].
    
    So sum_c U[j,c] V[j*2+k, c] W[k, c] = 1 for all j,k.
    
    With rank 5:
    For each j, set U[j, j] = 1, others 0.
    V[j*2+k, j] = 1 for k=0,1.
    W[k, j] = 1 for all j, k.
    Then sum_c U[j,c] V[j*2+k, c] W[k, c] = U[j,j] * V[j*2+k, j] * W[k, j] = 1*1*1 = 1. ✓
    
    So 1x5x2 has rank at most 5. Similarly 1x1x2 has rank 2.
    
    Revised total:
    Top 2 rows, j in {0,1}: Strassen 2x2x2, rank 7
    Top 2 rows, j in {2,3}: Strassen 2x2x2, rank 7  
    Top 2 rows, j=4: 2x1x2, rank 4
    Bottom row, j in {0,1}: 1x2x2, rank 4
    Bottom row, j in {2,3}: 1x2x2, rank 4
    Bottom row, j=4: 1x1x2, rank 2
    For each of these, we have 3 k-blocks (k in {0,1}, {2,3}, {4,5}).
    
    Total: 3 * (7 + 7 + 4 + 4 + 4 + 2) = 3 * 28 = 84. Still close to 90.
    
    Hmm, I need a better approach. Let me just implement the Strassen for the 2x2x2 blocks
    and schoolbook for the rest, and see what rank we get. The key insight is that we can
    potentially do better by sharing factors across k-blocks.
    
    Actually, let me just implement a clean Strassen-based construction and verify it works.
    Even if the rank isn't optimal, it should be better than 90.
    """
    # Strassen factors for 2x2x2
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Total rank: 3 k-blocks * (2 Strassen blocks + 1 schoolbook for j=4)
    # For each k-block:
    #   j in {0,1}: Strassen (7)
    #   j in {2,3}: Strassen (7) 
    #   j = 4: schoolbook 2x1x2 (4)
    # Per k-block: 7+7+4 = 18
    # For bottom row (i=2):
    #   j in {0,1}: schoolbook 1x2x2 (4)
    #   j in {2,3}: schoolbook 1x2x2 (4)
    #   j = 4: schoolbook 1x1x2 (2)
    # Per k-block for bottom: 4+4+2 = 10
    # Total per k-block: 18+10 = 28
    # 3 k-blocks: 84
    
    rank = 84
    U = np.zeros((n * m, rank))  # 15 x 84
    V = np.zeros((m * p, rank))  # 30 x 84
    W = np.zeros((p * n, rank))  # 18 x 84
    
    col = 0
    
    for k_block in range(3):
        k0 = k_block * 2  # start k index
        k1 = k0 + 1       # end k index
        
        # Top 2 rows (i=0,1), j in {0,1}: Strassen
        for c in range(7):
            # U: A[i,j] for i in {0,1}, j in {0,1}
            U[0*5 + 0, col] = sU[0, c]   # i=0, j=0
            U[0*5 + 1, col] = sU[1, c]   # i=0, j=1
            U[1*5 + 0, col] = sU[2, c]   # i=1, j=0
            U[1*5 + 1, col] = sU[3, c]   # i=1, j=1
            # V: B[j,k] for j in {0,1}, k in {k0,k1}
            V[0*6 + k0, col] = sV[0, c]   # j=0, k=k0
            V[0*6 + k1, col] = sV[1, c]   # j=0, k=k1
            V[1*6 + k0, col] = sV[2, c]   # j=1, k=k0
            V[1*6 + k1, col] = sV[3, c]   # j=1, k=k1
            # W: C[k,i] for k in {k0,k1}, i in {0,1}
            W[k0*3 + 0, col] = sW[0, c]   # k=k0, i=0
            W[k0*3 + 1, col] = sW[1, c]   # k=k0, i=1
            W[k1*3 + 0, col] = sW[2, c]   # k=k1, i=0
            W[k1*3 + 1, col] = sW[3, c]   # k=k1, i=1
            col += 1
        
        # Top 2 rows (i=0,1), j in {2,3}: Strassen
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {0,1}: schoolbook 1x2x2
        for j in range(2):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {2,3}: schoolbook 1x2x2
        for j in range(2, 4):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j=4: schoolbook 1x1x2
        for k in range(2):
            U[2*5 + 4, col] = 1
            V[4*6 + k0 + k, col] = 1
            W[(k0 + k)*3 + 2, col] = 1
            col += 1
    
    if col != rank:
        # Truncate or pad
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
