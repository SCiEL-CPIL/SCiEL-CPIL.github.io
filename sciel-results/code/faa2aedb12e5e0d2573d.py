"""Matrix multiplication 2x4x7 rank-minimal decomposition."""
import time
import itertools

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2x2.
    U, V, W are lists of 7 (row, col) triples for each factor.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # U rows: (i,j) for A, V rows: (j,k) for B, W rows: (k,i) for C
    # For each Strassen term, we need to express it as sum of products.
    # Each term is a sum/difference of individual entries, so each term
    # contributes multiple (U,V,W) triplets.
    
    # Actually, let me think about this differently.
    # The tensor T[i,j,k] = delta (identity) for matrix mult.
    # For 2x2x2, T has 8 non-zero entries: T[i,j,k]=1 for all i,j,k in {0,1}
    # No wait - the matrix multiplication tensor T_{ijk} where C_ik = sum_j A_ij B_jk
    # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for 2x2x2).
    # Rank of all-ones 2x2x2 tensor is 2 (since it's rank-1 as a tensor? No.)
    # Actually T[i,j,k] = 1 for all i,j,k. This is the tensor that maps
    # A ⊗ B -> C where C_ik = sum_j A_ij * B_jk.
    # T[i,j,k] = 1 for all i,j,k in {0,1}.
    # This tensor has rank 2? Let's see: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # If we take U[i,r]=1, V[j,r]=1, W[k,r]=1 for r=0,1... no that gives rank 1
    # but T[i,j,k]=1 which is indeed rank 1! 
    # Wait no. T[i,j,k] = 1 means the tensor is all ones. 
    # all-ones tensor of size 2x2x2 has rank 1: U=V=W=[1,1]^T.
    # But that's wrong for matrix multiplication!
    
    # Let me reconsider. The matrix multiplication tensor for n x m times m x p
    # is T[i, j, k] where i indexes rows of A (and C), j indexes cols of A (rows of B),
    # k indexes cols of B (and C). T[i,j,k] = 1 for all valid i,j,k.
    # Actually NO. T[i,j,k] is the coefficient such that C_ik = sum_{j} A_ij * B_jk * T[i,j,k]
    # So T[i,j,k] = 1 for all i,j,k. Yes it's the all-ones tensor.
    # For 2x2x2: all-ones tensor has rank 2? 
    # Rank of all-ones tensor of order 3 and size d is... 
    # Actually for 2x2x2 all-ones, the rank is 2.
    # Proof: T[i,j,k] = 1 = (1/2)(1+(-1)^{i+j+k})... no.
    # T[i,j,k] = 1. Write as sum_r U_r[i] V_r[j] W_r[k].
    # With r in {0,1}: U_0[i]=1, V_0[j]=1, W_0[k]=1/2; U_1[i]=1, V_1[j]=0, W_1[k]=1/2.
    # That gives T[i,j,k] = 1/2 + 1/2*0 = 1/2. Not right.
    # Try: T[i,j,k] = sum_r a_r b_r c_r where... 
    # Actually the all-ones 2x2x2 tensor: T[i,j,k]=1.
    # T = e1⊗e1⊗e1 + e1⊗e1⊗e2 + e1⊗e2⊗e1 + e1⊗e2⊗e2 + e2⊗e1⊗e1 + e2⊗e1⊗e2 + e2⊗e2⊗e1 + e2⊗e2⊗e2
    # That's rank 8 (schoolbook). But we can do better.
    # T[i,j,k] = 1 for all i,j,k. Note that 1 = (1/2)(1+1) so...
    # Actually T[i,j,k] = (1/2)(1+1)*1 = ... 
    # Let me use: T[i,j,k] = sum_{r=0}^{1} u_r(i) v_r(j) w_r(k)
    # We need: for all i,j,k in {0,1}: sum_r u_r(i)v_r(j)w_r(k) = 1
    # This is 8 equations, 18 unknowns (6 per factor). 
    # By symmetry, try u_0=u_1=[1,1], v_0=[1,0], v_1=[0,1], w_0=[1,1], w_1=[1,1]
    # T[i,j,k] = 1*1*1 + 1*1*1 = 2 for j=0, 1*0*1+1*1*1=1 for j=1. Not uniform.
    
    # OK the all-ones 2x2x2 tensor actually has rank 2? Let me check.
    # Flattening along first mode: 2x4 matrix. All ones 2x4 matrix has rank 1.
    # Flattening along second mode: 2x4 matrix, all ones, rank 1.
    # So the tensor rank is at least 1 (from flattening) and... 
    # Actually I recall that the all-ones tensor of format 2x2x2 has rank 2.
    # Construction: T[i,j,k] = (1/2)(1 + (-1)^i)(1 + (-1)^j)(1 + (-1)^k)... no.
    # Let me just use: T[i,j,k] = 1 = 1*1*1. So rank 1!
    # U[i,0]=1, V[j,0]=1, W[k,0]=1 for all i,j,k.
    # Then sum_r U[i,r]V[j,r]W[k,r] = U[i,0]V[j,0]W[k,0] = 1. YES!
    # So the 2x2x2 all-ones tensor has rank 1!
    
    # But wait - that can't be right for matrix multiplication.
    # Let me re-examine. For matrix multiplication C = AB where A is 2x2, B is 2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # 
    # In the tensor decomposition: C_ik = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Wait, the convention is W[k*n+i, r] so W is indexed by (k,i) for the output.
    # And U[i*m+j, r], V[j*p+k, r].
    # So C_ik = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # For this to give C_ik = sum_j A_ij B_jk, we need:
    # sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = sum_j A_ij B_jk
    # 
    # This means the tensor T[i,j,k] (where the sum over r and j gives the coefficient)
    # should satisfy: for the decomposition to be exact,
    # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] must equal delta(j is the intermediate index)
    # 
    # Actually the tensor is: T_{(i,j), (j,k), (k,i)} where the three indices are
    # (i*m+j), (j*p+k), (k*n+i). The tensor value is 1 for all valid combinations.
    # So it IS the all-ones tensor on the set of valid (i,j,k) triples.
    # 
    # For 2x2x2: the valid triples are all (i,j,k) with i,j,k in {0,1}, so 8 entries, all = 1.
    # The tensor is defined on a 4x4x4 grid but only 8 entries are non-zero.
    # Specifically T[i*m+j, j*p+k, k*n+i] = 1.
    # 
    # So it's NOT the full all-ones tensor. It's a sparse tensor.
    # For 2x2x2: T[0,0,0]=1, T[0,2,1]=1, T[1,0,2]=1, T[1,2,3]=1, T[2,1,0]=1, T[2,3,1]=1, T[3,1,2]=1, T[3,3,3]=1
    # (using 0-indexed: i*m+j where m=2, j*p+k where p=2, k*n+i where n=2)
    # 
    # Let me recompute: n=2, m=2, p=2
    # U index: i*m+j = 2i+j
    # V index: j*p+k = 2j+k
    # W index: k*n+i = 2k+i
    # 
    # (i=0,j=0,k=0): U[0], V[0], W[0]
    # (i=0,j=0,k=1): U[0], V[1], W[2]
    # (i=0,j=1,k=0): U[1], V[2], W[1]
    # (i=0,j=1,k=1): U[1], V[3], W[3]
    # (i=1,j=0,k=0): U[2], V[0], W[2]
    # (i=1,j=0,k=1): U[2], V[1], W[3]
    # (i=1,j=1,k=0): U[3], V[2], W[1]
    # (i=1,j=1,k=1): U[3], V[3], W[3]
    
    # This is the standard matrix multiplication tensor. Its rank for 2x2x2 is 7 (Strassen).
    
    # Strassen's 7 terms:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # 
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # In terms of the tensor decomposition, each Strassen term m_r corresponds to
    # one rank-1 term in the decomposition. For term r:
    # U_r[i*m+j] = coefficient of A_ij in m_r
    # V_r[j*p+k] = coefficient of B_jk in m_r
    # W_r[k*n+i] = coefficient in C_ik from m_r
    
    # m1 = (a+d)(e+h) = (A00+A11)(B00+B11)
    # U: A00->1, A11->1, others 0 => U[0]=1, U[3]=1
    # V: B00->1, B11->1 => V[0]=1, V[3]=1
    # W: contributes to C00 with +1, C11 with +1 => W[0]=1, W[3]=1
    
    # m2 = (c+d)e = (A10+A11)*B00
    # U: A10->1, A11->1 => U[2]=1, U[3]=1
    # V: B00->1 => V[0]=1
    # W: contributes to C10 with +1, C11 with -1 => W[2]=1, W[3]=-1
    
    # m3 = a(f-h) = A00*(B01-B11)
    # U: A00->1 => U[0]=1
    # V: B01->1, B11->-1 => V[1]=1, V[3]=-1
    # W: contributes to C01 with +1, C11 with +1 => W[1]=1, W[3]=1
    
    # m4 = d(g-e) = A11*(B10-B00)
    # U: A11->1 => U[3]=1
    # V: B10->1, B00->-1 => V[2]=1, V[0]=-1
    # W: contributes to C00 with +1, C10 with +1 => W[0]=1, W[2]=1
    
    # m5 = (a+b)h = (A00+A01)*B11
    # U: A00->1, A01->1 => U[0]=1, U[1]=1
    # V: B11->1 => V[3]=1
    # W: contributes to C00 with -1, C01 with +1 => W[0]=-1, W[1]=1
    
    # m6 = (c-a)(e+f) = (A10-A00)*(B00+B01)
    # U: A10->1, A00->-1 => U[2]=1, U[0]=-1
    # V: B00->1, B01->1 => V[0]=1, V[1]=1
    # W: contributes to C11 with +1 => W[3]=1
    
    # m7 = (b-d)(g+h) = (A01-A11)*(B10+B11)
    # U: A01->1, A11->-1 => U[1]=1, U[3]=-1
    # V: B10->1, B11->1 => V[2]=1, V[3]=1
    # W: contributes to C00 with +1 => W[0]=1
    
    # Now build the matrices (4x4 for U, V, W since n*m=4, m*p=4, p*n=4)
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1 (col 0)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    # m2 (col 1)
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    # m3 (col 2)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1
    # m4 (col 3)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    # m5 (col 4)
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    # m6 (col 5)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    # m7 (col 6)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1
    
    return U, V, W


def _schoolbook_2x2x1():
    """Rank-4 schoolbook for 2x2 times 2x1.
    n=2, m=2, p=1. U is 4x4, V is 2x4, W is 2x4.
    C_ik = sum_j A_ij B_jk. Here p=1 so k=0 only.
    C_i0 = A_i0*B_00 + A_i1*B_10
    """
    n, m, p = 2, 2, 1
    rank = n*m*p  # 4
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _combine_blocks(blocks):
    """Combine a list of (U, V, W) blocks into a single decomposition.
    Each block is for a sub-problem. We need to map the local indices to global.
    
    For the 2x4x7 problem, we split the 4 columns of A (and rows of B) into
    groups, and the 7 columns of B (and cols of C) into groups.
    
    Actually, let me think about this more carefully.
    The 2x4x7 multiplication: A is 2x4, B is 4x7, C is 2x7.
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We can partition the 4 intermediate indices j into groups, and the 7 output
    indices k into groups. But the tensor structure couples j and k.
    
    A better approach: partition the problem into independent sub-problems.
    If we group the j indices: j in {0,1} and j in {2,3}, and for each group
    we handle all 7 k values, we get two 2x2x7 sub-problems.
    Or: group k indices and handle all j for each group.
    
    Let me use: split the 4 j-indices into two groups of 2, giving two 2x2x7 blocks.
    Each 2x2x7 block can be further decomposed.
    
    Alternatively, split the 7 k-indices into groups: {0,1,2,3,4} and {5,6}
    giving a 2x4x5 and a 2x4x2 block.
    
    The directive says: "sum of two independent 2x4x4 blocks and one 2x4x3 block"
    So split the 7 k-indices into {0,1,2,3} and {4,5,6}.
    Block 1: 2x4x4 (A is 2x4, B is 4x4, C is 2x4)
    Block 2: 2x4x3 (A is 2x4, B is 4x3, C is 2x3)
    
    For 2x4x4: split j into {0,1} and {2,3}, giving two 2x2x4 blocks.
    Each 2x2x4: split k into {0,1} and {2,3}, giving two 2x2x2 blocks (rank 7 each).
    So 2x4x4 has rank 4*7 = 28.
    
    For 2x4x3: split j into {0,1} and {2,3}, giving two 2x2x3 blocks.
    Each 2x2x3: split k into {0,1} and {2}, giving one 2x2x2 (rank 7) and one 2x2x1 (rank 4).
    So 2x2x3 has rank 7+4 = 11.
    2x4x3 has rank 2*11 = 22.
    
    Total: 28 + 22 = 50.
    
    The directive mentions reducing from 50 to 49 by finding a cross-block dependency.
    Let me first implement the rank-50 version, then try to find the reduction.
    """
    pass


def construct():
    """Construct rank-minimal decomposition for 2x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Strategy: decompose into blocks
    # Split k indices: first 4, last 3
    # Block 1: 2x4x4, Block 2: 2x4x3
    
    # For each block, further decompose using Strassen for 2x2x2 sub-blocks
    # and schoolbook for 2x2x1 sub-blocks.
    
    # Let me build the full decomposition directly.
    # 
    # The 2x4x7 tensor: T[i,j,k] = 1 for i in {0,1}, j in {0,1,2,3}, k in {0,...,6}
    # U index: i*4+j (0..7), V index: j*7+k (0..27), W index: k*2+i (0..13)
    
    # Decomposition plan:
    # For j in {0,1} and k in {0,1,2,3}: use Strassen 2x2x2 (4 sub-blocks, 7 terms each = 28)
    # For j in {2,3} and k in {0,1,2,3}: use Strassen 2x2x2 (4 sub-blocks, 7 terms each = 28)
    # For j in {0,1} and k in {4,5,6}: use 2x2x3 (Strassen 2x2x2 + schoolbook 2x2x1 = 11)
    # For j in {2,3} and k in {4,5,6}: use 2x2x3 (Strassen 2x2x2 + schoolbook 2x2x1 = 11)
    # Total: 28 + 28 + 11 + 11 = 78? No that's wrong.
    
    # Wait, I need to be more careful. The Strassen decomposition works on
    # 2x2x2 sub-problems where the local indices are (i,j,k) with i,j,k in {0,1}.
    # But the global indices are (I,J,K) where I=i (0 or 1), J is the global j, K is global k.
    
    # Let me restructure. The key insight is that for a 2x2x2 sub-problem with
    # local A indices (i, j_local) where j_local in {0,1} maps to global j = j_offset + j_local,
    # and local B indices (j_local, k_local) where k_local in {0,1} maps to global k = k_offset + k_local,
    # the Strassen decomposition gives 7 rank-1 terms.
    
    # For the 2x4x4 block (j in {0,1,2,3}, k in {0,1,2,3}):
    #   j=0,1 & k=0,1: Strassen (j_offset=0, k_offset=0)
    #   j=0,1 & k=2,3: Strassen (j_offset=0, k_offset=2)
    #   j=2,3 & k=0,1: Strassen (j_offset=2, k_offset=0)
    #   j=2,3 & k=2,3: Strassen (j_offset=2, k_offset=2)
    #   4 * 7 = 28 terms
    
    # For the 2x4x3 block (j in {0,1,2,3}, k in {4,5,6}):
    #   j=0,1 & k=4,5: Strassen (j_offset=0, k_offset=4)
    #   j=0,1 & k=6: schoolbook 2x2x1 (j_offset=0, k_offset=6)
    #   j=2,3 & k=4,5: Strassen (j_offset=2, k_offset=4)
    #   j=2,3 & k=6: schoolbook 2x2x1 (j_offset=2, k_offset=6)
    #   2*7 + 2*4 = 22 terms
    
    # Total: 50 terms.
    
    # Now build the U, V, W matrices.
    total_rank = 50
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * total_rank for _ in range(p * n)]  # 14 x 50
    
    # Get Strassen and schoolbook templates
    sU, sV, sW = _strassen_2x2()  # each is 4x4 x 7
    bU, bV, bW = _schoolbook_2x2x1()  # U is 4x4, V is 2x4, W is 2x4
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(j_off, k_off, col):
        """Add Strassen 2x2x2 block for j in {j_off, j_off+1}, k in {k_off, k_off+1}."""
        for local_i in range(2):
            for local_j in range(2):
                for r in range(7):
                    U[local_i * 4 + (j_off + local_j)][col + r] = sU[local_i * 2 + local_j][r]
        for local_j in range(2):
            for local_k in range(2):
                for r in range(7):
                    V[(j_off + local_j) * 7 + (k_off + local_k)][col + r] = sV[local_j * 2 + local_k][r]
        for local_k in range(2):
            for local_i in range(2):
                for r in range(7):
                    W[(k_off + local_k) * 2 + local_i][col + r] = sW[local_k * 2 + local_i][r]
    
    # Helper to add a schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(j_off, k_off, col):
        """Add schoolbook 2x2x1 block for j in {j_off, j_off+1}, k = k_off."""
        for local_i in range(2):
            for local_j in range(2):
                for r in range(4):
                    U[local_i * 4 + (j_off + local_j)][col + r] = bU[local_i * 2 + local_j][r]
        for local_j in range(2):
            for r in range(4):
                V[(j_off + local_j) * 7 + k_off][col + r] = bV[local_j][r]
        for local_i in range(2):
            for r in range(4):
                W[k_off * 2 + local_i][col + r] = bW[local_i][r]
    
    # Add 2x4x4 block (4 Strassen sub-blocks)
    add_strassen(0, 0, col); col += 7
    add_strassen(0, 2, col); col += 7
    add_strassen(2, 0, col); col += 7
    add_strassen(2, 2, col); col += 7
    
    # Add 2x4x3 block (2 Strassen + 2 schoolbook)
    add_strassen(0, 4, col); col += 7
    add_schoolbook_2x2x1(0, 6, col); col += 4
    add_strassen(2, 4, col); col += 7
    add_schoolbook_2x2x1(2, 6, col); col += 4
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
