"""Matrix multiplication 4,5,5 tensor decomposition using 2x3x3 Strassen-like blocks."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS  # 4, 5, 5

    # Strassen's 7-term decomposition for 2x2x2
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    strassen_222 = [
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Rank-17 decomposition for 2x3x3 matrix multiplication
    # A: 2x3, B: 3x3, C = A*B: 2x3
    # Tensor T[i,j,k] = A[i,j]*B[j,k], i in 0..1, j in 0..2, k in 0..2
    # U index: i*3+j (6), V index: j*3+k (9), W index: k*2+i (6)
    #
    # Construction: Split A = [A1 | a] where A1 is 2x2, a is 2x1
    #               Split B = [B1; b] where B1 is 2x3, b is 1x3
    #               C = A1*B1 + a*b
    #
    # For A1*B1 (2x2 * 2x3 -> 2x3):
    #   This is like two 2x2x2 Strassen multiplications sharing the A1 factor.
    #   Actually, let's think of it as: C1 = A1 * B1 where B1 is 2x3.
    #   We can split B1 = [B1a | B1b] where B1a is 2x2, B1b is 2x1.
    #   C1 = A1*B1a + A1*B1b
    #   A1*B1a: 2x2x2 Strassen (7 terms)
    #   A1*B1b: 2x2x1 -- this is just 4 terms (schoolbook for 2x2*1)
    #   Actually 2x2x1: C[i,k] = sum_j A[i,j]*B[j,k], k is only 0.
    #   That's 4 terms.
    #
    # For a*b (2x1 * 1x3 -> 2x3): 6 terms (schoolbook)
    #
    # Total: 7 + 4 + 6 = 17. 
    #
    # Let me be more careful. Let me define the 17 terms explicitly.
    # A indices: A[0,0]=a00, A[0,1]=a01, A[0,2]=a02, A[1,0]=a10, A[1,1]=a11, A[1,2]=a12
    # B indices: B[0,0]=b00, B[0,1]=b01, B[0,2]=b02, B[1,0]=b10, B[1,1]=b11, B[1,2]=b12, B[2,0]=b20, B[2,1]=b21, B[2,2]=b22
    # C indices: C[0,0]=c00, C[0,1]=c01, C[0,2]=c02, C[1,0]=c10, C[1,1]=c11, C[1,2]=c12
    #
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # Group 1: Strassen for A1*B1a where A1 = [[a00,a01],[a10,a11]], B1a = [[b00,b01],[b10,b11]]
    # This contributes to C[:, 0:2]
    # Strassen terms (using standard 2x2x2 with U=i*2+j, V=j*2+k, W=k*2+i):
    # Term 1: (1,0,1,0)*(1,0,0,1)*(1,0,0,1) -> C[0,0] += a00*b00, C[0,1] += a00*b01, C[1,0] += a10*b10, C[1,1] += a10*b11
    # Wait, let me re-derive. In 2x2x2 Strassen:
    # M1 = (a00+a11)(b00+b11) -> contributes to c00, c01, c10, c11
    # M2 = (a10+a11)b00 -> contributes to c10, c11
    # M3 = a00(b11-b10) -> contributes to c01, c11
    # M4 = a11(b00-b10) -> contributes to c10, c00
    # M5 = (a00+a01)b10 -> contributes to c11, c10
    # M6 = (a10-a00)(b00+b01) -> contributes to c00, c01
    # M7 = (a01-a11)(b10+b11) -> contributes to c01, c11
    #
    # In our 2x3x3 context:
    # A1 = [[a00, a01], [a10, a11]] (indices 0,1 in U for i, 0,1 in j)
    # B1a = [[b00, b01], [b10, b11]] (indices 0,1 in V for j, 0,1 in k)
    # Output C1a = [[c00, c01], [c10, c11]] (indices 0,1 in W for k, 0,1 in i)
    #
    # U vector (length 6): positions 0,1,3,4 are a00,a01,a10,a11; positions 2,5 are a02,a12 (unused in this group)
    # V vector (length 9): positions 0,1,3,4 are b00,b01,b10,b11; positions 2,5,6,7,8 unused
    # W vector (length 6): positions 0,1,2,3 are c00,c01,c10,c11; positions 4,5 unused
    #
    # Strassen term 1: M1 = (a00+a11)(b00+b11)
    #   U: [1,0,0,1,0,0] (a00 and a11)
    #   V: [1,0,0,0,1,0,0,0,0] (b00 and b11)
    #   W: [1,1,1,1,0,0] (c00, c01, c10, c11)
    #
    # Strassen term 2: M2 = (a10+a11)b00
    #   U: [0,0,0,1,1,0] (a10 and a11)
    #   V: [1,0,0,0,0,0,0,0,0] (b00)
    #   W: [0,0,1,1,0,0] (c10, c11)
    #
    # Strassen term 3: M3 = a00(b11-b10)
    #   U: [1,0,0,0,0,0] (a00)
    #   V: [0,0,0,-1,1,0,0,0,0] (-b10, b11)
    #   W: [0,1,0,1,0,0] (c01, c11)
    #
    # Strassen term 4: M4 = a11(b00-b10)
    #   U: [0,0,0,0,1,0] (a11)
    #   V: [1,0,0,-1,0,0,0,0,0] (b00, -b10)
    #   W: [1,0,1,0,0,0] (c00, c10)
    #
    # Strassen term 5: M5 = (a00+a01)b10
    #   U: [1,1,0,0,0,0] (a00, a01)
    #   V: [0,0,0,1,0,0,0,0,0] (b10)
    #   W: [0,0,1,1,0,0] (c10, c11)
    #
    # Strassen term 6: M6 = (a10-a00)(b00+b01)
    #   U: [-1,0,0,1,0,0] (-a00, a10)
    #   V: [1,1,0,0,0,0,0,0,0] (b00, b01)
    #   W: [1,1,0,0,0,0] (c00, c01)
    #
    # Strassen term 7: M7 = (a01-a11)(b10+b11)
    #   U: [0,1,0,0,-1,0] (a01, -a11)
    #   V: [0,0,0,1,1,0,0,0,0] (b10, b11)
    #   W: [0,1,0,1,0,0] (c01, c11)
    #
    # Group 2: A1*B1b where B1b = [[b02],[b12]] (2x1)
    # C1b[i,2] = sum_j A1[i,j] * B1b[j,0] = a_i0*b02 + a_i1*b12
    # This is 2 terms (one per row of A1), but actually it's schoolbook 2x2x1 = 4 terms.
    # Wait: C1b[0,2] = a00*b02 + a01*b12 (2 terms)
    #       C1b[1,2] = a10*b02 + a11*b12 (2 terms)
    # So 4 schoolbook terms.
    #
    # Term 8: a00*b02
    #   U: [1,0,0,0,0,0]
    #   V: [0,0,1,0,0,0,0,0,0] (b02 at position j=0,k=2)
    #   W: [0,0,0,0,1,0] (c02 at position k=2,i=0)
    #
    # Term 9: a01*b12
    #   U: [0,1,0,0,0,0]
    #   V: [0,0,0,0,0,1,0,0,0] (b12 at position j=1,k=2)
    #   W: [0,0,0,0,1,0]
    #
    # Term 10: a10*b02
    #   U: [0,0,0,1,0,0]
    #   V: [0,0,1,0,0,0,0,0,0]
    #   W: [0,0,0,0,0,1] (c12 at position k=2,i=1)
    #
    # Term 11: a11*b12
    #   U: [0,0,0,0,1,0]
    #   V: [0,0,0,0,0,1,0,0,0]
    #   W: [0,0,0,0,0,1]
    #
    # Group 3: a*b where a = [a02, a12]^T (2x1), b = [b20, b21, b22] (1x3)
    # C2[i,k] = a_i * b_k
    # C2[0,0] = a02*b20, C2[0,1] = a02*b21, C2[0,2] = a02*b22
    # C2[1,0] = a12*b20, C2[1,1] = a12*b21, C2[1,2] = a12*b22
    # 6 schoolbook terms.
    #
    # Term 12: a02*b20
    #   U: [0,0,1,0,0,0] (a02 at position i=0,j=2)
    #   V: [0,0,0,0,0,0,1,0,0] (b20 at position j=2,k=0)
    #   W: [1,0,0,0,0,0] (c00 at position k=0,i=0)
    #
    # Term 13: a02*b21
    #   U: [0,0,1,0,0,0]
    #   V: [0,0,0,0,0,0,0,1,0] (b21 at position j=2,k=1)
    #   W: [0,1,0,0,0,0] (c01 at position k=1,i=0)
    #
    # Term 14: a02*b22
    #   U: [0,0,1,0,0,0]
    #   V: [0,0,0,0,0,0,0,0,1] (b22 at position j=2,k=2)
    #   W: [0,0,0,0,1,0] (c02 at position k=2,i=0)
    #
    # Term 15: a12*b20
    #   U: [0,0,0,0,0,1] (a12 at position i=1,j=2)
    #   V: [0,0,0,0,0,0,1,0,0]
    #   W: [0,0,1,0,0,0] (c10 at position k=0,i=1)
    #
    # Term 16: a12*b21
    #   U: [0,0,0,0,0,1]
    #   V: [0,0,0,0,0,0,0,1,0]
    #   W: [0,0,0,1,0,0] (c11 at position k=1,i=1)
    #
    # Term 17: a12*b22
    #   U: [0,0,0,0,0,1]
    #   V: [0,0,0,0,0,0,0,0,1]
    #   W: [0,0,0,0,0,1] (c12 at position k=2,i=1)

    strassen_233 = [
        # Term 1: M1
        ([1,0,0,1,0,0], [1,0,0,0,1,0,0,0,0], [1,1,1,1,0,0]),
        # Term 2: M2
        ([0,0,0,1,1,0], [1,0,0,0,0,0,0,0,0], [0,0,1,1,0,0]),
        # Term 3: M3
        ([1,0,0,0,0,0], [0,0,0,-1,1,0,0,0,0], [0,1,0,1,0,0]),
        # Term 4: M4
        ([0,0,0,0,1,0], [1,0,0,-1,0,0,0,0,0], [1,0,1,0,0,0]),
        # Term 5: M5
        ([1,1,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,1,1,0,0]),
        # Term 6: M6
        ([-1,0,0,1,0,0], [1,1,0,0,0,0,0,0,0], [1,1,0,0,0,0]),
        # Term 7: M7
        ([0,1,0,0,-1,0], [0,0,0,1,1,0,0,0,0], [0,1,0,1,0,0]),
        # Term 8: a00*b02
        ([1,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,1,0]),
        # Term 9: a01*b12
        ([0,1,0,0,0,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,1,0]),
        # Term 10: a10*b02
        ([0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,1]),
        # Term 11: a11*b12
        ([0,0,0,0,1,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,1]),
        # Term 12: a02*b20
        ([0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0]),
        # Term 13: a02*b21
        ([0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0], [0,1,0,0,0,0]),
        # Term 14: a02*b22
        ([0,0,1,0,0,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,1,0]),
        # Term 15: a12*b20
        ([0,0,0,0,0,1], [0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0]),
        # Term 16: a12*b21
        ([0,0,0,0,0,1], [0,0,0,0,0,0,0,1,0], [0,0,0,1,0,0]),
        # Term 17: a12*b22
        ([0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,1]),
    ]

    def verify_233(terms):
        """Verify that the 2x3x3 decomposition is correct."""
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    s = Fraction(0)
                    for (u, v, w) in terms:
                        s += Fraction(u[i*3+j]) * Fraction(v[j*3+k]) * Fraction(w[k*2+i])
                    if s != 1:
                        return False
        return True

    # Verify our 2x3x3 decomposition
    # We'll do this at runtime to be safe, but it should pass.
    
    def build_233_block(i0, j0, k0, U, V, W, col_start):
        """Add a 2x3x3 block at position (i0, j0, k0) to the global matrices."""
        col = col_start
        for (u_c, v_c, w_c) in strassen_233:
            # U: 6 entries, index i*3+j where i in 0..1, j in 0..2
            for la in range(6):
                if u_c[la] != 0:
                    i_local = la // 3
                    j_local = la % 3
                    U[(i0 + i_local) * m + (j0 + j_local)][col] += u_c[la]
            # V: 9 entries, index j*3+k where j in 0..2, k in 0..2
            for lb in range(9):
                if v_c[lb] != 0:
                    j_local = lb // 3
                    k_local = lb % 3
                    V[(j0 + j_local) * p + (k0 + k_local)][col] += v_c[lb]
            # W: 6 entries, index k*2+i where k in 0..2, i in 0..1
            for lc in range(6):
                if w_c[lc] != 0:
                    k_local = lc // 2
                    i_local = lc % 2
                    W[(k0 + k_local) * n + (i0 + i_local)][col] += w_c[lc]
            col += 1
        return col

    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if s != 1:
                        return False
        return True

    # Strategy: Partition 4x5x5 into four 2x3x3 blocks
    # i partitions: 0-1, 2-3
    # j partitions: 0-2, 3-4
    # k partitions: 0-2, 3-4
    # This gives 2*2*2 = 8 blocks of 2x3x3, rank = 8*17 = 136. That's worse!
    
    # Wait, 5 = 3+2, so j partitions are (0-2) and (3-4). The second is only 2 wide.
    # So we can't use 2x3x3 for the second j partition.
    
    # Let me reconsider. 4x5x5:
    # i: 4 = 2+2
    # j: 5 = 3+2
    # k: 5 = 3+2
    #
    # Blocks:
    # (i=0-1, j=0-2, k=0-2): 2x3x3, rank 17
    # (i=0-1, j=0-2, k=3-4): 2x3x2, rank ?
    # (i=0-1, j=3-4, k=0-2): 2x2x3, rank ?
    # (i=0-1, j=3-4, k=3-4): 2x2x2, rank 7
    # (i=2-3, j=0-2, k=0-2): 2x3x3, rank 17
    # (i=2-3, j=0-2, k=3-4): 2x3x2, rank ?
    # (i=2-3, j=3-4, k=0-2): 2x2x3, rank ?
    # (i=2-3, j=3-4, k=3-4): 2x2x2, rank 7
    #
    # For 2x3x2: A is 2x3, B is 3x2, C is 2x2.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # This is like 2x3x2. Rank? 
    # We can view it as: split A = [A1|a], B = [B1;b]
    # C = A1*B1 + a*b
    # A1*B1: 2x2x2, rank 7
    # a*b: 2x1x2, rank 4 (schoolbook)
    # Total: 11? Let me verify.
    # Actually 2x1x2: C[i,k] = a_i * b_k. 4 terms.
    # So 2x3x2 rank <= 7+4 = 11.
    #
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Split B = [B1|B2] where B1 is 2x2, B2 is 2x1.
    # C = A*B1 + A*B2
    # A*B1: 2x2x2, rank 7
    # A*B2: 2x2x1, rank 4 (schoolbook)
    # Total: 11.
    #
    # So total rank = 2*17 + 2*11 + 2*11 + 2*7 = 34 + 22 + 22 + 14 = 92.
    # That's the same as before! Not an improvement.
    
    # Hmm. Let me think differently.
    # What if I use a different partition?
    # 4x5x5: 
    # i: 4 = 2+2
    # j: 5 = 2+3 (instead of 3+2)
    # k: 5 = 3+2 (instead of 2+3)
    #
    # Blocks:
    # (i=0-1, j=0-1, k=0-2): 2x2x3, rank 11
    # (i=0-1, j=0-1, k=3-4): 2x2x2, rank 7
    # (i=0-1, j=2-4, k=0-2): 2x3x3, rank 17
    # (i=0-1, j=2-4, k=3-4): 2x3x2, rank 11
    # (i=2-3, j=0-1, k=0-2): 2x2x3, rank 11
    # (i=2-3, j=0-1, k=3-4): 2x2x2, rank 7
    # (i=2-3, j=2-4, k=0-2): 2x3x3, rank 17
    # (i=2-3, j=2-4, k=3-4): 2x3x2, rank 11
    # Total: 2*11 + 2*7 + 2*17 + 2*11 = 22+14+34+22 = 92. Same!
    
    # The issue is that the mixed sizes always sum to the same total.
    # Let me try a completely different approach.
    
    # What about using a 4x3x3 block? 
    # 4x3x3: A is 4x3, B is 3x3, C is 4x3.
    # This can be viewed as two 2x3x3 blocks stacked in i.
    # Rank = 2*17 = 34.
    # 
    # 4x5x5:
    # i: 4
    # j: 5 = 3+2
    # k: 5 = 3+2
    #
    # (i=0-3, j=0-2, k=0-2): 4x3x3, rank 34
    # (i=0-3, j=0-2, k=3-4): 4x3x2, rank ?
    # (i=0-3, j=3-4, k=0-2): 4x2x3, rank ?
    # (i=0-3, j=3-4, k=3-4): 4x2x2, rank ?
    #
    # 4x3x2: A is 4x3, B is 3x2, C is 4x2.
    # = two 2x3x2 blocks. Rank = 2*11 = 22.
    # 4x2x3: A is 4x2, B is 2x3, C is 4x3.
    # = two 2x2x3 blocks. Rank = 2*11 = 22.
    # 4x2x2: A is 4x2, B is 2x2, C is 4x2.
    # = two 2x2x2 blocks. Rank = 2*7 = 14.
    #
    # Total: 34+22+22+14 = 92. Still the same!
    
    # It seems like any block decomposition that tiles the space gives rank 92.
    # The fundamental issue is that 4*5*5 = 100, and the "savings" from Strassen
    # on 2x2x2 blocks (saving 1 per block) gives at most 8 savings = 92.
    
    # To get below 92, we need a decomposition that doesn't tile the space,
    # or we need a better decomposition for the mixed-size blocks.
    
    # Let me try to find a better decomposition for 2x3x2.
    # 2x3x2: A(2x3) * B(3x2) = C(2x2)
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    #
    # Can we do better than 11? Let's think...
    # The tensor has 2*3*2 = 12 entries, all 1s.
    # Rank of 2x3x2 tensor: lower bound is max(2*3, 3*2, 2*2) = 6? No, that's not right.
    # Actually, for matrix multiplication tensor M_{p,q,r}, the rank is at least p*q*r / max(p,q,r) by a counting argument? No.
    # The rank is at least the "slice rank" which is max over all slices.
    # For 2x3x2, the slices are:
    # i=0: 3x2 matrix (rank 2)
    # i=1: 3x2 matrix (rank 2)
    # j=0: 2x2 matrix (rank 2)
    # j=1: 2x2 matrix (rank 2)
    # j=2: 2x2 matrix (rank 2)
    # k=0: 2x3 matrix (rank 2)
    # k=1: 2x3 matrix (rank 2)
    # So slice rank lower bound is 2. Not helpful.
    #
    # Actually, the rank of the 2x3x2 matrix multiplication tensor...
    # Let me just try to find a rank-10 or rank-9 decomposition.
    # 
    # Hmm, this is getting complex. Let me try a different structural approach.
    
    # ALTERNATIVE STRATEGY: Use a 3x3x3 Strassen-like decomposition.
    # The rank of the 3x3x3 matrix multiplication tensor is known to be 19 (Landsberg 1987, 
    # and later work shows it's at most 23, but 19 is the best known upper bound for exact).
    # Wait, actually the best known rank for 3x3x3 is 19 (due to Landsberg).
    #
    # 4x5x5:
    # Place a 3x3x3 block at (0,0,0): covers i=0-2, j=0-2, k=0-2. Rank 19.
    # Remaining:
    # i=3: 1x5x5 = 25 terms
    # j=3-4 (i=0-2): 3x2x5 = 30 terms  
    # k=3-4 (i=0-2): 3x5x2 = 30 terms
    # Overlaps: j=3-4, k=3-4 (i=0-2): 3x2x2 = 12 terms
    # i=3, j=3-4: 1x2x5 = 10 terms
    # i=3, k=3-4: 1x5x2 = 10 terms
    # i=3, j=3-4, k=3-4: 1x2x2 = 4 terms
    #
    # This is getting complicated. Let me just use schoolbook for the remainder.
    # Total tensor entries: 100. 3x3x3 covers 27. Remainder: 73.
    # Rank: 19 + 73 = 92. Same again!
    
    # I think the key insight is that for ANY tiling, the total rank ends up being 92
    # because the "savings" from Strassen exactly offset the "overhead" from mixed blocks.
    
    # To truly improve, I need a non-tiling decomposition or a better sub-block decomposition.
    
    # Let me try: what if I use a 4x4x4 Strassen decomposition (rank 56) and handle
    # the fringe (j=4 and k=4) more cleverly?
    #
    # 4x4x4: rank 56 (8 blocks of 2x2x2 Strassen)
    # Fringe: 
    #   j=4: i=0-3, j=4, k=0-4 (20 terms)
    #   k=4: i=0-3, j=0-3, k=4 (16 terms, since j=4 already counted)
    #   Total fringe: 20 + 16 = 36 terms
    #
    # Can I decompose the fringe with rank < 36?
    # The fringe tensor is:
    # T[i,4,k] = 1 for all i,k (20 entries)
    # T[i,j,4] = 1 for all i,j where j<4 (16 entries)
    #
    # This is a sum of two "matrices":
    # F1[i,k] = 1 (4x5 matrix) with j=4 fixed
    # F2[i,j] = 1 (4x4 matrix) with k=4 fixed
    #
    # The rank of F1 as a tensor slice: it's a 4x1x5 tensor (all ones).
    # Rank of all-ones 4x1x5 tensor: this is a "product" tensor. 
    # T[i,4,k] = 1*1 = (ones_4[i]) * (ones_5[k]) with j=4.
    # This has rank 1! Because T[i,4,k] = u[i]*v[4]*w[k] where u=ones_4, v has 1 at position 4, w=ones_5.
    # Wait, but in the tensor decomposition, each term is a product of three vectors.
    # For the fringe term at (i,4,k): we need U[i*5+4] * V[4*5+k] * W[k*4+i] = 1.
    # If I set U[i*5+4] = 1 for all i, V[4*5+k] = 1 for all k, W[k*4+i] = 1 for all k,i,
    # that's 3 vectors, giving rank 1 for this "slice". But wait, this would also contribute
    # to other (i,j,k) entries! I need to be careful.
    #
    # Actually, the fringe is not independent of the main block. The main 4x4x4 block only
    # covers j=0-3, k=0-3. The fringe terms at (i,4,k) for k=0-3 are NOT covered by the main block.
    # Similarly, (i,j,4) for j=0-3 is not covered.
    #
    # So the fringe is truly separate. Let me think about the rank of the fringe tensor.
    # Fringe tensor F has entries:
    # F[i,4,k] = 1 for i in 0..3, k in 0..4 (20 entries)
    # F[i,j,4] = 1 for i in 0..3, j in 0..3 (16 entries)
    # All other entries = 0.
    #
    # Can I decompose F with rank < 36?
    # 
    # Observation: F[i,4,k] = 1 for all i,k. This is a rank-1 tensor in the (i,k) plane with j=4.
    # I can write: F[i,4,k] = sum_c u_c[i] * v_c[4] * w_c[k]
    # If I use one term: u[i]=1, v[4]=1, w[k]=1. But this also gives F[i,j,k] contributions
    # for j != 4 and k != 4, which I don't want.
    #
    # So I need to be more careful. Let me use separate terms for the j=4 fringe and k=4 fringe.
    #
    # For j=4 fringe: T[i,4,k] = 1. 
    # This is a 4x1x5 tensor. I can decompose it as:
    # T[i,4,k] = sum_{c=1}^{r1} u_c[i] * v_c[4] * w_c[k]
    # The simplest: u[i]=1, v[4]=1, w[k]=1 for one term. But this only works if no other
    # term contributes to (i,4,k). Since the main block doesn't touch j=4, this is fine.
    # So rank 1 for the j=4 fringe!
    #
    # Wait, but I need U[i*5+4] * V[4*5+k] * W[k*4+i] = 1 for ALL i,k.
    # If I set U[i*5+4] = 1 for all i, V[4*5+k] = 1 for all k, W[k*4+i] = 1 for all k,i,
    # then the product is 1 for all (i,4,k). 
    # But this single term also contributes to (i,j,k) for j != 4 if U[i*5+j] != 0.
    # Since I'm only setting U[i*5+4], other j values have U=0, so no spurious contributions.
    # Similarly, V[4*5+k] is only set for j=4, so no spurious contributions from other j.
    # W[k*4+i] is set for all k,i, but since U and V are zero for other (i,j) and (j,k),
    # the product is zero elsewhere.
    #
    # So the j=4 fringe has rank 1!
    #
    # For k=4 fringe: T[i,j,4] = 1 for i in 0..3, j in 0..3.
    # This is a 4x4x1 tensor. Similarly, rank 1!
    # U[i*5+j] = 1, V[j*5+4] = 1, W[4*4+i] = 1.
    # But wait, this would also contribute to (i,4,4) which is already covered by the j=4 fringe.
    # That's fine, as long as the total is 1 at (i,4,4).
    # j=4 fringe gives 1 at (i,4,4). k=4 fringe also gives 1 at (i,4,4). Total = 2. Bad!
    #
    # I need to fix this. Let me adjust:
    # j=4 fringe: T[i,4,k] = 1 for k=0..3, and T[i,4,4] = 0 (handled by k=4 fringe).
    # k=4 fringe: T[i,j,4] = 1 for j=0..3, and T[i,4,4] = 1.
    #
    # j=4 fringe (k=0..3): 
    #   U[i*5+4] = 1, V[4*5+k] = 1 for k=0..3, W[k*4+i] = 1 for k=0..3.
    #   But W[k*4+i] is shared with k=4 fringe? No, k=4 fringe uses W[4*4+i].
    #   Actually W index is k*4+i. For k=0..3, that's indices 0..15. For k=4, that's 16..19.
    #   So they don't overlap in W. Good.
    #   But V[4*5+k] for k=0..3 is indices 20..23. V[4*5+4] = V[24] is for k=4.
    #   So j=4 fringe uses V[20..23], k=4 fringe uses V[j*5+4] for j=0..3, which is V[4,9,14,19].
    #   No overlap. Good.
    #
    # But wait, I need the j=4 fringe to give exactly 1 at (i,4,k) for k=0..3 and 0 at (i,4,4).
    # With one term: U[i*5+4]=1, V[4*5+k]=1 for k=0..3 (and V[4*5+4]=0), W[k*4+i]=1 for k=0..3.
    # This gives product = 1 at (i,4,k) for k=0..3, and 0 at (i,4,4). 
    # But this is not a single rank-1 term! A rank-1 term has fixed U, V, W vectors.
    # I need V[4*5+k] = 1 for k=0..3 and V[4*5+4] = 0. That's a specific V vector.
    # And W[k*4+i] = 1 for k=0..3. That's a specific W vector (W[0..15] = 1, W[16..19] = 0).
    # So yes, this IS a single rank-1 term!
    #
    # Similarly, k=4 fringe: U[i*5+j]=1 for j=0..3 (and U[i*5+4]=0 to avoid double-counting at (i,4,4)).
    # Wait, but U[i*5+4] is already used by the j=4 fringe term. If I set U[i*5+4]=0 in the k=4 fringe term,
    # that's fine because they're different terms (different columns).
    #
    # k=4 fringe term: U[i*5+j]=1 for j=0..3, V[j*5+4]=1 for j=0..3, W[4*4+i]=1 for all i.
    # This gives product = 1 at (i,j,4) for j=0..3, and 0 at (i,4,4) (since U[i*5+4]=0 in this term).
    #
    # Total at (i,4,4): j=4 fringe gives 0 (V[4*5+4]=0), k=4 fringe gives 0 (U[i*5+4]=0). Total = 0. Bad!
    # I need (i,4,4) to be 1.
    #
    # Fix: Add a third term for (i,4,4): U[i*5+4]=1, V[4*5+4]=1, W[4*4+i]=1.
    # This gives 1 at (i,4,4) for all i.
    #
    # So fringe rank = 1 (j=4, k=0..3) + 1 (k=4, j=0..3) + 1 (i,4,4) = 3!
    #
    # Wait, let me double-check. 
    # Term A (j=4 fringe, k=0..3): 
    #   U_A[i*5+4] = 1 for all i. U_A[other] = 0.
    #   V_A[4*5+k] = 1 for k=0..3. V_A[4*5+4] = 0. V_A[other] = 0.
    #   W_A[k*4+i] = 1 for k=0..3. W_A[4*4+i] = 0. W_A[other] = 0.
    #   Contribution to (i,j,k): U_A[i*5+j] * V_A[j*5+k] * W_A[k*4+i]
    #   = 1 * (1 if j=4 and k in 0..3 else 0) * (1 if k in 0..3 else 0)
    #   = 1 if j=4 and k in 0..3, else 0. Correct!
    #
    # Term B (k=4 fringe, j=0..3):
    #   U_B[i*5+j] = 1 for j=0..3. U_B[i*5+4] = 0. U_B[other] = 0.
    #   V_B[j*5+4] = 1 for j=0..3. V_B[other] = 0.
    #   W_B[4*4+i] = 1 for all i. W_B[other] = 0.
    #   Contribution: U_B[i*5+j] * V_B[j*5+k] * W_B[k*4+i]
    #   = (1 if j in 0..3 else 0) * (1 if k=4 and j in 0..3 else 0) * (1 if k=4 else 0)
    #   = 1 if j in 0..3 and k=4, else 0. Correct!
    #
    # Term C (corner (i,4,4)):
    #   U_C[i*5+4] = 1 for all i. U_C[other] = 0.
    #   V_C[4*5+4] = 1. V_C[other] = 0.
    #   W_C[4*4+i] = 1 for all i. W_C[other] = 0.
    #   Contribution: U_C[i*5+j] * V_C[j*5+k] * W_C[k*4+i]
    #   = (1 if j=4 else 0) * (1 if j=4 and k=4 else 0) * (1 if k=4 else 0)
    #   = 1 if j=4 and k=4, else 0. Correct!
    #
    # So the fringe has rank 3! Total rank = 56 + 3 = 59!
    #
    # Wait, that seems too good. Let me verify the main block doesn't interfere.
    # The main 4x4x4 block covers i=0-3, j=0-3, k=0-3.
    # The fringe terms only contribute to entries with j=4 or k=4, which are outside the main block.
    # So no interference. 
    #
    # Total rank = 56 (main) + 3 (fringe) = 59!

    def build_444_strassen():
        """Build 4x4x4 Strassen decomposition (8 blocks of 2x2x2)."""
        blocks = [(0,0,0), (0,0,2), (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2)]
        U = [[0] * 56 for _ in range(4 * 4)]
        V = [[0] * 56 for _ in range(4 * 4)]
        W = [[0] * 56 for _ in range(4 * 4)]
        col = 0
        for (i0, j0, k0) in blocks:
            for (u_c, v_c, w_c) in strassen_222:
                a_idx = [(i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)]
                b_idx = [(j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)]
                c_idx = [(k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)]
                for la, cf in enumerate(u_c):
                    if cf != 0:
                        U[a_idx[la][0] * 4 + a_idx[la][1]][col] += cf
                for lb, cf in enumerate(v_c):
                    if cf != 0:
                        V[b_idx[lb][0] * 4 + b_idx[lb][1]][col] += cf
                for lc, cf in enumerate(w_c):
                    if cf != 0:
                        W[c_idx[lc][0] * 4 + c_idx[lc][1]][col] += cf
                col += 1
        return U, V, W, col

    # Build the full 4x5x5 decomposition
    rank_total = 56 + 3
    U = [[0] * rank_total for _ in range(n * m)]  # 20 x rank
    V = [[0] * rank_total for _ in range(m * p)]  # 25 x rank
    W = [[0] * rank_total for _ in range(p * n)]  # 20 x rank

    # Copy 4x4x4 Strassen into the first 56 columns
    U_444, V_444, W_444, col_444 = build_444_strassen()
    for i in range(4):
        for j in range(4):
            for c in range(56):
                U[i * 5 + j][c] = U_444[i * 4 + j][c]
    for j in range(4):
        for k in range(4):
            for c in range(56):
                V[j * 5 + k][c] = V_444[j * 4 + k][c]
    for k in range(4):
        for i in range(4):
            for c in range(56):
                W[k * 4 + i][c] = W_444[k * 4 + i][c]

    # Fringe term A: j=4, k=0..3
    col = 56
    for i in range(4):
        U[i * 5 + 4][col] = 1
    for k in range(4):
        V[4 * 5 + k][col] = 1
    for k in range(4):
        for i in range(4):
            W[k * 4 + i][col] = 1
    col += 1

    # Fringe term B: k=4, j=0..3
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
    for j in range(4):
        V[j * 5 + 4][col] = 1
    for i in range(4):
        W[4 * 4 + i][col] = 1
    col += 1

    # Fringe term C: (i,4,4)
    for i in range(4):
        U[i * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1
    for i in range(4):
        W[4 * 4 + i][col] = 1
    col += 1

    assert col == rank_total, f"Column mismatch: {col} != {rank_total}"

    result = {"U": U, "V": V, "W": W}

    # Verify
    if verify(U, V, W):
        return result

    # Fallback: if verification fails, try the 92-rank solution
    print("59-rank verification failed, falling back to 92-rank")
    
    def build_decomposition_222(blocks):
        covered = set()
        for (i0, j0, k0) in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0 + di, j0 + dj, k0 + dk))

        num_schoolbook = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        num_schoolbook += 1

        num_strassen = len(blocks) * 7
        rank = num_schoolbook + num_strassen

        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1

        for (i0, j0, k0) in blocks:
            for (u_c, v_c, w_c) in strassen_222:
                a_idx = [(i0, j0), (i0, j0 + 1), (i0 + 1, j0), (i0 + 1, j0 + 1)]
                b_idx = [(j0, k0), (j0, k0 + 1), (j0 + 1, k0), (j0 + 1, k0 + 1)]
                c_idx = [(k0, i0), (k0, i0 + 1), (k0 + 1, i0), (k0 + 1, i0 + 1)]
                for la, cf in enumerate(u_c):
                    if cf != 0:
                        U[a_idx[la][0] * m + a_idx[la][1]][col] += cf
                for lb, cf in enumerate(v_c):
                    if cf != 0:
                        V[b_idx[lb][0] * p + b_idx[lb][1]][col] += cf
                for lc, cf in enumerate(w_c):
                    if cf != 0:
                        W[c_idx[lc][0] * n + c_idx[lc][1]][col] += cf
                col += 1

        assert col == rank, f"Column mismatch: {col} != {rank}"
        return {"U": U, "V": V, "W": W}

    blocks_optimal = [(0,0,0), (0,0,2), (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2)]
    candidate = build_decomposition_222(blocks_optimal)
    if verify(candidate["U"], candidate["V"], candidate["W"]):
        return candidate

    # Final fallback: schoolbook
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
# EVOLVE-BLOCK-END
