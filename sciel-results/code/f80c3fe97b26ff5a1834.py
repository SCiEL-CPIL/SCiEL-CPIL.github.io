"""Matrix multiplication 2x5x5 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_strassen_46():
    """
    Build a rank-46 decomposition for 2x5x5 matrix multiplication.
    
    Strategy: Partition the 5 columns of A and 5 rows/cols of B into blocks of size 2,2,1.
    Use Strassen's algorithm (rank 7) for each 2x2x2 sub-block multiplication.
    For the 1x1 or 1x2 or 2x1 blocks, use direct multiplication.
    
    Partition:
    - A: 2 rows x 5 cols -> partition cols as [0,1], [2,3], [4]
    - B: 5 rows x 5 cols -> partition rows as [0,1], [2,3], [4], cols as [0,1], [2,3], [4]
    - C: 2 rows x 5 cols -> partition cols as [0,1], [2,3], [4]
    
    C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:4] @ B[2:4, 0:2] + A[:, 4:5] @ B[4:5, 0:2]
    C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:4] @ B[2:4, 2:4] + A[:, 4:5] @ B[4:5, 2:4]
    C[:, 4:5] = A[:, 0:2] @ B[0:2, 4:5] + A[:, 2:4] @ B[2:4, 4:5] + A[:, 4:5] @ B[4:5, 4:5]
    
    For each (i,j) block pair, we multiply A[:, cols_i] (2 x 2 or 2x1) by B[rows_j, cols_k] (2x2, 1x2, etc.)
    
    Let's index the column partitions of A as: c0=[0,1], c1=[2,3], c2=[4]
    Row partitions of B: r0=[0,1], r1=[2,3], r2=[4]
    Col partitions of B (and C): k0=[0,1], k1=[2,3], k2=[4]
    
    For each pair (ci, rj, kk), we need to compute A[:, ci] @ B[rj, kk] which contributes to C[:, kk].
    
    When ci is size 2 and rj is size 2: 2x2 @ 2x2 -> use Strassen (7 terms)
    When ci is size 2 and rj is size 1: 2x1 @ 1x2 -> 4 terms (direct)
    When ci is size 1 and rj is size 2: 2x2... wait, A[:, ci] has 2 rows. If ci is size 1, A[:, ci] is 2x1.
    
    Let me be more careful:
    - A[:, c0] is 2x2, A[:, c1] is 2x2, A[:, c2] is 2x1
    - B[r0, k0] is 2x2, B[r1, k1] is 2x2, B[r2, k2] is 1x1, etc.
    
    The product A[:, ci] @ B[rj, kk] has dimensions (2 x |ci|) @ (|rj| x |kk|) = 2 x |kk|, valid when |ci| == |rj|.
    
    So we need |ci| == |rj| for the matrix multiplication to be valid.
    - c0 (size 2) pairs with r0 (size 2): 2x2 @ 2x2 -> 7 Strassen terms
    - c0 (size 2) pairs with r1 (size 2): 2x2 @ 2x2 -> 7 Strassen terms
    - c0 (size 2) pairs with r2 (size 1): INVALID (2x1 @ 1x...)
    
    Wait, that's not right. In matrix multiplication C = A @ B where A is 2x5 and B is 5x5:
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    When we partition, for each column block of A and row block of B, the sizes must match.
    A[:, c0] (2x2) @ B[r0, kk] (2x|kk|) -> valid, gives 2x|kk|
    A[:, c0] (2x2) @ B[r1, kk] (2x|kk|) -> valid
    A[:, c0] (2x2) @ B[r2, kk] (1x|kk|) -> INVALID, size mismatch
    
    So actually, the block structure doesn't work this way directly. The issue is that A has 5 columns and B has 5 rows, and we partition them independently, but for the matrix product to work, the column partition of A must match the row partition of B.
    
    Let me reconsider. The correct block decomposition for C = A @ B:
    If we partition the 5 columns of A and 5 rows of B using the SAME partition [2, 2, 1], then:
    A = [A0, A1, A2] where A0 is 2x2, A1 is 2x2, A2 is 2x1
    B = [B0; B1; B2] where B0 is 2x5, B1 is 2x5, B2 is 1x5
    
    Then C = A0 @ B0 + A1 @ B1 + A2 @ B2
    
    Now for each term:
    - A0 @ B0: (2x2) @ (2x5). Partition B0's 5 columns into [2,2,1]: B0 = [B00, B01, B02]
      A0 @ B00: 2x2 @ 2x2 -> 7 terms (Strassen)
      A0 @ B01: 2x2 @ 2x2 -> 7 terms
      A0 @ B02: 2x2 @ 2x1 -> 4 terms (direct, since 2x2 @ 2x1 is just 2 scalar multiplications per column... actually 2x2 @ 2x1 gives 2x1, which is 4 multiplications)
      
    - A1 @ B1: similarly 7 + 7 + 4 = 18 terms
    - A2 @ B2: (2x1) @ (1x5). Partition B2's 5 columns into [2,2,1]: B2 = [B20, B21, B22]
      A2 @ B20: 2x1 @ 1x2 -> 4 terms
      A2 @ B21: 2x1 @ 1x2 -> 4 terms
      A2 @ B22: 2x1 @ 1x1 -> 2 terms
    
    Total: (7+7+4) + (7+7+4) + (4+4+2) = 18 + 18 + 10 = 46 terms.
    
    Let me implement this.
    """
    n, m, p = 2, 5, 5
    rank = 46
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Column partitions of A / Row partitions of B: [2, 2, 1] starting at indices 0, 2, 4
    # Col partitions of B / Col partitions of C: [2, 2, 1] starting at indices 0, 2, 4
    
    # For each (ci, rj) pair where ci==rj (same partition index), multiply A[:, ci] @ B[ci, :]
    # Wait no. Let me re-read.
    
    # A is 2x5, B is 5x5, C = A@B is 2x5.
    # Partition the 5 "inner" dimension (cols of A, rows of B) into [2,2,1].
    # A[:, 0:2] = A0 (2x2), A[:, 2:4] = A1 (2x2), A[:, 4:5] = A2 (2x1)
    # B[0:2, :] = B0 (2x5), B[2:4, :] = B1 (2x5), B[4:5, :] = B2 (1x5)
    # C = A0@B0 + A1@B1 + A2@B2
    
    # For A0@B0 (2x2 @ 2x5): partition cols of B0 into [2,2,1]
    #   A0 @ B0[:, 0:2] (2x2 @ 2x2): Strassen 7
    #   A0 @ B0[:, 2:4] (2x2 @ 2x2): Strassen 7
    #   A0 @ B0[:, 4:5] (2x2 @ 2x1): direct 4
    
    # For A1@B1 (2x2 @ 2x5): same structure
    #   A1 @ B1[:, 0:2]: Strassen 7
    #   A1 @ B1[:, 2:4]: Strassen 7
    #   A1 @ B1[:, 4:5]: direct 4
    
    # For A2@B2 (2x1 @ 1x5): partition cols of B2 into [2,2,1]
    #   A2 @ B2[:, 0:2] (2x1 @ 1x2): 4 terms
    #   A2 @ B2[:, 2:4] (2x1 @ 1x2): 4 terms
    #   A2 @ B2[:, 4:5] (2x1 @ 1x1): 2 terms
    
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        U[u_idx, col] = val
        V[v_idx, col] = val
        W[w_idx, col] = val
        col += 1
    
    def strassen_2x2(A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_col_offset):
        """
        Strassen for 2x2 @ 2x2.
        A_block = A[A_row_offset:A_row_offset+2, A_col_offset:A_col_offset+2]
        B_block = B[B_row_offset:B_row_offset+2, B_col_offset:B_col_offset+2]
        C_block contributes to C[:, C_col_offset:C_col_offset+2]
        
        For the 2x5x5 tensor:
        U index for A[i,j]: i*m + j = i*5 + j
        V index for B[j,k]: j*p + k = j*5 + k
        W index for C[i,k]: k*n + i = k*2 + i (trace convention W[k*n+i, r])
        
        Strassen 7 terms:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B01-B11)
        M4 = A11*(B10-B00)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B01)
        M7 = (A01-A11)*(B10+B11)
        
        C00 = M1 - M4 + M5 - M7
        C01 = M3 + M7
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        a00 = A_row_offset * 5 + A_col_offset
        a01 = A_row_offset * 5 + A_col_offset + 1
        a10 = (A_row_offset + 1) * 5 + A_col_offset
        a11 = (A_row_offset + 1) * 5 + A_col_offset + 1
        
        b00 = B_row_offset * 5 + B_col_offset
        b01 = B_row_offset * 5 + B_col_offset + 1
        b10 = (B_row_offset + 1) * 5 + B_col_offset
        b11 = (B_row_offset + 1) * 5 + B_col_offset + 1
        
        c00_col = C_col_offset
        c01_col = C_col_offset + 1
        c00_row = 0
        c01_row = 0
        c10_row = 1
        c11_row = 1
        
        # M1 = (A00+A11)*(B00+B11)
        # C00 += M1, C11 += M1
        # We need to express this as rank-1 terms.
        # M1 = (A00 + A11) * (B00 + B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # That's 4 rank-1 terms, but Strassen combines them into one "multiplication" 
        # which in the bilinear form is still 4 separate rank-1 contributions to different C entries.
        
        # Actually in the tensor decomposition, each rank-1 term is one triple (u,v,w).
        # Strassen reduces 8 multiplications to 7, but each "multiplication" M_i is a product
        # of two linear combinations, which expands to multiple rank-1 tensor terms.
        
        # Wait, I think I'm confusing myself. In the bilinear complexity context:
        # Each rank-1 term of the multiplication tensor corresponds to one scalar multiplication.
        # Strassen says we need only 7 scalar multiplications instead of 8 for 2x2x2.
        # But each scalar multiplication still produces one rank-1 tensor term.
        
        # So for 2x2x2 with Strassen, we have 7 rank-1 terms, not 28.
        
        # Let me reconsider. The tensor for 2x2x2 has 8 non-zero entries.
        # T[i,j,k] = 1 if the multiplication C[i,k] includes A[i,j]*B[j,k].
        # For 2x2x2: C[0,0]=A[0,0]B[0,0]+A[0,1]B[1,0], etc.
        # The tensor has 8 non-zero entries (one for each (i,j,k) triple).
        # Rank of this tensor is 7 (Strassen).
        
        # So each of the 7 Strassen terms IS a rank-1 tensor term, meaning one (u,v,w) triple
        # where u, v, w are 4-dimensional vectors (for 2x2).
        
        # For our 2x5x5 case, each rank-1 term has u of dim 10, v of dim 25, w of dim 10.
        
        # Let me redo this properly.
        
        # Strassen 2x2x2 decomposition (7 rank-1 terms):
        # Each term r has vectors u_r (dim 4), v_r (dim 4), w_r (dim 4) such that
        # sum_r u_r[i] * v_r[j] * w_r[k] = T[i,j,k] for the 2x2x2 tensor.
        
        # The standard Strassen decomposition:
        # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M_r is a scalar = sum_i u_r[i]*A[i] * sum_j v_r[j]*B[j]
        # And C[k] gets contribution sum_r w_r[k]*M_r
        # So the rank-1 term is u_r (for A), v_r (for B), w_r (for C)
        
        # u1 = [1,0,0,1], v1 = [1,1,1,1], w1 = [1,0,0,1]  (M1: (a+d)(e+h), C00+=M1, C11+=M1)
        # u2 = [0,0,1,1], v2 = [1,0,0,0], w2 = [0,0,1,0]   (M2: (c+d)e, C10+=M2)
        # u3 = [1,0,0,0], v3 = [0,1,-1,0], w3 = [0,1,0,0]   (M3: a(f-h), C01+=M3)
        # u4 = [0,0,0,1], v4 = [0,0,1,-1], w4 = [0,0,1,0]   (M4: d(g-e), C10+=M4, C00-=M4)
        # u5 = [1,1,0,0], v5 = [0,0,0,1], w5 = [1,0,0,0]   (M5: (a+b)h, C00+=M5)
        # u6 = [0,-1,1,0], v6 = [1,1,0,0], w6 = [0,0,0,1]   (M6: (c-a)(e+f), C11+=M6)
        # u7 = [0,1,0,-1], v7 = [0,0,1,1], w7 = [0,1,0,1]   (M7: (b-d)(g+h), C01+=M7, C00-=M7, C11-=M7)
        
        # Let me verify w vectors:
        # C00 = M1 - M4 + M5 - M7 => w1[0]=1, w4[0]=-1, w5[0]=1, w7[0]=-1
        # C01 = M3 + M7 => w3[1]=1, w7[1]=1
        # C10 = M2 + M4 => w2[2]=1, w4[2]=1
        # C11 = M1 - M2 + M3 + M6 => w1[3]=1, w2[3]=-1, w3[3]=1, w6[3]=1
        
        # So:
        # w1 = [1, 0, 0, 1]
        # w2 = [0, 0, 1, -1]
        # w3 = [0, 1, 0, 1]
        # w4 = [-1, 0, 1, 0]
        # w5 = [1, 0, 0, 0]
        # w6 = [0, 0, 0, 1]
        # w7 = [-1, 1, 0, 0]
        
        # u vectors (A indexing: a=00, b=01, c=10, d=11):
        # u1 = [1,0,0,1]  (a+d)
        # u2 = [0,0,1,1]  (c+d)
        # u3 = [1,0,0,0]  (a)
        # u4 = [0,0,0,1]  (d)
        # u5 = [1,1,0,0]  (a+b)
        # u6 = [0,-1,1,0] (c-a)
        # u7 = [0,1,0,-1] (b-d)
        
        # v vectors (B indexing: e=00, f=01, g=10, h=11):
        # v1 = [1,1,1,1]  (e+h)
        # v2 = [1,0,0,0]  (e)
        # v3 = [0,1,-1,0] (f-h)
        # v4 = [0,0,1,-1] (g-e)
        # v5 = [0,0,0,1]  (h)
        # v6 = [1,1,0,0]  (e+f)
        # v7 = [0,0,1,1]  (g+h)
        
        return None  # We'll handle this differently below
    
    # Actually, let me just directly enumerate all 46 rank-1 terms.
    
    # I'll define helper to add a rank-1 term with sparse u, v, w vectors.
    # u is indexed by (i, j) for A[i,j], flattened as i*5+j
    # v is indexed by (j, k) for B[j,k], flattened as j*5+k
    # w is indexed by (k, i) for C[i,k] with trace convention W[k*n+i] = W[2k+i]
    
    def add_rank1(u_entries, v_entries, w_entries):
        """u_entries: dict {(i,j): val}, v_entries: dict {(j,k): val}, w_entries: dict {(i,k): val}"""
        nonlocal col
        for (i, j), val in u_entries.items():
            U[i * m + j, col] = val
        for (j, k), val in v_entries.items():
            V[j * p + k, col] = val
        for (i, k), val in w_entries.items():
            W[k * n + i, col] = val
        col += 1
    
    # Strassen 2x2x2: 7 terms
    # A indices in block: (0,0),(0,1),(1,0),(1,1) relative to block
    # B indices in block: (0,0),(0,1),(1,0),(1,1)
    # C indices in block: (0,0),(0,1),(1,0),(1,1)
    
    def strassen_block(A_i0, A_j0, B_i0, B_j0, C_k0):
        """
        A_i0, A_j0: top-left (i,j) of A block
        B_i0, B_j0: top-left (j,k) of B block -- note B_i0 is row, B_j0 is col
        C_k0: top-left col of C block
        """
        # Local indices 0,1 for the 2x2 block
        # A local (a,b): global (A_i0+a, A_j0+b)
        # B local (c,d): global (B_i0+c, B_j0+d) where c is row (j dim), d is col (k dim)
        # C local (e,f): global row e, col C_k0+f
        
        # u1: A local (0,0)=1, (1,1)=1
        add_rank1(
            {(A_i0, A_j0): 1, (A_i0+1, A_j0+1): 1},
            {(B_i0, B_j0): 1, (B_i0, B_j0+1): 1, (B_i0+1, B_j0): 1, (B_i0+1, B_j0+1): 1},
            {(0, C_k0): 1, (1, C_k0+1): 1}
        )
        # u2: A local (1,0)=1, (1,1)=1
        add_rank1(
            {(A_i0+1, A_j0): 1, (A_i0+1, A_j0+1): 1},
            {(B_i0, B_j0): 1},
            {(1, C_k0): 1, (1, C_k0+1): -1}
        )
        # u3: A local (0,0)=1
        add_rank1(
            {(A_i0, A_j0): 1},
            {(B_i0, B_j0+1): 1, (B_i0+1, B_j0+1): -1},
            {(0, C_k0+1): 1, (1, C_k0+1): 1}
        )
        # u4: A local (1,1)=1
        add_rank1(
            {(A_i0+1, A_j0+1): 1},
            {(B_i0+1, B_j0): 1, (B_i0, B_j0): -1},
            {(0, C_k0): -1, (1, C_k0): 1}
        )
        # u5: A local (0,0)=1, (0,1)=1
        add_rank1(
            {(A_i0, A_j0): 1, (A_i0, A_j0+1): 1},
            {(B_i0+1, B_j0+1): 1},
            {(0, C_k0): 1}
        )
        # u6: A local (1,0)=1, (0,0)=-1
        add_rank1(
            {(A_i0+1, A_j0): 1, (A_i0, A_j0): -1},
            {(B_i0, B_j0): 1, (B_i0, B_j0+1): 1},
            {(1, C_k0+1): 1}
        )
        # u7: A local (0,1)=1, (1,1)=-1
        add_rank1(
            {(A_i0, A_j0+1): 1, (A_i0+1, A_j0+1): -1},
            {(B_i0+1, B_j0): 1, (B_i0+1, B_j0+1): 1},
            {(0, C_k0): -1, (0, C_k0+1): 1}
        )
    
    # Direct 2x2 @ 2x1 (4 terms)
    def direct_2x2_at_2x1(A_i0, A_j0, B_i0, B_k0, C_k0):
        """A is 2x2 at (A_i0, A_j0), B is 2x1 at (B_i0, B_k0), C col at C_k0"""
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    if b == c:  # A[:, b] dot B[b, :] where B is 2x1, so B[c, 0]
                        pass
        # C[0, C_k0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1, C_k0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        add_rank1(
            {(0, A_j0): 1},
            {(B_i0, B_k0): 1},
            {(0, C_k0): 1}
        )
        add_rank1(
            {(0, A_j0+1): 1},
            {(B_i0+1, B_k0): 1},
            {(0, C_k0): 1}
        )
        add_rank1(
            {(1, A_j0): 1},
            {(B_i0, B_k0): 1},
            {(1, C_k0): 1}
        )
        add_rank1(
            {(1, A_j0+1): 1},
            {(B_i0+1, B_k0): 1},
            {(1, C_k0): 1}
        )
    
    # Direct 2x1 @ 1x2 (4 terms)
    def direct_2x1_at_1x2(A_i0, A_j0, B_i0, B_k0, C_k0):
        """A is 2x1 at (A_i0, A_j0), B is 1x2 at (B_i0, B_k0), C cols at C_k0, C_k0+1"""
        # C[0, C_k0] = A[0,0]*B[0,0]
        # C[0, C_k0+1] = A[0,0]*B[0,1]
        # C[1, C_k0] = A[1,0]*B[0,0]
        # C[1, C_k0+1] = A[1,0]*B[0,1]
        add_rank1(
            {(0, A_j0): 1},
            {(B_i0, B_k0): 1},
            {(0, C_k0): 1}
        )
        add_rank1(
            {(0, A_j0): 1},
            {(B_i0, B_k0+1): 1},
            {(0, C_k0+1): 1}
        )
        add_rank1(
            {(1, A_j0): 1},
            {(B_i0, B_k0): 1},
            {(1, C_k0): 1}
        )
        add_rank1(
            {(1, A_j0): 1},
            {(B_i0, B_k0+1): 1},
            {(1, C_k0+1): 1}
        )
    
    # Direct 2x1 @ 1x1 (2 terms)
    def direct_2x1_at_1x1(A_i0, A_j0, B_i0, B_k0, C_k0):
        """A is 2x1 at (A_i0, A_j0), B is 1x1 at (B_i0, B_k0), C col at C_k0"""
        add_rank1(
            {(0, A_j0): 1},
            {(B_i0, B_k0): 1},
            {(0, C_k0): 1}
        )
        add_rank1(
            {(1, A_j0): 1},
            {(B_i0, B_k0): 1},
            {(1, C_k0): 1}
        )
    
    # Now assemble the 46 terms:
    # A0 (2x2) @ B0 (2x5):
    #   A0 @ B0[:, 0:2] (2x2 @ 2x2): Strassen
    strassen_block(A_i0=0, A_j0=0, B_i0=0, B_j0=0, C_k0=0)
    #   A0 @ B0[:, 2:4] (2x2 @ 2x2): Strassen
    strassen_block(A_i0=0, A_j0=0, B_i0=0, B_j0=2, C_k0=2)
    #   A0 @ B0[:, 4:5] (2x2 @ 2x1): direct 4
    direct_2x2_at_2x1(A_i0=0, A_j0=0, B_i0=0, B_k0=4, C_k0=4)
    
    # A1 (2x2) @ B1 (2x5):
    #   A1 @ B1[:, 0:2] (2x2 @ 2x2): Strassen
    strassen_block(A_i0=0, A_j0=2, B_i0=2, B_j0=0, C_k0=0)
    #   A1 @ B1[:, 2:4] (2x2 @ 2x2): Strassen
    strassen_block(A_i0=0, A_j0=2, B_i0=2, B_j0=2, C_k0=2)
    #   A1 @ B1[:, 4:5] (2x2 @ 2x1): direct 4
    direct_2x2_at_2x1(A_i0=0, A_j0=2, B_i0=2, B_k0=4, C_k0=4)
    
    # A2 (2x1) @ B2 (1x5):
    #   A2 @ B2[:, 0:2] (2x1 @ 1x2): 4 terms
    direct_2x1_at_1x2(A_i0=0, A_j0=4, B_i0=4, B_k0=0, C_k0=0)
    #   A2 @ B2[:, 2:4] (2x1 @ 1x2): 4 terms
    direct_2x1_at_1x2(A_i0=0, A_j0=4, B_i0=4, B_k0=2, C_k0=2)
    #   A2 @ B2[:, 4:5] (2x1 @ 1x1): 2 terms
    direct_2x1_at_1x1(A_i0=0, A_j0=4, B_i0=4, B_k0=4, C_k0=4)
    
    assert col == 46, f"Expected 46 terms, got {col}"
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    deadline = time.time() + 100  # Safety margin
    
    try:
        U, V, W = _build_strassen_46()
        return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback: schoolbook
        n, m, p = DIMENSIONS
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
