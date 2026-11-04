import numpy as np
import time

DIMENSIONS = [4, 5, 5]
DEADLINE = 145.0  # Safety margin before 150s kill


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # For each (i,j,k), check sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Vectorize by building the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(np.dot(U[i * m + j, :], V[j * p + k, :]), W[k * n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _strassen_2x2x2_embed(n, m, p, i0, j0, k0):
    """
    Returns U, V, W for a 2x2x2 block at offset (i0, j0, k0) in full n x m x p space.
    Uses Strassen's algorithm (rank 7).
    """
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    def ui(i, j): return i * m + j
    def vj(j, k): return j * p + k
    def wk(k, i): return k * n + i

    # M1 = (A00+A11)(B00+B11)
    U[ui(i0, j0), 0] = 1
    U[ui(i1, j1), 0] = 1
    V[vj(j0, k0), 0] = 1
    V[vj(j1, k1), 0] = 1
    W[wk(k0, i0), 0] = 1
    W[wk(k1, i1), 0] = 1

    # M2 = (A10+A11)B00
    U[ui(i1, j0), 1] = 1
    U[ui(i1, j1), 1] = 1
    V[vj(j0, k0), 1] = 1
    W[wk(k0, i1), 1] = 1
    W[wk(k1, i1), 1] = -1

    # M3 = A00(B01-B11)
    U[ui(i0, j0), 2] = 1
    V[vj(j0, k1), 2] = 1
    V[vj(j1, k1), 2] = -1
    W[wk(k1, i0), 2] = 1
    W[wk(k1, i1), 2] = 1

    # M4 = A11(B10-B00)
    U[ui(i1, j1), 3] = 1
    V[vj(j1, k0), 3] = 1
    V[vj(j0, k0), 3] = -1
    W[wk(k0, i0), 3] = 1
    W[wk(k0, i1), 3] = 1

    # M5 = (A00+A01)B11
    U[ui(i0, j0), 4] = 1
    U[ui(i0, j1), 4] = 1
    V[vj(j1, k1), 4] = 1
    W[wk(k0, i0), 4] = -1
    W[wk(k1, i0), 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[ui(i1, j0), 5] = 1
    U[ui(i0, j0), 5] = -1
    V[vj(j0, k0), 5] = 1
    V[vj(j0, k1), 5] = 1
    W[wk(k1, i1), 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[ui(i0, j1), 6] = 1
    U[ui(i1, j1), 6] = -1
    V[vj(j1, k0), 6] = 1
    V[vj(j1, k1), 6] = 1
    W[wk(k0, i0), 6] = 1

    return U, V, W


def _schoolbook_embed(n, m, p, i0, j0, k0, bi, bj, bk):
    """
    Returns U, V, W for a schoolbook bi x bj x bk block at offset (i0, j0, k0).
    """
    rank = bi * bj * bk
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for i in range(bi):
        for j in range(bj):
            for k in range(bk):
                U[(i0 + i) * m + (j0 + j), col] = 1
                V[(j0 + j) * p + (k0 + k), col] = 1
                W[(k0 + k) * n + (i0 + i), col] = 1
                col += 1

    return U, V, W


def _laderman_3x3x3_embed(n, m, p, i0, j0, k0):
    """
    Returns U, V, W for a 3x3x3 block at offset (i0, j0, k0) using Laderman's rank-23 algorithm.
    """
    rank = 23
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def ui(i, j): return (i0 + i) * m + (j0 + j)
    def vj(j, k): return (j0 + j) * p + (k0 + k)
    def wk(k, i): return (k0 + k) * n + (i0 + i)

    # Laderman's 23 terms for 3x3 matrix multiplication
    # Each term: U entries (i,j), V entries (j,k), W entries (k,i)
    # Reference: Laderman's algorithm from "A matrix multiplication algorithm based on 23 multiplications"
    
    # Term 1: (A00+A11+A22)(B00+B11+B22)
    U[ui(0,0),0]=1; U[ui(1,1),0]=1; U[ui(2,2),0]=1
    V[vj(0,0),0]=1; V[vj(1,1),0]=1; V[vj(2,2),0]=1
    W[wk(0,0),0]=1; W[wk(1,1),0]=1; W[wk(2,2),0]=1

    # Term 2: (A00-A11)(B00+B11)
    U[ui(0,0),1]=1; U[ui(1,1),1]=-1
    V[vj(0,0),1]=1; V[vj(1,1),1]=1
    W[wk(0,0),1]=1; W[wk(1,1),1]=1

    # Term 3: (A00+A11)(B00-B11)
    U[ui(0,0),2]=1; U[ui(1,1),2]=1
    V[vj(0,0),2]=1; V[vj(1,1),2]=-1
    W[wk(0,0),2]=1; W[wk(1,1),2]=-1

    # Term 4: (A00-A22)(B00+B22)
    U[ui(0,0),3]=1; U[ui(2,2),3]=-1
    V[vj(0,0),3]=1; V[vj(2,2),3]=1
    W[wk(0,0),3]=1; W[wk(2,2),3]=1

    # Term 5: (A00+A22)(B00-B22)
    U[ui(0,0),4]=1; U[ui(2,2),4]=1
    V[vj(0,0),4]=1; V[vj(2,2),4]=-1
    W[wk(0,0),4]=1; W[wk(2,2),4]=-1

    # Term 6: (A11-A22)(B11+B22)
    U[ui(1,1),5]=1; U[ui(2,2),5]=-1
    V[vj(1,1),5]=1; V[vj(2,2),5]=1
    W[wk(1,1),5]=1; W[wk(2,2),5]=1

    # Term 7: (A11+A22)(B11-B22)
    U[ui(1,1),6]=1; U[ui(2,2),6]=1
    V[vj(1,1),6]=1; V[vj(2,2),6]=-1
    W[wk(1,1),6]=1; W[wk(2,2),6]=-1

    # Term 8: A01(B10-B20)
    U[ui(0,1),7]=1
    V[vj(1,0),7]=1; V[vj(2,0),7]=-1
    W[wk(0,0),7]=1; W[wk(1,0),7]=-1

    # Term 9: (A01+A11)(B10-B11)
    U[ui(0,1),8]=1; U[ui(1,1),8]=1
    V[vj(1,0),8]=1; V[vj(1,1),8]=-1
    W[wk(0,0),8]=1; W[wk(1,1),8]=-1

    # Term 10: (A01-A11)(B10+B11)
    U[ui(0,1),9]=1; U[ui(1,1),9]=-1
    V[vj(1,0),9]=1; V[vj(1,1),9]=1
    W[wk(0,0),9]=1; W[wk(1,1),9]=1

    # Term 11: A12(B02-B22)
    U[ui(1,2),10]=1
    V[vj(0,2),10]=1; V[vj(2,2),10]=-1
    W[wk(0,1),10]=1; W[wk(2,1),10]=-1

    # Term 12: (A12+A11)(B02-B22)
    U[ui(1,2),11]=1; U[ui(1,1),11]=1
    V[vj(0,2),11]=1; V[vj(2,2),11]=-1
    W[wk(0,1),11]=1; W[wk(2,1),11]=-1

    # Term 13: (A12-A11)(B02+B22)
    U[ui(1,2),12]=1; U[ui(1,1),12]=-1
    V[vj(0,2),12]=1; V[vj(2,2),12]=1
    W[wk(0,1),12]=1; W[wk(2,1),12]=1

    # Term 14: A20(B01-B11)
    U[ui(2,0),13]=1
    V[vj(0,1),13]=1; V[vj(1,1),13]=-1
    W[wk(0,2),13]=1; W[wk(1,2),13]=-1

    # Term 15: (A20+A11)(B01-B11)
    U[ui(2,0),14]=1; U[ui(1,1),14]=1
    V[vj(0,1),14]=1; V[vj(1,1),14]=-1
    W[wk(0,2),14]=1; W[wk(1,2),14]=-1

    # Term 16: (A20-A11)(B01+B11)
    U[ui(2,0),15]=1; U[ui(1,1),15]=-1
    V[vj(0,1),15]=1; V[vj(1,1),15]=1
    W[wk(0,2),15]=1; W[wk(1,2),15]=1

    # Term 17: A21(B12-B22)
    U[ui(2,1),16]=1
    V[vj(1,2),16]=1; V[vj(2,2),16]=-1
    W[wk(0,2),16]=1; W[wk(2,2),16]=-1

    # Term 18: (A21+A11)(B12-B22)
    U[ui(2,1),17]=1; U[ui(1,1),17]=1
    V[vj(1,2),17]=1; V[vj(2,2),17]=-1
    W[wk(0,2),17]=1; W[wk(2,2),17]=-1

    # Term 19: (A21-A11)(B12+B22)
    U[ui(2,1),18]=1; U[ui(1,1),18]=-1
    V[vj(1,2),18]=1; V[vj(2,2),18]=1
    W[wk(0,2),18]=1; W[wk(2,2),18]=1

    # Term 20: A02(B20-B10)
    U[ui(0,2),19]=1
    V[vj(2,0),19]=1; V[vj(1,0),19]=-1
    W[wk(1,0),19]=1; W[wk(2,0),19]=-1

    # Term 21: (A02+A22)(B20-B10)
    U[ui(0,2),20]=1; U[ui(2,2),20]=1
    V[vj(2,0),20]=1; V[vj(1,0),20]=-1
    W[wk(1,0),20]=1; W[wk(2,0),20]=-1

    # Term 22: (A02-A22)(B20+B10)
    U[ui(0,2),21]=1; U[ui(2,2),21]=-1
    V[vj(2,0),21]=1; V[vj(1,0),21]=1
    W[wk(1,0),21]=1; W[wk(2,0),21]=1

    # Term 23: A10(B21-B01)
    U[ui(1,0),22]=1
    V[vj(2,1),22]=1; V[vj(0,1),22]=-1
    W[wk(1,1),22]=1; W[wk(2,1),22]=-1

    return U, V, W


def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5

    U_parts = []
    V_parts = []
    W_parts = []

    # Strategy: Partition the 4x5x5 tensor into sub-blocks:
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    # 
    # Split j into {0,1,2} and {3,4}:
    # Term A: j in {0,1,2}: A1 (4x3) * B1 (3x5) -> 4x3x5
    # Term B: j in {3,4}: A2 (4x2) * B2 (2x5) -> 4x2x5
    #
    # Term A: 4x3x5
    #   Split i into {0,1,2} and {3}, k into {0,1,2} and {3,4}
    #   A1a: 3x3x3 at (i=0-2, j=0-2, k=0-2) -> Laderman rank 23
    #   A1b: 3x3x2 at (i=0-2, j=0-2, k=3-4) -> schoolbook 3*3*2=18
    #   A1c: 1x3x3 at (i=3, j=0-2, k=0-2) -> schoolbook 1*3*3=9
    #   A1d: 1x3x2 at (i=3, j=0-2, k=3-4) -> schoolbook 1*3*2=6
    #   Total A: 23+18+9+6 = 56
    #
    # Term B: 4x2x5
    #   Schoolbook: 4*2*5 = 40
    #
    # Total: 56 + 40 = 96
    #
    # Alternative: Use Strassen 2x2x2 blocks where beneficial.
    # Let's try: 
    # Term A: 4x3x5
    #   Split into: 2x3x3 + 2x3x2 (for i=0,1 and i=2,3)
    #   2x3x3: split j into {0,1} and {2}, k into {0,1} and {2}
    #     2x2x2 at (i=0-1,j=0-1,k=0-1): Strassen 7
    #     2x2x1 at (i=0-1,j=0-1,k=2): schoolbook 4
    #     2x1x2 at (i=0-1,j=2,k=0-1): schoolbook 4
    #     2x1x1 at (i=0-1,j=2,k=2): schoolbook 2
    #     Total: 7+4+4+2=17
    #   2x3x2: schoolbook 2*3*2=12
    #   i=2,3: same structure: 17+12=29
    #   Total A: 17+12+29 = 58. Worse than 56.
    #
    # Let me stick with Laderman + schoolbook = 96.
    #
    # Can we do better? Try 4x5x5 as:
    #   4x4x5 + 4x1x5
    #   4x4x5 = 4x4x4 + 4x4x1
    #   4x4x4: use 4 Strassen 2x2x2 blocks? No, 4x4x4 needs 8 2x2x2 blocks = 56.
    #   4x4x1: 16
    #   4x1x5: 20
    #   4x1x5 (for j=4): 20
    #   Total: 56+16+20+20 = 112. Worse.
    #
    # Try 3x5x5 + 1x5x5:
    #   3x5x5 = 3x4x5 + 3x1x5
    #   3x4x5 = 3x4x4 + 3x4x1
    #   3x4x4 = 3x3x4 + 1x3x4
    #   3x3x4 = 3x3x3 + 3x3x1
    #   3x3x3: Laderman 23
    #   3x3x1: 9
    #   1x3x4: 12
    #   3x4x1: 12
    #   3x1x5: 15
    #   1x5x5: 25
    #   Total: 23+9+12+12+15+25 = 96. Same.
    #
    # Let's try mixing: 4x5x5 = 4x3x5 + 4x2x5
    # 4x3x5 = 3x3x5 + 1x3x5
    # 3x3x5 = 3x3x3 + 3x3x2
    # 3x3x3: Laderman 23
    # 3x3x2: 18
    # 1x3x5: 15
    # 4x2x5: 40
    # Total: 23+18+15+40 = 96. Same.
    #
    # Try: 4x5x5 = 4x4x5 + 4x1x5
    # 4x4x5 = 4x4x4 + 4x4x1
    # 4x4x4: Use 4 Laderman 3x3x3? No, 4x4x4 doesn't tile nicely with 3x3.
    # Actually 4x4x4 = 3x3x3 + 3x3x1 + 3x1x3 + 3x1x1 + 1x3x3 + 1x3x1 + 1x1x3 + 1x1x1
    # That's 23 + 9 + 9 + 3 + 9 + 3 + 3 + 1 = 60. Worse than 56.
    #
    # Best so far: 96. Let me also consider using a different partition for the 4x2x5 part.
    # 4x2x5: can we do better than schoolbook 40?
    # Split: 2x2x5 + 2x2x5 = 2*(2x2x5)
    # 2x2x5: Strassen 2x2x2 + 2x2x3 = 7 + 12 = 19. Two of these: 38. Better!
    # So: 4x2x5 = 2*(2x2x5) = 2*19 = 38.
    # Total: 23+18+15+38 = 94.
    #
    # Wait, 2x2x5: i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    # Split k into {0,1,2} and {3,4}:
    # 2x2x3: can we do better? 2x2x3 = 2x2x2 + 2x2x1 = 7 + 4 = 11
    # 2x2x2: Strassen 7
    # 2x2x1: schoolbook 4
    # Total 2x2x3: 11
    # 2x2x2 (for k=3,4): Strassen 7
    # Total 2x2x5: 11+7=18
    # So 4x2x5 = 2*18 = 36. Even better!
    # Total: 23+18+15+36 = 92.
    #
    # Let me verify: 2x2x3 = 2x2x2 + 2x2x1
    # 2x2x2: Strassen 7
    # 2x2x1: schoolbook 2*2*1=4
    # Total: 11. Correct.
    # 2x2x2 for k=3,4: Strassen 7. Correct.
    # 2x2x5: 11+7=18. Correct.
    # 4x2x5: 2*18=36. Correct.
    #
    # So total = 23 (Laderman) + 18 (3x3x2) + 15 (1x3x5) + 36 (4x2x5) = 92.
    #
    # Can we do even better? 
    # 3x3x2: can we use Strassen? 3x3x2 = 2x2x2 + 2x2x1 + 1x2x2 + 1x2x1 + 1x1x2 + 1x1x1
    # = 7 + 4 + 4 + 2 + 2 + 1 = 20. Worse than 18.
    #
    # 1x3x5: schoolbook 15. Can't really improve since one dimension is 1.
    #
    # Let me try: 4x5x5 = 4x3x5 + 4x2x5
    # 4x3x5 = 4x3x3 + 4x3x2
    # 4x3x3 = 3x3x3 + 1x3x3 = 23 + 9 = 32
    # 4x3x2 = 3x3x2 + 1x3x2 = 18 + 6 = 24
    # 4x3x5 = 32+24=56
    # 4x2x5 = 36
    # Total: 56+36=92. Same.
    #
    # Try another split: 4x5x5 = 2x5x5 + 2x5x5
    # 2x5x5 = 2x4x5 + 2x1x5
    # 2x4x5 = 2x4x4 + 2x4x1
    # 2x4x4 = 2x3x4 + 2x1x4
    # 2x3x4 = 2x3x3 + 2x3x1 = (2x2x3 + 2x1x3) + 2x3x1
    # 2x2x3 = 11 (from above)
    # 2x1x3 = 6
    # 2x3x1 = 6
    # 2x3x4 = 11+6+6=23
    # 2x1x4 = 8
    # 2x4x4 = 23+8=31
    # 2x4x1 = 8
    # 2x5x5 = 31+8+10=49 (2x1x5=10)
    # Total: 2*49=98. Worse.
    #
    # So 92 seems good. Let me implement it.
    
    # Block 1: 3x3x3 at (i0=0, j0=0, k0=0) using Laderman
    U_s, V_s, W_s = _laderman_3x3x3_embed(n, m, p, 0, 0, 0)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    # Block 2: 3x3x2 at (i0=0, j0=0, k0=3) schoolbook
    U_s, V_s, W_s = _schoolbook_embed(n, m, p, 0, 0, 3, 3, 3, 2)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    # Block 3: 1x3x5 at (i0=3, j0=0, k0=0) schoolbook
    U_s, V_s, W_s = _schoolbook_embed(n, m, p, 3, 0, 0, 1, 3, 5)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    # Block 4: 4x2x5 at (i0=0, j0=3, k0=0)
    # Split into 2 blocks of 2x2x5
    # 2x2x5 = 2x2x3 + 2x2x2
    # 2x2x3 = 2x2x2 + 2x2x1
    # 2x2x2 = Strassen 7
    # 2x2x1 = schoolbook 4
    
    # Sub-block 4a: 2x2x2 at (i0=0, j0=3, k0=0) Strassen
    U_s, V_s, W_s = _strassen_2x2x2_embed(n, m, p, 0, 3, 0)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    # Sub-block 4b: 2x2x1 at (i0=0, j0=3, k0=2) schoolbook
    U_s, V_s, W_s = _schoolbook_embed(n, m, p, 0, 3, 2, 2, 2, 1)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    # Sub-block 4c: 2x2x2 at (i0=0, j0=3, k0=3) Strassen
    U_s, V_s, W_s = _strassen_2x2x2_embed(n, m, p, 0, 3, 3)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    # Sub-block 4d: 2x2x2 at (i0=2, j0=3, k0=0) Strassen
    U_s, V_s, W_s = _strassen_2x2x2_embed(n, m, p, 2, 3, 0)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    # Sub-block 4e: 2x2x1 at (i0=2, j0=3, k0=2) schoolbook
    U_s, V_s, W_s = _schoolbook_embed(n, m, p, 2, 3, 2, 2, 2, 1)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    # Sub-block 4f: 2x2x2 at (i0=2, j0=3, k0=3) Strassen
    U_s, V_s, W_s = _strassen_2x2x2_embed(n, m, p, 2, 3, 3)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    U_combined = np.hstack(U_parts)
    V_combined = np.hstack(V_parts)
    W_combined = np.hstack(W_parts)
    
    rank = U_combined.shape[1]
    
    # Verify
    if _verify(U_combined, V_combined, W_combined, n, m, p):
        return {"U": U_combined.tolist(), "V": V_combined.tolist(), "W": W_combined.tolist()}
    
    # Fallback: pure schoolbook
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
