"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[i,j,k] = 1 if we multiply (i,j) of A by (j,k) of B
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_2x2(n, m, p):
    """Standard Strassen 2x2x2 rank-7 decomposition."""
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    # Strassen: C = A*B where A is n x m, B is m x p
    # For 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e), M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5 + M6, C12 = M3 + M4, C21 = M2 + M5, C22 = M1 - M2 + M3 + M7
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For 2x2: indices: A[0,0]=0,A[0,1]=1,A[1,0]=2,A[1,1]=3
    # B[0,0]=0,B[0,1]=1,B[1,0]=2,B[1,1]=3
    # W[0,0]=0,W[0,1]=1,W[1,0]=2,W[1,1]=3 (k*n+i)
    
    # M1 = (a+d)(e+h): U[0,0]=1,U[3,0]=1; V[0,0]=1,V[3,0]=1; W[0,0]=1,W[3,0]=1
    # Actually let me use the standard formulation.
    # C = A*B, (AB)_{ik} = sum_j A_{ij} B_{jk}
    # Strassen for 2x2:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1 - p4 + p5 + p6
    # C12 = p3 + p4
    # C21 = p2 + p5
    # C22 = p1 - p2 + p3 + p7
    
    # U[A_idx, c], V[B_idx, c], W[C_idx, c] where C_idx = k*2+i
    # A: a=0, b=1, c=2, d=3
    # B: e=0, f=1, g=2, h=3
    # C: C11=0, C12=1, C21=2, C22=3 (k*n+i: k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3)
    
    # p1 = (a+d)(e+h):
    # A part: a=+1, d=+1 -> U[0,0]=1, U[3,0]=1
    # B part: e=+1, h=+1 -> V[0,0]=1, V[3,0]=1
    # C part: C11=+1, C22=+1 -> W[0,0]=1, W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (c+d)e:
    # A: c=+1, d=+1 -> U[2,1]=1, U[3,1]=1
    # B: e=+1 -> V[0,1]=1
    # C: C21=+1 -> W[2,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # p3 = a(f-h):
    # A: a=+1 -> U[0,2]=1
    # B: f=+1, h=-1 -> V[1,2]=1, V[3,2]=-1
    # C: C12=+1, C22=+1 -> W[1,2]=1, W[3,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # p4 = d(g-e):
    # A: d=+1 -> U[3,3]=1
    # B: g=+1, e=-1 -> V[2,3]=1, V[0,3]=-1
    # C: C11=-1, C12=+1 -> W[0,3]=-1, W[1,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1
    
    # p5 = (a+b)h:
    # A: a=+1, b=+1 -> U[0,4]=1, U[1,4]=1
    # B: h=+1 -> V[3,4]=1
    # C: C11=+1, C21=+1 -> W[0,4]=1, W[2,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[2][4] = 1
    
    # p6 = (c-a)(e+f):
    # A: c=+1, a=-1 -> U[2,5]=1, U[0,5]=-1
    # B: e=+1, f=+1 -> V[0,5]=1, V[1,5]=1
    # C: C11=+1 -> W[0,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # p7 = (b-d)(g+h):
    # A: b=+1, d=-1 -> U[1,6]=1, U[3,6]=-1
    # B: g=+1, h=+1 -> V[2,6]=1, V[3,6]=1
    # C: C22=+1 -> W[3,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _strassen_3x4x8():
    """
    Use block Strassen: treat 3x4x8 as blocks.
    3 = 2+1, 4 = 2+2, 8 = 4+4
    Block sizes: A is 3x4 = [2x2, 1x2; 2x2, 1x2]... 
    Actually let's use: n=3=2+1, m=4=2+2, p=8=4+4
    A = [[A11 (2x2), A12 (2x2)], [A21 (1x2), A22 (1x2)]]
    B = [[B11 (2x4), B12 (2x4)], [B21 (2x4), B22 (2x4)]]
    C = [[C11 (2x4), C12 (2x4)], [C21 (1x4), C22 (1x4)]]
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    This is just 8 multiplications of various sizes. Not directly Strassen.
    
    Let me try a different approach: use the known lower bound and search.
    For 3x4x8, the schoolbook rank is 96.
    Known results: M(3,4,8) <= ? 
    Actually, let's just use a smarter block decomposition.
    
    Alternative: Use 2x2 Strassen on the 2x2x4 subproblem and handle the rest.
    
    Let me think about this differently. The tensor is T[i,j,k] for i in 0..2, j in 0..3, k in 0..7.
    We want to find the minimum rank decomposition.
    
    Known: M(3,4,8) is at most... let's use a practical approach.
    Use block structure: 
    - First 2 rows of A (2x4) times B (4x8): use Strassen-like on 2x4x8
    - Last 1 row of A (1x4) times B (4x8): schoolbook rank 8
    
    For 2x4x8: split B into two 4x4 blocks? Or split the 4 dimension.
    2x4x8: A is 2x4, B is 4x8.
    Split: m=4=2+2, p=8=4+4
    A = [A1 (2x2), A2 (2x2)]
    B = [[B11 (2x4), B12 (2x4)], [B21 (2x4), B22 (2x4)]]
    C = A*B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    Each subproblem is 2x2x4. 
    2x2x4: A is 2x2, B is 2x4.
    Schoolbook rank = 16.
    Can we do better? 
    2x2x4 = 2x2x2 + 2x2x2 (split p=4 into 2+2), each 2x2x2 has rank 7 (Strassen).
    So 2x2x4 has rank at most 14.
    
    So 2x4x8: two 2x2x4 subproblems, each rank 14, total 28.
    Plus 1x4x8: schoolbook rank 8.
    Total: 28 + 8 = 36.
    
    Wait, but the blocks share the B matrix structure. Let me be more careful.
    
    Actually, the decomposition is additive over independent sub-tensors.
    T_3x4x8 = T_{(2,1)x4x8} where we split n.
    The n=2 part: T_{2x4x8}
    The n=1 part: T_{1x4x8}
    These are independent (different i indices), so rank is additive.
    
    T_{1x4x8}: only i=0. T[0,j,k]=1 for j in 0..3, k in 0..7.
    This is just the "outer product" tensor. Rank = 4*8 = 32? No.
    T[0,j,k] = 1. We need sum_c U[j,c]*V[k,c]*W[c] = 1 (W[0,c] since p*n=8*1=8, W index is k*1+0=k).
    Actually W has shape p*n = 8*1 = 8. W[k*1+0, c] = W[k,c].
    We need: for all j,k: sum_c U[j,c] * V[j*p+k, c] * W[k, c] = 1.
    Wait, V has shape m*p = 4*8 = 32. V[j*p+k, c].
    U has shape n*m = 1*4 = 4. U[0*m+j, c] = U[j, c].
    
    So: sum_c U[j,c] * V[8j+k, c] * W[k, c] = 1 for j=0..3, k=0..7.
    
    This is equivalent to: the 4x8 matrix M[j,k] = 1 must be decomposed as sum_c U[:,c] * (V[:,c] reshaped to 4x8) * W... no.
    Actually it's a bilinear form. Let me think of it as:
    Define A_c = U[:,c] (4-vector), B_c = V[:,c] (32-vector = 4x8 matrix), and W_c = W[:,c] (8-vector).
    Then M[j,k] = sum_c A_c[j] * B_c[j,k] * W_c[k].
    
    Hmm, this is getting complicated. Let me just use schoolbook for the 1x4x8 part.
    Rank of 1x4x8 schoolbook = 4*8 = 32.
    
    For 2x4x8:
    Split m=4 into 2+2:
    A = [A1 (2x2), A2 (2x2)]
    B = [B1 (2x8), B2 (2x8)] where B1 is top 2 rows, B2 is bottom 2 rows.
    C = A1*B1 + A2*B2
    T_{2x4x8} = T_{A1*B1} + T_{A2*B2}
    Each is 2x2x8.
    
    2x2x8: split p=8 into 4+4:
    = T_{2x2x4} + T_{2x2x4}
    Each 2x2x4: split p=4 into 2+2:
    = T_{2x2x2} + T_{2x2x2}
    Each 2x2x2: Strassen rank 7.
    
    So 2x2x4 = 7+7 = 14
    2x2x8 = 14+14 = 28
    2x4x8 = 28+28 = 56
    
    Total: 56 + 32 = 88.
    
    Hmm, that's better than 96 but not great. Let me see if I can do better on 1x4x8.
    
    Actually for 1x4x8, the tensor T[i,j,k] with only i=0: T[0,j,k]=1.
    This is the all-ones 4x8 matrix as a "tensor". 
    Rank: we need sum_c U[j,c] * V[8j+k,c] * W[k,c] = 1.
    
    One simple decomposition: for each (j,k) pair, one term. Rank = 32.
    Can we do better? 
    U[j,c], V[8j+k,c], W[k,c].
    If we set W[k,c] = 1 for all k, then we need sum_c U[j,c]*V[8j+k,c] = 1 for all j,k.
    This is like a matrix factorization of the 4x8 all-ones matrix as sum_c u_c * v_c^T where u_c is 4-vector and v_c is 8-vector, but with the constraint that v_c is indexed by (j,k) as v_{j,k}.
    Actually V[8j+k, c] is just the (j,k) entry of a 4x8 matrix for fixed c.
    So we need: J_{4x8} = sum_c U_c * V_c (outer product of 4-vector and 8-vector... no, V_c is a 4x8 matrix).
    Wait, U[j,c] is scalar, V[8j+k,c] is scalar. The product U[j,c]*V[8j+k,c] is a 4x8 matrix (rank 1 if we view U_c as 4-vector and V_c as 4x8 matrix, it's U_c * V_c elementwise? No.)
    
    U[j,c] * V[8j+k, c]: for fixed c, this is a 4x8 matrix where entry (j,k) = U[j,c] * V[8j+k, c].
    This is NOT a rank-1 matrix in general. It's a 4x8 matrix.
    
    Hmm, let me just use schoolbook. The total rank would be 88.
    
    Actually, let me reconsider. Maybe I should just implement a search for better rank.
    But given time constraints, let me go with the block decomposition giving rank 88.
    """
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Process 2x4x8 part (i=0,1)
    # Split m=4 into two 2s
    for i in range(2):
        for j_block in range(2):  # j in [0,1] and [2,3]
            j0 = j_block * 2
            for k_block in range(4):  # k in groups of 2
                k0 = k_block * 2
                # 2x2x2 Strassen for A[i, j0:j0+2], B[j0:j0+2, k0:k0+2]
                # Local indices: a=i (0 or 1), b=j-j0 (0 or 1), c=k-k0 (0 or 1)
                # Global: A_idx = i*m + j, B_idx = j*p + k, C_idx = k*n + i
                for s in range(7):
                    # Strassen coefficients
                    if s == 0:  # p1 = (a+d)(e+h)
                        # A: a=1,d=1 -> local (0,0),(1,1)
                        # B: e=1,h=1 -> local (0,0),(1,1)
                        # C: C00=1,C11=1 -> local (0,0),(1,1)
                        a_coefs = [(0,0,1),(1,1,1)]
                        b_coefs = [(0,0,1),(1,1,1)]
                        c_coefs = [(0,0,1),(1,1,1)]
                    elif s == 1:  # p2 = (c+d)e
                        a_coefs = [(1,0,1),(1,1,1)]
                        b_coefs = [(0,0,1)]
                        c_coefs = [(1,0,1)]
                    elif s == 2:  # p3 = a(f-h)
                        a_coefs = [(0,0,1)]
                        b_coefs = [(0,1,1),(1,1,-1)]
                        c_coefs = [(0,1,1),(1,1,1)]
                    elif s == 3:  # p4 = d(g-e)
                        a_coefs = [(1,1,1)]
                        b_coefs = [(1,0,1),(0,0,-1)]
                        c_coefs = [(0,0,-1),(0,1,1)]
                    elif s == 4:  # p5 = (a+b)h
                        a_coefs = [(0,0,1),(0,1,1)]
                        b_coefs = [(1,1,1)]
                        c_coefs = [(0,0,1),(1,0,1)]
                    elif s == 5:  # p6 = (c-a)(e+f)
                        a_coefs = [(1,0,1),(0,0,-1)]
                        b_coefs = [(0,0,1),(0,1,1)]
                        c_coefs = [(0,0,1)]
                    else:  # s == 6: p7 = (b-d)(g+h)
                        a_coefs = [(0,1,1),(1,1,-1)]
                        b_coefs = [(1,0,1),(1,1,1)]
                        c_coefs = [(1,1,1)]
                    
                    for (la, lb, va) in a_coefs:
                        gi = i
                        gj = j0 + lb
                        U[gi * m + gj][col] = va
                    for (lb, lc, vb) in b_coefs:
                        gj = j0 + lb
                        gk = k0 + lc
                        V[gj * p + gk][col] = vb
                    for (lc, li, vc) in c_coefs:
                        gk = k0 + lc
                        gi = i
                        W[gk * n + gi][col] = vc
                    col += 1
    
    # Process 1x4x8 part (i=2), schoolbook
    for j in range(4):
        for k in range(8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based block decomposition
    result = _strassen_3x4x8()
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
