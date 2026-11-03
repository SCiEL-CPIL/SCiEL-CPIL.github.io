import time
import numpy as np

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 115

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m times m x p -> n x p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W, rank


def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Rank 7.
    A indices: 00,01,10,11 -> 0,1,2,3
    B indices: 00,01,10,11 -> 0,1,2,3
    W indices: k*n+i -> 0,1,2,3 (n=2)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # m1 = (a+d)(e+h)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0  # C00

    # m2 = (c+d)(e+f)
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0; V[1][1] = 1.0
    W[2][1] = 1.0  # C10
    W[3][1] = -1.0 # C11

    # m3 = a(f-h)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0  # C01
    W[3][2] = 1.0  # C11

    # m4 = d(g-e)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0  # C00
    W[2][3] = 1.0  # C10

    # m5 = (a+b)h
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0 # C00
    W[1][4] = 1.0  # C01

    # m6 = (c-d)e
    U[2][5] = 1.0; U[3][5] = -1.0
    V[0][5] = 1.0
    W[3][5] = 1.0  # C11

    # m7 = (b-d)(f+g)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[1][6] = 1.0; V[2][6] = 1.0
    W[0][6] = 1.0  # C00

    return U, V, W, rank


def _strassen_3x3():
    """
    Decomposition for 3x3 * 3x3 -> 3x3.
    We use a block approach: split into 2x2 + 1x1.
    This yields rank 7 (for 2x2 part) + 6 (for cross terms) + 1 (for 1x1 part) = 14?
    Wait, let's trace carefully.
    
    A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1.
    
    C11 (2x2) = A11 B11 + A12 B21
    C12 (2x1) = A11 B12 + A12 B22
    C21 (1x2) = A21 B11 + A22 B21
    C22 (1x1) = A21 B12 + A22 B22
    
    We need decompositions for:
    1. 2x2 * 2x2 -> 2x2 (Rank 7)
    2. 2x1 * 1x2 -> 2x2 (Rank 4? No, 2x1 * 1x2 is outer product. Rank 1? No, it's a matrix. 
       A12 is 2x1 (vector), B21 is 1x2 (vector). Product is 2x2 matrix. 
       This can be done with 4 scalar multiplications: A12[0]*B21[0], A12[0]*B21[1], A12[1]*B21[0], A12[1]*B21[1].
       So rank 4.)
    3. 2x1 * 1x1 -> 2x1 (Rank 2)
    4. 1x2 * 2x2 -> 1x2 (Rank 4)
    5. 1x1 * 1x2 -> 1x2 (Rank 2)
    6. 1x2 * 2x1 -> 1x1 (Rank 4? No, dot product. Rank 2? No, sum of 2 products. Rank 2.)
    7. 1x1 * 1x1 -> 1x1 (Rank 1)
    
    Total Rank = 7 + 4 + 2 + 4 + 2 + 2 + 1 = 22.
    Schoolbook is 27. So 22 is an improvement.
    """
    n, m, p = 3, 3, 3
    # We will construct this by concatenating sub-decompositions.
    # But _block_decompose is complex to set up for this specific split.
    # Let's just use schoolbook for 3x3 for now to ensure correctness, 
    # or implement the 22-rank one.
    # Given time, schoolbook is safer.
    return _schoolbook(3, 3, 3)


def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Use numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i,j,k,r] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+r, c]
    # We want T[i,j,k,r] = delta_{i,r} delta_{j,k}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+r][c]
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True


def _combine_decompositions(U1, V1, W1, U2, V2, W2, n, m, p):
    """
    Combine two decompositions that target the same output tensor.
    This is only valid if the decompositions are for disjoint parts of the tensor
    or if we are summing them (which is not standard for rank minimization unless we can cancel).
    Actually, to combine, we just concatenate the columns.
    The resulting decomposition is the sum of the two tensors.
    If T1 + T2 = T_target, this works.
    """
    r1 = len(W1[0])
    r2 = len(W2[0])
    r = r1 + r2
    U = []
    for row in range(n * m):
        U.append(U1[row] + U2[row])
    V = []
    for row in range(m * p):
        V.append(V1[row] + V2[row])
    W = []
    for row in range(p * n):
        W.append(W1[row] + W2[row])
    return U, V, W, r


def _decompose_4x5x5():
    """
    Decompose 4x5 * 5x5 -> 4x5.
    Strategy: Split n=4 into two 2x blocks.
    For each 2x5 * 5x5 -> 2x5 block, we can try to optimize.
    For a 2x5 * 5x5 problem:
    Split m=5 into 2+3, p=5 into 2+3.
    Sub-problems for 2x5x5:
    1. 2x2 * 2x2 -> 2x2 (Rank 7 via Strassen)
    2. 2x3 * 3x2 -> 2x2 (Rank 12 schoolbook, can we do better? 2x3 * 3x2 is rank 4 tensor? No, it's a 2x2 matrix from 2x3 and 3x2. Rank 4 is schoolbook. Min rank is likely 4.)
    3. 2x2 * 2x3 -> 2x3 (Rank 12 schoolbook. Min rank likely 4? No, 2x2 * 2x3 -> 2x3. Tensor size 2*2*3=12. Min rank? Probably 4 or 6.)
    4. 2x3 * 3x3 -> 2x3 (Rank 18 schoolbook.)
    
    Let's just use schoolbook for the sub-blocks for now, but use Strassen for the 2x2 part.
    
    Actually, let's build the 4x5x5 by splitting n into 2 and 2.
    Block 1: rows 0,1 of A.
    Block 2: rows 2,3 of A.
    
    For Block 1 (2x5 * 5x5 -> 2x5):
    We can split m into 2 and 3, and p into 2 and 3.
    This creates 4 sub-tensors.
    1. A11 (2x2) * B11 (2x2) -> C11 (2x2). Use Strassen (Rank 7).
    2. A12 (2x3) * B21 (3x2) -> C11 (2x2). Use Schoolbook (Rank 12).
    3. A11 (2x2) * B12 (2x3) -> C12 (2x3). Use Schoolbook (Rank 12).
    4. A12 (2x3) * B22 (3x3) -> C12 (2x3). Use Schoolbook (Rank 18).
    Total Rank for Block 1 = 7 + 12 + 12 + 18 = 49.
    
    For Block 2 (2x5 * 5x5 -> 2x5):
    Same structure, same rank 49.
    
    Total Rank = 49 + 49 = 98.
    Schoolbook is 100.
    So we save 2 ranks.
    """
    n, m, p = 4, 5, 5
    
    # Split n into n1=2, n2=2
    # Split m into m1=2, m2=3
    # Split p into p1=2, p2=3
    
    U = [[0.0] * 98 for _ in range(n * m)]
    V = [[0.0] * 98 for _ in range(m * p)]
    W = [[0.0] * 98 for _ in range(p * n)]
    
    col = 0
    
    # Block 1: i in [0, 2)
    # Sub 1.1: 2x2 * 2x2 -> 2x2 (Strassen)
    U1, V1, W1, r1 = _strassen_2x2()
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i * m + j][col + c] = U1[i * 2 + j][c]
    for j in range(2):
        for k in range(2):
            for c in range(r1):
                V[j * p + k][col + c] = V1[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k * n + i][col + c] = W1[k * 2 + i][c]
    col += r1
    
    # Sub 1.2: 2x3 * 3x2 -> 2x2 (Schoolbook)
    U2, V2, W2, r2 = _schoolbook(2, 3, 2)
    for i in range(2):
        for j in range(3):
            for c in range(r2):
                U[i * m + (2 + j)][col + c] = U2[i * 3 + j][c]
    for j in range(3):
        for k in range(2):
            for c in range(r2):
                V[(2 + j) * p + k][col + c] = V2[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(r2):
                W[k * n + i][col + c] += W2[k * 2 + i][c]
    col += r2
    
    # Sub 1.3: 2x2 * 2x3 -> 2x3 (Schoolbook)
    U3, V3, W3, r3 = _schoolbook(2, 2, 3)
    for i in range(2):
        for j in range(2):
            for c in range(r3):
                U[i * m + j][col + c] = U3[i * 2 + j][c]
    for j in range(2):
        for k in range(3):
            for c in range(r3):
                V[j * p + (2 + k)][col + c] = V3[j * 3 + k][c]
    for k in range(3):
        for i in range(2):
            for c in range(r3):
                W[(2 + k) * n + i][col + c] = W3[k * 2 + i][c]
    col += r3
    
    # Sub 1.4: 2x3 * 3x3 -> 2x3 (Schoolbook)
    U4, V4, W4, r4 = _schoolbook(2, 3, 3)
    for i in range(2):
        for j in range(3):
            for c in range(r4):
                U[i * m + (2 + j)][col + c] = U4[i * 3 + j][c]
    for j in range(3):
        for k in range(3):
            for c in range(r4):
                V[(2 + j) * p + (2 + k)][col + c] = V4[j * 3 + k][c]
    for k in range(3):
        for i in range(2):
            for c in range(r4):
                W[(2 + k) * n + i][col + c] += W4[k * 2 + i][c]
    col += r4
    
    # Block 2: i in [2, 4)
    # Sub 2.1: 2x2 * 2x2 -> 2x2 (Strassen)
    U1, V1, W1, r1 = _strassen_2x2()
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[(2 + i) * m + j][col + c] = U1[i * 2 + j][c]
    for j in range(2):
        for k in range(2):
            for c in range(r1):
                V[j * p + k][col + c] = V1[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k * n + (2 + i)][col + c] = W1[k * 2 + i][c]
    col += r1
    
    # Sub 2.2: 2x3 * 3x2 -> 2x2 (Schoolbook)
    U2, V2, W2, r2 = _schoolbook(2, 3, 2)
    for i in range(2):
        for j in range(3):
            for c in range(r2):
                U[(2 + i) * m + (2 + j)][col + c] = U2[i * 3 + j][c]
    for j in range(3):
        for k in range(2):
            for c in range(r2):
                V[(2 + j) * p + k][col + c] = V2[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(r2):
                W[k * n + (2 + i)][col + c] += W2[k * 2 + i][c]
    col += r2
    
    # Sub 2.3: 2x2 * 2x3 -> 2x3 (Schoolbook)
    U3, V3, W3, r3 = _schoolbook(2, 2, 3)
    for i in range(2):
        for j in range(2):
            for c in range(r3):
                U[(2 + i) * m + j][col + c] = U3[i * 2 + j][c]
    for j in range(2):
        for k in range(3):
            for c in range(r3):
                V[j * p + (2 + k)][col + c] = V3[j * 3 + k][c]
    for k in range(3):
        for i in range(2):
            for c in range(r3):
                W[(2 + k) * n + (2 + i)][col + c] = W3[k * 2 + i][c]
    col += r3
    
    # Sub 2.4: 2x3 * 3x3 -> 2x3 (Schoolbook)
    U4, V4, W4, r4 = _schoolbook(2, 3, 3)
    for i in range(2):
        for j in range(3):
            for c in range(r4):
                U[(2 + i) * m + (2 + j)][col + c] = U4[i * 3 + j][c]
    for j in range(3):
        for k in range(3):
            for c in range(r4):
                V[(2 + j) * p + (2 + k)][col + c] = V4[j * 3 + k][c]
    for k in range(3):
        for i in range(2):
            for c in range(r4):
                W[(2 + k) * n + (2 + i)][col + c] += W4[k * 2 + i][c]
    col += r4
    
    return U, V, W, col


def construct():
    n, m, p = DIMENSIONS
    # Try the 98-rank decomposition
    try:
        U, V, W, rank = _decompose_4x5x5()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
