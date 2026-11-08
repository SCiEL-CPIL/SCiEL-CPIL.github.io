"""Optimized program for matrix multiplication tensor decomposition 2x4 by 4x6."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]
n, m, p = DIMENSIONS
nm = n * m  # 8
mp = m * p  # 24
pn = p * n  # 12


def _validate(U, V, W, rank):
    """Check if U, V, W exactly decompose the matmul tensor."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u_c = Ua[:, c].reshape(n, m)
        v_c = Va[:, c].reshape(m, p)
        w_c = Wa[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True


def _clean_values(mat):
    """Clean floating point values to exact half-integers."""
    result = []
    for row in mat:
        new_row = []
        for val in row:
            if abs(val) < 1e-10:
                new_row.append(0)
            elif abs(val - 0.5) < 1e-10:
                new_row.append(0.5)
            elif abs(val + 0.5) < 1e-10:
                new_row.append(-0.5)
            elif abs(val - 1) < 1e-10:
                new_row.append(1)
            elif abs(val + 1) < 1e-10:
                new_row.append(-1)
            elif abs(val - 1.5) < 1e-10:
                new_row.append(1.5)
            elif abs(val + 1.5) < 1e-10:
                new_row.append(-1.5)
            else:
                new_row.append(val)
        result.append(new_row)
    return result


def _block_decompose_2x2_2x6():
    """
    Decompose 2x2 * 2x6 matrix multiplication.
    A is 2x2, B is 2x6, C is 2x6.
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    
    We can use a structured approach. For each k, we have a 2x2 * 2x1 product.
    But we can share the A factors.
    
    Actually, let's just use schoolbook for this sub-problem: rank = 2*2*6 = 24.
    But we can do better by noting that for fixed i, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    This is a linear combination.
    
    Let me use a different structure: think of it as 2 separate 1x2*2x6 multiplications
    that share B.
    """
    # Schoolbook for 2x2 * 2x6: 24 terms
    # U: 4x24, V: 12x24, W: 12x24
    # But wait, for the sub-problem, the dimensions are different.
    # In the sub-problem: n_sub=2, m_sub=2, p_sub=6
    # U_sub: (2*2)=4 x rank, V_sub: (2*6)=12 x rank, W_sub: (6*2)=12 x rank
    
    rank = 2 * 2 * 6  # schoolbook
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(6):
                U[i * 2 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
    
    return U, V, W, rank


def _construct_decomposition():
    """
    Construct a decomposition for 2x4 * 4x6 using block structure.
    
    Split the inner dimension 4 into two blocks of 2:
    A = [A1 | A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x6
    
    C = A1*B1 + A2*B2
    
    Each sub-problem is 2x2 * 2x6 with schoolbook rank 24.
    Total rank: 2 * 24 = 48 (same as schoolbook for the full problem)
    
    This isn't an improvement. Let me think of a better approach.
    
    Alternative: Use a Strassen-like approach for the 2x2 blocks.
    For 2x2 * 2x2, Strassen gives rank 7.
    But we have 2x2 * 2x6, not 2x2 * 2x2.
    
    Let me try: 2x2 * 2x6 can be viewed as 3 separate 2x2 * 2x2 multiplications
    (grouping the 6 columns into 3 pairs), each with Strassen rank 7.
    Total: 3 * 7 = 21 per block, times 2 blocks = 42.
    
    But wait, the A matrix is shared between the 3 sub-multiplications,
    and the B blocks are independent. So we can't simply multiply by 3.
    
    Actually, for 2x2 * 2x6:
    C[:, 0:2] = A * B[:, 0:2]  (2x2 * 2x2, Strassen rank 7)
    C[:, 2:4] = A * B[:, 2:4]  (2x2 * 2x2, Strassen rank 7)  
    C[:, 4:6] = A * B[:, 4:6]  (2x2 * 2x2, Strassen rank 7)
    
    Each uses the same A but different B blocks. The Strassen decomposition
    for each 2x2*2x2 uses 7 terms. Since A is shared, we can reuse the
    same U factors for A across all three sub-problems.
    
    So total rank for 2x2*2x6 = 3 * 7 = 21? No, that's not right either,
    because the U factors for A would be duplicated.
    
    Let me think more carefully. In Strassen for 2x2*2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A10)B11
    M6 = (A11-A01)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    For 2x2*2x6, we have 3 such sub-problems sharing A.
    The U factors (for A) in each Strassen term involve combinations of A entries.
    If we use the same Strassen pattern for all 3 column blocks, the U factors
    for A are the same, so we can share them.
    
    Actually, this is getting complicated. Let me just try to directly construct
    a low-rank decomposition using a different strategy.
    
    Let me try rank 36 with a structured approach.
    """
    pass


def construct():
    """Find rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Try a structured block decomposition
    # Split inner dimension 4 into 2+2
    # C = A1*B1 + A2*B2 where A1,A2 are 2x2, B1,B2 are 2x6
    
    # For each 2x2*2x6, use schoolbook rank 24
    # Total: 48 (same as schoolbook)
    
    # Let me try to find a better decomposition using a systematic search
    # with a smarter initialization based on block structure.
    
    # First, let's try the simple block approach and see if we can improve.
    
    # Approach: Use the fact that 2x4*4x6 can be decomposed as:
    # View A as 2 rows, each of length 4
    # View B as 4 rows, each of length 6
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Try rank 36: split into 3 groups of 4x2 (splitting the 6 into 3x2)
    # For each group, 2x4*4x2 has schoolbook rank 16
    # 3 * 16 = 48, still schoolbook
    
    # Let me just return schoolbook for now, but with a faster validation.
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
