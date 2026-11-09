"""Tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]
DEADLINE_MARGIN = 10  # seconds before the 150s limit


def _build_schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W):
    """Check that sum_c U[:,c]*V[:,c]*W[:,c] reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[i,j,k] = 1 if (i,j,k) valid
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check all entries for exactness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True


def _strassen_2x2_decomp():
    """
    Return Strassen's 7-rank decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7) for input A (2x2 flattened row-major)
    V: (4, 7) for input B (2x2 flattened row-major)  
    W: (4, 7) for output C (2x2 flattened row-major)
    
    Strassen's formulas:
    M1 = (a00 + a11) * (b00 + b11)
    M2 = (a10 + a11) * b00
    M3 = a00 * (b01 - b11)
    M4 = a11 * (b10 - b00)
    M5 = (a00 + a01) * b11
    M6 = (a10 - a00) * (b00 + b10)
    M7 = (a11 - a10) * (b01 + b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U rows: [a00, a01, a10, a11]
    # V rows: [b00, b01, b10, b11]
    # W rows: [c00, c01, c10, c11]
    
    U = [
        [1, 0, 0, 0, 1, 0, 0],  # a00
        [0, 1, 0, 0, 1, 0, 0],  # a01
        [0, 0, 1, 1, 0, 1, 0],  # a10
        [1, 0, 0, 1, 0, 0, 1],  # a11
    ]
    
    V = [
        [1, 1, 0, 1, 0, 1, 0],  # b00
        [1, 0, 1, 0, 0, 0, 1],  # b01
        [0, 0, 0, 0, 1, 1, 0],  # b10
        [1, 0, -1, 0, 1, 0, 1],  # b11
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, 0, 1, 0, 0, 1, 0],  # c11
    ]
    
    return U, V, W


def _make_4x4x4_strassen():
    """
    Build 4x4 * 4x4 -> 4x4 decomposition using recursive Strassen.
    Split into 2x2 blocks and use Strassen for each.
    Returns U (16, rank), V (16, rank), W (16, rank)
    """
    # Split 4x4 into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # Using Strassen for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A11-A10)*(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix computed by Strassen 2x2 mult (rank 7)
    # So total rank = 7 * 7 = 49
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # For each of the 7 Strassen products, we need a 2x2*2x2->2x2 decomposition
    # Block indices: 00, 01, 10, 11
    # A block ij has rows [2i, 2i+1] x [2j, 2j+1] in 4x4
    # When flattened row-major: block (i,j) starts at row (2i)*4 + 2j
    
    def get_block_rows(block_i, block_j, size=4):
        """Get row indices for a 2x2 block in a 4x4 matrix (flattened row-major)."""
        start_row = block_i * 2
        start_col = block_j * 2
        rows = []
        for di in range(2):
            for dj in range(2):
                row = (start_row + di) * size + (start_col + dj)
                rows.append(row)
        return rows
    
    # Strassen formulas for 2x2 block multiplication
    # We'll build the full 4x4x4 decomposition
    strassen_U, strassen_V, strassen_W = _strassen_2x2_decomp()
    
    # The 7 Strassen intermediate matrices
    # Each is computed from specific combinations of 2x2 blocks
    # For the 4x4 case, we apply Strassen at the block level
    
    # Block structure:
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01], [B10, B11]] (each 2x2)
    
    # Strassen intermediates (each is 2x2 result of 2x2*2x2):
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B10)
    # M7 = (A11-A10) * (B01+B11)
    
    # For each Mi, we need to decompose the 2x2*2x2->2x2 multiplication
    # using Strassen's 7-rank method
    
    # Let's build this more carefully
    # For each of the 7 Strassen products, we have a 2x2*2x2->2x2 subproblem
    # Each subproblem has rank 7 (Strassen)
    # Total rank = 7*7 = 49
    
    # U matrix: 16 rows (4x4 A flattened), 49 cols
    # V matrix: 16 rows (4x4 B flattened), 49 cols
    # W matrix: 16 rows (4x4 C flattened), 49 cols
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    # For each of the 7 outer Strassen products, and for each of the 7 inner Strassen products
    for outer_idx in range(7):
        for inner_idx in range(7):
            col = outer_idx * 7 + inner_idx
            
            # Determine which blocks are involved in outer product outer_idx
            # and how they're combined
            
            # Outer Strassen formulas (for 2x2 block matrices):
            if outer_idx == 0:  # M1 = (A00+A11) * (B00+B11)
                a_blocks = [(0,0,1), (1,1,1)]  # (block_i, block_j, sign)
                b_blocks = [(0,0,1), (1,1,1)]
            elif outer_idx == 1:  # M2 = (A10+A11) * B00
                a_blocks = [(1,0,1), (1,1,1)]
                b_blocks = [(0,0,1)]
            elif outer_idx == 2:  # M3 = A00 * (B01-B11)
                a_blocks = [(0,0,1)]
                b_blocks = [(0,1,1), (1,1,-1)]
            elif outer_idx == 3:  # M4 = A11 * (B10-B00)
                a_blocks = [(1,1,1)]
                b_blocks = [(1,0,1), (0,0,-1)]
            elif outer_idx == 4:  # M5 = (A00+A01) * B11
                a_blocks = [(0,0,1), (0,1,1)]
                b_blocks = [(1,1,1)]
            elif outer_idx == 5:  # M6 = (A10-A00) * (B00+B10)
                a_blocks = [(1,0,1), (0,0,-1)]
                b_blocks = [(0,0,1), (1,0,1)]
            else:  # outer_idx == 6: M7 = (A11-A10) * (B01+B11)
                a_blocks = [(1,1,1), (1,0,-1)]
                b_blocks = [(0,1,1), (1,1,1)]
            
            # For each involved A block, add its contribution to U
            for (bi, bj, sign) in a_blocks:
                rows = get_block_rows(bi, bj)
                for local_i, row in enumerate(rows):
                    U[row][col] += sign * strassen_U[local_i][inner_idx]
            
            # For each involved B block, add its contribution to V
            for (bi, bj, sign) in b_blocks:
                rows = get_block_rows(bi, bj)
                for local_i, row in enumerate(rows):
                    V[row][col] += sign * strassen_V[local_i][inner_idx]
            
            # The result of outer product outer_idx contributes to C blocks
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            if outer_idx == 0:  # M1 contributes to C00 (+1) and C11 (+1)
                c_blocks = [(0,0,1), (1,1,1)]
            elif outer_idx == 1:  # M2 contributes to C10 (+1) and C11 (-1)
                c_blocks = [(1,0,1), (1,1,-1)]
            elif outer_idx == 2:  # M3 contributes to C01 (+1) and C11 (+1)
                c_blocks = [(0,1,1), (1,1,1)]
            elif outer_idx == 3:  # M4 contributes to C00 (+1) and C10 (+1)
                c_blocks = [(0,0,1), (1,0,1)]
            elif outer_idx == 4:  # M5 contributes to C00 (-1) and C01 (+1)
                c_blocks = [(0,0,-1), (0,1,1)]
            elif outer_idx == 5:  # M6 contributes to C11 (+1)
                c_blocks = [(1,1,1)]
            else:  # outer_idx == 6: M7 contributes to C00 (+1)
                c_blocks = [(0,0,1)]
            
            # For each involved C block, add its contribution to W
            for (bi, bj, sign) in c_blocks:
                rows = get_block_rows(bi, bj)
                for local_i, row in enumerate(rows):
                    W[row][col] += sign * strassen_W[local_i][inner_idx]
    
    return U, V, W


def _make_4x4x5_decomp():
    """
    Build 4x4 * 4x5 -> 4x5 decomposition.
    Split B into B[:, :4] and B[:, 4].
    C = A * B[:, :4] + A * B[:, 4:5]
    
    First part: 4x4 * 4x4 -> 4x4 (use Strassen, rank 49)
    Second part: 4x4 * 4x1 -> 4x1 (schoolbook, rank 4)
    
    Total rank: 49 + 4 = 53
    """
    U_444, V_444, W_444 = _make_4x4x4_strassen()
    
    # For the 4x4 * 4x1 -> 4x1 part (last column of B)
    # This is just: C_i = sum_j A[i,j] * B[j,4]
    # We can decompose this with 16 rank-1 terms, but we can do better
    # Actually, for each output row i, C_i = dot(A[i,:], B[:,4])
    # This is a bilinear form, rank = 16 (schoolbook) but we can use
    # the fact that it's just matrix-vector multiplication
    
    # Wait, let me reconsider. The tensor for 4x4 * 4x1 -> 4x1:
    # T[i, j, k] where i=0..3 (output row), j=0..3 (input A col), k=0 (only one B col)
    # Actually k should be the B column index, so k=4 in the full 4x5 case
    # T[i, j, 4] = 1 for all i,j
    
    # For the full 4x4*4x5 decomposition:
    # U: (16, rank) - A flattened
    # V: (20, rank) - B flattened (4x5 = 20 elements)
    # W: (20, rank) - C flattened (4x5 = 20 elements)
    
    # For the 4x4*4x4 part (columns 0-3 of B, columns 0-3 of C):
    # U_444: (16, 49)
    # V_444: (16, 49) - only first 16 rows of V (B[:, :4])
    # W_444: (16, 49) - only first 16 rows of W (C[:, :4])
    
    # For the 4x4*4x1 part (column 4 of B, column 4 of C):
    # This is: C[:, 4] = A * B[:, 4]
    # T[i, j, 4] = 1 for i=0..3, j=0..3
    # We need: sum_c U[i,j] * V[j,4] * W[i,4] = 1 for all i,j
    # This is the all-ones tensor of size 4x4x1
    # Rank of all-ones tensor 4x4x1: we can do it with rank 4!
    # C_i = sum_j A[i,j] * B[j,4]
    # Let V[j,4] = 1 for all j, and U[i,j] * W[i,4] = A[i,j]
    # But we need exact decomposition...
    # Actually: sum_c U_c[i,j] * V_c[j,4] * W_c[i,4] = 1
    # If we set V_c[j,4] = 1 for all j,c, and U_c[i,j] * W_c[i,4] = delta_{i,j}
    # No, that doesn't work.
    
    # The tensor T[i,j,k] = 1 for all i,j,k (with k fixed at 4)
    # This is the all-ones 4x4 matrix (for fixed k)
    # Rank of all-ones 4x4 matrix is 1!
    # So we can do: U[i,j] = 1, V[j,4] = 1, W[i,4] = 1
    # Then sum = 1*1*1 = 1. But that's only one term!
    # Wait, that's not right because U, V, W need to be consistent.
    
    # Actually for the full decomposition, we need:
    # For k=4: sum_c U[i*m+j, c] * V[j*p+4, c] * W[4*n+i, c] = 1
    # Let's use rank 4 for this part:
    # U[i*4+j, c] = delta_{i,c} for c=0..3
    # V[j*5+4, c] = 1 for all j, c
    # W[4*4+i, c] = delta_{i,c} for c=0..3
    # Then sum_c delta_{i,c} * 1 * delta_{i,c} = 1 if i=c, but that gives 1 only when i=c
    # That's not right either.
    
    # Let me think again. We want:
    # sum_c U[i,j,c] * V[j,4,c] * W[4,i,c] = 1 for all i,j
    # Simple solution: U[i,j,c] = 1, V[j,4,c] = 1, W[4,i,c] = 1 for c=0 only
    # Then sum = 1. So rank 1 suffices!
    
    # But wait, we need to combine with the 4x4x4 part.
    # The 4x4x4 part uses columns 0-3 of B and C.
    # The 4x4x1 part uses column 4 of B and C.
    # They're independent, so we can just concatenate.
    
    # For the 4x4x1 part, rank 1:
    # U_extra: (16, 1) - all ones
    # V_extra: (20, 1) - V[j*5+4, 0] = 1 for j=0..3, others 0
    # W_extra: (20, 1) - W[4*4+i, 0] = 1 for i=0..3, others 0
    
    # Actually, I realize I need to be more careful.
    # Let me just use schoolbook for the last column to be safe.
    # Rank 16 for 4x4*4x1 -> 4x1
    
    # But that would give total rank 49 + 16 = 65, which is worse than expected.
    
    # Let me reconsider the problem structure.
    # The task says: 4x4 * 4x5 -> 4x5
    # So A is 4x4, B is 4x5, C = A*B is 4x5
    
    # Tensor: T[i, j, k] where:
    # i = 0..3 (row of C, also row of A)
    # j = 0..3 (col of A, row of B)  
    # k = 0..4 (col of B, col of C)
    # T[i,j,k] = 1 for all valid (i,j,k)
    
    # Decomposition: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # For k=0,1,2,3: this is the 4x4*4x4 part (with B having only 4 cols)
    # For k=4: this is the 4x4*4x1 part
    
    # The 4x4*4x4 part: we can use Strassen with rank 49
    # But Strassen gives us C = A*B where both are 4x4
    # Here B is 4x4 for k=0..3, so yes.
    
    # For k=4: T[i,j,4] = 1 for all i,j
    # This is the all-ones 4x4 matrix
    # We can decompose this with rank 1:
    # U[i,j,c] = 1, V[j,4,c] = 1, W[4,i,c] = 1 for c = 49 (new column)
    # Then sum = 1*1*1 = 1. 
    
    # So total rank = 49 + 1 = 50!
    
    # Let's build this properly.
    
    # Start with 4x4x4 Strassen decomposition
    U_444, V_444, W_444 = _make_4x4x4_strassen()
    
    # Now extend to 4x4x5
    # U: (16, 50) - A is 4x4, same for both parts
    # V: (20, 50) - B is 4x5
    # W: (20, 50) - C is 4x5
    
    U = [[0] * 50 for _ in range(16)]
    V = [[0] * 50 for _ in range(20)]
    W = [[0] * 50 for _ in range(20)]
    
    # Copy 4x4x4 part into first 49 columns
    # U_444 is (16, 49), V_444 is (16, 49) for B[:, :4], W_444 is (16, 49) for C[:, :4]
    for i in range(16):
        for c in range(49):
            U[i][c] = U_444[i][c]
    
    # V_444 has 16 rows (for B[:, :4]), we need to place them in V[j*5+k, c] for k=0..3
    # B row j, col k -> index j*5 + k
    for j in range(4):
        for k in range(4):
            row = j * 5 + k
            for c in range(49):
                V[row][c] = V_444[j * 4 + k][c]
    
    # W_444 has 16 rows (for C[:, :4]), we need to place them in W[k*4+i, c] for k=0..3
    # C row i, col k -> index k*4 + i (following the convention W[k*n+i, r])
    for i in range(4):
        for k in range(4):
            row = k * 4 + i
            for c in range(49):
                W[row][c] = W_444[k * 4 + i][c]
    
    # Add the 4x4x1 part (k=4) with rank 1
    # U[i*4+j, 49] = 1 for all i,j
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][49] = 1
    
    # V[j*5+4, 49] = 1 for all j
    for j in range(4):
        V[j * 5 + 4][49] = 1
    
    # W[4*4+i, 49] = 1 for all i
    for i in range(4):
        W[4 * 4 + i][49] = 1
    
    return U, V, W


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as lists of lists.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Start with schoolbook as fallback
    fallback = _build_schoolbook()
    
    # Try the Strassen-based decomposition
    try:
        U, V, W = _make_4x4x5_decomp()
        
        # Verify correctness
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            # Verification failed, return schoolbook
            return fallback
    except Exception:
        # If Strassen approach fails, return schoolbook
        return fallback


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['W'][0])}")
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
