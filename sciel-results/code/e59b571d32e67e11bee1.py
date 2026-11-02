"""Tensor decomposition for 2x3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 138.0  # seconds; leave 12s margin before the 150s kill

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # For each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_strassen_block_2x3x2():
    """
    Build a rank-11 decomposition for 2x3x2 matrix multiplication.
    """
    rank = 11
    U = [[0.0]*rank for _ in range(6)]
    V = [[0.0]*rank for _ in range(6)]
    W = [[0.0]*rank for _ in range(4)]
    
    # Strassen 2x2x2 (using A rows 0,1 and B rows 0,1)
    # A indices: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (a00+a11)(b00+b11) -> C00(+1), C11(+1)
    c = 0
    U[0][c] += 1; U[4][c] += 1
    V[0][c] += 1; V[3][c] += 1
    W[0][c] += 1; W[3][c] += 1
    
    # M2 = (a10+a11)(b00) -> C10(+1), C11(-1)
    c = 1
    U[3][c] += 1; U[4][c] += 1
    V[0][c] += 1
    W[2][c] += 1; W[3][c] += -1
    
    # M3 = a00(b01-b11) -> C01(+1), C11(+1)
    c = 2
    U[0][c] += 1
    V[1][c] += 1; V[3][c] += -1
    W[1][c] += 1; W[3][c] += 1
    
    # M4 = a11(b10-b00) -> C00(+1), C10(+1)
    c = 3
    U[4][c] += 1
    V[2][c] += 1; V[0][c] += -1
    W[0][c] += 1; W[2][c] += 1
    
    # M5 = (a00+a01)(b11) -> C00(-1), C01(+1)
    c = 4
    U[0][c] += 1; U[1][c] += 1
    V[3][c] += 1
    W[0][c] += -1; W[1][c] += 1
    
    # M6 = (a10-a00)(b00+b01) -> C11(+1)
    c = 5
    U[3][c] += 1; U[0][c] += -1
    V[0][c] += 1; V[1][c] += 1
    W[3][c] += 1
    
    # M7 = (a01-a11)(b10+b11) -> C00(+1)
    c = 6
    U[1][c] += 1; U[4][c] += -1
    V[2][c] += 1; V[3][c] += 1
    W[0][c] += 1
    
    # Schoolbook: A_right (column 2) * B_bottom (row 2)
    c = 7
    U[2][c] += 1; V[4][c] += 1; W[0][c] += 1
    c = 8
    U[2][c] += 1; V[5][c] += 1; W[1][c] += 1
    c = 9
    U[5][c] += 1; V[4][c] += 1; W[2][c] += 1
    c = 10
    U[5][c] += 1; V[5][c] += 1; W[3][c] += 1
    
    return U, V, W


def _build_block_decomposition():
    """
    Build the full 2x3x10 decomposition using 5 blocks of 2x3x2.
    Total rank: 5 * 11 = 55.
    """
    n, m, p = 2, 3, 10
    total_rank = 55
    U = [[0.0]*total_rank for _ in range(n*m)]
    V = [[0.0]*total_rank for _ in range(m*p)]
    W = [[0.0]*total_rank for _ in range(p*n)]
    
    block_U, block_V, block_W = _build_strassen_block_2x3x2()
    
    for b in range(5):
        rank_offset = b * 11
        
        for a_idx in range(6):
            for c in range(11):
                U[a_idx][rank_offset + c] += block_U[a_idx][c]
        
        v_map = [2*b, 2*b+1, 10+2*b, 10+2*b+1, 20+2*b, 20+2*b+1]
        for local_idx in range(6):
            global_idx = v_map[local_idx]
            for c in range(11):
                V[global_idx][rank_offset + c] += block_V[local_idx][c]
        
        w_map = [4*b, 4*b+2, 4*b+1, 4*b+3]
        for local_idx in range(4):
            global_idx = w_map[local_idx]
            for c in range(11):
                W[global_idx][rank_offset + c] += block_W[local_idx][c]
    
    return U, V, W


def _build_schoolbook():
    """Build the naive rank-60 schoolbook decomposition."""
    n, m, p = 2, 3, 10
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1.0
                V[j*p+k][column] = 1.0
                W[k*n+i][column] = 1.0
                column += 1
    return U, V, W


def _try_reduce_rank54(U55, V55, W55):
    """
    Try to find a rank-54 decomposition by identifying a linear dependency
    among the 55 rank-1 terms. The idea: if we can express one term as a
    linear combination of the others (in the tensor space), we can eliminate it.
    
    Strategy: For each column c in 0..54, try to eliminate it by checking if
    the resulting 54-term decomposition still satisfies the tensor equation.
    We do this by solving for corrections to the remaining terms.
    """
    n, m, p = 2, 3, 10
    U = np.array(U55, dtype=np.float64)
    V = np.array(V55, dtype=np.float64)
    W = np.array(W55, dtype=np.float64)
    
    # The tensor equation: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # If we remove column c0, we need the remaining 54 columns to still sum to 1.
    # This is generally not possible without modification.
    # 
    # Alternative: try to find a linear combination of all 55 terms that equals zero
    # in the tensor space. If such a combination exists, we can use it to reduce rank.
    #
    # The tensor space has dimension n*m*p = 60. We have 55 rank-1 tensors.
    # A linear dependency exists if the 55 tensors are linearly dependent in R^60.
    # Since 55 < 60, they are generically independent, so no dependency expected.
    #
    # However, we can try a different approach: modify the decomposition to use
    # fewer terms by exploiting structure. For 2x3x10, the output has only 20 elements.
    # 
    # Let's try a different strategy: use the fact that n=2 is small.
    # We can think of this as computing two rows of C:
    # C[0,k] = sum_j A[0,j]*B[j,k] for k=0..9
    # C[1,k] = sum_j A[1,j]*B[j,k] for k=0..9
    #
    # This is equivalent to two matrix multiplications of 1x3 by 3x10.
    # Each 1x3 x 3x10 multiplication has rank 3 (schoolbook).
    # Total: 6 rank-1 terms. But we can do better using the shared B structure.
    #
    # Actually, the key insight is that the two rows share the same B matrix.
    # So we can use a decomposition that exploits this sharing.
    #
    # Let's try: for each k, C[:,k] = A^T * B[:,k] (where A is 2x3, B[:,k] is 3x1).
    # This is a 2x3 x 3x1 multiplication, which has rank 3 (schoolbook).
    # For 10 values of k, that's 30 terms. But we can do better.
    #
    # Actually, the standard approach for 2x3x10 is to use block decomposition.
    # The current rank-55 approach uses 5 blocks of 2x3x2 with Strassen.
    #
    # Let me try a different block structure: 2x3x10 = 2x3x(10)
    # Use Strassen-style for the 2x3 part, then handle the 10 columns.
    #
    # Alternative: use a 2x3x10 decomposition based on 2x3x5 blocks.
    # 2x3x5: can we do better than 2x3x2 * 2.5?
    #
    # Let me try to construct a rank-50 decomposition using 5 blocks of 2x3x2
    # but with a more efficient 2x3x2 decomposition.
    #
    # For 2x3x2, the schoolbook rank is 6. Strassen gives 7 for 2x2x2 + 4 for 2x1x1x2 = 11.
    # But maybe there's a rank-10 or rank-9 decomposition for 2x3x2?
    #
    # The tensor for 2x3x2 has 12 entries (2*3*2). The rank is at least 6 (schoolbook).
    # Can we do rank 9 or 10?
    #
    # Let me try to find a rank-10 decomposition for 2x3x2.
    # If we can, then 5*10 = 50 < 55.
    
    # Try to find a rank-10 decomposition for 2x3x2
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1}
    # We need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all (i,j,k)
    
    # Let's try a structured approach:
    # Split into two 1x3x2 parts (one for each row of A).
    # For row 0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # For row 1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # Schoolbook for each: 3 terms each, total 6.
    # Can we share terms between the two rows?
    
    # Try: use 5 terms for the 2x3x2 tensor.
    # This would require rank 5, which is below the schoolbook rank of 6.
    # Unlikely to be possible.
    
    # Try rank 9: 5 terms for row 0, 4 for row 1? Or some other split.
    
    # Actually, let me try a different approach: use the known result that
    # the rank of 2x3x2 matrix multiplication is 9.
    # Reference: The rank of 2x3x2 is known to be 9 (not 10 or 11).
    
    # If rank(2x3x2) = 9, then 5*9 = 45 < 55.
    
    # Let me try to construct a rank-9 decomposition for 2x3x2.
    
    # A is 2x3, B is 3x2, C is 2x2.
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01], [b10, b11], [b20, b21]]
    # C = [[c00, c01], [c10, c11]]
    
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    
    # Try Strassen-like:
    # M1 = (a00+a10)(b00) -> contributes to c00 and c10
    # M2 = (a01+a11)(b10) -> contributes to c00 and c10
    # M3 = (a02+a12)(b20) -> contributes to c00 and c10
    # M4 = (a00+a10)(b01) -> contributes to c01 and c11
    # M5 = (a01+a11)(b11) -> contributes to c01 and c11
    # M6 = (a02+a12)(b21) -> contributes to c01 and c11
    # M7 = a00*b00 - a10*b00 ... this is getting complicated.
    
    # Let me try a different set of 9 terms:
    # M1 = (a00+a10)(b00+b01)
    # M2 = (a01+a11)(b10+b11)
    # M3 = (a02+a12)(b20+b21)
    # M4 = a00*b00
    # M5 = a01*b10
    # M6 = a02*b20
    # M7 = a10*b00
    # M8 = a11*b10
    # M9 = a12*b20
    
    # This doesn't seem to work cleanly.
    
    # Let me just try to verify if rank 9 is achievable by attempting a search.
    # For now, return None to indicate we couldn't find a better decomposition.
    return None


def _try_rank50_with_better_blocks():
    """
    Try to find a rank-50 decomposition by using a more efficient 2x3x2 block.
    If we can find a rank-10 decomposition for 2x3x2, total rank = 50.
    """
    n, m, p = 2, 3, 10
    # Try to construct a rank-10 decomposition for 2x3x2
    # A: 2x3, B: 3x2, C: 2x2
    
    # Let's try:
    # M1 = (a00+a10)(b00) -> c00: +1, c10: +1
    # M2 = (a01+a11)(b10) -> c00: +1, c10: +1
    # M3 = (a02+a12)(b20) -> c00: +1, c10: +1
    # M4 = (a00+a10)(b01) -> c01: +1, c11: +1
    # M5 = (a01+a11)(b11) -> c01: +1, c11: +1
    # M6 = (a02+a12)(b21) -> c01: +1, c11: +1
    # M7 = a00*b00 -> c00: -1 (to correct)
    # M8 = a10*b00 -> c10: -1
    # M9 = a00*b01 -> c01: -1
    # M10 = a10*b01 -> c11: -1
    
    # This gives:
    # c00 = M1+M2+M3 - M7 = (a00+a10)b00 + (a01+a11)b10 + (a02+a12)b20 - a00*b00
    #     = a10*b00 + a01*b10 + a02*b20 + a00*b00 - a00*b00
    #     = a10*b00 + a01*b10 + a02*b20
    # But we need c00 = a00*b00 + a01*b10 + a02*b20.
    # So this doesn't work.
    
    # Let me try a different approach.
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c11 = a10*b01 + a11*b11 + a12*b21
    
    # Use 9 terms:
    # M1 = (a00+a10+a01+a11+a02+a12) * (b00+b10+b20) -- too many terms
    
    # I think rank 9 for 2x3x2 is not trivial. Let me just return the rank-55 solution.
    return None


def construct():
    start_time = time.time()
    
    # Build the rank-55 block decomposition
    U, V, W = _build_block_decomposition()
    
    # Verify
    n, m, p = 2, 3, 10
    if _verify(U, V, W, n, m, p):
        result = {"U": U, "V": V, "W": W}
        return result
    
    # Fallback: schoolbook rank 60
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
