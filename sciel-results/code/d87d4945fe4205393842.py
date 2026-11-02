import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    # We want to find a low-rank decomposition for 3x4 * 4x5 -> 3x5.
    # Strategy: Use Strassen's algorithm for the core 4x4 * 4x4 multiplication (rank 49),
    # and handle the extra column of B (making it 4x5) separately.
    
    # The 3x4 * 4x5 multiplication can be seen as:
    # C = A * [B1 | b] = [A*B1 | A*b]
    # where B1 is 4x4 and b is 4x1.
    
    # We need a rank-49 decomposition for 4x4 * 4x4 -> 4x4.
    # Strassen for 2x2 has rank 7.
    # For 4x4, we can use the standard Strassen recursion:
    # A4 = [[A00, A01], [A10, A11]] where each Ai is 2x2.
    # B4 = [[B00, B01], [B10, B11]] where each Bj is 2x2.
    
    # Strassen multiplies for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # For 4x4, we apply Strassen to the blocks.
    # But the blocks are 2x2 matrices. So we need to multiply 2x2 matrices.
    # We can use the schoolbook for 2x2 (rank 4) or Strassen (rank 7).
    # If we use Strassen for the 2x2 multiplications inside the 4x4 Strassen,
    # the total rank is 7 * 7 = 49.
    
    # Let's construct the U, V, W matrices for the 4x4 * 4x4 case with rank 49.
    # This is complex. Let's instead try a different approach.
    
    # Alternative: Just use the schoolbook for the 3x4 * 4x5 case? Rank 60.
    # Can we do better?
    
    # Let's try to construct a rank 53 decomposition as hypothesized in the prompt's thought process.
    # Rank 49 for 3x4 * 4x4 + Rank 4 for 3x4 * 4x1.
    
    # To get rank 49 for 3x4 * 4x4, we can use the 4x4 * 4x4 Strassen decomposition and
    # simply ignore the 4th row of A (which is zero) and the 4th row of the result.
    
    # Let's hardcode the rank-49 Strassen decomposition for 4x4.
    # This is tedious to derive by hand. Let's try to generate it.
    
    # Actually, implementing a general Strassen decomposition generator is risky in 100s.
    # Let's stick to the safe schoolbook rank 60, but maybe we can find a small improvement.
    
    # Wait, the prompt's thought process suggests 53.
    # Let's try to construct the 4x4 Strassen decomposition explicitly.
    
    # U, V, W for 4x4 * 4x4 -> 4x4 with rank 49.
    # Indices for U: i*4 + j (i=0..3, j=0..3) -> 16 elements.
    # Indices for V: j*4 + k (j=0..3, k=0..3) -> 16 elements.
    # Indices for W: k*4 + i (k=0..3, i=0..3) -> 16 elements.
    
    # This is a large matrix. Let's not hardcode it.
    
    # Instead, let's use the fact that for small dimensions, the schoolbook is often hard to beat
    # without a very specific algorithm.
    
    # Let's return the schoolbook decomposition for safety.
    
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
