import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Indices helpers
    def ai(i, j): return i * m + j
    def bi(j, k): return j * p + k
    def ci(k, i): return k * n + i

    # We decompose the 3x3x7 multiplication into:
    # 1. 3x3x3 multiplication (cols 0-2 of B)
    # 2. 3x3x3 multiplication (cols 3-5 of B)
    # 3. 3x3x1 multiplication (col 6 of B)
    
    # The tensor T is the direct sum of these three tensors.
    # Rank(T) <= Rank(T1) + Rank(T2) + Rank(T3).
    # Schoolbook: 27 + 27 + 9 = 63.
    
    # We know Rank(3x3x3) = 23.
    # We know Rank(3x3x1) = 9 (since it's just 3 scalar multiplications for each of the 3 rows? No.
    # 3x3x1: A is 3x3, B is 3x1, C is 3x1.
    # C_i = sum_j A_ij B_j.
    # This is 3 dot products of length 3.
    # Can we do better than 9?
    # The tensor is sum_{i,j} A_ij B_j e_i.
    # This is a sum of 9 rank-1 terms in the (A, B, C) space?
    # Actually, for fixed j, the term is (A_{:,j} B_j) * e_{:}.
    # The rank of the 3x3x1 tensor is 9. Why?
    # Consider the matrix unfolding.
    # The tensor is essentially 3 independent 3x1 multiplications? No, the B vector is shared.
    # C = A B.
    # This is a linear map from (A, B) to C.
    # The rank of the multiplication tensor for m x n by n x 1 is mn.
    # Proof: The tensor is sum_{i,j} A_ij B_j e_i.
    # Each term A_ij B_j e_i is rank 1. There are 9 terms.
    # Are they independent? Yes, because the supports in the A-mode and B-mode are distinct for each (i,j) pair?
    # No, B_j is shared.
    # However, the rank of the tensor for matrix multiplication M_{m,n,1} is mn.
    # Reference: Landsberg, "Tensors: Geometry and Applications".
    # So Rank(3x3x1) = 9.
    
    # So the trivial bound is 23 + 23 + 9 = 55.
    # Is it possible to do better by sharing structure between the 3x3x3 blocks?
    # The two 3x3x3 blocks are independent (different columns of B, different columns of C).
    # So Rank(T) = Rank(T1) + Rank(T2) + Rank(T3) = 23 + 23 + 9 = 55.
    # This is a valid decomposition.
    
    # We need to construct the 23-term decomposition for 3x3x3.
    # Let's use the known algorithm.
    
    def get_23_mult():
        # Returns list of (u, v, w) for 3x3x3 multiplication
        # u, v, w are vectors of length 9
        # u[i*3+j], v[j*3+k], w[k*3+i]
        
        terms = []
        
        # We use the algorithm from:
        # "A 23-multiplication algorithm for 3x3 matrix multiplication"
        # This algorithm is often attributed to Winograd or derived from Strassen.
        # A specific set of 23 terms is:
        
        # Let's use the one from the paper "On the rank of the 3x3 matrix multiplication tensor"
        # or similar. 
        # Actually, let's use a verified set.
        
        # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
        u = [0]*9; v = [0]*9; w = [0]*9
        u[0] = 1; u[4] = 1  # a00, a11
        v[0] = 1; v[4] = 1  # b00, b11
        w[0] = 1; w[4] = 1  # c00, c11
        terms.append((u, v, w))
        
        # M2 = (a10+a11)b00 -> c10 += M2
        u = [0]*9; v = [0]*9; w = [0]*9
        u[3] = 1; u[4] = 1  # a10, a11
        v[0] = 1            # b00
        w[3] = 1            # c10
        terms.append((u, v, w))
        
        # M3 = a00(b01-b11) -> c01 += M3
        u = [0]*9; v = [0]*9; w = [0]*9
        u[0] = 1            # a00
        v[1] = 1; v[4] = -1 # b01, -b11
        w[1] = 1            # c01
        terms.append((u, v, w))
        
        # M4 = a11(b10-b00) -> c11 += M4
        u = [0]*9; v = [0]*9; w = [0]*9
        u[4] = 1            # a11
        v[3] = 1; v[0] = -1 # b10, -b00
        w[4] = 1            # c11
        terms.append((u, v, w))
        
        # M5 = (a00+a01)b11 -> c00 += M5
        u = [0]*9; v = [0]*9; w = [0]*9
        u[0] = 1; u[1] = 1  # a00, a01
        v[4] = 1            # b11
        w[0] = 1            # c00
        terms.append((u, v, w))
        
        # M6 = (a10-a00)(b00+b01) -> c10 += M6, c11 += M6
        u = [0]*9; v = [0]*9; w = [0]*9
        u[3] = 1; u[0] = -1 # a10, -a00
        v[0] = 1; v[1] = 1  # b00, b01
        w[3] = 1; w[4] = 1  # c10, c11
        terms.append((u, v, w))
        
        # M7 = (a01-a11)(b10+b11) -> c01 += M7
        u = [0]*9; v = [0]*9; w = [0]*9
        u[1] = 1; u[4] = -1 # a01, -a11
        v[3] = 1; v[4] = 1  # b10, b11
        w[1] = 1            # c01
        terms.append((u, v, w))
        
        # Now we have handled the 2x2 block.
        # We need to handle the interactions with the 3rd row/col.
        # The remaining terms involve indices 2.
        
        # The known 23-mult algorithm completes with 16 more terms?
        # No, 23 - 7 = 16.
        # But we need to be careful. The first 7 terms above are not a standard complete set for the 2x2 part in a way that easily extends.
        
        # Let's use a different, more standard 23-mult algorithm if possible.
        # One common source is the algorithm by S. Winograd (1978).
        
        # Alternative: Use the schoolbook 27 terms for 3x3x3.
        # Then total rank = 27 + 27 + 9 = 63.
        # This is safe but not an improvement.
        
        # Let's try to verify if we can use the 23 terms.
        # If I can't find a verified 23-term set quickly, I will fall back to 27.
        
        # Actually, let's try to construct the 23 terms from a known list.
        # Reference: "A 23-Multiplication Algorithm for 3x3 Matrix Multiplication"
        
        # Here is a set of 23 terms (indices 0-8 for 3x3 matrix, row-major):
        # a00=0, a01=1, a02=2
        # a10=3, a11=4, a12=5
        # a20=6, a21=7, a22=8
        
        # b00=0, b01=1, b02=2
        # b10=3, b11=4, b12=5
        # b20=6, b21=7, b22=8
        
        # c00=0, c01=1, c02=2
        # c10=3, c11=4, c12=5
        # c20=6, c21=7, c22=8
        
        # Note: The problem statement says W[k*n+i, r].
        # For 3x3, n=3. W index = k*3+i.
        # c00 -> k=0, i=0 -> 0
        # c01 -> k=1, i=0 -> 3
        # c02 -> k=2, i=0 -> 6
        # c10 -> k=0, i=1 -> 1
        # c11 -> k=1, i=1 -> 4
        # c12 -> k=2, i=1 -> 7
        # c20 -> k=0, i=2 -> 2
        # c21 -> k=1, i=2 -> 5
        # c22 -> k=2, i=2 -> 8
        
        # So the mapping is:
        # c[i,k] in standard math notation (row i, col k)
        # maps to W index k*3 + i.
        
        # Let's re-map my previous terms to this W index convention.
        # My previous terms used w[ci(k,i)] where ci(k,i) = k*3+i.
        # So w[0] is c00 (k=0, i=0).
        # w[1] is c10 (k=1, i=0)? No.
        # ci(k,i) = k*3+i.
        # If i=0, k=1 -> 3. So c01 (row 0, col 1) is index 3.
        # Wait, standard matrix C is n x p. C[i,k].
        # The problem says W[k*n+i].
        # So for 3x3x3, W index for C[i,k] is k*3+i.
        
        # Let's check my previous code:
        # def ci(k, i): return k * 3 + i
        # w[ci(0,0)] = 1 -> w[0] = 1. This is C[0,0].
        # w[ci(0,1)] = 1 -> w[3] = 1. This is C[1,0] in (k,i) notation?
        # No, ci(k,i) takes k as first arg, i as second.
        # So ci(0,1) means k=0, i=1. Index 1.
        # This corresponds to C[i=1, k=0] = C[1,0].
        
        # So the indices in the W vector are:
        # 0: C[0,0]
        # 1: C[1,0]
        # 2: C[2,0]
        # 3: C[0,1]
        # 4: C[1,1]
        # 5: C[2,1]
        # 6: C[0,2]
        # 7: C[1,2]
        # 8: C[2,2]
        
        # This is column-major storage of C.
        
        # Let's re-verify the 7 Strassen terms with this mapping.
        # M1 = (a00+a11)(b00+b11) -> C[0,0] += M1, C[1,1] += M1
        # C[0,0] is index 0. C[1,1] is index 4.
        # So w[0]=1, w[4]=1.
        
        # M2 = (a10+a11)b00 -> C[1,0] += M2
        # C[1,0] is index 1.
        # So w[1]=1.
        
        # M3 = a00(b01-b11) -> C[0,1] += M3
        # C[0,1] is index 3.
        # So w[3]=1.
        
        # M4 = a11(b10-b00) -> C[1,1] += M4
        # C[1,1] is index 4.
        # So w[4]=1.
        
        # M5 = (a00+a01)b11 -> C[0,0] += M5
        # C[0,0] is index 0.
        # So w[0]=1.
        
        # M6 = (a10-a00)(b00+b01) -> C[1,0] += M6, C[1,1] += M6
        # C[1,0] is index 1. C[1,1] is index 4.
        # So w[1]=1, w[4]=1.
        
        # M7 = (a01-a11)(b10+b11) -> C[0,1] += M7
        # C[0,1] is index 3.
        # So w[3]=1.
        
        # Let's check the sum for C[0,0] (index 0):
        # M1 + M5
        # M1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
        # M5 = (a00+a01)b11 = a00b11 + a01b11
        # Sum = a00b00 + 2a00b11 + a11b00 + a11b11 + a01b11
        # Expected C[0,0] = a00b00 + a01b10 + a02b20.
        # This does NOT match.
        
        # The 7 Strassen terms are for 2x2 multiplication.
        # They compute C[0,0], C[0,1], C[1,0], C[1,1] correctly IF we only consider the 2x2 submatrices.
        # But we are in 3x3. The terms involving a02, a12, a20, a21, a22, b02, b12, b20, b21, b22 are not touched.
        # And the 7 terms above might not sum correctly to the 3x3 result because of cross terms?
        # No, the 7 terms only involve a00, a01, a10, a11 and b00, b01, b10, b11.
        # So they only contribute to the 2x2 block of C.
        # For the 2x2 block, do they work?
        # C[0,0] (2x2) = a00b00 + a01b10.
        # My sum above: a00b00 + 2a00b11 + a11b00 + a11b11 + a01b11.
        # This is clearly wrong.
        
        # I made a mistake in the Strassen terms.
        # Standard Strassen for 2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Let's check C00:
        # M1 + M4 - M5 - M7
        # = (a00+a11)(b00+b11) + a11(b10-b00) - (a00+a01)b11 - (a01-a11)(b10+b11)
        # = a00b00 + a00b11 + a11b00 + a11b11 + a11b10 - a11b00 - a00b11 - a01b11 - a01b10 - a01b11 + a11b10 + a11b11
        # = a00b00 + a11b11 + a11b10 - a01b11 - a01b10 + a11b10 + a11b11
        # = a00b00 + 2a11b11 + 2a11b10 - a01b11 - a01b10
        # This is also wrong.
        
        # I am misremembering the Strassen formulas or the term definitions.
        # Let's look up the correct Strassen formulas.
        
        # Correct Strassen:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Let's check C11 (which is C[0,0] in 0-indexed):
        # M1 + M4 - M5 - M7
        # M1 = (a00+a11)(b00+b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M7 = (a01-a11)(b10+b11)
        
        # Sum:
        # (a00+a11)(b00+b11) + a11(b10-b00) - (a00+a01)b11 - (a01-a11)(b10+b11)
        # = a00b00 + a00b11 + a11b00 + a11b11 + a11b10 - a11b00 - a00b11 - a01b11 - a01b10 - a01b11 + a11b10 + a11b11
        # = a00b00 + a11b11 + a11b10 - a01b11 - a01b10 + a11b10 + a11b11
        # = a00b00 + 2a11b11 + 2a11b10 - a01b11 - a01b10
        
        # This is still not a00b00 + a01b10.
        
        # There is a sign error in my expansion or the formula.
        # Let's use a verified implementation from code.
        # Since I can't browse, I will rely on the fact that the rank of 3x3x3 is 23.
        # If I cannot construct it correctly, I will use rank 27 (schoolbook).
        
        # Given the complexity and risk of error, I will use the schoolbook decomposition for the 3x3x3 blocks.
        # Rank = 27 + 27 + 9 = 63.
        
        # Wait, the inspiration 3 mentioned trying to find a lower rank.
        # The current program uses schoolbook (63).
        # Can I do better?
        # If I use rank 23 for 3x3x3, I need a correct algorithm.
        
        # Let's try to find a correct 23-mult algorithm in my memory.
        # One source is the "Winograd" algorithm.
        
        # Actually, let's just use the schoolbook for the 3x3x1 part (rank 9) and see if I can optimize the 3x3x3 parts.
        
        # If I can't get 23, I'll stick to 63.
        
        return None

    # Since I'm unsure about the 23-mult algorithm, I will use the schoolbook for the whole thing.
    # Rank 63.
    
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
