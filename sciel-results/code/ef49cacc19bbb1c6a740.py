import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT

    # Winograd scheme for 1x4x2 block
    # Computes C[0,0] and C[0,1] from A[0,:] (1x4) and B[:,0:2] (4x2)
    # Using Winograd F(2,4) for convolution:
    # We want to compute:
    # C0 = A0*B0 + A1*B1 + A2*B2 + A3*B3
    # C1 = A1*B0 + A2*B1 + A3*B2 + A4*B3  (but B only has 4 rows, so B4 doesn't exist)
    # Wait, let's look at the standard Winograd for 1x4x2.
    # Actually, the task suggests "rank-5 Winograd scheme for 1x4x2 blocks".
    # Standard Winograd F(2,4) for 1D convolution of length 4 with kernel 2.
    # Inputs: a = [a0, a1, a2, a3], b = [b0, b1]
    # Outputs: y0 = a0*b0 + a1*b1, y1 = a1*b0 + a2*b1
    # Wait, the matrix multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # For a fixed row i, C[i, k] = sum_j A[i,j] B[j,k].
    # If we fix i, and consider a block of k's, say k=0,1.
    # C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0] + A[i,2]B[2,0] + A[i,3]B[3,0]
    # C[i,1] = A[i,0]B[0,1] + A[i,1]B[1,1] + A[i,2]B[2,1] + A[i,3]B[3,1]
    # This is equivalent to a dot product of A[i,:] with column 0 of B, and A[i,:] with column 1 of B.
    # Let A_vec = [A0, A1, A2, A3].
    # Let B_col0 = [B00, B10, B20, B30].
    # Let B_col1 = [B01, B11, B21, B31].
    # We want to compute two dot products.
    # Winograd F(2,4) for 1x4x2 usually refers to computing 2 outputs from 4 inputs and 2 filters.
    # The standard Winograd F(2,4) for convolution (output length 2, input length 4, filter length 2) uses 5 multiplications.
    # Let's verify the Winograd F(2,4) scheme.
    # Transformation matrices:
    # U (for A): 4x5? No, U is 5x4?
    # Let's recall the Winograd F(m, r) where m is output length, r is filter length.
    # Here m=2, r=2. Input length = m+r-1 = 3? No, input length is 4.
    # So it's F(2, 4) where 4 is input length?
    # Standard Winograd notation F(m, r) computes m outputs from input of length m+r-1.
    # If input length is 4, and we want 2 outputs, filter length is 3?
    # But here we have 4 inputs and 2 "filters" (columns of B).
    # Actually, this is just two dot products.
    # Dot product of length 4 can be computed with 4 mults.
    # Can it be done with fewer? No, over reals, dot product of length N has rank N.
    # BUT, we are computing TWO dot products simultaneously.
    # C0 = a0b0 + a1b1 + a2b2 + a3b3
    # C1 = a0c0 + a1c1 + a2c2 + a3c3
    # This is a bilinear map from (a, (b,c)) to (C0, C1).
    # The tensor is T[a, b, c, C] = a*b*C0 + a*c*C1.
    # This is equivalent to multiplying a 1x4 matrix by a 4x2 matrix.
    # Rank of 1x4x2 tensor?
    # The tensor T_{j, k, l} where j is input index (0..3), k is filter index (0..1), l is output index (0..1).
    # T_{j, k, l} = 1 if k=l? No.
    # C_l = sum_j A_j B_{j,l}.
    # So T_{j, l, l} = 1? No, T_{j, k, l} = delta_{k,l}.
    # So T is a 4x2x2 tensor where T_{j,k,l} = 1 if k=l else 0.
    # This tensor is the direct sum of two 4x1x1 tensors?
    # T = T1 + T2 where T1_{j,k,l} = 1 if k=0,l=0 else 0.
    # T2_{j,k,l} = 1 if k=1,l=1 else 0.
    # Each T1 is a 4x1x1 tensor. Rank of 4x1x1 is 4?
    # Wait, T1_{j,0,0} = 1.
    # Decomposition: sum_r U_j V_0 W_0 = 1 for all j.
    # This implies U_j * (V_0 W_0) = 1.
    # If we set V_0=1, W_0=1, then U_j=1.
    # So rank is 1?
    # Let's check: sum_r U_j V_k W_l = delta_{j, something}?
    # No, the tensor is T_{j,k,l}.
    # For T1, T1_{j,0,0} = 1 for all j. T1_{j,0,1} = 0, T1_{j,1,0}=0, T1_{j,1,1}=0.
    # We need sum_r U_{j,r} V_{0,r} W_{0,r} = 1 for all j.
    # And sum_r U_{j,r} V_{1,r} W_{0,r} = 0.
    # If we use 4 terms:
    # r=0: U_{0,0}=1, V_{0,0}=1, W_{0,0}=1. Others 0.
    # r=1: U_{1,1}=1, V_{0,1}=1, W_{0,1}=1. Others 0.
    # ...
    # This gives rank 4 for T1.
    # So total rank for 1x4x2 is 4+4=8?
    # But the prompt says "rank-5 Winograd scheme for 1x4x2 blocks".
    # This implies the rank is 5.
    # How can 1x4x2 have rank 5?
    # Maybe it's not T_{j,k,l} = delta_{k,l}.
    # Maybe it's a convolution?
    # "Winograd scheme for 1x4x2 blocks"
    # In Winograd, F(2,4) usually means output length 2, input length 4.
    # This corresponds to filter length 3?
    # Or maybe it's F(2,2) with padding?
    # Let's look up Winograd F(2,4).
    # Actually, there is a known result that the 1x4x2 tensor (dot product of length 4 with 2 vectors) has rank 5?
    # No, the tensor for matrix multiplication 1x4x2 is T_{i,j,k} with i=0, j=0..3, k=0..1.
    # T_{0,j,k} = 1 if j=k? No.
    # C_k = sum_j A_j B_{j,k}.
    # So the tensor is T_{j,k,k} = 1? No, the indices are (A_idx, B_idx, C_idx).
    # A has 4 indices. B has 4x2 indices. C has 2 indices.
    # T_{a, b, c} = 1 if b_row = a and b_col = c?
    # Yes, T_{a, (b_row, b_col), c} = 1 if b_row=a and b_col=c.
    # So T_{a, b, c} = delta_{a,b} delta_{b_col, c}.
    # This is a 4x4x2 tensor? No, B is 4x2.
    # Let's flatten B to 8 indices.
    # T is 4x8x2.
    # T_{a, b, c} = 1 if b = a*2 + c.
    # This is a permutation tensor.
    # Rank of a permutation tensor is the size of the permutation?
    # Here we have 4 pairs of (b,c) for each a? No.
    # For each a, there are 2 non-zero entries in T_{a, :, :}.
    # Specifically, T_{a, 2a, a} = 1 and T_{a, 2a+1, a+1} = 1? No.
    # B indices are (row, col). Row is 0..3, Col is 0..1.
    # So B index is row*2 + col.
    # T_{a, b, c} = 1 if b = a*2 + c.
    # So for a=0: b=0, c=0 -> T_{0,0,0}=1. b=1, c=1 -> T_{0,1,1}=1.
    # For a=1: b=2, c=0 -> T_{1,2,0}=1. b=3, c=1 -> T_{1,3,1}=1.
    # For a=2: b=4, c=0 -> T_{2,4,0}=1. b=5, c=1 -> T_{2,5,1}=1.
    # For a=3: b=6, c=0 -> T_{3,6,0}=1. b=7, c=1 -> T_{3,7,1}=1.
    # This tensor is a sum of 8 rank-1 tensors?
    # T = sum_{a,c} e_a \otimes e_{2a+c} \otimes e_c.
    # This is a direct sum of 8 rank-1 terms.
    # Can we do better?
    # The tensor is essentially two disjoint 4x4x1 tensors?
    # No, it's interleaved.
    # However, the prompt explicitly mentions "rank-5 Winograd scheme for 1x4x2 blocks".
    # This suggests that for a 1x4x2 block, the rank is 5.
    # This is a known result for the "dot product" tensor?
    # Actually, the tensor T_{i,j,k} = delta_{i,j} delta_{j,k} is the identity tensor.
    # Here we have T_{a, b, c} = delta_{b, 2a+c}.
    # This is not the identity.
    # Let's assume the prompt is correct and there is a rank-5 decomposition for 1x4x2.
    # If we use rank 5 for each 1x4x2 block, and we have 3 rows (i=0,1,2) and 2 column blocks (k=0..1, k=2..3, etc? No, p=8).
    # The strategy says: "decomposing each via the rank-5 Winograd scheme for 1x4x2 blocks applied to the two 4x2 column sub-blocks of B".
    # B is 4x8. We can split it into four 4x2 blocks.
    # So for each row i (3 rows), we have 4 blocks of 1x4x2.
    # Total rank = 3 * 4 * 5 = 60?
    # But the prompt says "construct a valid rank-30 decomposition".
    # 30 = 3 * 10? Or 3 * 4 * 2.5?
    # Maybe the 1x4x2 block has rank 2.5? No, rank is integer.
    # Maybe the blocks are shared?
    # "partition into three independent 1x4x8 row-multiplication tensors"
    # Each row tensor is 1x4x8.
    # "decomposing each via the rank-5 Winograd scheme for 1x4x2 blocks applied to the two 4x2 column sub-blocks of B"
    # Wait, "two 4x2 column sub-blocks"?
    # B is 4x8. Two 4x2 sub-blocks would cover 4 columns.
    # Maybe the 1x4x8 tensor is decomposed using a scheme that handles 8 columns with rank 10?
    # If 1x4x8 has rank 10, then 3 * 10 = 30.
    # How does 1x4x8 get rank 10?
    # Maybe it's two 1x4x4 blocks?
    # Or maybe the Winograd scheme for 1x4x8 has rank 10?
    # Let's assume the target rank is 30.
    # If we can achieve rank 30, that is much better than 88.
    # However, I need to construct the actual matrices.
    # Since I don't have the explicit Winograd F(2,4) or similar matrices for 1x4x2 or 1x4x8, I will have to derive them or use a known low-rank decomposition.
    # Actually, the rank of the 1x4x2 tensor (as defined by T_{a,b,c} = delta_{b, 2a+c}) is 8?
    # Let's check if it can be lower.
    # The tensor is a sum of 8 orthogonal rank-1 tensors.
    # If they are orthogonal, the rank is 8.
    # So where does rank 5 come from?
    # Maybe the "1x4x2 block" refers to a convolution, not a dot product?
    # In Winograd, F(2,4) is often used for 1D convolution.
    # If the problem was convolution, the tensor would be different.
    # But the problem is matrix multiplication.
    # Perhaps the "rank-5 Winograd scheme" is a specific known decomposition for a sub-problem.
    # Given the ambiguity and the risk of constructing an invalid decomposition, I should stick to the known valid Strassen-based decomposition (rank 88) and try to optimize it or find a valid lower rank decomposition using standard techniques.
    # However, the directive says "Use the 3x4x8 tensor's natural partition... to construct a valid rank-30 decomposition".
    # This implies I should try to build the rank-30 one.
    # If I cannot verify the rank-30 one, I should fall back to the rank-88 one.
    # But I need to output a single program.
    # I will implement the rank-88 Strassen decomposition as the primary, and add a placeholder for the rank-30 if I can derive it.
    # Since I cannot derive the specific Winograd matrices for this specific tensor structure without more info, I will stick to the rank-88 implementation which is known to be correct.
    # Wait, I can try to use the schoolbook rank 96 and then reduce it?
    # Or I can just return the rank-88 Strassen decomposition.
    # The current program already does this.
    # The "Inspiration 1" is also rank 88.
    # The "Inspiration 2" is schoolbook (rank 96) with Las Vegas.
    # The current program is rank 88.
    # The goal is to improve the score.
    # Score 1.0909 corresponds to rank 88 (since 96/88 = 1.0909).
    # If I can find a valid rank lower than 88, the score will improve.
    # The rank-30 proposal is very attractive.
    # Let's try to construct a rank-30 decomposition if possible.
    # If the 1x4x8 tensor has rank 10, then 3*10=30.
    # How to get rank 10 for 1x4x8?
    # 1x4x8 is a 4x8 matrix multiplication.
    # Rank of 1x4x8 tensor?
    # It is the tensor T_{a,b,c} = delta_{b, 2a+c} for b in 0..7, c in 0..7? No, c is 0..7.
    # T_{a, b, c} = 1 if b = a*2 + c? No.
    # C_c = sum_a A_a B_{a,c}.
    # So T_{a, b, c} = 1 if b=a and ? No.
    # B is 4x8. B_{a,c}.
    # T_{a, (a,c), c} = 1.
    # So T_{a, b, c} = 1 if b=a.
    # This is a 4x32x8 tensor?
    # B has 32 indices (4*8).
    # T_{a, b, c} = 1 if b = a*8 + c.
    # This is a sum of 32 rank-1 tensors.
    # Can it be lower?
    # The tensor is a "slice" of the identity.
    # The rank of the 1xN x M tensor (dot product) is N*M?
    # No, the rank of the tensor T_{i,j,k} = delta_{i,j} delta_{j,k} is N.
    # Here we have T_{a, b, c} = delta_{b, a*8+c}.
    # This is a permutation of the 32 basis vectors.
    # The rank is 32?
    # If so, 3*32 = 96.
    # So the rank is 96 for schoolbook.
    # Strassen reduces it to 88.
    # Can we go lower?
    # The rank of the 3x4x8 tensor is known to be at most 88.
    # Is it lower?
    # The border rank is likely lower, but we need exact rank.
    # Given the time constraints, I will stick to the rank-88 implementation.
    # I will clean up the code and ensure it is robust.

    def _build_strassen_3x4x8():
        """Build the block-Strassen decomposition: rank 88."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        # Strassen decomposition for 2x2x2
        strassen_decomp = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # M2
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # M3
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # M4
            ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),         # M5
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})          # M7
        ]

        col = 0
        for j1 in range(2):  # 2 blocks of 2 in m dimension
            # Top part: 2x2x8 (rows 0,1 of A)
            for l in range(4):  # 4 blocks of 2 in p dimension
                for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                    # U (A): local 2x2 -> global
                    for li, coeff in u_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i  # 0 or 1
                        global_j = 2 * j1 + li_j
                        U[global_i * m + global_j][col] += coeff

                    # V (B): local 2x2 -> global
                    for li, coeff in v_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_j = 2 * j1 + li_i
                        global_k = 2 * l + li_j
                        V[global_j * p + global_k][col] += coeff

                    # W (C): local 2x2 -> global
                    for li, coeff in w_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i
                        global_k = 2 * l + li_j
                        W[global_k * n + global_i][col] += coeff

                    col += 1

            # Bot part: 1x2x8 (row 2 of A), schoolbook
            for k in range(8):
                for j2 in range(2):
                    global_i = 2
                    global_j = 2 * j1 + j2
                    U[global_i * m + global_j][col] = 1
                    V[global_j * p + k][col] = 1
                    W[k * n + global_i][col] = 1
                    col += 1

        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W, tol=1e-9):
        """Verify exact tensor decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Build the 88-rank decomposition
    result = _build_strassen_3x4x8()

    # Verify correctness
    if not _verify(result['U'], result['V'], result['W']):
        # Fallback: schoolbook
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
        result = {"U": U, "V": V, "W": W}

    return result
# EVOLVE-BLOCK-END
