import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS
TOTAL_TIME_LIMIT = 120.0

def construct():
    start_time = time.time()
    
    # 1. Try to find a better decomposition using known fast matrix multiplication algorithms
    # We will try to use Strassen's algorithm for 2x2x2 blocks where possible.
    # 4x5 * 5x6 can be decomposed into blocks.
    # A is 4x5, B is 5x6.
    # Let's try to decompose this into 2x2 blocks of A (size 2x2) and 2x2 blocks of B (size 2x2) or similar.
    
    # Actually, let's look at the structure.
    # We can split A into [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x3, etc.
    # Or we can use a different block structure.
    
    # Let's try to construct a decomposition of rank < 120.
    # The trivial rank is 4*5*6 = 120.
    # We know that for 2x2x2, Strassen has rank 7 (vs 8).
    # For 4x4x4, the known rank is 49 (vs 64).
    
    # Let's try to use a block decomposition.
    # Split A (4x5) into two 2x5 matrices: A_top, A_bottom.
    # Split B (5x6) into two 5x3 matrices: B_left, B_right.
    # This doesn't immediately help with Strassen.
    
    # Better approach: Split the problem into smaller subproblems.
    # A (4x5) * B (5x6)
    # We can view this as (2x2 + 2x2 + 2x1) * (2x3 + 2x3 + 1x3)? No.
    
    # Let's try a 2x2 block structure for the dimensions that are even.
    # n=4 (even), m=5 (odd), p=6 (even).
    # This is tricky because m is odd.
    
    # Alternative: Use the known rank 49 for 4x4x4.
    # We can embed 4x5 * 5x6 into 4x6 * 6x6? No, that's larger.
    # We can view 4x5 * 5x6 as a sub-problem of 4x6 * 6x6?
    
    # Let's try to use the 4x4x4 rank 49 algorithm.
    # We need to handle the extra dimensions.
    # A is 4x5, B is 5x6.
    # Let A = [A1 (4x4), A2 (4x1)]
    # Let B = [B1 (4x6); B2 (1x6)]
    # Then AB = A1*B1 + A2*B2.
    # A1*B1 is 4x4 * 4x6.
    # A2*B2 is 4x1 * 1x6.
    
    # A1*B1: 4x4 * 4x6.
    # We can view this as 4x4 * (4x2 + 4x2 + 4x2).
    # So we need 3 copies of 4x4 * 4x2? No, that's not right.
    # Actually, 4x4 * 4x6 is just 6 columns of 4x4 * 4x1.
    # Rank of 4x4 * 4x4 is 49.
    # Rank of 4x4 * 4x1 is 16 (trivial) or maybe less?
    # Actually, multiplying by a vector is just a linear combination of columns.
    
    # Let's try a different split.
    # A (4x5) = A1 (4x2) + A2 (4x3) ? No, block multiplication requires matching inner dimension.
    # A (4x5) * B (5x6) = [A1 A2] * [B1; B2] = A1*B1 + A2*B2.
    # Let A1 be 4x2, A2 be 4x3.
    # Let B1 be 2x6, B2 be 3x6.
    # Term 1: A1 (4x2) * B1 (2x6). Rank of 4x2 * 2x6.
    # Term 2: A2 (4x3) * B2 (3x6). Rank of 4x3 * 3x6.
    
    # Rank of 4x2 * 2x6:
    # This is effectively 2 multiplications of 4x1 * 1x6? No.
    # It is sum of 2 terms: (col1 of A1) * (row1 of B1) + (col2 of A1) * (row2 of B1).
    # Each term is outer product of 4x1 and 1x6. Rank 1.
    # So total rank is 2 * (rank of 4x1 * 1x6)? No, the rank of the tensor for 4x1 * 1x6 is 4?
    # Wait. The tensor for C[i,k] = sum_j A[i,j] B[j,k].
    # For 4x2 * 2x6: C[i,k] = A[i,1]B[1,k] + A[i,2]B[2,k].
    # This is a sum of 2 rank-1 tensors? No.
    # A[i,1]B[1,k] is rank 1 in the (i,k) space? No, it depends on A and B.
    # The multiplication tensor T for 4x2 * 2x6 has rank 2 * 4 * 6 / ? No.
    # The rank is at most 4*2*6 = 48? No, that's the trivial rank.
    # Actually, for m=2, the rank is 2 * max(n,p)? No.
    # If m=1, rank = n*p.
    # If m=2, rank = 2 * n * p? No.
    # C[i,k] = A[i,1]B[1,k] + A[i,2]B[2,k].
    # This is just a bilinear form. The rank is n*p if the coefficients are independent?
    # Actually, for fixed i,k, it's linear in A and B.
    # The tensor is sum_{i,k} C_{ik} e_i \otimes e_k.
    # C_{ik} = A_{i1}B_{1k} + A_{i2}B_{2k}.
    # This can be written as sum_{j=1}^2 (sum_{i} A_{ij} e_i) \otimes (sum_{k} B_{jk} e_k).
    # So it is a sum of 2 rank-1 tensors.
    # Wait, is that right?
    # Rank 1 tensor is u \otimes v \otimes w.
    # Here the term is (A[:,j]) \otimes (B[j,:])?
    # In the context of matrix multiplication tensor T = sum C_{ik} E_{ik} \otimes A \otimes B.
    # The term for j is: (sum_i A_{ij} E_{ik})? No.
    # Let's check the rank of 4x1 * 1x6.
    # C[i,k] = A[i,1] B[1,k].
    # This is rank 1 in the tensor sense?
    # T = sum_{i,k} A_{i1} B_{1k} E_{ik}.
    # This is (sum_i A_{i1} E_{ik})? No, k is fixed?
    # No, T = (sum_i A_{i1} v_i) \otimes (sum_k B_{1k} u_k) \otimes W?
    # Actually, the rank of n x 1 * 1 x p is n*p.
    # Because C[i,k] depends on A[i,1] and B[1,k] independently for each (i,k).
    # So we need n*p multiplications.
    # So 4x1 * 1x6 has rank 24.
    # So 4x2 * 2x6 has rank 2 * 24 = 48?
    # Wait, if we use Strassen-like tricks?
    # For 2x1 * 1x2, rank is 4?
    # C[i,k] = A[i,1] B[1,k].
    # For 2x1 * 1x2: C[0,0]=A00B00, C[0,1]=A00B01, C[1,0]=A10B00, C[1,1]=A10B01.
    # These are 4 independent multiplications.
    # So yes, rank is n*p.
    
    # So splitting into A1 (4x2) * B1 (2x6) gives rank 2 * 24 = 48.
    # Splitting into A2 (4x3) * B2 (3x6) gives rank 3 * 24 = 72.
    # Total rank 48 + 72 = 120. Same as trivial.
    
    # We need to use fast matrix multiplication for larger blocks.
    # Let's try to use 4x4 * 4x4 rank 49.
    # Can we express 4x5 * 5x6 in terms of 4x4 * 4x4?
    # A = [A1 (4x4), A2 (4x1)]
    # B = [B1 (4x6); B2 (1x6)]
    # AB = A1*B1 + A2*B2.
    # A2*B2 is 4x1 * 1x6. Rank 24.
    # A1*B1 is 4x4 * 4x6.
    # B1 = [B11 (4x2), B12 (4x2), B13 (4x2)].
    # A1*B1 = A1*B11 + A1*B12 + A1*B13.
    # Each A1*B1j is 4x4 * 4x2.
    # Rank of 4x4 * 4x2?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This is effectively computing 2 columns of the product.
    # If we knew the rank of 4x4 * 4x2, we could sum them.
    # But the multiplications for different columns are independent?
    # No, they share the A factors.
    # Actually, the tensor for 4x4 * 4x2 is just the first 2 columns of the 4x4 * 4x4 tensor?
    # No, the output size is 4x2.
    # The rank of n x n * n x 2 is at most 2 * rank(n x n * n x n)?
    # Or is it rank(n x n * n x n) / 2?
    # Actually, we can compute 4x4 * 4x2 by embedding it into 4x4 * 4x4.
    # Pad B1j with 2 zero columns to make it 4x4.
    # Then A1 * [B1j 0] = [A1*B1j 0].
    # So the rank of 4x4 * 4x2 is at most the rank of 4x4 * 4x4, which is 49.
    # So A1*B1 can be done in 3 * 49 = 147 operations?
    # But we can do better.
    # We can compute A1 * B1 (4x4 * 4x6) by padding B1 to 4x8?
    # Or just use the rank 49 algorithm for 4x4 * 4x4 and apply it to the first 4 columns of B1?
    # B1 is 4x6. We can split it into 4x4 and 4x2.
    # A1*B1 = A1*B1_left (4x4) + A1*B1_right (4x2).
    # A1*B1_left uses rank 49.
    # A1*B1_right uses rank 49 (by padding).
    # Total for A1*B1 is 49 + 49 = 98.
    # Plus A2*B2 (rank 24).
    # Total rank 98 + 24 = 122. Worse than 120.
    
    # Let's try a different split.
    # A (4x5) = A1 (4x4) + A2 (4x1)
    # B (5x6) = B1 (4x6) + B2 (1x6)
    # This is what I just did.
    
    # What if we split into 2x2 blocks?
    # n=4, m=5, p=6.
    # Split n into 2+2, m into 2+2+1, p into 3+3.
    # This is getting complicated.
    
    # Let's try to use the 2x2x2 Strassen algorithm.
    # Rank 7 vs 8.
    # Can we tile the 4x5 * 5x6 with 2x2x2 blocks?
    # 4x5 is not divisible by 2 in the middle.
    
    # Let's stick to the trivial rank 120 for now, but try to find a small improvement.
    # Maybe we can optimize the 4x1 * 1x6 part?
    # No, that's just dot products.
    
    # Let's try to use a 3x3x3 algorithm? Rank 23?
    # 4x5 * 5x6.
    # Embed into 6x6 * 6x6? Rank 49?
    # 6x6 * 6x6 rank is 49? No, 49 is for 4x4.
    # Rank of 6x6 * 6x6 is 21? No, 3^3=27 is trivial.
    # Strassen for 2x2x2 is 7.
    # For 3x3x3, best known is 23.
    
    # Let's try to construct a decomposition using the 2x2x2 Strassen algorithm where possible.
    # We can split the 4x4 part into four 2x2 blocks.
    # The 4x4 * 4x4 multiplication can be done with rank 49.
    # We can construct the 4x4 * 4x4 decomposition explicitly.
    
    # Rank 49 algorithm for 4x4x4:
    # This is known as the "fast 4x4x4" algorithm.
    # It uses 49 multiplications.
    # I can hardcode the 49 terms.
    
    # However, implementing a full 4x4x4 rank 49 algorithm is complex.
    # Let's check if we can get a rank < 120 easily.
    # 4x5 * 5x6.
    # If we use 4x4 * 4x4 (rank 49) for the main part...
    # A = [A1 A2], B = [B1; B2].
    # A1*B1 (4x4 * 4x4) -> Rank 49.
    # A1*B2 (4x4 * 1x6) -> Rank 4*6 = 24.
    # A2*B1 (4x1 * 4x6) -> Rank 4*6 = 24.
    # A2*B2 (4x1 * 1x6) -> Rank 4*6 = 24.
    # Total: 49 + 24 + 24 + 24 = 121. Worse.
    
    # Wait, A1*B2 is 4x4 * 1x6? No, B is 5x6.
    # B = [B_top (4x6); B_bottom (1x6)].
    # A = [A_left (4x4), A_right (4x1)].
    # AB = A_left * B_top + A_right * B_bottom.
    # A_left * B_top is 4x4 * 4x6.
    # A_right * B_bottom is 4x1 * 1x6.
    
    # We established A_left * B_top <= 98.
    # A_right * B_bottom = 24.
    # Total 122.
    
    # What if we split B differently?
    # B (5x6) = B1 (3x6) + B2 (2x6)? No, dimensions don't match.
    # B is 5 rows.
    # Split into 3+2.
    # A (4x5) = A1 (4x3) + A2 (4x2).
    # AB = A1*B1 + A2*B2.
    # A1*B1: 4x3 * 3x6.
    # A2*B2: 4x2 * 2x6.
    
    # Rank of 4x3 * 3x6?
    # Trivial: 4*3*6 = 72.
    # Can we do better?
    # 3x3 * 3x3 is rank 23.
    # 4x3 * 3x6 can be viewed as 4x3 * (3x2 + 3x2 + 3x2).
    # So 3 * (4x3 * 3x2).
    # Rank of 4x3 * 3x2?
    # Pad to 4x4 * 4x4?
    # 4x3 * 3x2 -> Pad A to 4x4, B to 4x4.
    # Use rank 49.
    # So 4x3 * 3x6 <= 3 * 49 = 147. Worse.
    
    # It seems hard to beat 120 without a very specific decomposition.
    # Let's just return the trivial rank 120 decomposition. It is safe and correct.
    
    rank_total = n * m * p
    U = [[0.0] * rank_total for _ in range(n * m)]
    V = [[0.0] * rank_total for _ in range(m * p)]
    W = [[0.0] * rank_total for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time taken: {end - start}")
    # Basic verification
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    errors += 1
    if errors == 0:
        print("Verification PASSED")
    else:
        print(f"Verification FAILED with {errors} errors")
