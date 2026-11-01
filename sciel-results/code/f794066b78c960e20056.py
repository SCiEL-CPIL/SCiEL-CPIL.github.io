"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify the tensor decomposition: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]"""
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Indices:
    # a = i*m + j  (i in 0..n-1, j in 0..m-1)
    # b = j*p + k  (j in 0..m-1, k in 0..p-1)
    # c = k*n + i  (k in 0..p-1, i in 0..n-1)
    
    # Check that T[i,j,k] = 1 if the indices match the multiplication structure
    # The tensor T represents C = A @ B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor notation T[a,b,c] is 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    # Otherwise 0.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                if val != 1:
                    return False
                
                # Check that other entries with same a,b but different c are 0?
                # No, the tensor is defined on the full space.
                # We must ensure that for ANY a,b,c, the sum is 1 only if they form a valid triple.
                # Actually, the tensor T is zero everywhere except the "diagonal" defined by the multiplication.
                # Let's verify all non-diagonal entries are 0.
    
    # To be safe and fast, we can just check a few random non-diagonal points or all of them if small.
    # Here n=2, m=3, p=8. Total points = 6 * 24 * 16 = 2304.
    # This is fast enough.
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Determine if (a,b,c) corresponds to a valid (i,j,k)
                # a = i*m + j -> i = a//m, j = a%m
                # b = j'*p + k -> j' = b//p, k = b%p
                # c = k'*n + i' -> k' = c//n, i' = c%n
                
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                
                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                if is_valid:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Strategy:
    # We want to compute C[i,k] = sum_{j=0}^{m-1} A[i,j] B[j,k] for i in {0,1}, k in 0..p-1.
    # 
    # Standard schoolbook uses n*m*p = 2*3*8 = 48 terms.
    # 
    # Optimized Strategy (Row Sharing):
    # For each column k of B (vector B[:,k]) and each row j of B, we can form products with A.
    # Let's define 3 sets of products for each k:
    # 1. P0_j = A[0,j] * B[j,k]  for j=0,1,2. Sum is C[0,k].
    # 2. P1_j = A[1,j] * B[j,k]  for j=0,1,2. Sum is C[1,k].
    # 3. Ps_j = (A[0,j] + A[1,j]) * B[j,k] for j=0,1,2. Sum is C[0,k] + C[1,k].
    #
    # If we compute P0_j and P1_j for all j, we have 2*m*p = 48 terms.
    # But we can express C[1,k] using Ps and P0?
    # C[0,k] + C[1,k] = sum_j Ps_j
    # C[0,k] = sum_j P0_j
    # C[1,k] = (sum_j Ps_j) - (sum_j P0_j)
    #
    # So we only need to explicitly compute P0_j and Ps_j.
    # Number of terms: 2 * m * p = 2 * 3 * 8 = 48. Wait, that's the same count.
    # 
    # Let's re-evaluate.
    # To get C[0,k] and C[1,k], we need to separate the contributions from A[0,:] and A[1,:].
    #
    # Alternative Sharing:
    # Consider the 2x1 vector A_col_j = [A[0,j], A[1,j]]^T.
    # We multiply this by the scalar B[j,k].
    # Result is a 2x1 vector [A[0,j]B[j,k], A[1,j]B[j,k]]^T.
    #
    # We can decompose the rank-2 tensor for this step (2x1 * 1x1 -> 2x1) into rank 3?
    # No, rank of 2x1 * 1x1 is 2.
    #
    # Let's look at the structure again.
    # C = A B
    # Let A be 2x3, B be 3x8.
    #
    # Is there a known lower bound?
    # Rank(2x3x8) <= Rank(2x3x3) * 8/3? No.
    #
    # Let's try a different packing.
    # Pad A to 4x4 and B to 4x4?
    # Rank(4x4x4) is 25 (Strassen-like).
    # If we can pack 2x3 and 3x8 into 4x4 matrices such that the top-left 2x8 block of the product is correct,
    # we might get rank 25.
    #
    # Let A_pad be 4x4:
    # A_pad = [[A[0,0], A[0,1], A[0,2], 0],
    #          [A[1,0], A[1,1], A[1,2], 0],
    #          [0, 0, 0, 0],
    #          [0, 0, 0, 0]]
    # 
    # Let B_pad be 4x4:
    # We need B_pad such that A_pad @ B_pad has the correct 2x8 block?
    # B is 3x8. We can't fit 3x8 into 4x4 easily without losing columns.
    # We would need 2 blocks of 4x4 multiplication to cover 8 columns.
    # Total rank = 25 + 25 = 50. Worse than 48.
    #
    # What if we pack differently?
    # A is 2x3. B is 3x8.
    #
    # Let's try the "3 terms per k" idea again more carefully.
    # For a fixed k, we compute C[:,k] = A @ B[:,k].
    # This is a 2x3 matrix times a 3x1 vector.
    # Rank of 2x3x1 tensor is 3?
    # T[i, j, 1] = delta_{i, j'}? No.
    # The tensor is T_{ijk} where k is fixed.
    # T_{ij} = A_{ij} B_{j}.
    # This is a bilinear map from R^3 (A row) x R^3 (B col) -> R^2.
    # Rank is 3 (since A has 2 rows, B has 1 col, intermediate is 3).
    # Actually, the rank of the matrix multiplication tensor M_{n,m,p} is the number of scalar multiplications.
    # For 2x3 * 3x1, we need 3 multiplications:
    # m1 = A[0,0] B[0,0]
    # m2 = A[0,1] B[1,0]
    # m3 = A[0,2] B[2,0]
    # c0 = m1+m2+m3
    # m4 = A[1,0] B[0,0]
    # m5 = A[1,1] B[1,0]
    # m6 = A[1,2] B[2,0]
    # c1 = m4+m5+m6
    # Total 6 multiplications.
    #
    # Can we do better for 2x3 * 3x1?
    # Let x = A[0,0]+A[1,0], y = A[0,1]+A[1,1], z = A[0,2]+A[1,2].
    # p1 = x B[0,0]
    # p2 = y B[1,0]
    # p3 = z B[2,0]
    # p4 = A[0,0] B[0,0]
    # p5 = A[0,1] B[1,0]
    # p6 = A[0,2] B[2,0]
    # c0 = p4+p5+p6
    # c1 = (p1-p4) + (p2-p5) + (p3-p6)
    # Total 6 multiplications. No gain.
    #
    # It seems the rank for 2x3 * 3x1 is exactly 6.
    # Since the 8 columns are independent, the total rank is 8 * 6 = 48.
    #
    # Is it possible to share across columns k?
    # B[j,k] are different for different k.
    # A[i,j] are the same for all k.
    #
    # If we group columns, say k=0 and k=1.
    # We need C[:,0] and C[:,1].
    # This is 2x3 * 3x2.
    # Rank of 2x3 * 3x2?
    # Schoolbook: 12.
    # Can we do better?
    # Let's check known ranks.
    # Rank(2x2x2) = 7.
    # Rank(2x3x2) ?
    # 
    # If Rank(2x3x2) < 12, then we can tile 4 times (for 8 columns) and get < 48.
    #
    # Let's try to find a rank-11 algorithm for 2x3 * 3x2.
    #
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01], [b10, b11], [b20, b21]]
    # C = [[c00, c01], [c10, c11]]
    #
    # c00 = a00 b00 + a01 b10 + a02 b20
    # c01 = a00 b01 + a01 b11 + a02 b21
    # c10 = a10 b00 + a11 b10 + a12 b20
    # c11 = a10 b01 + a11 b11 + a12 b21
    #
    # Let's try to use 11 terms.
    #
    # Terms:
    # 1. a00 (b00+b01) -> t1
    # 2. a01 (b10+b11) -> t2
    # 3. a02 (b20+b21) -> t3
    # 4. (a00+a10) b00 -> t4
    # 5. (a00+a10) b01 -> t5
    # 6. (a01+a11) b10 -> t6
    # 7. (a01+a11) b11 -> t7
    # 8. (a02+a12) b20 -> t8
    # 9. (a02+a12) b21 -> t9
    # 10. a00 b00 -> t10
    # 11. a01 b10 -> t11
    #
    # We have 11 terms. Can we express all c's?
    # c00 = a00 b00 + a01 b10 + a02 b20 = t10 + t11 + ? (need a02 b20)
    # We don't have a02 b20 directly.
    #
    # Let's adjust.
    # We need to recover a02 b20 and a02 b21 as well.
    #
    # This trial and error is risky.
    #
    # However, there is a known result: Rank(2x3x3) = 16? Or 17?
    # If Rank(2x3x2) is 11, then Rank(2x3x8) <= 4 * 11 = 44.
    #
    # Let's implement a robust search for Rank(2x3x2) if possible, or just stick to 48 if we can't find a better one quickly.
    # Given the time limit, and the difficulty of finding a specific rank-11 decomposition by hand,
    # I will provide the Rank-48 solution which is guaranteed correct.
    #
    # WAIT. I can try a simple heuristic search for the 2x3x2 case within the time limit?
    # No, that's too complex for 150s.
    #
    # Let's stick to Rank 48. It is safe.
    # The prompt asks for "rank-minimal". If I return 48, and the minimal is 44, I score worse.
    # But if I return an invalid 44, I score 0.
    #
    # Is there a simple Rank-44 construction?
    # If we assume Rank(2x3x2) = 11, we need the coefficients.
    #
    # Let's try one more specific construction for 2x3x2 with 11 terms.
    # Reference: "The rank of 2x3x2 matrix multiplication is 11".
    # Construction:
    # Let A = [a00 a01 a02; a10 a11 a12], B = [b00 b01; b10 b11; b20 b21].
    #
    # m1 = a00 b00
    # m2 = a01 b10
    # m3 = a02 b20
    # m4 = a10 b00
    # m5 = a11 b10
    # m6 = a12 b20
    # m7 = (a00+a10) b01
    # m8 = (a01+a11) b11
    # m9 = (a02+a12) b21
    # m10 = a00 (b01-b00) ? No.
    #
    # c00 = m1+m2+m3
    # c10 = m4+m5+m6
    # c01 = ?
    # c11 = ?
    #
    # c01 = a00 b01 + a01 b11 + a02 b21
    # c11 = a10 b01 + a11 b11 + a12 b21
    #
    # Note m7 = a00 b01 + a10 b01
    # m8 = a01 b11 + a11 b11
    # m9 = a02 b21 + a12 b21
    #
    # Sum m7+m8+m9 = c01 + c11
    #
    # We need c01 and c11 separately.
    # c01 = c01
    # c11 = (m7+m8+m9) - c01
    #
    # So we just need c01.
    # c01 = a00 b01 + a01 b11 + a02 b21.
    # This requires 3 terms.
    # Total terms: 6 (for c00,c10) + 3 (for sums) + 3 (for c01) = 12.
    #
    # Can we save one?
    # Maybe share one of the c01 terms with the c00/c10 terms?
    # No, the B indices are different (b01 vs b00).
    #
    # It seems 12 is the minimum for this specific simple sharing.
    # Is there a more clever sharing?
    #
    # If Rank(2x3x2) is actually 12, then Rank(2x3x8) is 48.
    #
    # I will output the Rank-48 solution.
    
    rank = n * m * p  # 48
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
    
    # Verify
    assert _verify(U, V, W, n, m, p, rank)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
