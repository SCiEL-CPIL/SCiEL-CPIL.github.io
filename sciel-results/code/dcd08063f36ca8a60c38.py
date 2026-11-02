"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    num_blocks = p // m  # 3
    rank_per_block = 6
    total_rank = num_blocks * rank_per_block  # 18

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for block_idx in range(num_blocks):
        base = block_idx * rank_per_block
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    col = base + i * m + j + k * 0
                    # Use a canonical ordering: i, j, k
                    col = base + (i * m + j) + k * 0
                    # Let's use a flat index: i*m + j + k
                    # But i in [0,1], j in [0,2], k in [0,2]
                    # We need 6 distinct cols per block
                    # i*m + j gives 0..5? No, i*m+j: 0,1,2,3,4,5 for i=0,1 j=0,1,2
                    # Wait, k is the 3rd index. We need i, j, k to map to 6 values.
                    # But i=2, j=3, k=3. 2*3*3 = 18? No, rank per block is 6.
                    # The 2x3x3 tensor rank is 6.
                    # The indices are i in 0..1, j in 0..2, k in 0..2.
                    # Standard rank 6 decomposition:
                    # The tensor T_{i,j,k} = delta_{j,k} for 2x3x3? No.
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # Tensor T[i,j,k] = 1 if j==k? No, that's not right.
                    # T[i,j,k] corresponds to A[i,j]*B[j,k] -> C[i,k]
                    # The 2x3x3 matrix multiplication tensor has rank 6.
                    # A known rank-6 decomposition for 2x3x3:
                    # We can split into two 2x3x1 and 2x1x3? No.
                    # Let's stick to the schoolbook 6 terms for 2x3x3:
                    # Term (i,j,k) is 1 if it matches the pattern.
                    # Actually, the schoolbook 2x3x3 uses 6 terms:
                    # for i in 0..1, j in 0..2, k in 0..2:
                    #   if j == k? No.
                    # The 6 terms are:
                    # 1. A00*B00 -> C00
                    # 2. A01*B10 -> C01
                    # 3. A02*B20 -> C02
                    # 4. A10*B01 -> C10
                    # 5. A11*B11 -> C11
                    # 6. A12*B21 -> C12
                    # This only covers 6 of the 9 output entries.
                    # Wait, the 2x3x3 tensor rank is indeed 6?
                    # No, the rank of 2x3x3 matmul tensor is 6? 
                    # Actually, the rank of 2x3x3 is 6.
                    # Let's verify: 2x3x3 matmul requires 6 multiplications? 
                    # No, 2x3x3 matmul: C = A(2x3) * B(3x3).
                    # C has 6 entries. Each entry is a dot product of length 3.
                    # So 6*3 = 18 schoolbook multiplications.
                    # But the tensor rank is 6? That seems low.
                    # Actually, the rank of the 2x3x3 tensor is 6.
                    # Let's use the standard rank-6 decomposition.
                    
                    pass

    # Let's reconstruct the rank-6 block decomposition properly.
    # For 2x3x3, the tensor T[i,j,k] = 1 if the triple (i,j,k) is part of the matmul.
    # The matmul tensor for 2x3x3 has rank 6.
    # A rank-6 decomposition:
    # r=0: i=0,j=0,k=0 -> C[0,0] += A[0,0]*B[0,0]
    # r=1: i=0,j=1,k=0 -> C[0,0] += A[0,1]*B[1,0] ... wait, this is not rank 1.
    
    # The standard rank-6 for 2x3x3 is:
    # The 6 terms correspond to the 6 independent multiplications.
    # Actually, the rank of 2x3x3 is 6.
    # The decomposition is:
    # U[i,j] = 1, V[j,k] = 1, W[i,k] = 1 for specific (i,j,k).
    # The 6 non-zero triples in the rank-6 decomposition are:
    # (0,0,0), (0,1,1), (0,2,2), (1,0,0), (1,1,1), (1,2,2)
    # Let's check: C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # This requires 3 terms. So rank is at least 3 for each output, total 6?
    # No, the rank is 6.
    
    # Let's just use the schoolbook 18 for now to be safe, but the task says rank 18.
    # The current program uses rank 18.
    
    col = 0
    for t in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    U[i * m + j][col] = 1
                    V[j * p + (t * m + k)][col] = 1
                    W[(t * m + k) * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
