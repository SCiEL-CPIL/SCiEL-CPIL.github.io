DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a rank-60 decomposition: for each of the 3 rows, use 20 terms
    # by applying a Strassen-like optimization to a 2x2 sub-block of the 5x5 multiplication.
    # For each row i, we compute 5 dot products of length 5.
    # We optimize 5 of the 25 terms using Strassen on indices (0,1) x (0,1).
    
    # For a given row i, the 5 output entries are C[i,k] = sum_j A[i,j]*B[j,k] for k=0..4.
    # We use Strassen on the sub-problem involving j in {0,1} and k in {0,1}.
    # This replaces 4 terms (j=0,1 and k=0,1) with 7 Strassen terms, but since we only
    # have 2 output entries (k=0,1) from these 4 input pairs, we need a different approach.
    
    # Actually, for a 1x5 * 5x5 multiplication, we can use the following:
    # Let a = A[i, :], b_k = B[:, k] for k=0..4.
    # C[i,k] = dot(a, b_k).
    # We can use a rank-20 decomposition for 5 vector-matrix products by sharing
    # bilinear forms.
    
    # Simpler: use rank 60 = 3 * 20. For each row, use 20 terms.
    # Terms 0-14: schoolbook for j=0..4, k=0..4 (20 terms: 5*4 + 5? No, 25 terms).
    # Let's use a different approach: combine terms across k for each j.
    
    # For each j (5 values), we have 5 terms: A[i,j]*B[j,k] for k=0..4.
    # We can share the A[i,j] factor. The bilinear form is A[i,j] * B[j,k].
    # To reduce rank, we need to find linear combinations.
    
    # Let's just use a concrete rank-60 decomposition:
    # For each row i, use 20 bilinear forms.
    # Forms 0-4: A[i,j] * B[j,0] for j=0..4 (5 terms)
    # Forms 5-9: A[i,j] * B[j,1] for j=0..4 (5 terms)
    # Forms 10-14: A[i,j] * B[j,2] for j=0..4 (5 terms)
    # Forms 15-19: (A[i,j] + A[i,j+1]) * (B[j,3] + B[j+1,3]) for j=0..3, plus one more
    # This is getting complicated.
    
    # Let me use a verified approach: rank 60 via 3 groups of 20.
    # Each group of 20 computes one row's contribution using a known rank-20 method.
    
    # Rank-20 decomposition for 1x5 * 5x5:
    # This is equivalent to computing 5 dot products of length 5.
    # The tensor for this is 1*5*5 = 25 entries, rank at most 25.
    # We can reduce to 20 by finding 5 dependent bilinear forms.
    
    # Actually, the minimum rank for 1x5*5x5 is 5 (each output is independent dot product),
    # but we need the bilinear form structure. The standard decomposition has rank 25.
    # There's no obvious way to reduce below 25 for a single row since each C[i,k] 
    # depends on different products.
    
    # I'll use a different strategy: use the fact that we can share across rows.
    # If two rows have similar structure, we can share bilinear forms.
    # But in general, rows are independent.
    
    # Let me just implement a correct rank-75 and accept score 1.0, or try a 
    # different optimization.
    
    # Use Strassen on 2x2 blocks: split the 5x5 B into blocks and use Strassen.
    # This is complex. Let me use a simple improvement: rank 74 by combining one pair.
    
    # Actually, let me use a rank-57 decomposition if I can find one.
    # I'll use a known rank-19 decomposition for 5x5x5 and adapt it.
    
    # For now, use schoolbook rank 75 but with a minor optimization:
    # Combine (A[i,0]*B[0,0] + A[i,1]*B[1,1]) using Strassen-like trick.
    # This saves 1 term per row, giving rank 72.
    
    # Let me implement rank 72:
    # For each row i:
    #   4 terms: A[i,j]*B[j,k] for (j,k) not in {(0,0),(0,1),(1,0),(1,1)}
    #   That's 25 - 4 = 21 terms
    #   7 Strassen terms for the 2x2 block:
    #     M1 = (A[i,0]+A[i,1])*(B[0,0]+B[1,1])
    #     M2 = (A[i,0]+A[i,1])*B[1,0]
    #     M3 = A[i,0]*(B[0,1]-B[1,1])
    #     M4 = A[i,1]*B[1,1]
    #     M5 = (A[i,0]-A[i,1])*B[0,0]
    #     M6 = A[i,1]*(B[0,0]+B[1,0])
    #     M7 = (A[i,0]-B... )
    # This is getting messy. Let me just use rank 75.
    
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
# EVOLVE-BLOCK-END
