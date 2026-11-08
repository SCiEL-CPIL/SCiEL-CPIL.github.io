"""Matrix multiplication 3x4 x 4x4 via block decomposition."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    # Block decomposition:
    # A: 3x4, B: 4x4
    # Split A into 3 blocks of 1x4 (each row)
    # Split B into 4 blocks of 1x4 (each row)
    # 
    # We use the following strategy:
    # Treat the multiplication as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # For each (i,k) pair, the naive computation uses 4 multiplications.
    # Total: 3*4*4 = 48.
    #
    # We can reduce by grouping. Consider splitting B into two halves of 2x4.
    # Then C = A * B1 + A * B2 where B1 is first 2 rows of B, B2 is last 2 rows.
    # A is 3x2 (left half) and 3x2 (right half) if we split columns of A.
    #
    # Actually, let's think of it as:
    # A = [A1 | A2] where A1, A2 are 3x2
    # B = [B1; B2] where B1, B2 are 2x4
    # C = A1*B1 + A2*B2
    #
    # Each A1*B1 is (3x2)*(2x4) = 3x4, using 3*2*4=24 scalar mults.
    # Total: 48. No savings.
    #
    # Let's try a different approach. The key insight for small matrices is to
    # find explicit rank decompositions. For 3x4x4, the known upper bound is 48
    # (schoolbook), but there might be better ones.
    #
    # Actually, let's try to use the fact that we can group computations.
    # Consider computing 2x2 blocks. If we partition:
    # A into rows, B into rows, then each C[i,k] = sum_j A[i,j]B[j,k].
    #
    # For 3x4x4, let me just return the schoolbook for now but structured
    # to allow future improvement. The rank is 48.
    #
    # Wait - let me think about whether we can do better. 
    # The tensor rank of matrix multiplication M_{n,m,p} is at most n*m*p.
    # For small cases, exact values are known. For 3x4x4, I'm not certain
    # of the exact rank, but 48 is the trivial bound.
    #
    # Let me try a block approach: split into 2x2 blocks where possible.
    # A is 3x4: think of it as [[A11, A12], [A21, A22], [A31, A32]]
    # B is 4x4: think of it as [[B11, B12], [B21, B22]]
    # where A11, A12, etc. are 1x2 blocks.
    #
    # C[i,k] for i=0: A[0,:]*B[:,k] = A11*B11 + A12*B21 (if we align properly)
    # This is getting complex. Let me just do schoolbook.
    
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
# EVOLVE-BLOCK-END
