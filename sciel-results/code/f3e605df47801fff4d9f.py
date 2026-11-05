"""Improved program for 3x3x7 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    # Use a block decomposition.
    # Split into:
    #   Block A: 2x3x7 (A[0:2, :] @ B[:, 0:7])
    #   Block B: 1x3x7 (A[2:3, :] @ B[:, 0:7])
    # But that's just schoolbook split. Let's try to use Strassen on 2x2x2 parts.
    # Actually, for 3x3x7, let's use a known or constructed decomposition.
    # The schoolbook rank is 63. We need to do better.
    # Let's try a simple block approach:
    # Split B into B1 (3x3) and B2 (3x4).
    # C = A*B1 + A*B2
    # For A*B1 (3x3 * 3x3): can we do better than 27? Yes, rank 23 is known for 3x3x3.
    # But constructing that is complex. Let's try a simpler improvement.
    #
    # Alternative: Use the fact that 3x3x7 can be decomposed using
    # a mixture of smaller problems.
    #
    # Let's try: Split A into rows 0-1 and row 2.
    # A = [A1; A2] where A1 is 2x3, A2 is 1x3.
    # C = A1*B + A2*B
    # A1*B: 2x3 * 3x7. Rank of 2x3x7? Schoolbook is 42.
    # A2*B: 1x3 * 3x7. Rank is 7.
    # Total: 42 + 7 = 49. Better than 63.
    #
    # Can we do better for 2x3x7?
    # Split B into B1 (3x3) and B2 (3x4).
    # A1*B1: 2x3 * 3x3. Rank? Schoolbook 18.
    # A1*B2: 2x3 * 3x4. Rank? Schoolbook 24.
    # Total for A1*B: 18 + 24 = 42. Same.
    #
    # What if we use Strassen on 2x2x2 parts?
    # For 2x3x7: Split A1 into 2x2 and 2x1.
    # Split B into columns: 3x3 and 3x4.
    # This gets complicated. Let's just use the 49 decomposition.
    #
    # Actually, let's try an even simpler approach:
    # Just use schoolbook but with some sharing.
    # The simplest improvement: note that for 1x3x7, the rank is 7 (not 21 as in pure schoolbook
    # where you'd have 3*7=21 terms, but actually it's just 7 since it's a dot product per output).
    # Wait, 1x3 * 3x7: output is 1x7. Each output C[i] = sum_j A[j]*B[j][i].
    # So rank is 7 (one per output column).
    # Similarly, for 2x3 * 3x7: output is 2x7.
    # C[i][k] = sum_j A1[i][j] * B[j][k]
    # This is 2*7 = 14 dot products of length 3.
    # Schoolbook would use 2*3*7 = 42 terms.
    # But we can write it as: for each (i,k), C[i][k] = sum_j A1[i][j]*B[j][k].
    # This is a rank-3 tensor for each (i,k) pair, and there are 14 such pairs.
    # But the tensor is the direct sum of 14 rank-3 tensors? No, it's a single 2x3x7 tensor.
    # The rank of the 2x3x7 tensor: I believe it's at most 2*7 = 14 if we can share.
    # Actually, the tensor T[i,j,k] = A[i,j]*B[j,k]. This is the matmul tensor.
    # The rank of the (2,3,7) matmul tensor.
    # I know that M(2,3) = 2*3 = 6 is the schoolbook rank, but the actual rank might be lower.
    # For 2x3, the rank is 2*3=6? No, for m*n*p, the rank is at most min(m*p, n*p, m*n)... no.
    # The rank of the matmul tensor (m,n,p) is at most m*n*p (schoolbook) and at least max(m*n, n*p, m*p)/something.
    # Actually, for 2x3x7, I'll just use schoolbook for now to be safe, but that gives 42.
    # Plus 7 for the 1x3x7 part = 49.
    #
    # Let me just implement the 49-rank decomposition.
    U = []
    V = []
    W = []
    col = 0
    # Part 1: A[0:2, :] * B (2x3 * 3x7), schoolbook rank 42
    for i in range(2):
        for j in range(3):
            for k in range(7):
                u = [0] * 49
                v = [0] * 49
                w = [0] * 49
                u[i*3 + j] = 1
                v[j*7 + k] = 1
                w[k*3 + i] = 1
                U.append(u)
                V.append(v)
                W.append(w)
                col += 1
    # Part 2: A[2:3, :] * B (1x3 * 3x7), rank 7
    for k in range(7):
        u = [0] * 49
        v = [0] * 49
        w = [0] * 49
        u[2*3 + 0] = 1  # A[2,0]
        u[2*3 + 1] = 1  # A[2,1]
        u[2*3 + 2] = 1  # A[2,2]
        # Wait, this isn't right. For 1x3 * 3x7, each output C[0][k] = sum_j A[2][j]*B[j][k].
        # So we need 7 terms, one per k.
        # For term k:
        # U should have A[2][j] for j=0,1,2
        # V should have B[j][k] for j=0,1,2
        # W should have C[0][k]
        # But U, V, W are vectors of length 49.
        # U index for A[2][j] is 2*3+j = 6+j
        # V index for B[j][k] is j*7+k
        # W index for C[0][k] is k*3+0 = k
        pass
    # Let me redo this properly.
    U = []
    V = []
    W = []
    rank = 49
    # Part 1: 2x3x7 schoolbook
    for i in range(2):
        for j in range(3):
            for k in range(7):
                u = [0] * rank
                v = [0] * rank
                w = [0] * rank
                u[i*3 + j] = 1
                v[j*7 + k] = 1
                w[k*3 + i] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    # Part 2: 1x3x7, 7 terms
    for k in range(7):
        u = [0] * rank
        v = [0] * rank
        w = [0] * rank
        for j in range(3):
            u[2*3 + j] = 1
            v[j*7 + k] = 1
        w[k*3 + 2] = 1
        U.append(u)
        V.append(v)
        W.append(w)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
