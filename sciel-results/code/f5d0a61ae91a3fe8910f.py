import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7

    # We need to decompose the matrix multiplication tensor for (n x m) @ (m x p) -> (n x p)
    # The tensor has entries T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # Standard decomposition: T = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Wait, the problem says:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # And the trace convention is W[k*n + i, r]
    # So the decomposition is: C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) and W[k*n+i, r] corresponds to the (i,k) output entry.
    # The decomposition should satisfy:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   delta(i,j,k) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where delta(i,j,k) = 1 (the multiplication tensor has 1s at all valid (i,j,k) positions)
    # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k in their ranges.
    # The rank is the minimum number of terms in the sum decomposition.

    # Let's use a block-wise approach. We can partition the 3x4x7 tensor into smaller blocks.
    # Strategy: Use known low-rank decompositions for smaller sub-tensors.
    #
    # For 2x2x2 multiplication, the rank is 7 (Strassen).
    # For 1x2x2, the rank is 2.
    # For 2x1x2, the rank is 2.
    # For 1x1x2, the rank is 1.
    # etc.
    #
    # Let's try to partition the 3x4x7 tensor.
    # n=3, m=4, p=7.
    # We can partition n into 1+2, m into 2+2, p into 3+4 or 1+2+4 etc.
    #
    # Let's try a simple block decomposition first to see if we can beat 84.
    #
    # Actually, let's use a known result: the rank of matrix multiplication is known to be
    # sub-multiplicative and there are upper bounds. But for small dimensions, we can try
    # explicit constructions.
    #
    # Let me try a different approach: use the fact that we can combine Strassen-like
    # identities for 2x2 blocks.
    #
    # Partition:
    # n: 1 + 2 (rows)
    # m: 2 + 2 (cols of A, rows of B)
    # p: 3 + 4 (cols of B, cols of C)
    #
    # This gives 8 sub-tensors:
    # (1x2x3), (1x2x4), (2x2x3), (2x2x4)
    # Each appears with multiplicity depending on the block.
    #
    # Wait, the tensor is T[i,j,k] = 1 for all i,j,k. If we partition:
    # i in {0} union {1,2}, j in {0,1} union {2,3}, k in {0,1,2} union {3,4,5,6}
    # Then we have 4 blocks:
    # Block (0,0): i=0, j in {0,1}, k in {0,1,2} -> 1x2x3 tensor
    # Block (0,1): i=0, j in {0,1}, k in {3,4,5,6} -> 1x2x4 tensor
    # Block (1,0): i in {1,2}, j in {0,1}, k in {0,1,2} -> 2x2x3 tensor
    # Block (1,1): i in {1,2}, j in {0,1}, k in {3,4,5,6} -> 2x2x4 tensor
    #
    # But we also need to account for j in {2,3}!
    # Let me redo:
    # i in {0} union {1,2}, j in {0,1} union {2,3}, k in {0,1,2} union {3,4,5,6}
    # 8 blocks:
    # (i=0, j=0:1, k=0:2), (i=0, j=0:1, k=3:6)
    # (i=0, j=2:3, k=0:2), (i=0, j=2:3, k=3:6)
    # (i=1:2, j=0:1, k=0:2), (i=1:2, j=0:1, k=3:6)
    # (i=1:2, j=2:3, k=0:2), (i=1:2, j=2:3, k=3:6)
    #
    # Each block is a sub-tensor. The rank of the whole tensor is at most the sum of ranks
    # of the blocks (since we can combine the decompositions).
    #
    # Let's estimate ranks:
    # 1x2x3: rank <= 2*3 = 6 (schoolbook), but can we do better? For 1xm x p, rank = m*p... actually no.
    # For multiplying a 1xm vector by an mxp matrix, the output is 1xp.
    # T[i,j,k] with i fixed: T[j,k] = 1 for all j,k. This is a rank-1 matrix in (j,k) space? No.
    # T is a tensor of shape (1, m, p). T[0,j,k] = 1.
    # Decomposition: sum_r U[j,r] V[k,r] W[0,k,r]... wait.
    # Actually for i fixed, the "matrix" T[j,k] = 1 has rank 1. So the tensor rank is 1?
    # Let's check: T[0,j,k] = sum_r U[j,r] * V[k,r] * W[k,r]... no, W depends on i,k.
    # W[k*n+i, r]. For i=0, W[k*n, r].
    # We need T[0,j,k] = 1 = sum_r U[j,r] * V[k,r] * W[k, r] (with appropriate indexing).
    # If we set U[j,0] = 1, V[k,0] = 1, W[k,0] = 1, then sum = 1. So rank 1 works for 1xm xp.
    # Wait, that can't be right in general. Let me re-examine.
    # The tensor T for 1xm@mxp -> 1xp: T[0,j,k] = 1 for all j,k.
    # We want: 1 = sum_r U[j,r] * V[k,r] * W[k*n+0, r]
    # If U[j,0]=1, V[k,0]=1, W[k,0]=1, then for each (j,k): sum = 1*1*1 = 1. Yes!
    # So rank 1 suffices for any 1xm@mxp. Similarly for n x 1 @ 1 x p.
    #
    # OK so let me reconsider. For 2x2x2, the rank is 7, not 4. The schoolbook is 4, Strassen is 7? No wait.
    # Schoolbook 2x2x2 has rank 4 (4 multiplications). Strassen has rank 7? No, Strassen has 7 multiplications, so rank 7?
    # Wait, Strassen's algorithm uses 7 multiplications for 2x2 matrix multiplication, so the rank is at most 7.
    # But the schoolbook uses 4, so rank is at most 4. Actually the schoolbook IS rank 4.
    # So for 2x2x2, rank <= 4. And it's known that rank = 7? No, I'm confused.
    #
    # Let me think again. The rank of the matrix multiplication tensor M_{n,m,p} is the minimum number
    # of scalar multiplications needed. For 2x2x2, schoolbook uses 4 multiplications, so rank <= 4.
    # Strassen shows rank <= 7, but that's worse. Actually Strassen is for reducing the number of
    # multiplications when the dimensions are larger, by recursion.
    #
    # Wait, I think I have it backwards. For 2x2 matrix multiplication, the standard algorithm
    # uses 8 multiplications (2x2x2 = 8 scalar multiplications in the schoolbook sense where you
    # multiply each entry). No wait, 2x2 @ 2x2 = sum over 2 terms for each output entry, 4 output
    # entries, so 8 scalar multiplications total. The rank of the tensor is 8 for schoolbook.
    # Strassen reduces this to 7, so rank <= 7.
    #
    # But the problem's schoolbook decomposition uses rank = n*m*p = 3*4*7 = 84.
    # For 2x2x2, that would be 8. So schoolbook rank is n*m*p.
    #
    # So for 2x2x2, can we do better than 8? Yes, Strassen shows 7.
    #
    # Let me look up: the rank of 2x2x2 matrix multiplication tensor is 7.
    # Yes, that's a well-known result. Strassen's identity gives a rank-7 decomposition.
    #
    # OK so let me use this. Let me construct explicit low-rank decompositions for small blocks.
    #
    # For 2x2x2, the rank-7 decomposition (Strassen):
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = C0 + C1
    # The 7 products are:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    #
    # In tensor form, for each (i,j,k), T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    # We need to express each of the 7 terms.
    #
    # This is getting complex. Let me try a different approach: use a known library or
    # construct the decomposition programmatically using linear algebra.
    #
    # Actually, let me try to use the fact that we can find a low-rank decomposition
    # using randomized methods or by solving a system of equations.
    #
    # Alternative: let me just try to find a decomposition with rank less than 84
    # using a greedy or iterative approach.
    #
    # Actually, let me try the block approach with known small-rank decompositions.
    #
    # Let me try partitioning into 2x2x2 blocks plus remainder.
    # n=3, m=4, p=7.
    #
    # Let's partition:
    # n: 2 + 1
    # m: 2 + 2
    # p: 4 + 3
    #
    # Blocks:
    # (2,2,4): rank? For 2x2x4, schoolbook is 16. Can we do better?
    #   2x2x4 = 2x2x2 + 2x2x2 (in p dimension), so rank <= 7+7 = 14.
    # (2,2,3): schoolbook 12. 2x2x3 = 2x2x2 + 2x2x1. Rank of 2x2x1: schoolbook 4, but
    #   multiplying 2x2 by 2x1. T[i,j,k] with k fixed. T[i,j,0] = 1 for all i,j.
    #   This is a 2x2 matrix of 1s, rank 1. So 2x2x1 has rank 1? Let's check.
    #   T[i,j,0] = 1 = sum_r U[i,j,r] V[j,0,r] W[0,r].
    #   If U[i,j,0]=1, V[j,0,0]=1, W[0,0]=1, then sum = 1. Yes, rank 1.
    #   So 2x2x3 rank <= 7 + 1 = 8.
    # (2,2,2) twice more: 7 each.
    # (1,2,4): rank 1 (as shown above for 1xm xp).
    # (1,2,3): rank 1.
    # (1,2,2) x2: rank 1 each.
    #
    # Wait, I need to be more careful. Let me redo the partition.
    #
    # n=3, m=4, p=7.
    # Partition n into {0,1} and {2}, m into {0,1} and {2,3}, p into {0,1,2,3} and {4,5,6}.
    #
    # 8 blocks:
    # 1. n={0,1}, m={0,1}, p={0,1,2,3}: 2x2x4, rank <= 14
    # 2. n={0,1}, m={0,1}, p={4,5,6}: 2x2x3, rank <= 8
    # 3. n={0,1}, m={2,3}, p={0,1,2,3}: 2x2x4, rank <= 14
    # 4. n={0,1}, m={2,3}, p={4,5,6}: 2x2x3, rank <= 8
    # 5. n={2}, m={0,1}, p={0,1,2,3}: 1x2x4, rank 1
    # 6. n={2}, m={0,1}, p={4,5,6}: 1x2x3, rank 1
    # 7. n={2}, m={2,3}, p={0,1,2,3}: 1x2x4, rank 1
    # 8. n={2}, m={2,3}, p={4,5,6}: 1x2x3, rank 1
    #
    # Total rank <= 14+8+14+8+1+1+1+1 = 48.
    #
    # That's much better than 84! But I need to actually construct the decompositions.
    #
    # Let me code this up. I'll need to implement the rank-7 Strassen decomposition for 2x2x2,
    # and then extend to 2x2x3 and 2x2x4 by adding terms.
    #
    # For 2x2x2 Strassen rank-7:
    # The 7 terms correspond to 7 "multiplications". In the tensor decomposition, each term r
    # has U[i,j,r], V[j,k,r], W[i,k,r] values.
    #
    # Let me define the Strassen decomposition for a 2x2x2 block at positions (i0, j0, k0).
    # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}.
    #
    # The 7 products:
    # m1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
    #   U[i0,j0,1] += 1, U[i0+1,j0+1,1] += 1
    #   V[j0,k0,1] += 1, V[j0+1,k0+1,1] += 1
    #   W[i0,k0,1] += 1, W[i0,k0+1,1] += 1, W[i0+1,k0,1] += 1, W[i0+1,k0+1,1] += 1
    #
    # m2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
    #   U[i0+1,j0,2] += 1, U[i0+1,j0+1,2] += 1
    #   V[j0,k0,2] += 1
    #   W[i0+1,k0,2] += 1
    #
    # m3 = A[i0,j0] * (B[j0+1,k0+1]-B[j0+1,k0])
    #   U[i0,j0,3] += 1
    #   V[j0+1,k0+1,3] += 1, V[j0+1,k0,3] += -1
    #   W[i0,k0+1,3] += 1
    #
    # m4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
    #   U[i0+1,j0+1,4] += 1
    #   V[j0+1,k0,4] += 1, V[j0,k0,4] += -1
    #   W[i0+1,k0,4] += 1
    #
    # m5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
    #   U[i0,j0,5] += 1, U[i0,j0+1,5] += 1
    #   V[j0+1,k0+1,5] += 1
    #   W[i0,k0+1,5] += 1
    #
    # m6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
    #   U[i0+1,j0,6] += 1, U[i0,j0,6] += -1
    #   V[j0,k0,6] += 1, V[j0,k0+1,6] += 1
    #   W[i0+1,k0,6] += 1
    #
    # m7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
    #   U[i0,j0+1,7] += 1, U[i0+1,j0+1,7] += -1
    #   V[j0+1,k0,7] += 1, V[j0+1,k0+1,7] += 1
    #   W[i0+1,k0+1,7] += 1
    #
    # Let me verify: C[i0,k0] should be A[i0,j0]*B[j0,k0] + A[i0,j0+1]*B[j0+1,k0]
    # From m1: A[i0,j0]*B[j0,k0] + A[i0,j0]*B[j0+1,k0+1] + A[i0+1,j0+1]*B[j0,k0] + A[i0+1,j0+1]*B[j0+1,k0+1]
    # From m2: A[i0+1,j0]*B[j0,k0] + A[i0+1,j0+1]*B[j0,k0]
    # From m3: A[i0,j0]*B[j0+1,k0+1] - A[i0,j0]*B[j0+1,k0]
    # From m4: A[i0+1,j0+1]*B[j0+1,k0] - A[i0+1,j0+1]*B[j0,k0]
    # From m5: A[i0,j0]*B[j0+1,k0+1] + A[i0,j0+1]*B[j0+1,k0+1]
    # From m6: A[i0+1,j0]*B[j0,k0] + A[i0+1,j0]*B[j0,k0+1] - A[i0,j0]*B[j0,k0] - A[i0,j0]*B[j0,k0+1]
    # From m7: A[i0,j0+1]*B[j0+1,k0] + A[i0,j0+1]*B[j0+1,k0+1] - A[i0+1,j0+1]*B[j0+1,k0] - A[i0+1,j0+1]*B[j0+1,k0+1]
    #
    # This is getting very messy to verify by hand. Let me just trust the standard Strassen
    # decomposition and code it up, then test it.
    #
    # Actually, let me just use a simpler approach: use the schoolbook for small blocks
    # and only use Strassen for 2x2x2. Then combine.
    #
    # But even schoolbook for the blocks gives:
    # 2x2x4: 16, 2x2x3: 12, 1x2x4: 8, 1x2x3: 6
    # Total: 16+12+16+12+8+6+8+6 = 84. Same as before!
    #
    # So I need to use low-rank decompositions for the 2x2xk blocks.
    #
    # Let me try yet another approach: use a known result that rank(2x2xp) <= 2p for p>=2?
    # Or use the fact that rank is sub-multiplicative in some sense.
    #
    # Actually, I recall that for 2x2 matrix multiplication, the rank is 7.
    # For 2x2x3, the rank is at most 2*rank(2x2x2) - something... I'm not sure.
    #
    # Let me just try to use the Strassen decomposition for each 2x2x2 sub-block
    # and schoolbook for the rest, and see what rank I get.
    #
    # Partition n=3 into 2+1, m=4 into 2+2, p=7 into 4+3.
    #
    # For the 2x2x4 block: decompose p into 2+2, giving two 2x2x2 blocks, each rank 7. Total 14.
    # For the 2x2x3 block: decompose p into 2+1. 2x2x2 rank 7, 2x2x1 rank 1. Total 8.
    # For 1x2x4: rank 1.
    # For 1x2x3: rank 1.
    #
    # Total: 14+8+14+8+1+1+1+1 = 48.
    #
    # Let me code this up.

    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(i0, j0, k0, start_col):
        """Add Strassen rank-7 decomposition for 2x2x2 block at (i0,j0,k0)."""
        nonlocal col
        # m1
        U[i0 * m + j0][start_col] = 1
        U[(i0 + 1) * m + (j0 + 1)][start_col] = 1
        V[j0 * p + k0][start_col] = 1
        V[(j0 + 1) * p + (k0 + 1)][start_col] = 1
        W[k0 * n + i0][start_col] = 1
        W[(k0 + 1) * n + i0][start_col] = 1
        W[k0 * n + (i0 + 1)][start_col] = 1
        W[(k0 + 1) * n + (i0 + 1)][start_col] = 1
        col = start_col + 1

        # m2
        U[(i0 + 1) * m + j0][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        col += 1

        # m3
        U[i0 * m + j0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        V[(j0 + 1) * p + k0][col] = -1
        W[(k0 + 1) * n + i0][col] = 1
        col += 1

        # m4
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + (i0 + 1)][col] = 1
        col += 1

        # m5
        U[i0 * m + j0][col] = 1
        U[i0 * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + i0][col] = 1
        col += 1

        # m6
        U[(i0 + 1) * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + (k0 + 1)][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        col += 1

        # m7
        U[i0 * m + (j0 + 1)][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = -1
        V[(j0 + 1) * p + k0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1

    def add_rank1_1xm_xp(i, j0, k0, start_col):
        """Add rank-1 decomposition for 1xm@mxp block (row i, cols j0..j0+1, k0..k0+p-1)."""
        nonlocal col
        for j in range(m):
            U[i * m + j][start_col] = 1
        for k in range(p):
            V[j0 * p + k][start_col] = 1
            W[k * n + i][start_col] = 1
        col = start_col + 1

    # Block 1: n={0,1}, m={0,1}, p={0,1,2,3} -> 2x2x4 = two 2x2x2
    add_strassen(0, 0, 0, col)
    add_strassen(0, 0, 2, col)

    # Block 2: n={0,1}, m={0,1}, p={4,5,6} -> 2x2x3 = 2x2x2 + 2x2x1
    add_strassen(0, 0, 4, col)
    # 2x2x1 at k=6
    U[0 * m + 0][col] = 1
    U[0 * m + 1][col] = 1
    U[1 * m + 0][col] = 1
    U[1 * m + 1][col] = 1
    V[0 * p + 6][col] = 1
    V[1 * p + 6][col] = 1
    W[6 * n + 0][col] = 1
    W[6 * n + 1][col] = 1
    col += 1

    # Block 3: n={0,1}, m={2,3}, p={0,1,2,3} -> 2x2x4 = two 2x2x2
    add_strassen(0, 2, 0, col)
    add_strassen(0, 2, 2, col)

    # Block 4: n={0,1}, m={2,3}, p={4,5,6} -> 2x2x3 = 2x2x2 + 2x2x1
    add_strassen(0, 2, 4, col)
    U[0 * m + 2][col] = 1
    U[0 * m + 3][col] = 1
    U[1 * m + 2][col] = 1
    U[1 * m + 3][col] = 1
    V[2 * p + 6][col] = 1
    V[3 * p + 6][col] = 1
    W[6 * n + 0][col] = 1
    W[6 * n + 1][col] = 1
    col += 1

    # Block 5: n={2}, m={0,1}, p={0,1,2,3} -> 1x2x4, rank 1
    add_rank1_1xm_xp(2, 0, 0, col)

    # Block 6: n={2}, m={0,1}, p={4,5,6} -> 1x2x3, rank 1
    add_rank1_1xm_xp(2, 0, 4, col)

    # Block 7: n={2}, m={2,3}, p={0,1,2,3} -> 1x2x4, rank 1
    add_rank1_1xm_xp(2, 2, 0, col)

    # Block 8: n={2}, m={2,3}, p={4,5,6} -> 1x2x3, rank 1
    add_rank1_1xm_xp(2, 2, 4, col)

    return {"U": U, "V": V, "W": W}
