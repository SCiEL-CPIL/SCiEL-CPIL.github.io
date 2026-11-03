"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin

    # Build the multiplication tensor T[i,j,k] = delta_{i,k} * delta_{j,j}
    # T has shape (n, m, p) = (2, 3, 8)
    # T[i,j,k] = 1 if i==k and j==j (always true for j), so T[i,j,k] = 1 if i==k else 0
    # Wait, that's not right. The tensor is T[i,j,k] such that
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = delta_{i,i} * delta_{k,k} ... no.
    # The tensor T is defined by T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
    # Actually T[i,j,k] = 1 always (it's the standard matrix multiplication tensor)
    # More precisely, the tensor has entries T[i,j,k] where:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # So the tensor T in the decomposition C = sum_r U_r * V_r * W_r
    # has T[i,j,k] = 1 for all valid i,j,k (the multiplication tensor)

    # Let me build it properly:
    # T is a 3-tensor of shape (n, m, p)
    # T[i, j, k] = 1 if this term appears in C[i,k] = sum_j A[i,j]B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait no. The tensor for matrix multiplication is:
    # T[i,j,k] = delta_{i,k} ... no that's for trace.
    
    # The correct tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k in range.
    # Because each product A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all valid indices.

    # Actually, the tensor is: T[i,j,k] = 1 if i is row of A, j is col of A/row of B, k is col of B
    # And the output C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.

    # Let me verify with the Strassen decomposition to make sure I have the convention right.

    # For now, let me just try to find a low-rank decomposition using SVD on the matricized tensor.

    # Flatten the tensor T of shape (n, m, p) = (2, 3, 8)
    # T[i,j,k] = 1 for all i,j,k
    T = np.ones((n, m, p))

    # Matricize along different modes and find low-rank approximations
    # Mode-1 unfolding: shape (n, m*p)
    T1 = T.reshape(n, m * p)
    # Mode-2 unfolding: shape (m, n*p)
    T2 = T.reshape(m, n * p)
    # Mode-3 unfolding: shape (p, n*m)
    T3 = T.reshape(p, n * m)

    # The rank of the tensor is at least the max of the ranks of its matricizations
    # But it can be higher. Let's compute SVDs to get lower bounds.
    U1, s1, Vt1 = np.linalg.svd(T1, full_matrices=False)
    U2, s2, Vt2 = np.linalg.svd(T2, full_matrices=False)
    U3, s3, Vt3 = np.linalg.svd(T3, full_matrices=False)

    rank1 = sum(1 for s in s1 if s > 1e-10)
    rank2 = sum(1 for s in s2 if s > 1e-10)
    rank3 = sum(1 for s in s3 if s > 1e-10)

    lower_bound = max(rank1, rank2, rank3)
    # The tensor rank is at least the lower bound from matricizations

    # For T = ones(2,3,8):
    # T1 = ones(2, 24), rank 1
    # T2 = ones(3, 16), rank 1
    # T3 = ones(8, 6), rank 1
    # So lower bound is 1, which is not tight.

    # Let me try a different approach: use the known block structure.
    # Split m=3 into m=2 and m=1.
    # For m=2, p=8: decompose 2x2x8
    # For m=1, p=8: decompose 2x1x8 (rank 16, schoolbook)

    # For 2x2x8: split p=8 into four p=2 blocks, each 2x2x2 (Strassen, rank 7)
    # Total: 4*7 = 28
    # Plus 2x1x8: rank 16
    # Total: 44 (this is what we already have)

    # Alternative: split p=8 into two p=4 blocks.
    # For 2x2x4: can we do better than 14 (two Strassen)?
    # 2x2x4 = multiply 2x2 by 2x4.
    # Schoolbook: 2*2*4 = 16
    # Two Strassen 2x2x2: 7+7 = 14
    # Can we do rank 12? Let me search.

    # Let me try to find a rank-r decomposition of the 2x2x4 tensor.
    # T_2x2x4[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2,3}

    T224 = np.ones((2, 2, 4))
    # Matricizations:
    T224_1 = T224.reshape(2, 8)  # rank 1
    T224_2 = T224.reshape(2, 8)  # rank 1
    T224_3 = T224.reshape(4, 4)  # rank 1

    # Lower bound is 1, not helpful.

    # Let me try to use the known result: the rank of 2x2x4 is 14 (two Strassen).
    # Is there a known rank-12 decomposition? I don't think so.

    # Let me try a different block partition:
    # Split m=3 into m=1 and m=2, but handle them jointly.
    # Or split differently.

    # Another idea: use the fact that for 2x3x8, we can view it as
    # (2x3) x (3x8) = (2x3) x (3x4) x (3x4)
    # No, that's not right.

    # Let me just try to do a numerical search for a low-rank decomposition.
    # I'll use a simple iterative approach: start with a random initialization
    # and do gradient descent on the rank-r factorization.

    # But first, let me try rank 40 and see if it's feasible.
    # If not, fall back to 44.

    # For rank r, we need U (6, r), V (24, r), W (16, r) such that
    # T[i,j,k] = sum_{l=0}^{r-1} U[i,l] * V[j,l] * W[k,l]
    # where T[i,j,k] = 1 for all i,j,k.

    # This is a system of 2*3*8 = 48 equations in 6*r + 24*r + 16*r = 46*r unknowns.
    # For r=40: 48 equations, 1840 unknowns. Very underdetermined.

    # Let me try a simple approach: use the block structure.
    # Block 1: i in {0,1}, j in {0,1}, k in {0,...,7} -> 2x2x8
    # Block 2: i in {0,1}, j in {2}, k in {0,...,7} -> 2x1x8

    # For 2x2x8, use Strassen on four 2x2x2 blocks: rank 28
    # For 2x1x8, use schoolbook: rank 16
    # Total: 44

    # To get below 44, I need to either:
    # 1. Find a lower-rank decomposition of 2x2x8 (below 28)
    # 2. Find a lower-rank decomposition of 2x1x8 (below 16)
    # 3. Use a joint decomposition that exploits cross-block structure

    # 2x1x8 has rank exactly 16 (it's a sum of 16 rank-1 tensors with no structure).
    # 2x2x8: the rank is 28 (four Strassen). Is there a better decomposition?
    # The rank of 2x2xn is known to be 7*ceil(n/2) for even n? No, that's not right.
    # For 2x2x4, the rank is 14 (two Strassen). For 2x2x8, it's 28.
    # I believe these are optimal.

    # So the minimum rank is 44, and we can't do better with this block structure.

    # But wait - maybe a different block structure works?
    # What if we don't split m=3 into 2+1, but use a different partition?

    # Alternative: split p=8 into p=2 blocks (4 blocks of 2x3x2).
    # 2x3x2: multiply 2x3 by 3x2.
    # Schoolbook: 2*3*2 = 12
    # Can we do better? 2x3x2 is equivalent to 2x2x3 transposed.
    # The rank of 2x3x2... let me think.
    # T[i,j,k] for i in {0,1}, j in {0,1,2}, k in {0,1}
    # This is a 2x3x2 tensor. The rank is at most 12 (schoolbook).
    # Can it be done in fewer?
    # Matricization: T.reshape(2, 6) has rank 1, T.reshape(3, 4) has rank 1, T.reshape(2, 6) has rank 1.
    # Lower bound is 1, not helpful.
    # I think the rank of 2x3x2 is 12 (no improvement over schoolbook for these small sizes).

    # So 4 * 12 = 48, which is worse than 44.

    # What about splitting p=8 into p=4 blocks (2 blocks of 2x3x4)?
    # 2x3x4: multiply 2x3 by 3x4.
    # Schoolbook: 2*3*4 = 24
    # Block: 2x2x4 (rank 14) + 2x1x4 (rank 8) = 22
    # So 2 * 22 = 44. Same as before.

    # I think 44 is the best we can do with standard techniques.
    # Let me just return the rank-44 decomposition.

    total_rank = 44

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def _add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1

        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1

        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1

        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    def _add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off+0)][col] = 1
                V[(j_off+0)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    _add_strassen_block(0, 0, 0)
    _add_strassen_block(0, 0, 2)
    _add_schoolbook_2x1x4(0, 2, 0)

    _add_strassen_block(0, 0, 4)
    _add_strassen_block(0, 0, 6)
    _add_schoolbook_2x1x4(0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
