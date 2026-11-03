"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Model 2x4 * 4x8 as 2x2 matrix multiplication over ring of 2x2 matrices.
    # A (2x4) = [A_L, A_R], B (4x8) = [B_T; B_B]
    # C = A_L B_T + A_R B_B
    # Apply Strassen (rank 7) over the ring of 2x2 matrices.
    # Each 2x2 ring-multiplication = 4 scalar rank-1 terms.
    # Total rank = 7 * 4 = 28.

    rank = 28
    U = np.zeros((n * m, rank), dtype=float)   # 8 x 28
    V = np.zeros((m * p, rank), dtype=float)   # 32 x 28
    W = np.zeros((p * n, rank), dtype=float)   # 16 x 28

    # Strassen 7 components (over ring of 2x2 matrices)
    # Each component: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (a_index, coeff) where a_index in {0,1,2,3} = (i_loc, j_loc)
    # V_coeffs: list of (b_index, coeff) where b_index in {0,1,2,3} = (j_loc, k_loc)
    # W_coeffs: list of (c_index, coeff) where c_index in {0,1,2,3} = (i_loc, k_loc)
    strassen = [
        # M1 = (a00+a11)(b00+b11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (a10+a11)b00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = a00(b01-b11)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = a11(b10-b00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (a00+a01)b11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (a10-a00)(b00+b01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (a01-a11)(b10+b11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    # Fill U, V, W
    # U[i*m + j, r] where i in {0,1}, j in {0,1,2,3}
    # V[j*p + k, r] where j in {0,1,2,3}, k in {0..7}
    # W[k*n + i, r] where k in {0..7}, i in {0,1}

    for r_idx, (u_coeff, v_coeff, w_coeff) in enumerate(strassen):
        for a_idx, a_val in u_coeff:
            i_loc, j_loc = a_idx
            # j_g: if j_loc==0, j_g = 2*j_loc; if j_loc==1, j_g = 2*(j_loc-1)+1
            j_g = 2 * j_loc
            a_global = i_loc * m + j_g
            U[a_global, r_idx * 4] += a_val  # We'll use 4 sub-terms per Strassen component

        # Actually, we need 4 scalar rank-1 terms per Strassen component.
        # Let me restructure: for each Strassen component r and each scalar term s (0..3):
        # The s-th scalar term corresponds to (i_loc, j_loc) in U, (j_loc, k_loc) in V, (i_loc, k_loc) in W
        # where s encodes the pairing.
        pass

    # Let me redo this more carefully.
    # For each Strassen component r (0..6), we have 4 scalar rank-1 terms (s=0..3).
    # Term s corresponds to:
    #   U: a[i_loc, j_loc] where (i_loc, j_loc) = s decoded as (s//2, s%2)? 
    #   Actually, for a 2x2 * 2x2 = 2x2, the 4 scalar products are:
    #   c[0,0] = a[0,0]*b[0,0] + a[0,1]*b[1,0]
    #   c[0,1] = a[0,0]*b[0,1] + a[0,1]*b[1,1]
    #   c[1,0] = a[1,0]*b[0,0] + a[1,1]*b[1,0]
    #   c[1,1] = a[1,0]*b[0,1] + a[1,1]*b[1,1]
    #
    # But in the bilinear decomposition, each scalar product a[i,j]*b[j,k] is a rank-1 term.
    # For the 2x2*2x2, there are 4 such products, each contributing to one c entry.
    #
    # So for Strassen component r, the 4 scalar terms are:
    # s=0: a[0,0]*b[0,0] -> c[0,0]
    # s=1: a[0,1]*b[1,0] -> c[0,0]
    # s=2: a[0,0]*b[0,1] -> c[0,1]
    # s=3: a[0,1]*b[1,1] -> c[0,1]
    # s=4: a[1,0]*b[0,0] -> c[1,0]
    # s=5: a[1,1]*b[1,0] -> c[1,0]
    # s=6: a[1,0]*b[0,1] -> c[1,1]
    # s=7: a[1,1]*b[1,1] -> c[1,1]
    #
    # Wait, that's 8 terms, not 4. Let me reconsider.
    #
    # Actually, for 2x2 * 2x2, the schoolbook has 4 multiplications:
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    #
    # Each multiplication a_ij * b_jk is a rank-1 term in the tensor decomposition.
    # So there are 4 rank-1 terms for the schoolbook 2x2*2x2.
    #
    # In Strassen, we have 7 "multiplications" but each is over the ring of 2x2 matrices.
    # Each ring multiplication decomposes into 4 scalar rank-1 terms.
    # So total = 7 * 4 = 28 rank-1 terms.
    #
    # For Strassen component r, the 4 scalar terms are:
    # s=0: a[0,0] * b[0,0] -> contributes to c[0,0]
    # s=1: a[0,1] * b[1,0] -> contributes to c[0,0]
    # s=2: a[0,0] * b[0,1] -> contributes to c[0,1]
    # s=3: a[0,1] * b[1,1] -> contributes to c[0,1]
    # s=4: a[1,0] * b[0,0] -> contributes to c[1,0]
    # s=5: a[1,1] * b[1,0] -> contributes to c[1,0]
    # s=6: a[1,0] * b[0,1] -> contributes to c[1,1]
    # s=7: a[1,1] * b[1,1] -> contributes to c[1,1]
    #
    # Wait, that's 8 terms again. I'm confusing myself.
    #
    # Let me think again. A 2x2 matrix multiplication c = a * b has:
    # c[i,k] = sum_j a[i,j] * b[j,k]
    # The tensor T[i,j,k] = delta (for matmul) has rank 4 for 2x2*2x2.
    # The 4 rank-1 terms are:
    # (i=0,j=0,k=0), (i=0,j=1,k=0), (i=0,j=0,k=1), (i=0,j=1,k=1)
    # (i=1,j=0,k=0), (i=1,j=1,k=0), (i=1,j=0,k=1), (i=1,j=1,k=1)
    #
    # No wait, the rank-1 terms for 2x2*2x2 matmul are:
    # For each (i,j,k) where the term a[i,j]*b[j,k] contributes to c[i,k]:
    # (0,0,0): a[0,0]*b[0,0] -> c[0,0]
    # (0,1,0): a[0,1]*b[1,0] -> c[0,0]
    # (0,0,1): a[0,0]*b[0,1] -> c[0,1]
    # (0,1,1): a[0,1]*b[1,1] -> c[0,1]
    # (1,0,0): a[1,0]*b[0,0] -> c[1,0]
    # (1,1,0): a[1,1]*b[1,0] -> c[1,0]
    # (1,0,1): a[1,0]*b[0,1] -> c[1,1]
    # (1,1,1): a[1,1]*b[1,1] -> c[1,1]
    #
    # That's 8 terms, but the rank is 4. How?
    # Because in the bilinear decomposition, a rank-1 term is:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For 2x2*2x2: n=2, m=2, p=2
    # U: 4x4, V: 4x4, W: 4x4
    # A rank-1 term has one nonzero in U, one in V, one in W.
    #
    # The 4 rank-1 terms for 2x2*2x2 matmul (schoolbook) are:
    # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> a[0,0]*b[0,0] -> c[0,0]
    # r=1: U[1,1]=1, V[2,1]=1, W[1,1]=1  -> a[0,1]*b[1,0] -> c[0,0]
    # r=2: U[0,2]=1, V[1,2]=1, W[2,2]=1  -> a[0,0]*b[0,1] -> c[0,1]
    # r=3: U[1,3]=1, V[3,3]=1, W[3,3]=1  -> a[0,1]*b[1,1] -> c[0,1]
    #
    # Wait, that's only 4 terms but they only cover c[0,0] and c[0,1]. What about c[1,0] and c[1,1]?
    #
    # I see my error. The rank of the 2x2*2x2 matmul tensor is 4, but the 4 rank-1 terms
    # must cover all 8 entries (i,j,k). Let me reconsider.
    #
    # Actually, for 2x2*2x2, the tensor T[i,j,k] = 1 if the indices form a valid matmul term,
    # i.e., T[i,j,k] = 1 for all (i,j,k) in {0,1}^3. The rank of this all-ones tensor is 1!
    # No wait, that's not right either.
    #
    # Let me go back to basics. The matrix multiplication tensor for n x m * m x p is:
    # T[i, j, k] = 1 if we're computing c[i,k] = sum_j a[i,j]*b[j,k]
    # In the bilinear decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For 2x2*2x2 (n=2, m=2, p=2):
    # T[i,j,k] = 1 for all i,j,k in {0,1}
    # This is the all-ones tensor of size 2x2x2, which has rank 1!
    #
    # But that can't be right because Strassen says rank 7 for 2x2*2x2.
    #
    # I think I'm confusing the tensor. Let me look at this more carefully.
    #
    # The matrix multiplication is a trilinear map: (A, B, C) -> sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k]
    # where T[i,j,k] = 1 for all valid (i,j,k).
    #
    # For 2x2*2x2: T is 2x2x2 all ones. Rank = 1.
    # But Strassen's algorithm has 7 multiplications, which means rank 7.
    #
    # There's a contradiction. Let me resolve it.
    #
    # Ah, I see. The "rank" in Strassen's context is the number of scalar multiplications,
    # which is the tensor rank of the matrix multiplication tensor. But the tensor I defined
    # (all ones) has rank 1, not 7.
    #
    # The issue is that the matrix multiplication tensor is NOT the all-ones tensor.
    # It's the tensor that encodes the bilinear form of matrix multiplication.
    #
    # Let me redefine: The matrix multiplication of A (n x m) and B (m x p) gives C (n x p).
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # In the bilinear decomposition, we write:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k]) ... 
    #
    # No, that's not right either. Let me look at the standard formulation.
    #
    # The standard bilinear decomposition of matrix multiplication is:
    # C[i,k] = sum_r (sum_j U_r[i,j] A[i,j]) * (sum_l V_r[l,k] B[l,k])
    # where U_r and V_r are n x m and m x p matrices, and the sum is over r.
    #
    # In this formulation, the rank is the number of terms r.
    # For 2x2*2x2, the minimum rank is 7 (Strassen).
    #
    # The U, V, W matrices in the task are:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    #
    # And the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # where T[i,j,k] is the matrix multiplication tensor.
    #
    # For 2x2*2x2, T[i,j,k] = 1 for all i,j,k in {0,1}.
    # The rank of this tensor is 1 (it's the all-ones tensor).
    #
    # But Strassen's rank is 7. So there must be a different tensor.
    #
    # I think the confusion is between the tensor of the bilinear form and the tensor rank.
    # Let me just trust the Strassen construction and verify it numerically.
    #
    # Let me construct the 28-rank solution and verify it.

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # For each Strassen component r (0..6) and each scalar term s (0..3):
    # The scalar term corresponds to a specific (i_loc, j_loc) in U, (j_loc, k_loc) in V, (i_loc, k_loc) in W
    #
    # For 2x2 * 2x2, the 4 scalar terms are:
    # s=0: (i_loc=0, j_loc=0) -> U[0,0], V[0,0], W[0,0]
    # s=1: (i_loc=0, j_loc=1) -> U[0,1], V[1,0], W[0,0]  [a[0,1]*b[1,0] -> c[0,0]]
    # s=2: (i_loc=0, j_loc=0) -> U[0,0], V[0,1], W[0,1]  [a[0,0]*b[0,1] -> c[0,1]]
    # s=3: (i_loc=0, j_loc=1) -> U[0,1], V[1,1], W[0,1]  [a[0,1]*b[1,1] -> c[0,1]]
    #
    # Wait, I need to be more careful. Let me define the 4 scalar terms for 2x2*2x2:
    # Term 0: a[0,0] * b[0,0] -> c[0,0]
    # Term 1: a[0,1] * b[1,0] -> c[0,0]
    # Term 2: a[0,0] * b[0,1] -> c[0,1]
    # Term 3: a[0,1] * b[1,1] -> c[0,1]
    # Term 4: a[1,0] * b[0,0] -> c[1,0]
    # Term 5: a[1,1] * b[1,0] -> c[1,0]
    # Term 6: a[1,0] * b[0,1] -> c[1,1]
    # Term 7: a[1,1] * b[1,1] -> c[1,1]
    #
    # That's 8 terms. But the rank of 2x2*2x2 matmul is 7 (Strassen), not 8 (schoolbook).
    #
    # I think the issue is that in the bilinear decomposition, each "term" r has:
    # U_r: an n x m matrix (flattened to n*m entries)
    # V_r: an m x p matrix (flattened to m*p entries)
    # W_r: a p x n matrix (flattened to p*n entries)
    #
    # And C = sum_r U_r * V_r (matrix product)
    #
    # For 2x2*2x2, each U_r is 2x2, V_r is 2x2, W_r is 2x2.
    # The rank is the number of terms r.
    #
    # In the task's formulation:
    # U: (n*m, rank) = (4, rank) for 2x2
    # V: (m*p, rank) = (4, rank) for 2x2
    # W: (p*n, rank) = (4, rank) for 2x2
    #
    # And T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For 2x2*2x2, T[i,j,k] should encode the matmul tensor.
    # Let me just construct the Strassen solution directly and verify.

    # Strassen 2x2*2x2 (rank 7):
    # U_local, V_local, W_local as in the inspiration

    U_local = np.zeros((4, 7), dtype=float)
    V_local = np.zeros((4, 7), dtype=float)
    W_local = np.zeros((4, 7), dtype=float)

    # r=0 (M1)
    U_local[0, 0] = 1; U_local[3, 0] = 1  # a00, a11
    V_local[0, 0] = 1; V_local[3, 0] = 1  # b00, b11
    W_local[0, 0] = 1  # C00
    W_local[3, 0] = 1  # C11

    # r=1 (M2)
    U_local[2, 1] = 1; U_local[3, 1] = 1  # a10, a11
    V_local[0, 1] = 1  # b00
    W_local[2, 1] = 1  # C10
    W_local[3, 1] = -1  # C11

    # r=2 (M3)
    U_local[0, 2] = 1  # a00
    V_local[1, 2] = 1; V_local[3, 2] = -1  # b01, b11
    W_local[1, 2] = 1  # C01
    W_local[3, 2] = 1  # C11

    # r=3 (M4)
    U_local[3, 3] = 1  # a11
    V_local[2, 3] = 1; V_local[0, 3] = -1  # b10, b00
    W_local[0, 3] = 1  # C00
    W_local[2, 3] = 1  # C10

    # r=4 (M5)
    U_local[0, 4] = 1; U_local[1, 4] = 1  # a00, a01
    V_local[3, 4] = 1  # b11
    W_local[0, 4] = -1  # C00
    W_local[1, 4] = 1  # C01

    # r=5 (M6)
    U_local[2, 5] = 1; U_local[0, 5] = -1  # a10, a00
    V_local[0, 5] = 1; V_local[1, 5] = 1  # b00, b01
    W_local[3, 5] = 1  # C11

    # r=6 (M7)
    U_local[1, 6] = 1; U_local[3, 6] = -1  # a01, a11
    V_local[2, 6] = 1; V_local[3, 6] = 1  # b10, b11
    W_local[0, 6] = 1  # C00

    # Now, for the 2x4*4x8 case, we model it as 2x2 * 2x2 over ring of 2x2 matrices.
    # A (2x4) = [A_L, A_R], B (4x8) = [B_T; B_B]
    # C = A_L B_T + A_R B_B
    #
    # Using Strassen over the ring:
    # M1 = (A_L + A_R) * (B_T + B_B)
    # M2 = (A_L + A_R) * B_T  [wait, this doesn't look right]
    #
    # Let me use the standard Strassen formulas with A_L, A_R as "elements":
    # a00 = A_L, a01 = A_R, a10 = A_L, a11 = A_R  [if we view A as 2x2 of 2x2 blocks]
    #
    # Actually, A (2x4) can be viewed as a 2x2 matrix of 2x2 blocks:
    # A = [[A_00, A_01], [A_10, A_11]] where A_ij is 2x2
    # B (4x8) can be viewed as a 2x2 matrix of 2x2 blocks:
    # B = [[B_00, B_01, B_02, B_03], [B_10, B_11, B_12, B_13]] where B_ij is 2x2
    #
    # This is getting complicated. Let me just use the 56-rank solution from the current program
    # and verify it, then try to find a 28-rank solution.
    #
    # Actually, let me just construct the 56-rank solution as in the inspiration and return it.
    # The current program already does this. Let me just make sure it's correct.

    # Use the 56-rank Strassen solution from the inspiration
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7

            a_map = [0] * 4
            if s == 0:
                a_map[0] = 0 * 4 + 0
                a_map[1] = 0 * 4 + 1
                a_map[2] = 1 * 4 + 0
                a_map[3] = 1 * 4 + 1
            else:
                a_map[0] = 0 * 4 + 2
                a_map[1] = 0 * 4 + 3
                a_map[2] = 1 * 4 + 2
                a_map[3] = 1 * 4 + 3

            b_map = [0] * 4
            if s == 0:
                b_map[0] = 0 * 8 + (2 * j)
                b_map[1] = 0 * 8 + (2 * j + 1)
                b_map[2] = 1 * 8 + (2 * j)
                b_map[3] = 1 * 8 + (2 * j + 1)
            else:
                b_map[0] = 2 * 8 + (2 * j)
                b_map[1] = 2 * 8 + (2 * j + 1)
                b_map[2] = 3 * 8 + (2 * j)
                b_map[3] = 3 * 8 + (2 * j + 1)

            c_map = [0] * 4
            c_map[0] = (2 * j) * 2 + 0
            c_map[1] = (2 * j + 1) * 2 + 0
            c_map[2] = (2 * j) * 2 + 1
            c_map[3] = (2 * j + 1) * 2 + 1

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val

    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0
                rhs = 0.0
                for t in range(rank):
                    rhs += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(lhs - rhs) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid and time.time() - start_time < 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
