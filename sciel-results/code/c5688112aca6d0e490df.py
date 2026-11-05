import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = 4, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Trivial decomposition: rank = n*m*p = 120
    trivial_rank = n * m * p
    U_triv = np.zeros((n * m, trivial_rank))
    V_triv = np.zeros((m * p, trivial_rank))
    W_triv = np.zeros((p * n, trivial_rank))

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_triv[a, idx] = 1
                V_triv[b, idx] = 1
                W_triv[c, idx] = 1
                idx += 1

    # Try to find a better decomposition using Strassen for 2x2x2 sub-blocks
    # Strategy: Split the 4x5 * 5x6 multiplication into blocks where we can apply Strassen
    # 
    # Split approach:
    # - Split A (4x5) into two 2x5 blocks: A_top (rows 0,1), A_bot (rows 2,3)
    # - Split B (5x6) into two 5x3 blocks: B_left (cols 0,1,2), B_right (cols 3,4,5)
    # - C = A_top * B_left + A_bot * B_right + cross terms
    # 
    # Actually, let's use a different approach. We'll use the fact that:
    # M_{4,5,6} can be decomposed by splitting the middle dimension m=5 into 2+3.
    # 
    # Let A = [A_L, A_R] where A_L is 4x2, A_R is 4x3
    # Let B = [B_T; B_B] where B_T is 2x6, B_B is 3x6
    # 
    # C = A_L * B_T + A_R * B_B
    # 
    # These are independent tensor products (disjoint A and B indices), so we can concatenate.
    # 
    # For A_L * B_T: M_{4,2,6}
    # For A_R * B_B: M_{4,3,6}
    #
    # Now decompose M_{4,2,6} and M_{4,3,6} recursively.
    #
    # M_{4,2,6}: Split n=4 into 2+2.
    #   M_{2,2,6}: Split p=6 into 2+2+2. Three M_{2,2,2} blocks.
    #   Each M_{2,2,2} has rank 7 (Strassen).
    #   So M_{2,2,6} has rank 3*7 = 21.
    #   M_{4,2,6} = two M_{2,2,6} = 2*21 = 42.
    #
    # M_{4,3,6}: Split n=4 into 2+2.
    #   M_{2,3,6}: Split p=6 into 3+3. Two M_{2,3,3} blocks.
    #   M_{2,3,3}: This is smaller. Let's use trivial: rank = 2*3*3 = 18.
    #   So M_{2,3,6} = 2*18 = 36.
    #   M_{4,3,6} = 2*36 = 72.
    #
    # Total: 42 + 72 = 114. Better than 120.
    #
    # But we can do better on M_{2,3,3}. Let's see if Strassen helps there.
    # M_{2,3,3}: We can try to find a better rank. The trivial rank is 18.
    # Known bounds: rank(M_{2,3,3}) is between 15 and 18.
    # Let's just use trivial for now and see if we can improve.

    # Helper: create trivial decomposition for M_{n,m,p}
    def trivial_decomp(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return U, V, W

    # Helper: Strassen decomposition for M_{2,2,2}
    def strassen_decomp():
        # Strassen's 7 multiplications for 2x2 * 2x2
        # U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = A * B, C = [[c00, c01], [c10, c11]]
        #
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # c00 = M1 - M4 + M5 + M6
        # c01 = M3 + M4
        # c10 = M2 + M5
        # c11 = M1 - M2 + M3 + M7
        #
        # U[i*m+j, r] = coefficient of A[i,j] in r-th multiplication
        # V[j*p+k, r] = coefficient of B[j,k] in r-th multiplication
        # W[k*n+i, r] = coefficient of C[i,k] in r-th multiplication (output index)
        #
        # Wait, W[k*n+i, r] is the output. So for each r, we get a contribution to C.
        # c00 gets M1 - M4 + M5 + M6, so W[0*2+0, r] = W[0, r] for r=0..6
        # Let's define:
        # r=0: M1, r=1: M2, r=2: M3, r=3: M4, r=4: M5, r=5: M6, r=6: M7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # U indices: a00=0, a01=1, a10=2, a11=3
        # V indices: b00=0, b01=1, b10=2, b11=3
        # W indices: c00=0, c01=1, c10=2, c11=3

        # M1 = (a00+a11)(b00+b11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # c00 += M1, c11 += M1

        # M2 = (a10+a11)b00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1  # c10 += M2, c11 -= M2

        # M3 = a00(b10-b11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1  # c01 += M3, c11 += M3

        # M4 = a11(b01-b00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[1, 3] = 1  # c00 -= M4, c01 += M4

        # M5 = (a00+a01)b11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[2, 4] = 1  # c00 += M5, c10 += M5

        # M6 = (a10-a00)(b00+b01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = 1  # c00 += M6

        # M7 = (a01-a11)(b10+b11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[3, 6] = 1  # c11 += M7

        return U, V, W

    # Helper: combine decompositions for independent tensor products
    # If we have (U1,V1,W1) for C1 = A1*B1 and (U2,V2,W2) for C2 = A2*B2
    # where A1 and A2 have disjoint row indices, and B1 and B2 have disjoint column indices,
    # then C = C1 + C2 can be decomposed by concatenating the rank dimensions.
    # We need to embed U1, V1, W1 into full-size matrices.
    def combine_independent(U1, V1, W1, n1, m1, p1, n2, m2, p2, n, m, p,
                             a1_offset, b1_offset, c1_offset,
                             a2_offset, b2_offset, c2_offset):
        rank1 = U1.shape[1]
        rank2 = U2.shape[1] if False else W1.shape[1]  # Actually need to pass U2,V2,W2 too
        # This function signature is getting complex. Let me restructure.
        pass

    # Let me restructure with a cleaner approach.
    # We'll build the decomposition directly.

    # Strategy:
    # 1. Split m=5 into 2+3.
    # 2. For M_{4,2,6}: split n=4 into 2+2, giving two M_{2,2,6}.
    #    For each M_{2,2,6}: split p=6 into 2+2+2, giving three M_{2,2,2}.
    #    Each M_{2,2,2} uses Strassen (rank 7).
    #    So M_{2,2,6} = 3*7 = 21, M_{4,2,6} = 2*21 = 42.
    # 3. For M_{4,3,6}: split n=4 into 2+2, giving two M_{2,3,6}.
    #    For each M_{2,3,6}: split p=6 into 3+3, giving two M_{2,3,3}.
    #    Each M_{2,3,3} uses trivial (rank 18).
    #    So M_{2,3,6} = 2*18 = 36, M_{4,3,6} = 2*36 = 72.
    # Total rank: 42 + 72 = 114.

    # Let's implement this step by step.
    # We'll create the full U, V, W matrices by filling in the appropriate blocks.

    total_rank = 114
    U = np.zeros((n * m, total_rank))  # (20, 114)
    V = np.zeros((m * p, total_rank))  # (30, 114)
    W = np.zeros((p * n, total_rank))  # (24, 114)

    rank_offset = 0

    # Part 1: M_{4,2,6} (A_L is 4x2, B_T is 2x6)
    # A_L rows: 0,1,2,3 (all), cols: 0,1
    # B_T rows: 0,1, cols: 0,1,2,3,4,5
    # C1 = A_L * B_T, shape 4x6

    # Split n=4 into 2+2: rows 0,1 and rows 2,3
    for n_block_start in [0, 2]:
        # M_{2,2,6}
        # Split p=6 into 2+2+2: cols 0,1 and cols 2,3 and cols 4,5
        for p_block_start in [0, 2, 4]:
            # M_{2,2,2}
            U_s, V_s, W_s = strassen_decomp()
            for r in range(7):
                # Map Strassen indices to global indices
                # A indices: i in {n_block_start, n_block_start+1}, j in {0,1}
                # B indices: j in {0,1}, k in {p_block_start, p_block_start+1}
                # C indices: i in {n_block_start, n_block_start+1}, k in {p_block_start, p_block_start+1}
                for i_s in range(2):
                    for j_s in range(2):
                        i = n_block_start + i_s
                        j = j_s
                        a_idx = i * m + j
                        U[a_idx, rank_offset + r] += U_s[i_s * 2 + j_s, r]
                for j_s in range(2):
                    for k_s in range(2):
                        j = j_s
                        k = p_block_start + k_s
                        b_idx = j * p + k
                        V[b_idx, rank_offset + r] += V_s[j_s * 2 + k_s, r]
                for k_s in range(2):
                    for i_s in range(2):
                        k = p_block_start + k_s
                        i = n_block_start + i_s
                        c_idx = k * n + i
                        W[c_idx, rank_offset + r] += W_s[k_s * 2 + i_s, r]
            rank_offset += 7

    # Part 2: M_{4,3,6} (A_R is 4x3, B_B is 3x6)
    # A_R rows: 0,1,2,3 (all), cols: 2,3,4
    # B_B rows: 2,3,4, cols: 0,1,2,3,4,5
    # C2 = A_R * B_B, shape 4x6

    # Split n=4 into 2+2: rows 0,1 and rows 2,3
    for n_block_start in [0, 2]:
        # M_{2,3,6}
        # Split p=6 into 3+3: cols 0,1,2 and cols 3,4,5
        for p_block_start in [0, 3]:
            # M_{2,3,3}
            U_t, V_t, W_t = trivial_decomp(2, 3, 3)
            for r in range(18):
                # Map trivial indices to global indices
                # A indices: i in {n_block_start, n_block_start+1}, j in {2,3,4}
                # B indices: j in {2,3,4}, k in {p_block_start, p_block_start+1, p_block_start+2}
                # C indices: i in {n_block_start, n_block_start+1}, k in {p_block_start, p_block_start+1, p_block_start+2}
                for i_s in range(2):
                    for j_s in range(3):
                        i = n_block_start + i_s
                        j = 2 + j_s
                        a_idx = i * m + j
                        U[a_idx, rank_offset + r] += U_t[i_s * 3 + j_s, r]
                for j_s in range(3):
                    for k_s in range(3):
                        j = 2 + j_s
                        k = p_block_start + k_s
                        b_idx = j * p + k
                        V[b_idx, rank_offset + r] += V_t[j_s * 3 + k_s, r]
                for k_s in range(3):
                    for i_s in range(2):
                        k = p_block_start + k_s
                        i = n_block_start + i_s
                        c_idx = k * n + i
                        W[c_idx, rank_offset + r] += W_t[k_s * 2 + i_s, r]
            rank_offset += 18

    # Verify: rank_offset should be 114
    # Part 1: 2 * 3 * 7 = 42
    # Part 2: 2 * 2 * 18 = 72
    # Total: 114

    # Check if we're within deadline
    if time.time() > deadline:
        return {'U': U_triv.tolist(), 'V': V_triv.tolist(), 'W': W_triv.tolist()}

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

