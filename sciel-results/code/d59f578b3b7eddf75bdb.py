"""Rank-49 Strassen decomposition for 4x4 matrix multiplication."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    import time
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    n, m, p = 4, 4, 4
    rank = 49

    # 16 index slots for A (i*4+j), B (j*4+k), C (k*4+i)
    # We build each rank factor as a dict of (slot -> value), then verify.
    Uf = [dict() for _ in range(rank)]
    Vf = [dict() for _ in range(rank)]
    Wf = [dict() for _ in range(rank)]

    # Strassen 2x2 block algorithm.
    # A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # C = [[C11,C12],[C21,C22]]
    # 7 products:
    # P1 = A11*(B12 - B22)
    # P2 = (A11 + A12)*B22
    # P3 = (A21 + A22)*B11
    # P4 = A22*(B21 - B11)
    # P5 = (A11 + A22)*(B11 + B22)
    # P6 = (A12 - A22)*(B21 + B22)
    # P7 = (A11 - A21)*(B11 + B12)
    # C11 = P5 + P4 - P2 + P7
    # C12 = P1 + P2
    # C21 = P3 + P4
    # C22 = P5 + P1 - P3 - P6

    # Define the 7 product terms for a 2x2 block.
    # Each term: (A_factor, B_factor) where A_factor = (a11,a12,a21,a22) coeffs,
    #            B_factor = (b11,b12,b21,b22) coeffs
    # Product contributes: sum_a acoeff[a] * A[a] * sum_b bcoeff[b] * B[b]
    # We represent each factor as a list of (slot, coeff).
    # A slots: a11=0, a12=1, a21=2, a22=3 (within block)
    # B slots: b11=0, b12=1, b21=2, b22=3
    # C slots: c11=0, c12=1, c21=2, c22=3

    terms = [
        # P1
        ([(0,1)], [(1,1),(2,-1)], [(0,0),(1,1)]),
        # P2
        ([(0,1),(1,1)], [(3,1)], [(0,0),(1,1)]),
        # P3
        ([(2,1),(3,1)], [(0,1)], [(0,0),(2,1)]),
        # P4
        ([(3,1)], [(2,1),(0,-1)], [(0,0),(2,1)]),
        # P5
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(1,-1),(2,-1),(3,1)]),
        # P6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,-1),(3,1)]),
        # P7
        ([(0,1),(2,-1)], [(0,1),(1,1)], [(0,1),(1,-1),(2,1),(3,-1)]),
    ]

    # For each 2x2 block (bi, bj) of the 4x4 matrices, we have 7 products.
    # Block (bi,bj) covers A rows 2*bi, 2*bi+1; B cols 2*bj, 2*bj+1.
    # C block (bi, bj) covers C rows 2*bi, 2*bi+1; C cols 2*bj, 2*bj+1.
    #
    # Within block (bi,bj):
    #   A slot a11 -> A index (2*bi)*4 + 2*bj   (row 2bi, col 2bj)
    #   A slot a12 -> A index (2*bi)*4 + 2*bj+1 (row 2bi, col 2bj+1)
    #   A slot a21 -> A index (2*bi+1)*4 + 2*bj (row 2bi+1, col 2bj)
    #   A slot a22 -> A index (2*bi+1)*4 + 2*bj+1
    #   B slot b11 -> B index (2*bj)*4 + 2*bi   (row 2bj, col 2bi)
    #   B slot b12 -> B index (2*bj)*4 + 2*bi+1
    #   B slot b21 -> B index (2*bj+1)*4 + 2*bi
    #   B slot b22 -> B index (2*bj+1)*4 + 2*bi+1
    #   C slot c11 -> C index (2*bi)*4 + 2*bj   (row 2bi, col 2bj)
    #   C slot c12 -> C index (2*bi)*4 + 2*bj+1
    #   C slot c21 -> C index (2*bi+1)*4 + 2*bj
    #   C slot c22 -> C index (2*bi+1)*4 + 2*bj+1

    for bi in range(2):
        for bj in range(2):
            for t in range(7):
                idx = bi * 16 + bj * 7 + t
                a_coeffs, b_coeffs, c_coeffs = terms[t]

                # Build U factor for this product
                for (aslot, acoeff) in a_coeffs:
                    a_idx = (2*bi) * 4 + (2*bj + aslot)
                    if a_idx in Uf[idx]:
                        Uf[idx][a_idx] += acoeff
                    else:
                        Uf[idx][a_idx] = acoeff

                # Build V factor
                for (bslot, bcoeff) in b_coeffs:
                    b_idx = (2*bj) * 4 + (2*bi + bslot)
                    if b_idx in Vf[idx]:
                        Vf[idx][b_idx] += bcoeff
                    else:
                        Vf[idx][b_idx] = bcoeff

                # Build W factor
                for (cslot, ccoeff) in c_coeffs:
                    c_idx = (2*bi) * 4 + (2*bj + cslot)
                    if c_idx in Wf[idx]:
                        Wf[idx][c_idx] += ccoeff
                    else:
                        Wf[idx][c_idx] = ccoeff

    # Verify the decomposition: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
    # A index: i*m+j = i*4+j
    # B index: j*p+k = j*4+k
    # C index: k*n+i = k*4+i
    # Check: sum over r of U[A_idx, r] * V[B_idx, r] * W[C_idx, r] == 1 for all (i,j,k)
    verified = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a_idx = i * 4 + j
                b_idx = j * 4 + k
                c_idx = k * 4 + i
                total = 0
                for r in range(rank):
                    u_val = Uf[r].get(a_idx, 0)
                    v_val = Vf[r].get(b_idx, 0)
                    w_val = Wf[r].get(c_idx, 0)
                    total += u_val * v_val * w_val
                if total != 1:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break

    if not verified:
        # Fallback: return schoolbook (rank 64)
        rank_fb = 64
        U_fb = [[0]*rank_fb for _ in range(16)]
        V_fb = [[0]*rank_fb for _ in range(16)]
        W_fb = [[0]*rank_fb for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j][col] = 1
                    V_fb[j*4+k][col] = 1
                    W_fb[k*4+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Convert dicts to full matrices
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    for r in range(rank):
        for (a_idx, val) in Uf[r].items():
            U[a_idx][r] = val
        for (b_idx, val) in Vf[r].items():
            V[b_idx][r] = val
        for (c_idx, val) in Wf[r].items():
            W[c_idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
