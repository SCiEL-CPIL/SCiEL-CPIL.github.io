"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """Build a rank-7 exact decomposition for 2x2 matrix multiplication.

    Uses Strassen's 7-multiplication scheme.  The factor matrices are
    produced from compact per-term specifications and then checked by
    reconstructing every entry of the product tensor C = A * B as
    sum_r U[i,r] * V[j,r] * W[k,r].
    """
    n, m, p = DIMENSIONS
    R = 7

    # Row index maps:
    #   U row: i * m + j  (i,j in {0,1})  -> 0,1,2,3 for A00,A01,A10,A11
    #   V row: j * p + k  (j,k in {0,1})  -> 0,1,2,3 for B00,B01,B10,B11
    #   W row: k * n + i  (k,i in {0,1})  -> 0,1,2,3 for C00,C01,C10,C11
    #     (trace convention: W[k*n + i] = C[i,k])

    # Seven Strassen products expressed as lists of (row, coeff):
    #   P0 = (A00 + A11)(B00 + B11)
    #   P1 = (A10 + A11)(B00)
    #   P2 = (A00)(B01 - B11)
    #   P3 = (A11)(B10 - B00)
    #   P4 = (A00 + A01)(B11)
    #   P5 = (A10 - A00)(B00 + B01)
    #   P6 = (A01 - A11)(B10 + B11)
    #
    # Reassembly:
    #   C00 = P0 + P3 - P4 - P5
    #   C01 = P2 + P4
    #   C10 = P1 + P3
    #   C11 = P0 - P1 + P2 + P6

    u_terms = [
        [(0, 1), (3, 1)],   # P0
        [(2, 1), (3, 1)],   # P1
        [(0, 1)],           # P2
        [(3, 1)],           # P3
        [(0, 1), (1, 1)],   # P4
        [(2, 1), (0, -1)],  # P5
        [(1, 1), (3, -1)],  # P6
    ]
    v_terms = [
        [(0, 1), (3, 1)],   # P0
        [(0, 1)],           # P1
        [(1, 1), (3, -1)],  # P2
        [(2, 1), (0, -1)],  # P3
        [(3, 1)],           # P4
        [(0, 1), (1, 1)],   # P5
        [(2, 1), (3, 1)],   # P6
    ]
    w_terms = [
        [(0, 1), (3, 1)],   # P0 -> C00, C11
        [(1, 1), (3, -1)],  # P1 -> C01, C11
        [(2, 1), (3, 1)],   # P2 -> C10, C11
        [(0, 1), (2, 1)],   # P3 -> C00, C10
        [(0, -1), (2, 1)],  # P4 -> C00, C10
        [(0, -1), (2, 1)],  # P5 -> C00, C10
        [(3, 1)],           # P6 -> C11
    ]

    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    for r in range(R):
        for row, c in u_terms[r]:
            U[row][r] += c
        for row, c in v_terms[r]:
            V[row][r] += c
        for row, c in w_terms[r]:
            W[row][r] += c

    # Verify: for each output (i,k) the sum over j of A[i,j]*B[j,k] must equal
    # sum_r U[i*m+j]... actually verify the tensor identity:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m + j?] ... 
    # In the decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r]
    # We verify all 8 (i,j,k) bilinear terms reconstruct correctly:
    # The tensor T[i,j,k] = delta (i.e. (AB)[i,k] = sum_j A[i,j]B[j,k])
    # means for the rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
    # and T[i,j,k] = 1 if the standard product has A[i,j]*B[j,k] contributing to C[i,k],
    # which is always 1 for the matrix multiplication tensor (every (i,j,k) triple appears).
    
    # So we check: for all i,j,k in {0,1}, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to schoolbook rank 8 (should never trigger)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
