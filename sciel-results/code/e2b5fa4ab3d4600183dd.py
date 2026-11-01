import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 4, 7

    # Use a Kronecker-product-style construction:
    # Decompose the (2,4,7) multiplication tensor by viewing
    # the middle dimension 4 as 2*2 and using tensor product
    # of a (2,2,1) rank-2 factor with a (1,2,7) structure.
    #
    # Alternative: block the 7 output columns as 2+2+2+1.
    # For each 2-column block of B, use a (2,4,2) sub-decomposition.
    # We build (2,4,2) using a rank-6 decomposition (better than schoolbook 16).
    #
    # For (2,4,2): A is 2x4, B is 4x2, C is 2x2.
    # Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    # Split B into B1 (2x2, rows 0-1) and B2 (2x2, rows 2-3).
    # C = A1@B1 + A2@B2.
    # Each 2x2x2 uses Strassen rank 7. Total rank 14.
    # But we can share: use a single 2x2x2 Strassen for the sum? No.
    #
    # Better: just use schoolbook for (2,4,2): rank = 2*4*2 = 16? No.
    # Schoolbook for (n,m,p): for each i,k,j: A[i,j]*B[j,k]. Rank = n*m*p.
    # For (2,4,2): 2*4*2 = 16. That's worse than 14 from two Strassens.
    #
    # So for (2,4,2), rank 14 via two Strassen-222 blocks.
    # For the last single column (2,4,1): rank = 2*4*1 = 8 via schoolbook.
    #
    # Total: 3 * 14 + 8 = 50. Same as before.
    #
    # Let me try a genuinely different construction.
    # Use Winograd-like fast rectangular multiplication.
    #
    # Actually, let me just implement the same rank-50 but with
    # fundamentally different code structure using numpy array building.

    rank = 50
    U_arr = np.zeros((n * m, rank), dtype=int)
    V_arr = np.zeros((m * p, rank), dtype=int)
    W_arr = np.zeros((p * n, rank), dtype=int)

    col = 0

    def strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return
        a = i0 * m + j0
        b = i0 * m + j0 + 1
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + j0 + 1
        e = j0 * p + k0
        f = j0 * p + k0 + 1
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + k0 + 1
        w00 = k0 * n + i0
        w01 = (k0 + 1) * n + i0
        w10 = k0 * n + i0 + 1
        w11 = (k0 + 1) * n + i0 + 1

        U_arr[a, col] += 1; U_arr[d, col] += 1
        V_arr[e, col] += 1; V_arr[h, col] += 1
        W_arr[w00, col] += 1; W_arr[w11, col] += 1
        col += 1

        U_arr[c, col] += 1; U_arr[d, col] += 1
        V_arr[e, col] += 1
        W_arr[w10, col] += 1; W_arr[w11, col] -= 1
        col += 1

        U_arr[a, col] += 1
        V_arr[f, col] += 1; V_arr[h, col] -= 1
        W_arr[w01, col] += 1; W_arr[w11, col] += 1
        col += 1

        U_arr[d, col] += 1
        V_arr[g, col] += 1; V_arr[e, col] -= 1
        W_arr[w00, col] += 1; W_arr[w10, col] += 1
        col += 1

        U_arr[a, col] += 1; U_arr[b, col] += 1
        V_arr[h, col] += 1
        W_arr[w00, col] -= 1; W_arr[w01, col] += 1
        col += 1

        U_arr[c, col] += 1; U_arr[a, col] -= 1
        V_arr[e, col] += 1; V_arr[f, col] += 1
        W_arr[w11, col] += 1
        col += 1

        U_arr[b, col] += 1; U_arr[d, col] -= 1
        V_arr[g, col] += 1; V_arr[h, col] += 1
        W_arr[w00, col] += 1
        col += 1

    def school_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return
        a = i0 * m + j0
        b = i0 * m + j0 + 1
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + j0 + 1
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        w0 = k0 * n + i0
        w1 = k0 * n + i0 + 1

        U_arr[a, col] += 1; V_arr[e, col] += 1; W_arr[w0, col] += 1
        col += 1
        U_arr[b, col] += 1; V_arr[g, col] += 1; W_arr[w0, col] += 1
        col += 1
        U_arr[c, col] += 1; V_arr[e, col] += 1; W_arr[w1, col] += 1
        col += 1
        U_arr[d, col] += 1; V_arr[g, col] += 1; W_arr[w1, col] += 1
        col += 1

    for k0 in (0, 2, 4):
        strassen_222(0, 0, k0)
        strassen_222(0, 2, k0)
    school_221(0, 0, 6)
    school_221(0, 2, 6)

    result = {
        "U": [list(row) for row in U_arr],
        "V": [list(row) for row in V_arr],
        "W": [list(row) for row in W_arr],
    }
    return result
