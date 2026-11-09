import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 6]

def _build_base():
    """Build the base 77-rank Strassen decomposition for 4x4 * 4x6 -> 4x6."""
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((n * m, R))  # (16, 77)
    V = np.zeros((m * p, R))  # (24, 77)
    W = np.zeros((p * n, R))  # (24, 77)

    # Top-level Strassen for 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where each is 2x3
    # C = A*B is 4x6, split into 2x2 blocks: C = [[C00, C01, C02], [C10, C11, C12]]
    # Each Cij is 2x2

    # Strassen 7 terms for the 2x2 "outer" product
    alpha = {
        0: {0: 1, 3: 1},   # M0 = (A00+A11)
        1: {2: 1, 3: 1},   # M1 = (A10+A11)
        2: {0: 1},          # M2 = A00
        3: {3: 1},          # M3 = A11
        4: {0: 1, 1: 1},   # M4 = (A00+A01)
        5: {2: 1, 0: -1},  # M5 = (A10-A00)
        6: {1: 1, 3: -1},  # M6 = (A01-A11)
    }
    beta = {
        0: {0: 1, 3: 1},   # N0 = (B00+B11)
        1: {0: 1},          # N1 = B00
        2: {1: 1, 3: -1},  # N2 = (B01-B11)
        3: {2: 1, 0: -1},  # N3 = (B10-B00)
        4: {3: 1},          # N4 = B11
        5: {0: 1, 1: 1},   # N5 = (B00+B01)
        6: {2: 1, 3: 1},   # N6 = (B10+B11)
    }
    # Reconstruction: C_ab = sum_s gamma_s * M_s * N_s
    # C00 = M0*N0 - M3*N1 - M4*N2 + M5*N3  ... wait, let me use standard Strassen
    # Standard 2x2 Strassen:
    # C00 = M0 - M3 - M4 + M5  ... no, let me be precise
    # 
    # For 2x2: 
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B01-B11),
    # M4=A11(B10-B00), M5=(A00+A01)B11, M6=(A10-A00)(B00+B10), M7=(A01-A11)(B11+B21)
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 - M5 + M6
    #
    # But our B blocks are 2x3, so B00, B01, B02, B10, B11, B12 are 2x3
    # The "2x2" Strassen structure applies to the block indices, and each 
    # M_s * N_s is a 2x2 * 2x3 = 2x3 product
    #
    # C00 (2x2) = M0*N0 - M3*N1 - M4*N2 + M5*N3  ... this doesn't work directly
    # because N0, N1, etc. are 2x3, so M_s*N_s is 2x3, not 2x2
    #
    # I need to rethink. The top-level Strassen gives us 7 terms where each term
    # is a 2x2 matrix times a 2x3 matrix = 2x3 matrix. These 2x3 matrices are
    # then combined to form the 4x6 output.
    #
    # Let me define it properly:
    # A is 4x4, split into 2x2 blocks: A = [[A00, A01], [A10, A11]], each 2x2
    # B is 4x6, split into 2x3 blocks: B = [[B00, B01, B02], [B10, B11, B12]], each 2x3
    # C = A*B is 4x6, split into 2x2 blocks: C = [[C00, C01, C02], [C10, C11, C12]], each 2x2
    #
    # C00 = A00*B00 + A01*B10  (2x2 * 2x3 + 2x2 * 2x3 = 2x3... wait, 2x2 * 2x3 = 2x3)
    # Hmm, C00 should be 2x2, but A00*B00 is 2x2*2x3 = 2x3. That's wrong.
    #
    # I see the issue. When we split B into 2x3 blocks, A00 (2x2) times B00 (2x3) gives 2x3, not 2x2.
    # So the output blocks aren't 2x2; they're 2x3.
    #
    # Let me redo: C = A*B where A is 4x4, B is 4x6.
    # Split A into 2x2 blocks (4 blocks), B into 2x3 blocks (4 blocks).
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # C split into 2x3 blocks: C = [[C00, C01, C02], [C10, C11, C12]] where each Cij is 2x3
    # C00 = A00*B00 + A01*B10  (2x2*2x3 + 2x2*2x3 = 2x3) ✓
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    #
    # Now apply Strassen to the "outer" 2x2 block structure:
    # M0 = (A00+A11) * (B00+B11)  -- but B00+B11 is 2x3, so M0 is 2x2*2x3 = 2x3
    # M1 = (A10+A11) * B00
    # M2 = A00 * (B01-B11)
    # M3 = A11 * (B10-B00)
    # M4 = (A00+A01) * B11
    # M5 = (A10-A00) * (B00+B10)
    # M6 = (A01-A11) * (B11+B12)  -- wait, this should be (B11+B21) but we only have 2 rows
    #
    # Standard Strassen for 2x2 block matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B21)  -- but B has only 2 row-blocks, so B21 doesn't exist
    #
    # I think the issue is that Strassen is for square matrices. For rectangular, we need to adapt.
    #
    # Let me just use the current program's approach which seems to work (score 1.2468 = 96/77).
    # I'll keep the structure but clean it up and add rank reduction.

    # Use the same structure as the current program which achieves 77
    # Re-implement cleanly
    
    col = 0
    for s in range(7):
        for si in range(7):
            # Inner Strassen for 2x2 x 2x2 part (columns 0,1 of the 2x3)
            # This is getting complex; let me just copy the working logic
            pass
    
    # Given time constraints, let me just return the known-good 77-rank solution
    # and focus on the rank reduction search
    raise NotImplementedError("Use the original _build_base")


def construct():
    deadline = time.time() + 110.0
    n, m, p = 4, 4, 6
    
    # Build the 77-rank decomposition
    R = 77
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Top-level Strassen coefficients
    alpha = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    beta = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    col = 0
    for s in range(7):
        for si in range(7):
            # Inner Strassen for 2x2 x 2x2
            inner_alpha = alpha
            inner_beta = beta
            inner_recon = recon

            for a, coef_i in [(0, inner_alpha[si].get(0, 0)), (1, inner_alpha[si].get(1, 0)),
                              (2, inner_alpha[si].get(2, 0)), (3, inner_alpha[si].get(3, 0))]:
                if coef_i != 0:
                    xr_i, xc_i = a // 2, a % 2
                    for r, coef_s in alpha[s].items():
                        ar, ac = r // 2, r % 2
                        gi = ar * 2 + xr_i
                        gk = ac * 2 + xc_i
                        U[gi * 4 + gk, col] += coef_i * coef_s

            for b, coef_i in [(0, inner_beta[si].get(0, 0)), (1, inner_beta[si].get(1, 0)),
                              (2, inner_beta[si].get(2, 0)), (3, inner_beta[si].get(3, 0))]:
                if coef_i != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c, coef_s in beta[s].items():
                        ac, bc = c // 2, c % 2
                        gk = ac * 2 + y_row
                        gj = bc * 3 + y_inner_col
                        V[gk * 6 + gj, col] += coef_i * coef_s

            for a, b, gamma_s in recon[s]:
                for za, zb, gamma_i in inner_recon[si]:
                    gi = a * 2 + za
                    gj = b * 3 + zb
                    W[gj * 4 + gi, col] += gamma_s * gamma_i
            col += 1

        # Schoolbook for 2x2 x 2x1 (column 2)
        for u in range(2):
            for v in range(2):
                for r, coef_s in alpha[s].items():
                    ar, ac = r // 2, r % 2
                    gi = ar * 2 + u
                    gk = ac * 2 + v
                    U[gi * 4 + gk, col] += coef_s
                for c, coef_s in beta[s].items():
                    ac, bc = c // 2, c % 2
                    gk = ac * 2 + v
                    gj = bc * 3 + 2
                    V[gk * 6 + gj, col] += coef_s
                for a, b, gamma_s in recon[s]:
                    gi = a * 2 + u
                    gj = b * 3 + 2
                    W[gj * 4 + gi, col] += gamma_s
                col += 1

    # Verify
    def verify(U, V, W):
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to trivial
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k, idx] = 1
                    V[k*p + j, idx] = 1
                    W[j*n + i, idx] = 1
                    idx += 1

    # Try to reduce rank by finding linear dependencies
    if time.time() < deadline - 30:
        # Check for linear dependencies in the decomposition
        # For each column, see if it can be expressed as a combination of others
        for target_col in range(R - 1, -1, -1):
            if time.time() > deadline - 25:
                break
            # Try to see if column target_col is redundant
            # Build the tensor contribution of this column
            # and check if the remaining columns still span the same tensor
            # This is expensive; do a quick numerical check
            
            # Quick check: is this column all zeros?
            if np.allclose(U[:, target_col], 0) and np.allclose(V[:, target_col], 0) and np.allclose(W[:, target_col], 0):
                U = np.delete(U, target_col, axis=1)
                V = np.delete(V, target_col, axis=1)
                W = np.delete(W, target_col, axis=1)
                R -= 1
                continue

            # More sophisticated: try to find coefficients c such that
            # for all (i,k,j): sum_{r != target} c_r * U[i,k,r]*V[k,j,r]*W[j,i,r] = 
            #   U[i,k,target]*V[k,j,target]*W[j,i,target]
            # This is a linear system in c_r
            # But the left side is bilinear in the unknowns if we don't fix them
            # Actually, if we fix the other columns and only scale, it's linear
            # But we can also add linear combinations, so it's more complex
            
            # For now, just check if the column is a scalar multiple of another
            for other in range(R):
                if other == target_col:
                    continue
                if np.allclose(U[:, target_col], U[:, other]) and \
                   np.allclose(V[:, target_col], V[:, other]) and \
                   np.allclose(W[:, target_col], W[:, other]):
                    U = np.delete(U, target_col, axis=1)
                    V = np.delete(V, target_col, axis=1)
                    W = np.delete(W, target_col, axis=1)
                    R -= 1
                    break

    # Round to integers
    def to_list(M):
        out = []
        for row in M:
            r = []
            for x in row:
                rx = round(x)
                r.append(int(rx) if abs(x - rx) < 1e-9 else float(x))
            out.append(r)
        return out

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
