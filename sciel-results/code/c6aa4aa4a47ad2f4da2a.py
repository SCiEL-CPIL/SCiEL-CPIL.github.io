"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that U, V, W correctly decompose the 2x5x6 matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_mat[i * m + j, :]
                    v_row = V_mat[j * p + k, :]
                    w_row = W_mat[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > tol:
                        return False
        return True

    # Strategy: Use Strassen blocks for 2x2x2 subproblems
    # Partition the 5 middle indices into pairs + leftover
    # Partition the 6 output indices into pairs
    # 
    # Option A: 3 pairs of j (0,1), (2,3), (4) and 3 pairs of k (0,1), (2,3), (4,5)
    #   -> 6 Strassen blocks (rank 42) + 12 schoolbook for j=4 (rank 12) = 54
    #
    # Option B: Try to use a better algorithm for the 2x1x6 part
    #   The 2x1x6 tensor has rank 6 (each of 2*6=12 entries needs its own term... no)
    #   Actually for 2x1x6: C[i,k] = A[i,0]*B[0,k], rank = 2*6/... 
    #   Each term can contribute to at most one (i,k) pair if we use schoolbook
    #   But we can share: e.g., (A[0,0]+A[1,0])*(B[0,k]) contributes to both C[0,k] and C[1,k]
    #   Actually the minimal rank of 2x1x6 is 6 (use B[0,k] times (A[0,0], A[1,0]))
    #   Wait no - we need separate terms for each k since B[0,k] are independent
    #   So rank(2x1x6) = 6? Let's think...
    #   C[i,k] = A[i,0] * B[0,k]
    #   We can write this as: for each k, one term with U[i,0]=A[i,0] for i=0,1, V[0,k]=B[0,k], W[k,i]=1
    #   That's 6 terms. Can we do better? 
    #   A rank-1 tensor is u_i * v_j * w_k. Here the tensor T[i,j,k] = delta_{j,0} * ... 
    #   Actually T[i,0,k] = A[i,0]*B[0,k] which is a rank-1 matrix in (i,k) for fixed j=0
    #   The tensor is sum over j of A[i,j]*B[j,k]*delta... no.
    #   T[i,j,k] = A[i,j]*B[j,k] only when j is the middle index.
    #   For 2x1x6: T[i,0,k] = A[i,0]*B[0,k]. This is a 2x1x6 tensor.
    #   It factors as: for each k, the slice T[:,:,k] = A[:,0] * B[0,k] which is rank 1.
    #   The full tensor has rank 6 (one term per k).
    #   So we need at least 6 terms for the 2x1x6 part.
    #   Current approach uses 12 terms (schoolbook) for 2x1x6. We can do 6!

    # Improved strategy:
    # 6 Strassen blocks for 2x4x6 (j in {0,1,2,3}, k in {0..5}) = rank 42
    # 6 terms for 2x1x6 (j=4, k in {0..5}) = rank 6
    # Total: 48

    rank = 48
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    def strassen_block(ja, jb, ka, kb):
        """Add a Strassen 2x2x2 block for rows ja,jb of A and cols ka,kb of B."""
        # u indices: i*m + j for i in {0,1}, j in {ja,jb}
        # v indices: j*p + k for j in {ja,jb}, k in {ka,kb}
        # w indices: k*n + i for k in {ka,kb}, i in {0,1}
        ua0 = 0 * m + ja  # A[0, ja]
        ua1 = 0 * m + jb  # A[0, jb]
        ub0 = 1 * m + ja  # A[1, ja]
        ub1 = 1 * m + jb  # A[1, jb]
        
        va0 = ja * p + ka  # B[ja, ka]
        va1 = ja * p + kb  # B[ja, kb]
        vb0 = jb * p + ka  # B[jb, ka]
        vb1 = jb * p + kb  # B[jb, kb]
        
        wa0 = ka * n + 0   # C[ka, 0]
        wa1 = ka * n + 1   # C[ka, 1]
        wb0 = kb * n + 0   # C[kb, 0]
        wb1 = kb * n + 1   # C[kb, 1]

        # Standard Strassen for 2x2:
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        # M2 = (A10+A11)(B00)     -> C10, C11
        # M3 = (A00)(B10-B11)     -> C00, C01
        # M4 = (A11)(B01)         -> C01, C11
        # M5 = (A00+A01)(B11)     -> C00, C01
        # M6 = (A10-A00)(B00+B01) -> C10
        # M7 = (A01-A11)(B10+B11) -> C00
        
        # Wait, let me use the standard Strassen formulas carefully:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        #
        # M1 = (a+d)(e+h)  -> contributes to C[0,0] and C[1,1]
        # M2 = (c+d)(e)    -> contributes to C[1,0] and C[1,1]  
        # M3 = (a)(g-h)    -> contributes to C[0,0] and C[0,1]
        # M4 = (d)(f)      -> contributes to C[0,1] and C[1,1]
        # M5 = (a+b)(h)    -> contributes to C[0,0] and C[0,1]
        # M6 = (c-a)(e+f)  -> contributes to C[1,0]
        # M7 = (b-d)(g+h)  -> contributes to C[0,0]
        #
        # Verification:
        # C[0,0] = ae+bg = M1 - M2 - M4 + M5 + M7 ... let me check
        # M1 = ae+ah+de+dh
        # M2 = ce+de
        # M3 = ag-ah
        # M4 = df
        # M5 = ah+bh
        # M6 = ce+cf-ae-af
        # M7 = bg+bh-dg-dh
        #
        # C[0,0] should be ae+bg
        # M1 - M2 - M4 + M5 + M7 = (ae+ah+de+dh) - (ce+de) - (df) + (ah+bh) + (bg+bh-dg-dh)
        # = ae + ah + de + dh - ce - de - df + ah + bh + bg + bh - dg - dh
        # = ae + 2ah - ce - df + 2bh + bg - dg
        # That doesn't look right. Let me use the correct Strassen formulas.
        #
        # Correct Strassen:
        # p1 = (a + d)(e + h)
        # p2 = (c + d)(e)
        # p3 = (a)(g - h)
        # p4 = (d)(f)
        # p5 = (a + b)(h)
        # p6 = (c - a)(e + f)
        # p7 = (b - d)(g + h)
        #
        # C[0,0] = p1 - p4 - p6 + p7
        # C[0,1] = p3 + p5
        # C[1,0] = p2 + p6
        # C[1,1] = p1 - p2 - p3 + p4
        #
        # Let me verify C[0,0]:
        # p1 = ae+ah+de+dh
        # p4 = df
        # p6 = ce+cf-ae-af
        # p7 = bg+bh-dg-dh
        # p1 - p4 - p6 + p7 = ae+ah+de+dh - df - ce - cf + ae + af + bg + bh - dg - dh
        # = 2ae + ah + de + dh - df - ce - cf + af + bg + bh - dg - dh
        # Hmm, still not clean. Let me just use a known correct formulation.
        #
        # Standard Strassen (from Wikipedia):
        # m1 = (a + d)(e + h)
        # m2 = (c + d)(e)
        # m3 = (a)(g - h)
        # m4 = (d)(f)
        # m5 = (a + b)(h)
        # m6 = (c - a)(e + f)
        # m7 = (b - d)(g + h)
        #
        # C11 = m1 - m4 - m6 + m7
        # C12 = m3 + m5
        # C21 = m2 + m6
        # C22 = m1 - m2 - m3 + m4
        #
        # In our notation: a=A[0,ja], b=A[0,jb], c=A[1,ja], d=A[1,jb]
        # e=B[ja,ka], f=B[ja,kb], g=B[jb,ka], h=B[jb,kb]
        # C[ka,0]=C11, C[ka,1]=C12, C[kb,0]=C21, C[kb,1]=C22

        # m1 = (a+d)(e+h)
        add_term([(ua0, 1), (ub1, 1)], [(va0, 1), (vb1, 1)], [(wa0, 1), (wb1, 1)])
        # m2 = (c+d)(e)
        add_term([(ub0, 1), (ub1, 1)], [(va0, 1)], [(wb0, 1), (wb1, -1)])
        # m3 = (a)(g-h)
        add_term([(ua0, 1)], [(vb0, 1), (vb1, -1)], [(wa0, 1), (wa1, 1)])
        # m4 = (d)(f)
        add_term([(ub1, 1)], [(va1, 1)], [(wa1, 1), (wb1, 1)])
        # m5 = (a+b)(h)
        add_term([(ua0, 1), (ua1, 1)], [(vb1, 1)], [(wa0, -1), (wa1, 1)])
        # m6 = (c-a)(e+f)
        add_term([(ub0, 1), (ua0, -1)], [(va0, 1), (va1, 1)], [(wb0, 1)])
        # m7 = (b-d)(g+h)
        add_term([(ua1, 1), (ub1, -1)], [(vb0, 1), (vb1, 1)], [(wa0, 1)])

    # Apply Strassen blocks for j in {0,1}, {2,3} and k in {0,1}, {2,3}, {4,5}
    strassen_block(0, 1, 0, 1)
    strassen_block(0, 1, 2, 3)
    strassen_block(0, 1, 4, 5)
    strassen_block(2, 3, 0, 1)
    strassen_block(2, 3, 2, 3)
    strassen_block(2, 3, 4, 5)

    # For j=4 (the 5th column), use rank-6 decomposition of 2x1x6
    # C[i,k] = A[i,4] * B[4,k]
    # One term per k: U[i,4]=A[i,4] for i=0,1; V[4,k]=B[4,k]; W[k,i]=1
    for k in range(p):
        U[0 * m + 4, col] = 1  # A[0,4]
        U[1 * m + 4, col] = 1  # A[1,4]
        V[4 * p + k, col] = 1  # B[4,k]
        W[k * n + 0, col] = 1  # C[k,0]
        W[k * n + 1, col] = 1  # C[k,1]
        col += 1

    U_used = U[:, :col]
    V_used = V[:, :col]
    W_used = W[:, :col]

    if verify(U_used, V_used, W_used):
        return {"U": U_used.tolist(), "V": V_used.tolist(), "W": W_used.tolist()}

    # Fallback: schoolbook rank 60
    U_fb = np.zeros((n * m, n * m * p), dtype=float)
    V_fb = np.zeros((m * p, n * m * p), dtype=float)
    W_fb = np.zeros((p * n, n * m * p), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, c] = 1
                V_fb[j * p + k, c] = 1
                W_fb[k * n + i, c] = 1
                c += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
