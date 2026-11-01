"""Matrix multiplication 2x3x7: hybrid Strassen/Schoolbook decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 2, 3, 7

    def to_list(mat):
        return mat.tolist()

    def verify_full(U, V, W):
        """Verify exact decomposition over all entries (integer arithmetic)."""
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]:
            return False
        # Required entries (i,j,k) -> 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = int(np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]))
                    if actual != 1:
                        return False
        # Forbidden entries -> 0
        for a in range(n * m):
            i1, j1 = divmod(a, m)
            for b in range(m * p):
                j2, k1 = divmod(b, p)
                if j1 != j2:
                    continue
                for c in range(p * n):
                    k2, i2 = divmod(c, n)
                    if i1 == i2 and k1 == k2:
                        continue
                    actual = int(np.sum(U[a, :] * V[b, :] * W[c, :]))
                    if actual != 0:
                        return False
        return True

    # ---- Strassen 2x2x2 decomposition (rank 7) ----
    # M[i,j,k] = A[i,j]*B[j,k]
    # Strassen: M = U*V*W where
    #   U[i,j,r]:  r0: A00+A11, r1: A10+A11, r2: A00, r3: A01, r4: B00-B10, r5: B01-B11, r6: A11
    #   V[j,k,r]:  r0: B00,     r1: B11,     r2: B00+B11, r3: B01, r4: A10, r5: A01, r6: B10+B11
    #   W[k,i,r]:  r0: W00-W11, r1: W10,     r2: W11,     r3: W01, r4: W00, r5: W01, r6: W10-W00
    def strassen_222(U_base, V_base, W_base, i_off, j_off, k_off):
        """Add Strassen rank-7 contribution for 2x2x2 block at offset."""
        # U rows: (i0,j0)=base+0, (i0,j1)=base+1, (i1,j0)=base+2, (i1,j1)=base+3
        # V rows: (j0,k0)=base+0, (j0,k1)=base+1, (j1,k0)=base+2, (j1,k1)=base+3
        # W rows: (k0,i0)=base+0, (k0,i1)=base+1, (k1,i0)=base+2, (k1,i1)=base+3
        u00, u01, u10, u11 = U_base, U_base+1, U_base+2, U_base+3
        v00, v01, v10, v11 = V_base, V_base+1, V_base+2, V_base+3
        w00, w01, w10, w11 = W_base, W_base+1, W_base+2, W_base+3

        # r0: (A00+A11)*B00*(W00-W11)
        U[u00, 0] += 1; U[u11, 0] += 1
        V[v00, 0] += 1
        W[w00, 0] += 1; W[w11, 0] -= 1
        # r1: (A10+A11)*B11*W10
        U[u10, 1] += 1; U[u11, 1] += 1
        V[v11, 1] += 1
        W[w10, 1] += 1
        # r2: A00*(B00+B11)*W11
        U[u00, 2] += 1
        V[v00, 2] += 1; V[v11, 2] += 1
        W[w11, 2] += 1
        # r3: A01*B01*W01
        U[u01, 3] += 1
        V[v01, 3] += 1
        W[w01, 3] += 1
        # r4: (B00-B10)*A10*W00
        V[v00, 4] += 1; V[v10, 4] -= 1
        U[u10, 4] += 1
        W[w00, 4] += 1
        # r5: (B01-B11)*A01*W01
        V[v01, 5] += 1; V[v11, 5] -= 1
        U[u01, 5] += 1
        W[w01, 5] += 1
        # r6: A11*(B10+B11)*(W10-W00)
        U[u11, 6] += 1
        V[v10, 6] += 1; V[v11, 6] += 1
        W[w10, 6] += 1; W[w00, 6] -= 1

    # ---- Build hybrid decomposition ----
    # Block 1: 2x2x3 (i=0,1; j=0,1; k=0,1,2) -> split into 2x2x2 (k=0,1) + 2x2x1 (k=2)
    # Block 2: 2x2x4 (i=0,1; j=2; k=0..3) -> split into 2x2x2 (k=0,1) + 2x2x2 (k=2,3)
    # Total: 7 (Strassen) + 2 (schoolbook for 2x2x1) + 7 (Strassen) + 7 (Strassen) = 23

    rank = 23
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Block 1a: Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1}
    # U rows: i*m+j -> 0,1,2,3 (j=0,1)
    # V rows: j*p+k -> 0,1 (j=0,k=0,1), 7,8 (j=1,k=0,1)
    # W rows: k*n+i -> 0,1 (k=0,i=0,1), 2,3 (k=1,i=0,1)
    strassen_222(U_base=0, V_base=0, W_base=0, i_off=0, j_off=0, k_off=0)
    # Fix: V_base for j=1 should be p=7, not 2
    # Redo: V rows for (j=0,k=0)=0, (j=0,k=1)=1, (j=1,k=0)=7, (j=1,k=1)=8
    # My strassen_222 assumed consecutive V rows. Let me handle non-consecutive.
    # Actually let me rebuild with explicit indices.

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    def strassen_explicit(u_idx, v_idx, w_idx, r_start):
        """u_idx = [u00,u01,u10,u11], v_idx = [v00,v01,v10,v11], w_idx = [w00,w01,w10,w11]"""
        u00, u01, u10, u11 = u_idx
        v00, v01, v10, v11 = v_idx
        w00, w01, w10, w11 = w_idx
        r = r_start
        U[u00, r] += 1; U[u11, r] += 1; V[v00, r] += 1; W[w00, r] += 1; W[w11, r] -= 1; r += 1
        U[u10, r] += 1; U[u11, r] += 1; V[v11, r] += 1; W[w10, r] += 1; r += 1
        U[u00, r] += 1; V[v00, r] += 1; V[v11, r] += 1; W[w11, r] += 1; r += 1
        U[u01, r] += 1; V[v01, r] += 1; W[w01, r] += 1; r += 1
        V[v00, r] += 1; V[v10, r] -= 1; U[u10, r] += 1; W[w00, r] += 1; r += 1
        V[v01, r] += 1; V[v11, r] -= 1; U[u01, r] += 1; W[w01, r] += 1; r += 1
        U[u11, r] += 1; V[v10, r] += 1; V[v11, r] += 1; W[w10, r] += 1; W[w00, r] -= 1; r += 1
        return r

    def schoolbook_221(u_idx, v_idx, w_idx, r_start):
        """2x2x1: two terms for (i0,j,k) and (i1,j,k) -- actually 2 terms for 2 rows x 1 col"""
        # 2x2x1 means 2 i-values, 1 j-value, 1 k-value -> 2 entries
        u0, u1 = u_idx  # two i-rows for same j
        v0 = v_idx[0]
        w0, w1 = w_idx  # two i-values for same k
        r = r_start
        U[u0, r] += 1; V[v0, r] += 1; W[w0, r] += 1; r += 1
        U[u1, r] += 1; V[v0, r] += 1; W[w1, r] += 1; r += 1
        return r

    # Block 1a: 2x2x2, i={0,1}, j={0,1}, k={0,1}
    # U: i*m+j: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V: j*p+k: (0,0)=0, (0,1)=1, (1,0)=7, (1,1)=8
    # W: k*n+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    r = strassen_explicit(
        u_idx=[0, 1, 2, 3],
        v_idx=[0, 1, 7, 8],
        w_idx=[0, 1, 2, 3],
        r_start=0
    )

    # Block 1b: 2x2x1, i={0,1}, j={1}, k={2}
    # U: i*m+1: (0,1)=1, (1,1)=3
    # V: 1*p+2 = 9
    # W: 2*n+i: (2,0)=4, (2,1)=5
    r = schoolbook_221(
        u_idx=[1, 3],
        v_idx=[9],
        w_idx=[4, 5],
        r_start=r
    )

    # Block 2a: 2x2x2, i={0,1}, j={2}, k={0,1}
    # U: i*m+2: (0,2)=4, (1,2)=5
    # V: 2*p+k: (2,0)=14, (2,1)=15, (2,0)=14, (2,1)=15 -> v00=14, v01=15, v10=14, v11=15
    # Wait, j is fixed at 2, so v_idx for 2x2x2 needs j in {j0,j1} but here j is single.
    # This is actually 2x1x2, not 2x2x2. Let me reconsider.
    # For j=2 only: it's 2x1x2 = 4 entries. Schoolbook would be 4 terms.
    # But I can still use a Strassen-like structure if I treat it as 2x2x2 with j1=j0.
    # Actually no -- Strassen requires 2x2 in all dimensions. For 2x1x2, just use schoolbook (4 terms).
    # Hmm, but that gives 7+2+4+4 = 17. Let me reconsider the split.

    # Better split: 
    # Block A: i={0,1}, j={0,1}, k={0,1,2} = 2x2x3 = 6 entries
    #   -> Strassen 2x2x2 (k=0,1) + schoolbook 2x2x1 (k=2) = 7 + 2 = 9
    # Block B: i={0,1}, j={2}, k={0,1,2,3,4,5,6} = 2x1x7 = 14 entries
    #   -> schoolbook = 14
    # Total: 9 + 14 = 23

    # Let me redo Block B as schoolbook (14 terms)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Block A: 2x2x3
    r = strassen_explicit(
        u_idx=[0, 1, 2, 3],
        v_idx=[0, 1, 7, 8],
        w_idx=[0, 1, 2, 3],
        r_start=0
    )
    r = schoolbook_221(
        u_idx=[1, 3],
        v_idx=[9],
        w_idx=[4, 5],
        r_start=r
    )

    # Block B: 2x1x7, schoolbook
    for i in range(2):
        for k in range(7):
            U[i * m + 2, r] = 1
            V[2 * p + k, r] = 1
            W[k * n + i, r] = 1
            r += 1

    assert r == rank, f"Expected {rank}, got {r}"

    # Verify
    if verify_full(U, V, W):
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Fallback: schoolbook rank 42
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=np.int64)
    V_sb = np.zeros((m * p, rank_sb), dtype=np.int64)
    W_sb = np.zeros((p * n, rank_sb), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
# EVOLVE-BLOCK-END
