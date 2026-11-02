"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = _start_time + 140.0

def _remaining():
    return _DEADLINE - time.time()

def _build_strassen_2x2x2_block():
    """Return (U,V,W) lists for a single 2x2x2 Strassen block.
    U: 4x7, V: 4x7, W: 4x7 (W indexed by p*n = 2*2=4, layout W[k*n+i] -> here n=2, so W[2*j + i]).
    We treat A as 2x2 (A[i,k] with i in {0,1} row, k in {0,1} col) => U indexed by n*m=4 as A[row][col] -> flat row-major: index = i*m + k = i*2 + k.
    B is 2x2 (B[k,j]) => V indexed by m*p=4 as B[row][col] = k*p + j = k*2 + j.
    C is 2x2 (C[i,j]) => W indexed by p*n=4 as W[j*n + i] = j*2 + i.
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    # A00=U[0], A01=U[1], A10=U[2], A11=U[3]
    # B00=V[0], B01=V[1], B10=V[2], B11=V[3]
    # C00=W[0], C01=W[1], C10=W[2], C11=W[3]
    def A(i,k): return i*m + k
    def B(k,j): return k*p + j
    def C(i,j): return j*n + i
    r = 0
    # P1 = (A00+A11)(B00+B11)
    U[A(0,0)][r] += 1; U[A(1,1)][r] += 1
    V[B(0,0)][r] += 1; V[B(1,1)][r] += 1
    W[C(0,0)][r] += 1
    W[C(1,1)][r] += 1
    r += 1
    # P2 = (A10+A11)(B00)
    U[A(1,0)][r] += 1; U[A(1,1)][r] += 1
    V[B(0,0)][r] += 1
    W[C(1,0)][r] += 1
    W[C(1,1)][r] += -1
    r += 1
    # P3 = A00 (B01 - B11)
    U[A(0,0)][r] += 1
    V[B(0,1)][r] += 1; V[B(1,1)][r] += -1
    W[C(0,1)][r] += 1
    r += 1
    # P4 = A11 (B10 - B00)
    U[A(1,1)][r] += 1
    V[B(1,0)][r] += 1; V[B(0,0)][r] += -1
    W[C(1,0)][r] += 1
    W[C(0,0)][r] += 1
    r += 1
    # P5 = (A00+A01) B11
    U[A(0,0)][r] += 1; U[A(0,1)][r] += 1
    V[B(1,1)][r] += 1
    W[C(0,1)][r] += 1
    r += 1
    # P6 = (A10 - A00)(B00+B01)
    U[A(1,0)][r] += 1; U[A(0,0)][r] += -1
    V[B(0,0)][r] += 1; V[B(0,1)][r] += 1
    W[C(1,1)][r] += 1
    r += 1
    # P7 = (A01 - A11)(B10+B11)
    U[A(0,1)][r] += 1; U[A(1,1)][r] += -1
    V[B(1,0)][r] += 1; V[B(1,1)][r] += 1
    W[C(0,0)][r] += 1
    r += 1
    return U, V, W

def _build_trivial_block(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    r = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i*m + k][r] = 1
                V[k*p + j][r] = 1
                W[j*n + i][r] = 1
                r += 1
    return U, V, W

def _concat_blocks(blocks):
    """blocks: list of (U,V,W). Concatenate along rank axis."""
    total = 0
    for U,V,W in blocks:
        total += len(U[0])
    n = len(blocks[0][2][0]) // 1  # placeholder
    return total

def _verify_decomposition(U, V, W, n, m, p):
    """Check that sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] == 1 iff (i,j) reachable... 
    Actually the matrix multiplication tensor T_{i,k,j} (A index (i,k), B index (k,j), C index (j,i)) must equal
    sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r].
    The true tensor value is 1 when the indices are consistent for the product: T is nonzero (=1) exactly for each (i,k,j) triple? 
    For matrix mult, T[i,k;j] = delta... The tensor M_{n,m,p} has T_{i,k,j} = 1 for all i,k,j? No.
    The multiplication tensor: C[i,j] = sum_k A[i,k] B[k,j]. So the tensor indexed by (i,k) for A, (k,j) for B, (j,i) for C:
    T[(i,k),(k,j),(j,i)] should be 1 for all i,k,j (it's the coefficient of A[i,k] B[k,j] in C[i,j]).
    So we need sum_r U[i*m+k,r] V[k*p+j,r] W[j*n+i,r] == 1 for ALL i in [n], k in [m], j in [p].
    """
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the contraction: for each r, outer product of reshaped U (n x m), V (m x p), W (p x n)
    # sum_r U_r[i,k] V_r[k,j] W_r[j,i] should equal 1 for all i,k,j.
    acc = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        # (i,k) * (k,j) * (j,i): sum over k,j of ... no, it's a full 3-tensor product
        # acc[i,k,j] += Ur[i,k] * Vr[k,j] * Wr[j,i]
        # Compute via: for each i,k: Ur[i,k]; for each j: Vr[k,j]*Wr[j,i]
        for i in range(n):
            for k in range(m):
                uik = Ur[i, k]
                if uik == 0:
                    continue
                for j in range(p):
                    acc[i, k, j] += uik * Vr[k, j] * Wr[j, i]
    # Should be all ones
    if not np.allclose(acc, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2,2,15
    import numpy as np

    # Fallback: known rank-53 solution (7 Strassen blocks + 1 trivial single-column block)
    def _fallback_53():
        rank = 53
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        def add_strassen_block(base_r, k0, k1):
            r1 = base_r; r2 = base_r+1; r3 = base_r+2; r4 = base_r+3
            r5 = base_r+4; r6 = base_r+5; r7 = base_r+6
            def A(i,k): return i*m + k
            def B(k,j): return k*p + j
            def C(i,j): return j*n + i
            # P1 (A00+A11)(B0k0+B1k1)
            U[A(0,0)][r1] += 1; U[A(1,1)][r1] += 1
            V[B(0,k0)][r1] += 1; V[B(1,k1)][r1] += 1
            W[C(0,k0)][r1] += 1
            W[C(1,k1)][r1] += 1
            # P2 (A10+A11)(B0k0)
            U[A(1,0)][r2] += 1; U[A(1,1)][r2] += 1
            V[B(0,k0)][r2] += 1
            W[C(1,k0)][r2] += 1
            W[C(1,k1)][r2] += -1
            # P3 A00 (B0k1 - B1k1)
            U[A(0,0)][r3] += 1
            V[B(0,k1)][r3] += 1; V[B(1,k1)][r3] += -1
            W[C(0,k1)][r3] += 1
            # P4 A11 (B1k0 - B0k0)
            U[A(1,1)][r4] += 1
            V[B(1,k0)][r4] += 1; V[B(0,k0)][r4] += -1
            W[C(1,k0)][r4] += 1
            W[C(0,k0)][r4] += 1
            # P5 (A00+A01) B1k1
            U[A(0,0)][r5] += 1; U[A(0,1)][r5] += 1
            V[B(1,k1)][r5] += 1
            W[C(0,k1)][r5] += 1
            # P6 (A10-A00)(B0k0+B0k1)
            U[A(1,0)][r6] += 1; U[A(0,0)][r6] += -1
            V[B(0,k0)][r6] += 1; V[B(0,k1)][r6] += 1
            W[C(1,k1)][r6] += 1
            # P7 (A01-A11)(B1k0+B1k1)
            U[A(0,1)][r7] += 1; U[A(1,1)][r7] += -1
            V[B(1,k0)][r7] += 1; V[B(1,k1)][r7] += 1
            W[C(0,k0)][r7] += 1
        for i in range(7):
            add_strassen_block(7*i, 2*i, 2*i+1)
        # trivial block for column 14
        base_r = 49
        for idx, (i, k) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            r = base_r + idx
            j = 14
            U[i*m + k][r] = 1
            V[k*p + j][r] = 1
            W[j*n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    best = _fallback_53()
    best_rank = 53

    # Attempt 1: rank-10 M_{2,2,3} then tensor with scalar x5 -> rank 50
    def _try_m223_rank10():
        # M_{2,2,3}: A 2x2, B 2x3, C 2x3
        # We attempt a structured search. Start from rank 11 (Strassen 2 cols + trivial 1 col) and try to merge.
        # Build rank-11 base:
        n3, m3, p3 = 2, 2, 3
        U3 = [[0]*11 for _ in range(n3*m3)]
        V3 = [[0]*11 for _ in range(m3*p3)]
        W3 = [[0]*11 for _ in range(p3*n3)]
        def A3(i,k): return i*m3 + k
        def B3(k,j): return k*p3 + j
        def C3(i,j): return j*n3 + i
        # Strassen on columns 0,1
        base = 0
        r1=base;r2=base+1;r3=base+2;r4=base+3;r5=base+4;r6=base+5;r7=base+6
        k0,k1=0,1
        U3[A3(0,0)][r1]+=1;U3[A3(1,1)][r1]+=1
        V3[B3(0,k0)][r1]+=1;V3[B3(1,k1)][r1]+=1
        W3[C3(0,k0)][r1]+=1;W3[C3(1,k1)][r1]+=1
        U3[A3(1,0)][r2]+=1;U3[A3(1,1)][r2]+=1
        V3[B3(0,k0)][r2]+=1
        W3[C3(1,k0)][r2]+=1;W3[C3(1,k1)][r2]+=-1
        U3[A3(0,0)][r3]+=1
        V3[B3(0,k1)][r3]+=1;V3[B3(1,k1)][r3]+=-1
        W3[C3(0,k1)][r3]+=1
        U3[A3(1,1)][r4]+=1
        V3[B3(1,k0)][r4]+=1;V3[B3(0,k0)][r4]+=-1
        W3[C3(1,k0)][r4]+=1;W3[C3(0,k0)][r4]+=1
        U3[A3(0,0)][r5]+=1;U3[A3(0,1)][r5]+=1
        V3[B3(1,k1)][r5]+=1
        W3[C3(0,k1)][r5]+=1
        U3[A3(1,0)][r6]+=1;U3[A3(0,0)][r6]+=-1
        V3[B3(0,k0)][r6]+=1;V3[B3(0,k1)][r6]+=1
        W3[C3(1,k1)][r6]+=1
        U3[A3(0,1)][r7]+=1;U3[A3(1,1)][r7]+=-1
        V3[B3(1,k0)][r7]+=1;V3[B3(1,k1)][r7]+=1
        W3[C3(0,k0)][r7]+=1
        # trivial for column 2
        for idx,(i,k) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            r = 7+idx; j=2
            U3[i*m3+k][r]=1; V3[k*p3+j][r]=1; W3[j*n3+i][r]=1
        # Verify rank-11 base
        if not _verify_decomposition(U3,V3,W3,n3,m3,p3):
            return None
        # Try to reduce to rank 10 by finding a linear dependency / null vector among the 11 rank terms
        # Build matrix of the 11 tensors as vectors in R^{n3*m3*p3=12}
        import numpy as np
        vecs = []
        for r in range(11):
            v = np.zeros(n3*m3*p3)
            Ur = np.array(U3[:,r]).reshape(n3,m3)
            Vr = np.array(V3[:,r]).reshape(m3,p3)
            Wr = np.array(W3[:,r]).reshape(p3,n3)
            for i in range(n3):
                for k in range(m3):
                    for j in range(p3):
                        v[(i*m3+k)*p3 + j] = Ur[i,k]*Vr[k,j]*Wr[j,i]
            vecs.append(v)
        M = np.array(vecs).T  # 12 x 11
        # Find null space: M @ x = 0
        try:
            Usv, s, Vt = np.linalg.svd(M)
        except Exception:
            return None
        tol = 1e-7
        # rank of M
        rankM = int(np.sum(s > tol))
        if rankM < 11:
            # null vector
            nv = Vt[-1]
            # try dropping the component with largest magnitude
            idx_drop = int(np.argmax(np.abs(nv)))
            # new rank-10: keep all except idx_drop, but we need the sum of kept * nv_kept to still give the all-ones tensor
            # Actually if M @ nv = 0, then sum_r nv[r] T_r = 0, so the T_r are dependent. Dropping one term requires that term to be a combo of others, which the null vector gives only if we can scale.
            # Simpler: the tensor = sum_r T_r (all coeff 1). We found dependence sum nv[r] T_r = 0.
            # This doesn't directly let us drop a term unless nv has a zero somewhere or we can re-express.
            # We'll just accept rank 11 for M_{2,2,3} and not claim 10.
        return None  # cannot reliably get rank 10; skip

    _try_m223_rank10()

    # Attempt 2: direct structured search for M_{2,2,15} below 53 using a "sliding window" ansatz
    # Decompose 15 columns as overlapping windows. We'll attempt a rank-50 construction:
    # 5 windows of 3 columns (0-2,3-5,...,12-14) each with a rank-10 M_{2,2,3} would be 50 if rank-10 exists.
    # Since we couldn't confirm rank-10, instead try: 4 Strassen blocks (cols 0-1,2-3,4-5,6-7) = 28, 
    # then cols 8-14 (7 cols) as 3 Strassen (8-9,10-11,12-13)=21 + col14 trivial=4 -> total 28+21+4=53. No gain.
    # 
    # Try a genuinely different tiling: treat as M_{2,2,15} = M_{2,2,3} x M_{1,1,5} tensor product.
    # If we had rank-10 M_{2,2,3}, rank would be 50. Without it, use rank-11 M_{2,2,3} -> 55 (worse).
    # 
    # So the realistic best remains 53. Attempt a local search to see if 52 is possible via merging two blocks,
    # but this is heuristic; skip to save time.

    # Validate the fallback
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        # Should not happen; but if it does, return trivial rank 60
        U,V,W = _build_trivial_block(n,m,p)
        return {"U": U, "V": V, "W": W}

    return best

# EVOLVE-BLOCK-END
