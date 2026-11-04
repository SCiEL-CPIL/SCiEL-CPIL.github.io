"""Block-based rank-minimal tensor decomposition for 4x4 * 4x7 multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _build_block(n, m, p, algo):
    """Build U,V,W factor matrices (as dense lists) for an n x m * m x p product.
    Returns (U,V,W,rank). U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    W uses the trace convention W[k*n + i, r].
    """
    if algo == "schoolbook":
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return U, V, W, rank
    elif algo == "strassen222":
        # 2x2x2 Strassen, rank 7.
        assert (n, m, p) == (2, 2, 2)
        # Strassen products:
        # M1 = (A11-A22)(B21+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B11-B22)
        # M4 = A22(B22-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        # We need U (A factors), V (B factors), W (combine into C) with trace convention.
        # Index A[i,j] -> U index i*m+j ; B[j,k] -> V index j*p+k ; C[k,i] -> W index k*n+i.
        rank = 7
        U = np.zeros((4, rank), dtype=float)
        V = np.zeros((4, rank), dtype=float)
        W = np.zeros((4, rank), dtype=float)
        # A indices: A11=0, A12=1, A21=2, A22=3
        # B indices: B11=0, B12=1, B21=2, B22=3
        # C indices (trace convention C[k,i]): C11 -> k=0,i=0 ->0 ; C12 -> k=0,i=1 ->1 ; C21 -> k=1,i=0 ->2 ; C22 -> k=1,i=1 ->3
        # Each rank column corresponds to one Strassen product (product of a linear A-combo and B-combo).
        # Column 0: M1 = (A11-A22)(B21+B22)
        U[0,0]=1; U[3,0]=-1
        V[2,0]=1; V[3,0]=1
        W[0,0]=1; W[1,0]=0; W[2,0]=0; W[3,0]=0
        # C11 gets M1, C22 gets M1
        W[0,0]=1; W[3,0]=1
        # Column 1: M2 = (A21+A22)B11
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=0; W[0,1]=0; W[1,1]=0
        # C21 gets M2
        W[2,1]=1
        # Column 2: M3 = A11(B11-B22)
        U[0,2]=1
        V[0,2]=1; V[3,2]=-1
        W[1,2]=1; W[0,2]=0; W[2,2]=0; W[3,2]=0
        # C12 gets M3, C22 gets M3
        W[1,2]=1; W[3,2]=1
        # Column 3: M4 = A22(B22-B11)
        U[3,3]=1
        V[3,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1; W[1,3]=0; W[3,3]=0
        # C11 gets M4, C21 gets M4
        W[0,3]=1; W[2,3]=1
        # Column 4: M5 = (A11+A12)B22
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[1,4]=1; W[0,4]=0; W[2,4]=0; W[3,4]=0
        # C12 gets M5, C11 gets -M5
        W[1,4]=1; W[0,4]=-1
        # Column 5: M6 = (A21-A11)(B11+B12)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1; W[0,5]=0; W[1,5]=0; W[2,5]=0
        # C22 gets M6
        W[3,5]=1
        # Column 6: M7 = (A12-A22)(B21+B22)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1; W[3,6]=1; W[1,6]=0; W[2,6]=0
        # C11 gets M7, C22 gets M7
        W[0,6]=1; W[3,6]=1
        return U, V, W, rank
    else:
        raise ValueError("unknown algo")


def _combine_blocks(block_specs):
    """block_specs: list of (n_i, m_i, p_i, algo, row_offset_i, col_offset_j, out_offset_k).
    We decompose the 4x4 * 4x7 into blocks along rows (i: 0..4), cols (j:0..4), out (k:0..7).
    For each block we build local U,V,W and place into global matrices.
    Returns global U,V,W,rank.
    """
    n, m, p = DIMENSIONS
    total_rank = 0
    for (ni, mi, pi, algo, i0, j0, k0) in block_specs:
        U, V, W, r = _build_block(ni, mi, pi, algo)
        total_rank += r
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    col = 0
    for (ni, mi, pi, algo, i0, j0, k0) in block_specs:
        U, V, W, r = _build_block(ni, mi, pi, algo)
        # U local index (i*m + j) -> global ( (i0+i)*m + (j0+j) )
        for i in range(ni):
            for j in range(mi):
                for c in range(r):
                    U[(i0 + i) * m + (j0 + j), col + c] = U[i * mi + j, c]
        # V local index (j*p + k) -> global ( (j0+j)*p + (k0+k) )
        for j in range(mi):
            for k in range(pi):
                for c in range(r):
                    V[(j0 + j) * p + (k0 + k), col + c] = V[j * pi + k, c]
        # W local index (k*ni + i) -> global ( (k0+k)*n + (i0+i) )
        for k in range(pi):
            for i in range(ni):
                for c in range(r):
                    W[(k0 + k) * n + (i0 + i), col + c] = W[k * ni + i, c]
        col += r
    return U, V, W, total_rank


def _verify(U, V, W, tol=1e-6):
    """Verify that the decomposition reproduces matrix multiplication for random inputs."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Build the bilinear form: C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) (sum_j V[j*p+k,c] B[j,k]) W[k*n+i,c]
    # Test with random matrices.
    rng = np.random.RandomState(0)
    for _ in range(3):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        C_true = A @ B
        C = np.zeros((n, p))
        for c in range(rank):
            ua = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    ua[i, j] = U[i * m + j, c]
            vb = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    vb[j, k] = V[j * p + k, c]
            wc = np.zeros((p, n))
            for k in range(p):
                for i in range(n):
                    wc[k, i] = W[k * n + i, c]
            C += (ua @ A) * (B @ vb) * wc
        if not np.allclose(C, C_true, atol=tol):
            return False
    return True


def _to_list(M):
    # Convert to list of lists with integer/half-integer entries.
    out = []
    for row in M:
        r = []
        for v in row:
            # Round to nearest half-integer
            rv = round(v * 2) / 2
            if abs(rv - round(rv)) < 1e-9:
                r.append(int(round(rv)))
            else:
                r.append(rv)
        out.append(r)
    return out


def construct():
    start = time.time()
    deadline = start + 120.0  # safety margin under 150s kill
    n, m, p = DIMENSIONS

    # Candidate block decompositions. We want to cover i in [0,4), j in [0,4), k in [0,7).
    # Strategy: use 2x2 blocks in (i,j) and 3x3 / 3x1 / 1x3 in k.
    candidates = []

    # Block grid: i blocks of size 2,2 ; j blocks 2,2 ; k blocks 3,3,1
    i_sizes = [2, 2]
    j_sizes = [2, 2]
    k_sizes = [3, 3, 1]

    def build_grid(ik_algo, kj_algo):
        specs = []
        i0 = 0
        for ni in i_sizes:
            j0 = 0
            for mi in j_sizes:
                k0 = 0
                for pi in k_sizes:
                    algo = ik_algo if pi >= 2 else "schoolbook"
                    # For pi==2 use strassen222; else schoolbook
                    if pi == 2:
                        a = "strassen222"
                    else:
                        a = "schoolbook"
                    specs.append((ni, mi, pi, a, i0, j0, k0))
                    k0 += pi
                j0 += mi
            i0 += ni
        return specs

    # Grid A: k blocks 3,3,1 with schoolbook for k-blocks (no strassen since k not 2)
    specsA = build_grid("schoolbook", "schoolbook")
    U, V, W, r = _combine_blocks(specsA)
    if _verify(U, V, W):
        candidates.append((r, U, V, W))

    # Grid B: k blocks 2,2,2,1 with strassen for k=2 blocks
    i_sizes2 = [2, 2]
    j_sizes2 = [2, 2]
    k_sizes2 = [2, 2, 2, 1]
    def build_grid2():
        specs = []
        i0 = 0
        for ni in i_sizes2:
            j0 = 0
            for mi in j_sizes2:
                k0 = 0
                for pi in k_sizes2:
                    if pi == 2:
                        a = "strassen222"
                    else:
                        a = "schoolbook"
                    specs.append((ni, mi, pi, a, i0, j0, k0))
                    k0 += pi
                j0 += mi
            i0 += ni
        return specs
    specsB = build_grid2()
    U, V, W, r = _combine_blocks(specsB)
    if _verify(U, V, W):
        candidates.append((r, U, V, W))

    # Also keep the pure schoolbook as a fallback (always valid).
    U, V, W, r = _build_block(n, m, p, "schoolbook")
    candidates.append((r, U, V, W))

    # Pick the best (lowest rank) valid candidate.
    best = None
    for (r, U, V, W) in candidates:
        if best is None or r < best[0]:
            best = (r, U, V, W)

    r, U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
