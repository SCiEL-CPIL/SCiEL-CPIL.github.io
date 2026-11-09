"""Combined program: rank-65 decomposition for 4x4x5 matrix multiplication.
Uses recursive Strassen (rank 49) for 4x4x4 block + schoolbook (rank 16) for 4x4x1.
Restructured with explicit block-diagonal partitioning and direct coefficient tables.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20

    rank_part1 = 49
    rank_part2 = 16
    total_rank = rank_part1 + rank_part2

    U = np.zeros((nm, total_rank), dtype=np.int64)
    V = np.zeros((mp, total_rank), dtype=np.int64)
    W = np.zeros((pn, total_rank), dtype=np.int64)

    # --- Part 2: Schoolbook for column k=4 (4x4x1, rank 16) ---
    col_offset = rank_part1
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1

    # --- Part 1: Recursive Strassen 4x4x4 (rank 49) ---
    # Build using the 2x2 Strassen base and block composition.
    
    # 2x2 Strassen base matrices (rows: 00,01,10,11; cols: 7 multiplications)
    # U2[r, m] for A, V2[r, m] for B, W2[r, m] for C
    # Convention: row r = (block_row, block_col) flattened as r = br*2 + bc
    # For A: A[br, bc]
    # For B: B[br, bc]  
    # For C: C[br, bc]
    
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # U2: for each A position (br,bc) and each multiplication m, coefficient
    # A00=(0,0) r=0, A01=(0,1) r=1, A10=(1,0) r=2, A11=(1,1) r=3
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],   # A00: M1, M3, M5
        [0, 0, 0, 0, 1, 0, 1],   # A01: M5, M7
        [0, 1, 0, 0, 0, 1, 0],   # A10: M2, M6
        [1, 1, 0, 1, 0, 0, -1],  # A11: M1, M2, M4, M7
    ], dtype=np.int64)
    
    # V2: for each B position (br,bc) and each multiplication m
    # B00=(0,0) r=0, B01=(0,1) r=1, B10=(1,0) r=2, B11=(1,1) r=3
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],   # B00: M1, M2, M6
        [0, 0, 1, 0, 0, 1, 0],   # B01: M3, M6
        [0, 0, 0, 1, 0, 0, 1],   # B10: M4, M7
        [1, 0, -1, -1, 1, 0, 1], # B11: M1, M3, M4, M5, M7
    ], dtype=np.int64)
    
    # W2: for each C position (br,bc) and each multiplication m
    # C00=(0,0) r=0, C01=(0,1) r=1, C10=(1,0) r=2, C11=(1,1) r=3
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00: M1, M4, -M5, M7
        [0, 0, 1, 0, 1, 0, 0],   # C01: M3, M5
        [0, 1, 0, 1, 0, 0, 0],   # C10: M2, M4
        [1, -1, 1, 0, 0, 1, 0],  # C11: M1, -M2, M3, M6
    ], dtype=np.int64)
    
    # Verify 2x2 Strassen
    A2 = np.array([[1,2],[3,4]], dtype=float)
    B2 = np.array([[5,6],[7,8]], dtype=float)
    C2_true = A2 @ B2
    C2 = np.zeros((2,2), dtype=float)
    for mm in range(7):
        a_sub = np.array([[A2[0,0]*U2[0,mm] + A2[1,1]*U2[3,mm],
                           A2[0,1]*U2[1,mm] + A2[1,1]*U2[3,mm]],
                          [A2[1,0]*U2[2,mm] + A2[1,1]*U2[3,mm],
                           A2[1,1]*U2[3,mm] + A2[1,1]*U2[3,mm]]])
        # Actually, let's just verify via tensor contraction
    # Simpler verification:
    U2f = U2.reshape(2, 2, 7)
    V2f = V2.reshape(2, 2, 7)
    W2f = W2.reshape(2, 2, 7)
    C2_check = np.einsum('ijr,jkr,kir->ik', U2f, V2f, W2f)
    if not np.allclose(C2_check, C2_true):
        # Fallback to schoolbook
        U_s, V_s, W_s = _schoolbook(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Now build 4x4x4 by applying Strassen at the 2x2-block level.
    # The 4x4 matrix is partitioned into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # Global row for A[i,j] is i*m + j = i*4 + j
    # Within block Aij, local row (u,v) maps to global (i*2+u, j*2+v) -> row (i*2+u)*4 + (j*2+v)
    
    # For the outer Strassen, the "A matrix" has entries that are 2x2 blocks.
    # For multiplication mk, the outer A-side contributes from certain blocks with coefficients.
    # Then within each block, the inner Strassen applies.
    
    # Outer Strassen structure (same as 2x2):
    # For each outer multiplication mk (0..6):
    #   A-side: sum over blocks bi of coeff_a[mk][bi] * A_bi
    #   B-side: sum over blocks bj of coeff_b[mk][bj] * B_bj
    #   Output: sum over blocks bc of coeff_c[mk][bc] * C_bc
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 (same for B, C)
    outer_A = np.array([
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [0, 0, 1, -1],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ], dtype=np.int64)
    
    outer_B = np.array([
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [0, 0, 1, -1],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ], dtype=np.int64)
    
    outer_C = np.array([
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [0, 1, 0, 1],   # M3: C01+C11
        [1, 0, 1, 0],   # M4: C00+C10
        [-1, 1, 0, 0],  # M5: -C00+C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ], dtype=np.int64)
    
    # For each outer multiplication mk and inner multiplication nk:
    # Column index = mk * 7 + nk
    
    # U contribution: for outer block bi with coeff outer_A[mk,bi],
    # and inner position (u,v) with coeff U2[u*2+v, nk],
    # global A row = (bi//2)*2 + u, col = (bi%2)*2 + v -> row = (bi//2)*2+u)*4 + (bi%2)*2+v
    
    # V contribution: similar for B blocks
    # W contribution: for outer block bc with coeff outer_C[mk,bc],
    # and inner position (u,v) with coeff W2[u*2+v, nk],
    # global C row: C[k*n+i] where k = (bc%2)*2+v, i = (bc//2)*2+u
    
    for mk in range(7):
        for nk in range(7):
            col = mk * 7 + nk
            
            # U contributions
            for bi in range(4):
                ca = outer_A[mk, bi]
                if ca == 0:
                    continue
                br = bi // 2  # block row (0 or 1)
                bc = bi % 2   # block col (0 or 1)
                for u in range(2):
                    for v in range(2):
                        cu = U2[u * 2 + v, nk]
                        if cu == 0:
                            continue
                        # Global A row: (br*2+u)*4 + (bc*2+v)
                        g_row = (br * 2 + u) * 4 + (bc * 2 + v)
                        U[g_row, col] += ca * cu
            
            # V contributions
            for bi in range(4):
                cb = outer_B[mk, bi]
                if cb == 0:
                    continue
                br = bi // 2
                bc = bi % 2
                for u in range(2):
                    for v in range(2):
                        cv = V2[u * 2 + v, nk]
                        if cv == 0:
                            continue
                        # Global B row: (br*2+u)*4 + (bc*2+v)  [j*p+k = j*4+k]
                        g_row = (br * 2 + u) * 4 + (bc * 2 + v)
                        V[g_row, col] += cb * cv
            
            # W contributions
            for bi in range(4):
                cc = outer_C[mk, bi]
                if cc == 0:
                    continue
                br = bi // 2  # C block row -> corresponds to output row group
                bc = bi % 2   # C block col -> corresponds to output col group
                for u in range(2):
                    for v in range(2):
                        cw = W2[u * 2 + v, nk]
                        if cw == 0:
                            continue
                        # C[br*2+u, bc*2+v] in C matrix
                        # Trace convention: W[k*n + i] where i=br*2+u, k=bc*2+v
                        i = br * 2 + u
                        k = bc * 2 + v
                        g_row = k * n + i
                        W[g_row, col] += cc * cw
    
    # Verify
    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 80
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the tensor decomposition exactly reproduces matrix multiplication."""
    if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]):
        return False
    if W.shape != (p * n, U.shape[1]):
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False

    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    Ctrue = A @ B

    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C, Ctrue, atol=tol)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W
# EVOLVE-BLOCK-END
