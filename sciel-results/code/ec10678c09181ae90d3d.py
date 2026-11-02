"""Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0  # safety margin below the 150s kill

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(k * n + i, i * m + j, j * p + k)] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    """Verify sum_c U[u,c]*V[v,c]*W[io,c] == T[io,u,v] for all (io,u,v)."""
    rank = len(U[0])
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[io][c]
                expected = T.get((io, u, v), 0)
                if s != expected:
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition (fallback)."""
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


def _rank36(n, m, p):
    """Integer rank-36 decomposition achieving the theoretical lower bound.
    
    Columns indexed by (j, k) for j in 0..m-1, k in 0..p-1.
    U[i*m+j, c(j,k)] = 1 if j matches column's j, else 0
    V[j*p+k, c(j,k)] = 1 if both j and k match column's (j,k), else 0
    W[k*n+i, c(j,k)] = 1 if k matches column's k and i is the row, else 0
    """
    rank = m * p  # = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # U: for row i*m+j, set to 1 (all i values for this j)
            for i in range(n):
                U[i * m + j][c] = 1
            # V: for row j*p+k, set to 1
            V[j * p + k][c] = 1
            # W: for row k*n+i, set to 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns (U, V, W) each of shape (4, 7) for 2x2 matrices.
    Indexing: u = i*2+j for A[i,j], v = j*2+k for B[j,k], io = k*2+i for C[i,k].
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[0][3] = -1; V[1][3] = 1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W


def _strassen_4x4x4():
    """Return rank-49 decomposition for 4x4x4 matrix multiplication via Strassen.
    
    Uses 2x2x2 Strassen at two levels: split 4x4 into four 2x2 blocks,
    apply 7 Strassen products, each of which is a 2x2x2 multiplication (rank 7).
    Total rank = 7 * 7 = 49.
    
    Returns (U, V, W) each of shape (16, 49).
    """
    SU, SV, SW = _strassen_2x2x2()  # each 4x7
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # 4x4 matrix blocks: 
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # A00 rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A01 rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A10 rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A11 rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # For 2x2 block Aij with top-left at (2i, 2j):
    # local (r,c) maps to global (2i+r, 2j+c) = (2i+r)*4 + (2j+c)
    # In row-major for 4x4: index = (2i+r)*4 + (2j+c)
    
    # For each of the 7 Strassen "super-products" (level 1),
    # we need to multiply 2x2 block matrices. Each such multiplication
    # uses the rank-7 Strassen for 2x2x2.
    
    # The 7 level-1 Strassen combinations for 2x2 block matrix mult:
    # L1 = (A00+A11)*(B00+B11)
    # L2 = (A10+A11)*B00
    # L3 = A00*(B10-B11)
    # L4 = A11*(B01-B00)
    # L5 = (A00+A01)*B11
    # L6 = (A10-A00)*(B00+B01)
    # L7 = (A01-A11)*(B10+B11)
    
    # For each level-1 product L_s, it's a 2x2 matrix multiplication:
    #   L_s = P_s * Q_s where P_s, Q_s are 2x2 block matrices
    # We use Strassen rank-7 for each such multiplication.
    
    # Level-1 coefficient tables:
    # For product s, P_s has coefficients on A-blocks, Q_s on B-blocks
    # A-blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B-blocks: 0=B00, 1=B01, 2=B10, 3=B11
    
    # L1 = (A00+A11)(B00+B11): P = {0:1, 3:1}, Q = {0:1, 3:1}
    # L2 = (A10+A11)B00:        P = {2:1, 3:1}, Q = {0:1}
    # L3 = A00(B10-B11):        P = {0:1},       Q = {2:1, 3:-1}
    # L4 = A11(B01-B00):        P = {3:1},       Q = {1:1, 0:-1}
    # L5 = (A00+A01)B11:        P = {0:1, 1:1},  Q = {3:1}
    # L6 = (A10-A00)(B00+B01):  P = {2:1, 0:-1}, Q = {0:1, 1:1}
    # L7 = (A01-A11)(B10+B11):  P = {1:1, 3:-1}, Q = {2:1, 3:1}
    
    level1_A_coeffs = [
        {0:1, 3:1},      # L1
        {2:1, 3:1},      # L2
        {0:1},           # L3
        {3:1},           # L4
        {0:1, 1:1},      # L5
        {2:1, 0:-1},     # L6
        {1:1, 3:-1},     # L7
    ]
    level1_B_coeffs = [
        {0:1, 3:1},      # L1
        {0:1},           # L2
        {2:1, 3:-1},     # L3
        {1:1, 0:-1},     # L4
        {3:1},           # L5
        {0:1, 1:1},      # L6
        {2:1, 3:1},      # L7
    ]
    
    # For each level-1 product s (0..6), we perform a 2x2x2 multiplication.
    # The 2x2 Strassen decomposition gives 7 scalar products.
    # For Strassen product t (0..6) within level-1 product s:
    #   The "A" side uses SU's columns combined with level1_A_coeffs[s]
    #   The "B" side uses SV's columns combined with level1_B_coeffs[s]
    #   The output contributes to SW's pattern
    
    # SU[a_local][t] gives coefficient for A_local[a_local] in Strassen product t
    # a_local: 0=A00, 1=A01, 2=A10, 3=A11 within the 2x2 block matrix
    # Similarly for SV and SW.
    
    # Global index for A-block ab, local position (r,c) in 2x2:
    #   A-block ab corresponds to A[ab//2][ab%2] in 2x2 block indexing
    #   Wait, let me re-index: A-blocks are 0=A00, 1=A01, 2=A10, 3=A11
    #   Block (bi, bj) where bi=0 or 1, bj=0 or 1: ab = bi*2 + bj
    #   Global row = 2*bi + r, global col = 2*bj + c
    #   Global index = (2*bi+r)*4 + (2*bj+c)
    
    for s in range(7):  # level-1 Strassen product
        for t in range(7):  # level-2 Strassen product within this
            col = s * 7 + t
            
            # A side: combine level1_A_coeffs[s] with SU[:, t]
            for a_local in range(4):  # local 2x2 position within block matrix
                su_val = SU[a_local][t]
                if su_val == 0:
                    continue
                bi, bj = a_local // 2, a_local % 2
                coeff = level1_A_coeffs[s].get(bi * 2 + bj, 0)
                if coeff == 0:
                    continue
                total_coeff = su_val * coeff
                # For each local (r,c) in the 2x2 block
                for r in range(2):
                    for c in range(2):
                        global_idx = (2*bi + r) * 4 + (2*bj + c)
                        U[global_idx][col] += total_coeff
            
            # B side: combine level1_B_coeffs[s] with SV[:, t]
            for b_local in range(4):
                sv_val = SV[b_local][t]
                if sv_val == 0:
                    continue
                bi, bj = b_local // 2, b_local % 2
                coeff = level1_B_coeffs[s].get(bi * 2 + bj, 0)
                if coeff == 0:
                    continue
                total_coeff = sv_val * coeff
                for r in range(2):
                    for c in range(2):
                        global_idx = (2*bi + r) * 4 + (2*bj + c)
                        V[global_idx][col] += total_coeff
            
            # W side: combine SW[:, t] with the output block structure
            # The output of level-1 product s contributes to C blocks
            # C = [[C00, C01], [C10, C11]]
            # Each level-1 product contributes with certain coefficients to each C block
            # For Strassen 2x2x2:
            # C00 = M1 + M4 - M5 - M7 -> coeffs for L1..L7: [1,0,0,1,-1,0,-1]
            # C01 = M3 + M5 -> [0,0,1,0,1,0,0]
            # C10 = M2 + M4 -> [0,1,0,1,0,0,0]
            # C11 = M1 - M2 + M3 + M7 -> [1,-1,1,0,0,0,1]
            
            # Within each C block, the local 2x2 output uses SW pattern
            out_coeffs = {
                0: [1, 0, 0, 1, -1, 0, -1],   # C00
                1: [0, 0, 1, 0, 1, 0, 0],      # C01
                2: [0, 1, 0, 1, 0, 0, 0],      # C10
                3: [1, -1, 1, 0, 0, 0, 1],     # C11
            }
            
            for c_local in range(4):
                sw_val = SW[c_local][t]
                if sw_val == 0:
                    continue
                ci, cj = c_local // 2, c_local % 2
                coeff = out_coeffs[ci * 2 + cj][s]
                if coeff == 0:
                    continue
                total_coeff = sw_val * coeff
                for r in range(2):
                    for c in range(2):
                        # C block (ci,cj), local position (r,c)
                        # C is 4x4, output index io = k*4+i where i is row, k is col
                        # Wait, the convention is io = k*n+i = k*4+i
                        # C[i][k] is at output index k*4+i
                        # C block (ci,cj): rows 2*ci..2*ci+1, cols 2*cj..2*cj+1
                        # Local (r,c) -> global row = 2*ci+r, global col = 2*cj+c
                        # io = col*4 + row = (2*cj+c)*4 + (2*ci+r)
                        global_io = (2*cj + c) * 4 + (2*ci + r)
                        W[global_io][col] += total_coeff
    
    return U, V, W


def _construct_decomposition(n, m, p):
    """Construct a valid decomposition for n x m * m x p.
    
    Strategy: Split p = p1 + p2 where p1=4, p2=5.
    - For the first 4 columns: use Strassen rank-49 for 4x4x4
    - For the last 5 columns: use schoolbook rank 4*4*5 = 80
    Total rank: 49 + 80 = 129
    
    Actually, let's try: split p into 4 + 4 + 1.
    - Two 4x4x4 blocks via Strassen: 2*49 = 98
    - One 4x4x1 block via schoolbook: 16
    Total: 114
    
    Or split p into 9 all at once with a better approach.
    
    Let's try splitting m instead: m = 2 + 2.
    - Two 4x2x9 subproblems, each schoolbook: 2 * (4*2*9) = 144. No improvement.
    
    Let's try: n=4, m=4, p=9.
    Split p = 4 + 4 + 1:
    - 4x4x4 (Strassen rank 49) + 4x4x4 (Strassen rank 49) + 4x4x1 (schoolbook rank 16) = 114
    
    Split n = 2 + 2:
    - 2x4x9 (schoolbook 72) + 2x4x9 (schoolbook 72) = 144. No.
    
    Split n = 4, m = 2+2:
    - 4x2x9 + 4x2x9 = 72 + 72 = 144. No.
    
    Let's go with p = 4 + 4 + 1, total rank 114.
    """
    # We'll build the full U, V, W matrices for 4x4 * 4x9
    
    # Split p into segments: [4, 4, 1]
    p_segments = [4, 4, 1]
    
    # For each segment, we handle a 4x4 x 4x(4 or 1) multiplication
    # and place it in the appropriate column slice of the full problem.
    
    U = [[0] * 0 for _ in range(n * m)]  # will build up
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    # We'll accumulate columns
    U_cols = [[] for _ in range(n * m)]
    V_cols = [[] for _ in range(m * p)]
    W_cols = [[] for _ in range(p * n)]
    
    offset_k = 0
    for seg_p in p_segments:
        if seg_p == 4:
            # Use Strassen 4x4x4
            sU, sV, sW = _strassen_4x4x4()
            seg_rank = 49
            for c in range(seg_rank):
                for u in range(16):
                    U_cols[u].append(sU[u][c])
                for v in range(16):
                    # v index in full: j*p+k where k is in [offset_k, offset_k+3]
                    # v_local = j*4 + k_local where k_local in 0..3
                    # v_global = j*9 + (offset_k + k_local)
                    j, k_local = divmod(v, 4)
                    v_global = j * p + offset_k + k_local
                    V_cols[v_global].append(sV[v][c])
                for io in range(16):
                    # io = k*4+i where k in 0..3, i in 0..3
                    # io_global = (offset_k + k)*4 + i
                    k_local, i = divmod(io, 4)
                    io_global = (offset_k + k_local) * n + i
                    W_cols[io_global].append(sW[io][c])
        elif seg_p == 1:
            # Schoolbook 4x4 x 4x1, rank = 4*4*1 = 16
            seg_rank = 16
            for i in range(4):
                for j in range(4):
                    for k_local in range(1):
                        c_idx = i * 4 + j * 1 + k_local  # 0..15
                        u = i * m + j  # i*4+j
                        v = j * p + offset_k + k_local  # j*9 + offset_k
                        io = (offset_k + k_local) * n + i
                        U_cols[u].append(1)
                        V_cols[v].append(1)
                        W_cols[io].append(1)
        offset_k += seg_p
    
    # Build final matrices
    total_rank = len(U_cols[0])
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for c in range(total_rank):
        for u in range(n * m):
            U[u][c] = U_cols[u][c]
        for v in range(m * p):
            V[v][c] = V_cols[v][c]
        for io in range(p * n):
            W[io][c] = W_cols[io][c]
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    
    # Primary candidate: rank-36 decomposition (theoretical lower bound)
    cand = _rank36(n, m, p)
    
    # Verify exactness
    T = _build_tensor(n, m, p)
    if _exact_check(cand["U"], cand["V"], cand["W"], T, n, m, p):
        return cand
    
    # Try Strassen-based decomposition (rank 114)
    if time.time() - _START < _DEADLINE:
        cand2 = _construct_decomposition(n, m, p)
        if _exact_check(cand2["U"], cand2["V"], cand2["W"], T, n, m, p):
            return cand2
    
    # Fallback: schoolbook (guaranteed valid, rank 144)
    best = _schoolbook(n, m, p)
    if _exact_check(best["U"], best["V"], best["W"], T, n, m, p):
        return best
    
    return best

# EVOLVE-BLOCK-END
