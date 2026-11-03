import time
import numpy as np

N, M, P = 3, 3, 8
SCHOOLBOOK = N * M * P  # 72


def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[a,c] V[b,c] W[c] == 1 iff a,b,c correspond to a product."""
    try:
        rank = U.shape[1]
        for a in range(N * M):
            for b in range(M * P):
                acc = np.zeros(P * N)
                for c in range(rank):
                    u = U[a, c]
                    if u == 0:
                        continue
                    v = V[b, c]
                    if v == 0:
                        continue
                    acc += u * v * W[:, c]
                i, j = divmod(a, M)
                j2, k2 = divmod(b, P)
                expected_idx = k2 * N + i
                if abs(acc[expected_idx] - 1.0) > tol:
                    return False
                mask = np.ones(P * N, dtype=bool)
                mask[expected_idx] = False
                if np.any(np.abs(acc[mask]) > tol):
                    return False
        return True
    except Exception:
        return False


def _build_schoolbook():
    """Build the 72-rank schoolbook decomposition."""
    U = np.zeros((N * M, SCHOOLBOOK), dtype=np.float64)
    V = np.zeros((M * P, SCHOOLBOOK), dtype=np.float64)
    W = np.zeros((P * N, SCHOOLBOOK), dtype=np.float64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Each of U, V, W has shape (4, 7).
    U indices: 0=A00, 1=A01, 2=A10, 3=A11
    V indices: 0=B00, 1=B01, 2=B10, 3=B11
    W indices: 0=C00, 1=C01, 2=C10, 3=C11
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11 (positive), C01, C10 (negative)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    W[1, 0] = -1; W[2, 0] = -1

    # M2 = (A10+A11)B00 -> contributes to C10 (positive), C11 (negative)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3 = A00(B01-B11) -> contributes to C01 (positive), C11 (positive)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4 = A11(B10-B00) -> contributes to C00 (positive), C10 (positive)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # M5 = (A00+A01)B11 -> contributes to C00 (negative), C01 (positive)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    # M6 = (A10-A00)(B00+B01) -> contributes to C11 (positive)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)(B10+B11) -> contributes to C00 (positive), C10 (negative)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = -1

    return U, V, W


def _build_kronecker_strassen():
    """
    Build a decomposition using Kronecker product structure:
    - Split the 3x3 matrix A into a 2x2 block (top-left) + remaining rows/cols
    - Use Strassen (rank 7) for the 2x2 * 2x2 block with each of the 4 column-pairs of B
    - Use schoolbook for the remaining parts
    
    Specifically:
    - A is 3x3, B is 3x8
    - Split A into A[0:2,0:2] (2x2), A[0:2,2] (2x1), A[2,0:2] (1x2), A[2,2] (1x1)
    - Split B into B[0:2,0:2] (2x2), B[0:2,2:4] (2x2), B[0:2,4:6] (2x2), B[0:2,6:8] (2x2),
                B[2,0:2] (1x2), B[2,2:4] (1x2), B[2,4:6] (1x2), B[2,6:8] (1x2)
    
    For the top-left 2x2 block of A times each 2x2 block of B: use Strassen (7 terms each)
    That's 4 blocks * 7 = 28 terms
    
    For A[0:2,2] (2x1) times B[2,:] (1x8): 2*8 = 16 terms (schoolbook)
    For A[2,0:2] (1x2) times B[0:2,:] (2x8): 2*8 = 16 terms (schoolbook)
    For A[2,2] (1x1) times B[2,:] (1x8): 1*8 = 8 terms (schoolbook)
    
    Total: 28 + 16 + 16 + 8 = 68 terms
    
    Wait, let me reconsider. The output C is 3x8.
    C[0:2,0:2] = A[0:2,0:2] * B[0:2,0:2] + A[0:2,2] * B[2,0:2]
    C[0:2,2:4] = A[0:2,0:2] * B[0:2,2:4] + A[0:2,2] * B[2,2:4]
    C[0:2,4:6] = A[0:2,0:2] * B[0:2,4:6] + A[0:2,2] * B[2,4:6]
    C[0:2,6:8] = A[0:2,0:2] * B[0:2,6:8] + A[0:2,2] * B[2,6:8]
    C[2,0:2] = A[2,0:2] * B[0:2,0:2] + A[2,2] * B[2,0:2]
    C[2,2:4] = A[2,0:2] * B[0:2,2:4] + A[2,2] * B[2,2:4]
    C[2,4:6] = A[2,0:2] * B[0:2,4:6] + A[2,2] * B[2,4:6]
    C[2,6:8] = A[2,0:2] * B[0:2,6:8] + A[2,2] * B[2,6:8]
    
    So:
    - 4 Strassen blocks for A[0:2,0:2] * B[0:2,2:4:6:8] (4 blocks of 2x2*2x2)
    - 16 schoolbook terms for A[0:2,2] * B[2,0:8] (2 rows of A col 2, 8 cols of B row 2)
    - 16 schoolbook terms for A[2,0:2] * B[0:2,0:8] (1 row of A, 2 rows of B, 8 cols)
    - 8 schoolbook terms for A[2,2] * B[2,0:8] (1x1 * 1x8)
    
    Total: 4*7 + 16 + 16 + 8 = 28 + 40 = 68
    """
    # Strassen factors for 2x2 * 2x2
    S_U, S_V, S_W = _strassen_2x2x2()
    
    rank = 4 * 7 + 16 + 16 + 8  # 68
    U = np.zeros((N * M, rank), dtype=np.float64)
    V = np.zeros((M * P, rank), dtype=np.float64)
    W = np.zeros((P * N, rank), dtype=np.float64)
    
    col = 0
    
    # Part 1: A[0:2,0:2] * B[0:2, k:k+2] for k in [0,2,4,6]
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,k]=k, B[0,k+1]=k+1, B[1,k]=4+k, B[1,k+1]=5+k
    # C indices: C[0,k]=k*3+0, C[0,k+1]=k*3+1, C[1,k]=k*3+3, C[1,k+1]=k*3+4
    for k in [0, 2, 4, 6]:
        a_idx = [0, 1, 2, 3]  # A[0,0], A[0,1], A[1,0], A[1,1]
        b_idx = [k, k+1, 4+k, 5+k]  # B[0,k], B[0,k+1], B[1,k], B[1,k+1]
        c_idx = [k*3+0, k*3+1, k*3+3, k*3+4]  # C[0,k], C[0,k+1], C[1,k], C[1,k+1]
        
        for r in range(7):
            # U
            for a_pos, a_val in enumerate([1, 1, 2, 3]):
                # S_U has non-zeros at specific positions
                pass
            
            # Let's just add the Strassen contribution directly
            for s in range(4):
                u_val = S_U[s, r]
                if u_val != 0:
                    U[a_idx[s], col] += u_val
                v_val = S_V[s, r]
                if v_val != 0:
                    V[b_idx[s], col] += v_val
                w_val = S_W[s, r]
                if w_val != 0:
                    W[c_idx[s], col] += w_val
            col += 1
    
    # Part 2: A[0:2,2] * B[2, 0:8]
    # A[0,2]=5, A[1,2]=6
    # B[2,k]=16+k for k in 0..7
    # C[0,k]=k*3+0, C[1,k]=k*3+3
    for i in range(2):  # i=0,1 for A row
        for k in range(8):  # k=0..7 for B col
            u_idx = 5 + i  # A[i,2]
            v_idx = 16 + k  # B[2,k]
            w_idx = k * 3 + i  # C[i,k]
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    # Part 3: A[2,0:2] * B[0:2, 0:8]
    # A[2,0]=6, A[2,1]=7
    # B[i,k]=i*8+k for i in 0,1; k in 0..7
    # C[2,k]=k*3+2
    for j in range(2):  # j=0,1 for A col
        for k in range(8):  # k=0..7 for B col
            for i in range(2):  # i=0,1 for B row
                u_idx = 6 + j  # A[2,j]
                v_idx = i * 8 + k  # B[i,k]
                w_idx = k * 3 + 2  # C[2,k]
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    # Part 4: A[2,2] * B[2, 0:8]
    # A[2,2]=8
    # B[2,k]=16+k for k in 0..7
    # C[2,k]=k*3+2
    for k in range(8):
        u_idx = 8  # A[2,2]
        v_idx = 16 + k  # B[2,k]
        w_idx = k * 3 + 2  # C[2,k]
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
        col += 1
    
    return U[:, :col], V[:, :col], W[:, :col]


def _build_known_68():
    """Replicate the 68-rank decomposition from the current program."""
    N, M, P = 3, 3, 8
    rank = 68
    U = np.zeros((N * M, rank), dtype=np.float64)
    V = np.zeros((M * P, rank), dtype=np.float64)
    W = np.zeros((P * N, rank), dtype=np.float64)
    col = 0

    def add_strassen(u_base, v_base, w_base):
        nonlocal col
        a00, a01, a10, a11 = u_base
        b00, b01, b10, b11 = v_base
        c00, c01, c10, c11 = w_base
        # M1
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        W[c01, col] -= 1; W[c10, col] -= 1
        col += 1
        # M2
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        # M3
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M4
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M5
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        # M6
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        # M7
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c10, col] -= 1
        col += 1

    # Block 1: A_L * B1_T1
    add_strassen([0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 3, 4])
    # Block 2: A_L * B1_T2
    add_strassen([0, 1, 2, 3], [2, 3, 6, 7], [6, 7, 9, 10])
    # Block 3: A_R * B1_B
    u_r = [1, 3]
    v_b = [16, 17, 18, 19]
    w_c = [0, 1, 3, 4, 6, 7, 9, 10]
    for i, u_idx in enumerate(u_r):
        for j, v_idx in enumerate(v_b):
            w_idx = w_c[i * 4 + j]
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    # Block 4: A_L * B2_T1
    add_strassen([0, 1, 2, 3], [8, 9, 12, 13], [12, 13, 15, 16])
    # Block 5: A_L * B2_T2
    add_strassen([0, 1, 2, 3], [10, 11, 14, 15], [18, 19, 21, 22])
    # Block 6: A_R * B2_B
    u_r = [1, 3]
    v_b = [20, 21, 22, 23]
    w_c = [12, 13, 15, 16, 18, 19, 21, 22]
    for i, u_idx in enumerate(u_r):
        for j, v_idx in enumerate(v_b):
            w_idx = w_c[i * 4 + j]
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    # Block 7: A_bot * B (schoolbook)
    for k in range(8):
        w_idx = k * 3 + 2
        for i in range(3):
            u_idx = 6 + i
            v_idx = i * 8 + k
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    return U[:, :col], V[:, :col], W[:, :col]


def construct():
    start = time.time()
    deadline = start + 130.0

    # Try the Kronecker/Strassen decomposition first (new approach)
    U, V, W = _build_kronecker_strassen()
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try the known 68-rank decomposition
    U, V, W = _build_known_68()
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    U, V, W = _build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
