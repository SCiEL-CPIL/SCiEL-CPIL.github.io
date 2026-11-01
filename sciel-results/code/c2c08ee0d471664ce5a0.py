"""Matrix multiplication 2,5,6 — exact tensor decomposition (Strassen-based block construction)."""
import numpy as np

DIMENSIONS = [2, 5, 6]


def _strassen_222():
    """Return (U, V, W) for 2x2x2 multiplication using Strassen's 7-multiplication algorithm.

    U: shape (4, 7) — maps A (row-major 2x2) to factors
    V: shape (4, 7) — maps B (row-major 2x2) to factors
    W: shape (4, 7) — maps products to C (trace convention: W[k*2+i, r])
    """
    # Strassen products M1..M7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22) B11
    # M3 = A11 (B12-B22)
    # M4 = A22 (B21-B11)
    # M5 = (A11+A12) B22
    # M6 = (A21-A11) (B11+B12)
    # M7 = (A12-A22) (B21+B22)

    # A indices: (0,0)=11, (0,1)=12, (1,0)=21, (1,1)=22
    # B indices: same
    # C indices: trace convention W[k*n + i, r] where n=2
    # C[0,0]=11, C[0,1]=12, C[1,0]=21, C[1,1]=22

    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[1, 0] = 1  # B12
    V[2, 0] = 1  # B21
    V[3, 0] = 1  # B22

    # M2 = (A21+A22) B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11

    # M3 = A11 (B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B12
    V[3, 2] = -1  # -B22

    # M4 = A22 (B21-B11)
    U[3, 3] = 1  # A22
    V[0, 3] = -1  # -B11
    V[2, 3] = 1  # B21

    # M5 = (A11+A12) B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22

    # M6 = (A21-A11) (B11+B12)
    U[2, 5] = 1  # A21
    U[0, 5] = -1  # -A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12

    # M7 = (A12-A22) (B21+B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1  # -A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22

    # C combination:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # W[k*2+i, r] where k is column index of C, i is row index
    # C[0,0] = C11: row 0, col 0 -> W[0*2+0, r] = W[0, r]
    # C[0,1] = C12: row 0, col 1 -> W[1*2+0, r] = W[2, r]  (trace convention: W[k*n+i])
    # C[1,0] = C21: row 1, col 0 -> W[0*2+1, r] = W[1, r]
    # C[1,1] = C22: row 1, col 1 -> W[1*2+1, r] = W[3, r]

    # C11 = M1 + M4 - M5 + M7
    W[0, 0] = 1   # M1
    W[0, 3] = 1   # M4
    W[0, 4] = -1  # -M5
    W[0, 6] = 1   # M7

    # C12 = M3 + M5
    W[2, 2] = 1   # M3
    W[2, 4] = 1   # M5

    # C21 = M2 + M4
    W[1, 1] = 1   # M2
    W[1, 3] = 1   # M4

    # C22 = M1 - M2 + M3 + M6
    W[3, 0] = 1   # M1
    W[3, 1] = -1  # -M2
    W[3, 2] = 1   # M3
    W[3, 5] = 1   # M6

    return U, V, W


def _schoolbook_block(n_loc, m_loc, p_loc):
    """Schoolbook decomposition for n_loc x m_loc times m_loc x p_loc."""
    rank = n_loc * m_loc * p_loc
    U = np.zeros((n_loc * m_loc, rank))
    V = np.zeros((m_loc * p_loc, rank))
    W = np.zeros((p_loc * n_loc, rank))
    col = 0
    for i in range(n_loc):
        for j in range(m_loc):
            for k in range(p_loc):
                U[i * m_loc + j, col] = 1
                V[j * p_loc + k, col] = 1
                W[k * n_loc + i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6

    # We'll build the global U, V, W by partitioning the m and p dimensions into 2x2 blocks
    # where possible, using Strassen for those blocks, and schoolbook for the rest.
    #
    # Strategy: partition m=5 into [2,2,1] and p=6 into [2,2,2]
    # For each pair of (m_block, p_block), if both are size 2, use Strassen (rank 7 per 2x2x2).
    # Otherwise use schoolbook for that sub-block.
    #
    # m blocks: sizes [2, 2, 1]
    # p blocks: sizes [2, 2, 2]
    #
    # 3x3 = 9 sub-blocks:
    # (m=2,p=2): Strassen, rank 7, count = 2*2 = 4 such blocks -> 4*7 = 28
    # (m=2,p=2): Strassen, rank 7, count = 2*2 = 4 such blocks -> 4*7 = 28
    # (m=2,p=1): schoolbook 2*2*1=4, count = 2*1 = 2 such blocks -> 2*4 = 8
    # (m=1,p=2): schoolbook 2*1*2=4, count = 1*2 = 2 such blocks -> 2*4 = 8
    # (m=1,p=2): schoolbook 2*1*2=4, count = 1*2 = 2 such blocks -> 2*4 = 8
    # (m=1,p=2): schoolbook 2*1*2=4, count = 1*2 = 2 such blocks -> 2*4 = 8
    #
    # Wait, let me recount. m blocks: [2,2,1], p blocks: [2,2,2]
    # Block (0,0): m=2, p=2 -> Strassen, rank 7
    # Block (0,1): m=2, p=2 -> Strassen, rank 7
    # Block (0,2): m=2, p=2 -> Strassen, rank 7
    # Block (1,0): m=2, p=2 -> Strassen, rank 7
    # Block (1,1): m=2, p=2 -> Strassen, rank 7
    # Block (1,2): m=2, p=2 -> Strassen, rank 7
    # Block (2,0): m=1, p=2 -> schoolbook, rank 2*1*2 = 4
    # Block (2,1): m=1, p=2 -> schoolbook, rank 4
    # Block (2,2): m=1, p=2 -> schoolbook, rank 4
    #
    # Total rank = 6*7 + 3*4 = 42 + 12 = 54

    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 2]

    # Precompute offsets
    m_offsets = [0]
    for s in m_sizes[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_sizes[:-1]:
        p_offsets.append(p_offsets[-1] + s)

    total_rank = 0
    for mb in range(len(m_sizes)):
        for pb in range(len(p_sizes)):
            ms = m_sizes[mb]
            ps = p_sizes[pb]
            if ms == 2 and ps == 2:
                total_rank += 7
            else:
                total_rank += n * ms * ps

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    for mb in range(len(m_sizes)):
        ms = m_sizes[mb]
        m_off = m_offsets[mb]
        for pb in range(len(p_sizes)):
            ps = p_sizes[pb]
            p_off = p_offsets[pb]

            if ms == 2 and ps == 2:
                # Strassen block
                u_loc, v_loc, w_loc = _strassen_222()
                r = 7
                for i in range(n):
                    for j in range(ms):
                        # Global U row: i*m + (m_off + j)
                        U[i * m + (m_off + j), col:col+r] = u_loc[i * 2 + j, :]
                    for k in range(ps):
                        # Global V row: (m_off + j)*p + (p_off + k) — but V is indexed by (j, k)
                        # V row index: j_global * p + k_global where j_global = m_off + j, k_global = p_off + k
                        # Actually V has shape (m*p, rank), indexed as V[j*p+k, :]
                        pass
                # V: for each local (j, k) in 2x2, global j = m_off + j, global k = p_off + k
                for j in range(ms):
                    for k in range(ps):
                        V[(m_off + j) * p + (p_off + k), col:col+r] = v_loc[j * 2 + k, :]
                # W: for each local (k, i) in 2x2 (p_loc x n_loc), global k = p_off + k, global i = i
                # W index: k*n + i
                for k in range(ps):
                    for i in range(n):
                        W[(p_off + k) * n + i, col:col+r] = w_loc[k * 2 + i, :]
                col += r
            else:
                # Schoolbook block
                u_loc, v_loc, w_loc = _schoolbook_block(n, ms, ps)
                r = n * ms * ps
                for i in range(n):
                    for j in range(ms):
                        U[i * m + (m_off + j), col:col+r] = u_loc[i * ms + j, :]
                for j in range(ms):
                    for k in range(ps):
                        V[(m_off + j) * p + (p_off + k), col:col+r] = v_loc[j * ps + k, :]
                for k in range(ps):
                    for i in range(n):
                        W[(p_off + k) * n + i, col:col+r] = w_loc[k * n + i, :]
                col += r

    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
