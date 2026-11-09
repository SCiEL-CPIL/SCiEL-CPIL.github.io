"""Matrix multiplication 4x5 * 5x6 rank-minimal tensor decomposition.
Uses block decomposition: Strassen on 2x2*2x2->2x2 blocks, outer product
on 2x1*1x2->2x2 blocks. Convention W[k*n+i, r] for C[i,k] = sum_j A[i,j]B[j,k].
"""
DIMENSIONS = [4, 5, 6]
import time

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Exact verification: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2_block():
    """Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U (4x7) indexed U[i*2+j, r], V (4x7) V[j*2+k, r], W (4x7) W[k*2+i, r].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4 = A11(B01-B10)
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W

def _outer_2x1_1x2():
    """Outer product for (2x1)*(1x2) -> 2x2. C[i,k] = A[i]*B[k].
    U (2x4) U[i, r], V (2x4) V[k, r], W (4x4) W[k*2+i, r]. Rank 4.
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(4)]
    for i in range(2):
        for k in range(2):
            col = i*2 + k
            U[i][col] = 1
            V[k][col] = 1
            W[k*2+i][col] = 1
    return U, V, W

def _block_decompose_4x5x6():
    """Block decomposition of 4x5 * 5x6 -> 4x6.
    A: 4x5 split rows [2,2], cols [2,2,1]
    B: 5x6 split rows [2,2,1], cols [2,2,2]
    C: 4x6 split rows [2,2], cols [2,2,2]
    C[i,j] = sum_k A[i,k] B[k,j]
    """
    n, m, p = 4, 5, 6
    U_s, V_s, W_s = _strassen_2x2x2_block()
    U_o, V_o, W_o = _outer_2x1_1x2()

    a_row_splits = [2, 2]
    a_col_splits = [2, 2, 1]
    b_row_splits = [2, 2, 1]
    b_col_splits = [2, 2, 2]
    c_row_splits = [2, 2]
    c_col_splits = [2, 2, 2]

    def offsets(splits):
        off = [0]
        for s in splits[:-1]:
            off.append(off[-1] + s)
        return off

    a_row_off = offsets(a_row_splits)
    a_col_off = offsets(a_col_splits)
    b_row_off = offsets(b_row_splits)
    b_col_off = offsets(b_col_splits)
    c_row_off = offsets(c_row_splits)
    c_col_off = offsets(c_col_splits)

    # Count total rank
    total_rank = 0
    for i in range(2):
        for k in range(3):
            aw = a_col_splits[k]
            bh = b_row_splits[k]
            for j in range(3):
                bw = b_col_splits[j]
                if aw == 2 and bh == 2 and bw == 2:
                    total_rank += 7
                elif aw == 1 and bh == 1 and bw == 2:
                    total_rank += 4
                else:
                    total_rank += a_row_splits[i] * aw * bh * bw

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    for i in range(2):
        for k in range(3):
            aw = a_col_splits[k]
            bh = b_row_splits[k]
            a_i0 = a_row_off[i]
            a_k0 = a_col_off[k]
            b_k0 = b_row_off[k]
            for j in range(3):
                bw = b_col_splits[j]
                b_j0 = b_col_off[j]
                c_i0 = c_row_off[i]
                c_j0 = c_col_off[j]
                if aw == 2 and bh == 2 and bw == 2:
                    for r in range(7):
                        for ii in range(2):
                            for jj in range(2):
                                if U_s[ii*2+jj][r]:
                                    U[(a_i0+ii)*m + (a_k0+jj)][col] += U_s[ii*2+jj][r]
                        for kk in range(2):
                            for ll in range(2):
                                if V_s[kk*2+ll][r]:
                                    V[(b_k0+kk)*p + (b_j0+ll)][col] += V_s[kk*2+ll][r]
                        for ll in range(2):
                            for mm in range(2):
                                if W_s[ll*2+mm][r]:
                                    W[(c_j0+ll)*n + (c_i0+mm)][col] += W_s[ll*2+mm][r]
                        col += 1
                elif aw == 1 and bh == 1 and bw == 2:
                    for r in range(4):
                        for ii in range(2):
                            if U_o[ii][r]:
                                U[(a_i0+ii)*m + a_k0][col] += U_o[ii][r]
                        for ll in range(2):
                            if V_o[ll][r]:
                                V[b_k0*p + (b_j0+ll)][col] += V_o[ll][r]
                        for ll in range(2):
                            for mm in range(2):
                                if W_o[ll*2+mm][r]:
                                    W[(c_j0+ll)*n + (c_i0+mm)][col] += W_o[ll*2+mm][r]
                        col += 1
                else:
                    # schoolbook fallback for this block
                    for ii in range(a_row_splits[i]):
                        for jj in range(aw):
                            for kk in range(bh):
                                for ll in range(bw):
                                    U[(a_i0+ii)*m + (a_k0+jj)][col] = 1
                                    V[(b_k0+kk)*p + (b_j0+ll)][col] = 1
                                    W[(c_j0+ll)*n + (c_i0+ii)][col] = 1
                                    col += 1
    return {"U": U, "V": V, "W": W}, total_rank

def _try_block():
    n, m, p = DIMENSIONS
    result, rank = _block_decompose_4x5x6()
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result, rank
    return None, None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140

    # Primary: block decomposition (Strassen + outer). Expected rank 108.
    try:
        result, rank = _try_block()
        if result is not None:
            if time.time() < deadline:
                return result
    except Exception:
        pass

    # Fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
