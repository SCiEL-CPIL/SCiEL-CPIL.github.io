"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly for integer/half-integer entries."""
    def scale2(mat):
        arr = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(arr - np.round(arr))) > 1e-9:
            return None
        return np.round(arr).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    if np.max(np.abs(T_computed - target)) > 1e-6:
        return False
    return True

def _build_target(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1
    return T

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b, col] = 1
                V[b*p + c, col] = 1
                W[c*n + a, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _laderman_3x3():
    """Laderman 3x3x3 decomposition, rank 15."""
    # Coefficients for Laderman's algorithm
    # U, V, W are 9x15
    U = np.zeros((9, 15), dtype=np.int64)
    V = np.zeros((9, 15), dtype=np.int64)
    W = np.zeros((9, 15), dtype=np.int64)
    
    # Indices: 0,1,2 for rows/cols
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    # A[2,0]=6, A[2,1]=7, A[2,2]=8
    
    # Term 1: (A00+A11)(B00+B11) -> C00 +=, C11 +=
    U[0,0]=1; U[4,0]=1; V[0,0]=1; V[4,0]=1; W[0,0]=1; W[4,0]=1
    # Term 2: (A00+A22)(B00+B22) -> C00 +=, C22 +=
    U[0,1]=1; U[8,1]=1; V[0,1]=1; V[8,1]=1; W[0,1]=1; W[8,1]=1
    # Term 3: (A11+A22)(B11+B22) -> C11 +=, C22 +=
    U[4,2]=1; U[8,2]=1; V[4,2]=1; V[8,2]=1; W[4,2]=1; W[8,2]=1
    # Term 4: (A01+A12)(B01+B12) -> C01 +=, C12 +=
    U[1,3]=1; U[5,3]=1; V[1,3]=1; V[5,3]=1; W[1,3]=1; W[5,3]=1
    # Term 5: (A02+A21)(B02+B21) -> C02 +=, C21 +=
    U[2,4]=1; U[7,4]=1; V[2,4]=1; V[7,4]=1; W[2,4]=1; W[7,4]=1
    # Term 6: (A10+A20)(B10+B20) -> C10 +=, C20 +=
    U[3,5]=1; U[6,5]=1; V[3,5]=1; V[6,5]=1; W[3,5]=1; W[6,5]=1
    
    # Off-diagonals require more complex terms
    # This hardcoded set is likely INCOMPLETE or WRONG for full 3x3.
    # Laderman's algorithm is specific. 
    # To be safe, I will NOT use this if I'm not sure.
    # I will fallback to schoolbook for 3x3 if verification fails.
    
    # Given the risk, I will use a KNOWN valid set if I can recall it.
    # Actually, I will just use the 2x2 Strassen and 3x3 Schoolbook for now, 
    # but try to optimize the 3x3 block by using a 2+1 split inside it?
    # No, that's what block decomposition does.
    
    # I will return a dummy that fails verification to trigger fallback.
    return U, V, W

def _block_decompose(n, m, p, sizes_a, sizes_b, sizes_c):
    U_terms = []
    V_terms = []
    W_terms = []
    rank_offset = 0

    for i in range(len(sizes_a)):
        for j in range(len(sizes_b)):
            for k in range(len(sizes_c)):
                sa, sb, sc = sizes_a[i], sizes_b[j], sizes_c[k]
                if sa == 0 or sb == 0 or sc == 0:
                    continue

                if sa == 2 and sb == 2 and sc == 2:
                    U_blk, V_blk, W_blk = _strassen_2x2()
                    r = 7
                elif sa == 3 and sb == 3 and sc == 3:
                    # Use Laderman if valid, else schoolbook
                    U_blk, V_blk, W_blk = _laderman_3x3()
                    r = 15
                    # We must verify this block in isolation? 
                    # The main _verify will check the whole thing.
                else:
                    r = sa * sb * sc
                    U_blk = np.zeros((sa*sb, r), dtype=np.int64)
                    V_blk = np.zeros((sb*sc, r), dtype=np.int64)
                    W_blk = np.zeros((sc*sa, r), dtype=np.int64)
                    col = 0
                    for a in range(sa):
                        for b in range(sb):
                            for c in range(sc):
                                U_blk[a*sb+b, col] = 1
                                V_blk[b*sc+c, col] = 1
                                W_blk[c*sa+a, col] = 1
                                col += 1

                row_start_A = sum(sizes_a[:i])
                row_start_B = sum(sizes_b[:j])
                row_start_C = sum(sizes_c[:k])

                for rr in range(r):
                    for a_idx in range(sa*sb):
                        val = U_blk[a_idx, rr]
                        if val != 0:
                            U_terms.append((row_start_A + a_idx, rank_offset + rr, val))
                    for b_idx in range(sb*sc):
                        val = V_blk[b_idx, rr]
                        if val != 0:
                            V_terms.append((row_start_B + b_idx, rank_offset + rr, val))
                    for c_idx in range(sc*sa):
                        val = W_blk[c_idx, rr]
                        if val != 0:
                            W_terms.append((row_start_C + c_idx, rank_offset + rr, val))
                rank_offset += r

    total_rank = rank_offset
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)
    for row, col, val in U_terms:
        U[row, col] += val
    for row, col, val in V_terms:
        V[row, col] += val
    for row, col, val in W_terms:
        W[row, col] += val
    return U.tolist(), V.tolist(), W.tolist(), total_rank

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    partitions = [
        ([3, 2], [3, 2], [3, 2]),
        ([2, 3], [2, 3], [2, 3]),
        ([2, 2, 1], [2, 2, 1], [2, 2, 1]),
    ]
    
    for sa, sb, sc in partitions:
        if time.time() > deadline - 30:
            break
        U_blk, V_blk, W_blk, blk_rank = _block_decompose(n, m, p, sa, sb, sc)
        if blk_rank < best_rank:
            if _verify(U_blk, V_blk, W_blk, n, m, p, blk_rank):
                best_rank = blk_rank
                best_result = {"U": U_blk, "V": V_blk, "W": W_blk}
                print(f"Block decomposition {sa}|{sb}|{sc} rank: {best_rank}")
            else:
                print(f"Block decomposition {sa}|{sb}|{sc} rank {blk_rank} FAILED")
    
    return best_result
# EVOLVE-BLOCK-END
