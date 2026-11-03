"""Matrix multiplication 3x4x5 with block-diagonal Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_entries():
    """Return (U, V, W) lists for 2x2x2 Strassen decomposition, rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Convention: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1: (A00+A11)(B00+B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[2][1] += 1

    # M2: (A10+A11)B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][0] += 1

    # M3: A00(B10-B11)
    U[0][2] += 1
    V[2][2] += 1; V[3][2] -= 1
    W[2][1] += 1; W[1][0] += 1

    # M4: A11(B01-B10)
    U[3][3] += 1
    V[1][3] += 1; V[2][3] -= 1
    W[0][0] += 1; W[2][1] += 1

    # M5: (A00+A01)B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[2][1] -= 1; W[1][0] += 1

    # M6: (A10-A00)(B00+B01)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][0] += 1

    # M7: (A01-A11)(B10+B11)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][0] += 1

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def _build_block_decomposition(n, m, p, n_split, m_split, p_split):
    """Build a block-diagonal decomposition using Strassen on (2,2,2) blocks where possible.
    
    Returns (U, V, W) as lists of lists with the correct global dimensions.
    """
    n_blocks = [n_split[0], n_split[1]]
    m_blocks = [m_split[0], m_split[1]]
    p_blocks = [p_split[0], p_split[1]]
    
    # Calculate total rank
    total_rank = 0
    block_ranks = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a, b, c = n_blocks[i], m_blocks[j], p_blocks[k]
                if a == 2 and b == 2 and c == 2:
                    block_ranks[(i,j,k)] = 7
                else:
                    block_ranks[(i,j,k)] = a * b * c
                total_rank += block_ranks[(i,j,k)]
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a, b, c = n_blocks[i], m_blocks[j], p_blocks[k]
                rank = block_ranks[(i,j,k)]
                
                # Offsets in global indices
                n_off = sum(n_blocks[:i])
                m_off = sum(m_blocks[:j])
                p_off = sum(p_blocks[:k])
                
                if a == 2 and b == 2 and c == 2:
                    # Use Strassen
                    sU, sV, sW = _strassen_2x2_entries()
                    for local_i in range(2):
                        for local_j in range(2):
                            global_i = n_off + local_i
                            global_j = m_off + local_j
                            u_idx = local_i * 2 + local_j
                            for r in range(7):
                                U[global_i * m + global_j][col_offset + r] += sU[u_idx][r]
                    for local_j in range(2):
                        for local_k in range(2):
                            global_j = m_off + local_j
                            global_k = p_off + local_k
                            v_idx = local_j * 2 + local_k
                            for r in range(7):
                                V[global_j * p + global_k][col_offset + r] += sV[v_idx][r]
                    for local_k in range(2):
                        for local_i in range(2):
                            global_k = p_off + local_k
                            global_i = n_off + local_i
                            w_idx = local_k * 2 + local_i
                            for r in range(7):
                                W[global_k * n + global_i][col_offset + r] += sW[w_idx][r]
                else:
                    # Schoolbook for this block
                    for local_i in range(a):
                        for local_j in range(b):
                            for local_k in range(c):
                                global_i = n_off + local_i
                                global_j = m_off + local_j
                                global_k = p_off + local_k
                                U[global_i * m + global_j][col_offset] = 1
                                V[global_j * p + global_k][col_offset] = 1
                                W[global_k * n + global_i][col_offset] = 1
                                col_offset += 1
                    continue
                
                col_offset += rank
    
    return U, V, W, total_rank


def _try_decomposition(n, m, p, n_split, m_split, p_split):
    """Try a specific block decomposition and return (U, V, W, rank) if valid, else None."""
    try:
        U, V, W, rank = _build_block_decomposition(n, m, p, n_split, m_split, p_split)
        if _verify(U, V, W, n, m, p):
            return U, V, W, rank
    except Exception:
        pass
    return None


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 118  # Safety margin
    
    # Fallback: Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try various block splits
    # n=3: splits (1,2) or (2,1)
    # m=4: splits (1,3), (2,2), (3,1)
    # p=5: splits (1,4), (2,3), (3,2), (4,1)
    
    best_result = None
    best_rank = rank_school
    
    splits_n = [(1,2), (2,1)]
    splits_m = [(1,3), (2,2), (3,1)]
    splits_p = [(1,4), (2,3), (3,2), (4,1)]
    
    for sn in splits_n:
        for sm in splits_m:
            for sp in splits_p:
                if time.time() > deadline:
                    break
                result = _try_decomposition(n, m, p, sn, sm, sp)
                if result is not None:
                    U, V, W, rank = result
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W, rank)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
