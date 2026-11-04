"""Rewritten program for 4x5x6 matrix multiplication tensor rank minimization."""
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0])
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i = a // m
                j = a % m
                j2 = b // p
                k2 = b % p
                k3 = c // n
                i3 = c % n
                expected = 1 if (j == j2 and i == i3 and k2 == k3) else 0
                s = 0
                for col in range(rank):
                    s += U[a][col] * V[b][col] * W[c][col]
                if s != expected:
                    return False
    return True


def _strassen_2x2x2():
    """Return the 7-rank Strassen decomposition for 2x2x2 matmul.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c], each shape (4, 7).
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    # M2: (a10+a11)b00
    U[2][1]=1; U[3][1]=1; V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    # M3: a00(b01-b11)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1
    # M4: a11(b10-b00)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    # M5: (a00+a01)b11
    U[0][4]=1; U[1][4]=1; V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    # M6: (a10-a00)(b00+b01)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1
    W[3][5]=1
    # M7: (a01-a11)(b10+b11)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1
    W[0][6]=1
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m x p matmul. Returns U, V, W."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _block_decompose(n, m, p, n_splits, m_splits, p_splits):
    """Decompose n x m x p matmul by splitting into blocks.
    n_splits: list of sizes summing to n
    m_splits: list of sizes summing to m  
    p_splits: list of sizes summing to p
    Returns U, V, W for the full problem.
    """
    rank_total = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Build index mappings
    n_offsets = []
    off = 0
    for ns in n_splits:
        n_offsets.append(off)
        off += ns
    m_offsets = []
    off = 0
    for ms in m_splits:
        m_offsets.append(off)
        off += ms
    p_offsets = []
    off = 0
    for ps in p_splits:
        p_offsets.append(off)
        off += ps
    
    # For each block combination, get its decomposition
    block_decs = []
    for ni, ns in enumerate(n_splits):
        for mi, ms in enumerate(m_splits):
            for pi, ps in enumerate(p_splits):
                # This block computes C[ni_block, pi_block] = A[ni_block, mi_block] * B[mi_block, pi_block]
                # A block: ns x ms, B block: ms x ps, C block: ns x ps
                # Use schoolbook for now
                U_b, V_b, W_b = _schoolbook_block(ns, ms, ps)
                block_decs.append((ni, mi, pi, ns, ms, ps, U_b, V_b, W_b))
    
    # Total rank
    rank_total = sum(len(U_b[0]) for _, _, _, _, _, _, U_b, _, _ in block_decs)
    
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    col_offset = 0
    for ni, mi, pi, ns, ms, ps, U_b, V_b, W_b in block_decs:
        r_b = len(U_b[0])
        n_off = n_offsets[ni]
        m_off = m_offsets[mi]
        p_off = p_offsets[pi]
        
        for i in range(ns):
            for j in range(ms):
                for c in range(r_b):
                    U[(n_off+i)*m + (m_off+j)][col_offset + c] = U_b[i*ms+j][c]
        
        for j in range(ms):
            for k in range(ps):
                for c in range(r_b):
                    V[(m_off+j)*p + (p_off+k)][col_offset + c] = V_b[j*ps+k][c]
        
        for k in range(ps):
            for i in range(ns):
                for c in range(r_b):
                    W[(p_off+k)*n + (n_off+i)][col_offset + c] = W_b[k*ns+i][c]
        
        col_offset += r_b
    
    return U, V, W, rank_total


def _strassen_block_decompose(n, m, p, n_splits, m_splits, p_splits):
    """Like _block_decompose but uses Strassen for 2x2x2 blocks."""
    rank_total = 0
    block_decs = []
    
    n_offsets = []
    off = 0
    for ns in n_splits:
        n_offsets.append(off)
        off += ns
    m_offsets = []
    off = 0
    for ms in m_splits:
        m_offsets.append(off)
        off += ms
    p_offsets = []
    off = 0
    for ps in p_splits:
        p_offsets.append(off)
        off += ps
    
    for ni, ns in enumerate(n_splits):
        for mi, ms in enumerate(m_splits):
            for pi, ps in enumerate(p_splits):
                if ns == 2 and ms == 2 and ps == 2:
                    U_b, V_b, W_b = _strassen_2x2x2()
                else:
                    U_b, V_b, W_b = _schoolbook_block(ns, ms, ps)
                block_decs.append((ni, mi, pi, ns, ms, ps, U_b, V_b, W_b))
    
    rank_total = sum(len(U_b[0]) for _, _, _, _, _, _, U_b, _, _ in block_decs)
    
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    col_offset = 0
    for ni, mi, pi, ns, ms, ps, U_b, V_b, W_b in block_decs:
        r_b = len(U_b[0])
        n_off = n_offsets[ni]
        m_off = m_offsets[mi]
        p_off = p_offsets[pi]
        
        for i in range(ns):
            for j in range(ms):
                for c in range(r_b):
                    U[(n_off+i)*m + (m_off+j)][col_offset + c] = U_b[i*ms+j][c]
        
        for j in range(ms):
            for k in range(ps):
                for c in range(r_b):
                    V[(m_off+j)*p + (p_off+k)][col_offset + c] = V_b[j*ps+k][c]
        
        for k in range(ps):
            for i in range(ns):
                for c in range(r_b):
                    W[(p_off+k)*n + (n_off+i)][col_offset + c] = W_b[k*ns+i][c]
        
        col_offset += r_b
    
    return U, V, W, rank_total


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook_block(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try block decompositions with Strassen for 2x2x2 blocks
    # Strategy 1: Split n=4 as 2+2, m=5 as 2+2+1, p=6 as 2+2+2
    # Blocks: 2x2x2 (Strassen, rank 7), 2x2x2 (7), 2x2x2 (7),
    #         2x2x2 (7), 2x2x2 (7), 2x2x2 (7),
    #         2x1x2 (schoolbook, rank 2), 2x1x2 (2), 2x1x2 (2)
    # Total: 6*7 + 3*2 = 42 + 6 = 48
    
    if time.time() < deadline:
        U1, V1, W1, r1 = _strassen_block_decompose(n, m, p, [2,2], [2,2,1], [2,2,2])
        if r1 < best_rank:
            if _verify(U1, V1, W1, n, m, p):
                best_result = {"U": U1, "V": V1, "W": W1}
                best_rank = r1
    
    # Strategy 2: Split n=4 as 2+2, m=5 as 3+2, p=6 as 3+3
    # Blocks: 2x3x3 (schoolbook 18), 2x2x3 (schoolbook 12), 2x3x3 (18), 2x2x3 (12)
    # Total: 60 - not better
    
    # Strategy 3: Split n=4 as 2+2, m=5 as 2+3, p=6 as 2+2+2
    # Blocks: 2x2x2(7), 2x2x2(7), 2x2x2(7), 2x3x2(12), 2x3x2(12), 2x3x2(12)
    # Total: 3*7 + 3*12 = 21+36 = 57
    
    if time.time() < deadline:
        U2, V2, W2, r2 = _strassen_block_decompose(n, m, p, [2,2], [2,3], [2,2,2])
        if r2 < best_rank:
            if _verify(U2, V2, W2, n, m, p):
                best_result = {"U": U2, "V": V2, "W": W2}
                best_rank = r2
    
    # Strategy 4: Split n=4 as 2+2, m=5 as 1+2+2, p=6 as 2+2+2
    # Same as strategy 1 but different m ordering
    # Total: 6*7 + 3*2 = 48 (same)
    
    # Strategy 5: Try n=4 as 1+1+2, m=5 as 2+2+1, p=6 as 2+2+2
    # 1x2x2(4) x3 + 1x2x2(4) x3 + 2x2x2(7) x3 + 1x1x2(2) x3 + 1x1x2(2) x3 + 2x1x2(2) x3
    # = 3*4 + 3*4 + 3*7 + 3*2 + 3*2 + 3*2 = 12+12+21+6+6+6 = 63 - worse
    
    # Strategy 6: n=4 as 2+2, m=5 as 2+2+1, p=6 as 1+2+3
    # 2x2x1(4), 2x2x2(7), 2x2x3(12), 2x2x1(4), 2x2x2(7), 2x2x3(12), 2x1x1(2), 2x1x2(4), 2x1x3(6)
    # Total: 4+7+12+4+7+12+2+4+6 = 58
    
    # Strategy 1 is best so far at 48. Let me verify it's correct.
    # The key insight: when we split all three dimensions, each (ni,mi,pi) block
    # independently computes C[ni_block, pi_block] += A[ni_block, mi_block] * B[mi_block, pi_block]
    # Since different (ni,pi) blocks write to different C positions, they don't interfere.
    # And for a fixed (ni,pi), different mi blocks sum their contributions correctly.
    
    return best_result

# EVOLVE-BLOCK-END
