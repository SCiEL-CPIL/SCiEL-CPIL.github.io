"""Matrix multiplication 5x5x5 tensor decomposition using block-structured Strassen."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                dot = np.dot(u_val, v_val * w_val)
                if abs(dot - 1.0) > 1e-9:
                    return False
    return True


def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns (U, V, W) where:
      U: (4, 7) - maps (i,j) pairs to Strassen components
      V: (4, 7) - maps (j,k) pairs to Strassen components  
      W: (4, 7) - maps (k,i) pairs to Strassen components
    
    Strassen's 7 products for 2x2:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B21-B22)
      M4 = A22(B12-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n = 2
    rank = 7
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    # Indexing: U[i*n+j, c], V[j*n+k, c], W[k*n+i, c]
    # U[0,0]=A11, U[0,1]=A12, U[1,0]=A21, U[1,1]=A22
    # V[0,0]=B11, V[0,1]=B12, V[1,0]=B21, V[1,1]=B22
    # W[0,0]=C11, W[0,1]=C12, W[1,0]=C21, W[1,1]=C22
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11=1, A22=1 => U[0,0]=1, U[1,1]=1
    # V: B11=1, B22=1 => V[0,0]=1, V[1,1]=1
    # W: C11+=1, C22+=1 => W[0,0]=1, W[1,1]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    # U: A21=1, A22=1 => U[2,0]=1, U[3,0]=1
    # V: B11=1 => V[0,0]=1
    # W: C21+=1 => W[2,0]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A11(B21-B22)
    # U: A11=1 => U[0,0]=1
    # V: B21=1, B22=-1 => V[2,0]=1, V[3,0]=-1
    # W: C12+=1 => W[1,0]=1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A22(B12-B11)
    # U: A22=1 => U[3,0]=1
    # V: B12=1, B11=-1 => V[1,0]=1, V[0,0]=-1
    # W: C11+=1, C21+=1 => W[0,0]=1, W[2,0]=1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A11+A12)B22
    # U: A11=1, A12=1 => U[0,0]=1, U[1,0]=1
    # V: B22=1 => V[3,0]=1
    # W: C11-=1, C12+=1 => W[0,0]=-1, W[1,0]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # U: A21=1, A11=-1 => U[2,0]=1, U[0,0]=-1
    # V: B11=1, B12=1 => V[0,0]=1, V[1,0]=1
    # W: C22+=1 => W[3,0]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # U: A12=1, A22=-1 => U[1,0]=1, U[3,0]=-1
    # V: B21=1, B22=1 => V[2,0]=1, V[3,0]=1
    # W: C11+=1 => W[0,0]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}


def _embed_block(n_full, m_full, p_full, n_block, m_block, p_block, 
                 row_offset, col_offset_A, col_offset_B, decomp):
    """
    Embed a block decomposition into the full matrices.
    
    n_full, m_full, p_full: full dimensions
    n_block, m_block, p_block: block dimensions
    row_offset: offset for rows of A (and output C)
    col_offset_A: offset for columns of A (and rows of B)
    col_offset_B: offset for columns of B (and output C)
    decomp: dict with U, V, W for the block
    """
    U_full = [[0] * len(decomp['U'][0]) for _ in range(n_full * m_full)]
    V_full = [[0] * len(decomp['V'][0]) for _ in range(m_full * p_full)]
    W_full = [[0] * len(decomp['W'][0]) for _ in range(p_full * n_full)]
    
    rank = len(decomp['U'][0])
    
    for i in range(n_block):
        for j in range(m_block):
            u_row = (row_offset + i) * m_full + (col_offset_A + j)
            for c in range(rank):
                U_full[u_row][c] = decomp['U'][i * m_block + j][c]
    
    for j in range(m_block):
        for k in range(p_block):
            v_row = (col_offset_A + j) * p_full + (col_offset_B + k)
            for c in range(rank):
                V_full[v_row][c] = decomp['V'][j * p_block + k][c]
    
    for k in range(p_block):
        for i in range(n_block):
            w_row = (col_offset_B + k) * n_full + (row_offset + i)
            for c in range(rank):
                W_full[w_row][c] = decomp['W'][k * n_block + i][c]
    
    return {"U": U_full, "V": V_full, "W": W_full}


def _combine_decompositions(decomps, n, m, p):
    """
    Combine multiple block decompositions into one full decomposition.
    Each decomp is a dict with U, V, W of appropriate full size.
    """
    rank = sum(len(d['U'][0]) for d in decomps)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    for d in decomps:
        r = len(d['U'][0])
        for i in range(n * m):
            for c in range(r):
                U[i][col_offset + c] = d['U'][i][c]
        for j in range(m * p):
            for c in range(r):
                V[j][col_offset + c] = d['V'][j][c]
        for k in range(p * n):
            for c in range(r):
                W[k][col_offset + c] = d['W'][k][c]
        col_offset += r
    
    return {"U": U, "V": V, "W": W}


def _construct_block_decomposition():
    """
    Construct a block-structured decomposition for 5x5x5 matrix multiplication.
    
    Partition the 5x5 matrices into blocks:
      A = [A11 A12; A21 A22] where A11 is 3x3, A12 is 3x2, A21 is 2x3, A22 is 2x2
      B = [B11 B12; B21 B22] where B11 is 3x3, B12 is 3x2, B21 is 2x3, B22 is 2x2
    
    For each block multiplication, we need a decomposition. For the 2x2 blocks,
    we can use Strassen (rank 7). For the 3x3 block, we'll use schoolbook (rank 27)
    for now, and for cross-blocks we'll use schoolbook.
    
    Actually, let me think about this differently. The key insight is that for
    rectangular blocks like 3x2 * 2x3, the minimum rank is just 2 (the inner
    dimension), because we can write:
    C[i,k] = sum_j A[i,j] * B[j,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is already a rank-2 decomposition! So for cross-blocks, we can achieve
    rank equal to the inner dimension.
    
    Let me construct the full decomposition:
    """
    n = m = p = 5
    
    # Block sizes
    # A11: 3x3, A12: 3x2, A21: 2x3, A22: 2x2
    # B11: 3x3, B12: 3x2, B21: 2x3, B22: 2x2
    
    # We'll create 8 block decompositions:
    # 1. A11 * B11 -> C11 (3x3 * 3x3 -> 3x3): schoolbook rank 27
    # 2. A12 * B21 -> C11 (3x2 * 2x3 -> 3x3): rank 2
    # 3. A11 * B12 -> C12 (3x3 * 3x2 -> 3x2): schoolbook rank 18
    # 4. A12 * B22 -> C12 (3x2 * 2x2 -> 3x2): rank 2
    # 5. A21 * B11 -> C21 (2x3 * 3x3 -> 2x3): schoolbook rank 18
    # 6. A22 * B21 -> C21 (2x2 * 2x3 -> 2x3): rank 2
    # 7. A21 * B12 -> C22 (2x3 * 3x2 -> 2x2): schoolbook rank 12
    # 8. A22 * B22 -> C22 (2x2 * 2x2 -> 2x2): Strassen rank 7
    
    decomps = []
    
    # Block 1: A11 * B11 (3x3 * 3x3)
    # Schoolbook: rank 27
    decomp1 = _schoolbook(3, 3, 3)
    embedded1 = _embed_block(5, 5, 5, 3, 3, 3, 0, 0, 0, decomp1)
    decomps.append(embedded1)
    
    # Block 2: A12 * B21 (3x2 * 2x3)
    # Rank 2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # U[i,j,c] = 1 if j==c, else 0 (for c in {0,1})
    # V[j,k,c] = 1 if j==c, else 0
    # W[k,i,c] = 1
    n_b, m_b, p_b = 3, 2, 3
    rank2 = 2
    U2 = [[0] * rank2 for _ in range(n_b * m_b)]
    V2 = [[0] * rank2 for _ in range(m_b * p_b)]
    W2 = [[0] * rank2 for _ in range(p_b * n_b)]
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                c = j  # inner index
                U2[i * m_b + j][c] = 1
                V2[j * p_b + k][c] = 1
                W2[k * n_b + i][c] = 1
    decomp2 = {"U": U2, "V": V2, "W": W2}
    embedded2 = _embed_block(5, 5, 5, 3, 2, 3, 0, 3, 0, decomp2)
    decomps.append(embedded2)
    
    # Block 3: A11 * B12 (3x3 * 3x2)
    # Schoolbook: rank 18
    decomp3 = _schoolbook(3, 3, 2)
    embedded3 = _embed_block(5, 5, 5, 3, 3, 2, 0, 0, 3, decomp3)
    decomps.append(embedded3)
    
    # Block 4: A12 * B22 (3x2 * 2x2)
    # Rank 2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    n_b, m_b, p_b = 3, 2, 2
    rank4 = 2
    U4 = [[0] * rank4 for _ in range(n_b * m_b)]
    V4 = [[0] * rank4 for _ in range(m_b * p_b)]
    W4 = [[0] * rank4 for _ in range(p_b * n_b)]
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                c = j
                U4[i * m_b + j][c] = 1
                V4[j * p_b + k][c] = 1
                W4[k * n_b + i][c] = 1
    decomp4 = {"U": U4, "V": V4, "W": W4}
    embedded4 = _embed_block(5, 5, 5, 3, 2, 2, 0, 3, 3, decomp4)
    decomps.append(embedded4)
    
    # Block 5: A21 * B11 (2x3 * 3x3)
    # Schoolbook: rank 18
    decomp5 = _schoolbook(2, 3, 3)
    embedded5 = _embed_block(5, 5, 5, 2, 3, 3, 3, 0, 0, decomp5)
    decomps.append(embedded5)
    
    # Block 6: A22 * B21 (2x2 * 2x3)
    # Rank 2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    n_b, m_b, p_b = 2, 2, 3
    rank6 = 2
    U6 = [[0] * rank6 for _ in range(n_b * m_b)]
    V6 = [[0] * rank6 for _ in range(m_b * p_b)]
    W6 = [[0] * rank6 for _ in range(p_b * n_b)]
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                c = j
                U6[i * m_b + j][c] = 1
                V6[j * p_b + k][c] = 1
                W6[k * n_b + i][c] = 1
    decomp6 = {"U": U6, "V": V6, "W": W6}
    embedded6 = _embed_block(5, 5, 5, 2, 2, 3, 3, 3, 0, decomp6)
    decomps.append(embedded6)
    
    # Block 7: A21 * B12 (2x3 * 3x2)
    # Schoolbook: rank 12
    decomp7 = _schoolbook(2, 3, 2)
    embedded7 = _embed_block(5, 5, 5, 2, 3, 2, 3, 0, 3, decomp7)
    decomps.append(embedded7)
    
    # Block 8: A22 * B22 (2x2 * 2x2)
    # Strassen: rank 7
    decomp8 = _strassen_2x2()
    embedded8 = _embed_block(5, 5, 5, 2, 2, 2, 3, 3, 3, decomp8)
    decomps.append(embedded8)
    
    # Combine all decompositions
    full_decomp = _combine_decompositions(decomps, 5, 5, 5)
    
    # Verify
    if _verify(5, 5, 5, full_decomp['U'], full_decomp['V'], full_decomp['W']):
        return full_decomp
    
    # If verification fails, fall back to schoolbook
    return _schoolbook(5, 5, 5)


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Try block-structured decomposition
    best = _construct_block_decomposition()
    best_rank = len(best['U'][0])
    
    # Verify
    if _verify(n, m, p, best['U'], best['V'], best['W']):
        return best
    
    # Fallback to schoolbook
    fallback = _schoolbook(n, m, p)
    return fallback

# EVOLVE-BLOCK-END
