import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    n=2, m=2, p=2.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1 = (A00 + A11) * (B00 + B11)
    # C00 += M1, C11 += M1, C10 -= M1, C01 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[2][0] = -1; W[1][0] = 1
    
    # M2 = (A10 + A11) * B00
    # C10 += M2, C11 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00 * (B01 - B11)
    # C01 += M3, C11 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11 * (B10 - B00)
    # C00 += M4, C10 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00 + A01) * B11
    # C00 -= M5, C01 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    # C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_221():
    """
    Returns U, V, W for 2x2x1 matrix multiplication.
    n=2, m=2, p=1.
    U shape: (4, 4)
    V shape: (2, 4)
    W shape: (2, 4)
    """
    n, m, p = 2, 2, 1
    rank = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # C00 = A00*B00 + A01*B10
    # Term 1: A00*B00 -> C00
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # Term 2: A01*B10 -> C00
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # C10 = A10*B00 + A11*B10
    # Term 3: A10*B00 -> C10
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # Term 4: A11*B10 -> C10
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W

def _concat_tensors(tensors, n, m, p):
    """
    Concatenates a list of tensors (U, V, W) horizontally.
    All tensors must have the same n, m, p.
    """
    if not tensors:
        return [], [], []
    
    U_list, V_list, W_list = zip(*tensors)
    
    U = [row[:] for row in U_list[0]]
    V = [row[:] for row in V_list[0]]
    W = [row[:] for row in W_list[0]]
    
    total_rank = sum(len(U_t[0]) for U_t in U_list)
    
    # Initialize with zeros
    U = [[0] * total_rank for _ in range(len(U[0]) * 0 + n * m)] # Wait, len(U[0]) is not n*m
    # Let's just use the first tensor's dimensions
    n_rows_U = len(U_list[0])
    n_rows_V = len(V_list[0])
    n_rows_W = len(W_list[0])
    
    U = [[0] * total_rank for _ in range(n_rows_U)]
    V = [[0] * total_rank for _ in range(n_rows_V)]
    W = [[0] * total_rank for _ in range(n_rows_W)]
    
    col_offset = 0
    for U_t, V_t, W_t in tensors:
        r = len(U_t[0])
        for i in range(n_rows_U):
            for j in range(r):
                U[i][col_offset + j] = U_t[i][j]
        for i in range(n_rows_V):
            for j in range(r):
                V[i][col_offset + j] = V_t[i][j]
        for i in range(n_rows_W):
            for j in range(r):
                W[i][col_offset + j] = W_t[i][j]
        col_offset += r
        
    return U, V, W

def _build_223():
    """
    Builds 2x2x3 tensor by concatenating 2x2x2 and 2x2x1.
    """
    t1 = _strassen_222()
    t2 = _schoolbook_221()
    
    # We need to map 2x2x1 to 2x2x3?
    # No, 2x2x1 has p=1. 2x2x3 has p=3.
    # We cannot directly concatenate if dimensions differ.
    # We must embed the 2x2x1 tensor into the 2x2x3 space.
    
    # For 2x2x3, p=3.
    # The 2x2x1 tensor uses columns 0 and 1 of B?
    # If we split B (2x3) into B1 (2x1) and B2 (2x2),
    # then C = A B1 + A B2.
    # So we need a 2x2x1 tensor and a 2x2x2 tensor.
    # The 2x2x1 tensor corresponds to the first column of B.
    # The 2x2x2 tensor corresponds to the last two columns of B.
    
    # So we need to shift the indices of the 2x2x2 tensor in the V and W dimensions?
    # V index: j*p + k.
    # For 2x2x1: p=1. V row j*1 + k.
    # For 2x2x3: p=3. V row j*3 + k.
    
    # Let's create a helper to embed a tensor into a larger p dimension.
    
    def embed_UVW(U_in, V_in, W_in, n_in, m_in, p_in, n_out, m_out, p_out, j_offset, k_offset):
        """
        Embeds a tensor with dims (n_in, m_in, p_in) into (n_out, m_out, p_out).
        Assumes n_in == n_out, m_in == m_out.
        j_offset, k_offset are used to map local j, k to global j, k.
        """
        rank = len(U_in[0])
        U_out = [[0] * rank for _ in range(n_out * m_out)]
        V_out = [[0] * rank for _ in range(m_out * p_out)]
        W_out = [[0] * rank for _ in range(p_out * n_out)]
        
        for i in range(n_in):
            for j_local in range(m_in):
                j_global = j_offset + j_local
                for c in range(rank):
                    U_out[i * m_out + j_global][c] = U_in[i * m_in + j_local][c]
                    
        for j_local in range(m_in):
            j_global = j_offset + j_local
            for k_local in range(p_in):
                k_global = k_offset + k_local
                for c in range(rank):
                    V_out[j_global * p_out + k_global][c] = V_in[j_local * p_in + k_local][c]
                    
        for k_local in range(p_in):
            k_global = k_offset + k_local
            for i in range(n_in):
                for c in range(rank):
                    W_out[k_global * n_out + i][c] = W_in[k_local * n_in + i][c]
                    
        return U_out, V_out, W_out

    # Embed 2x2x2 into 2x2x3 with k_offset=1 (last 2 columns)
    U2, V2, W2 = _strassen_222()
    U2_emb, V2_emb, W2_emb = embed_UVW(U2, V2, W2, 2, 2, 2, 2, 2, 3, 0, 1)
    
    # Embed 2x2x1 into 2x2x3 with k_offset=0 (first column)
    U1, V1, W1 = _schoolbook_221()
    U1_emb, V1_emb, W1_emb = embed_UVW(U1, V1, W1, 2, 2, 1, 2, 2, 3, 0, 0)
    
    return _concat_tensors([(U2_emb, V2_emb, W2_emb), (U1_emb, V1_emb, W1_emb)], 2, 2, 3)

def _build_224():
    """
    Builds 2x2x4 tensor by concatenating two 2x2x2 tensors.
    """
    def embed_UVW(U_in, V_in, W_in, n_in, m_in, p_in, n_out, m_out, p_out, j_offset, k_offset):
        rank = len(U_in[0])
        U_out = [[0] * rank for _ in range(n_out * m_out)]
        V_out = [[0] * rank for _ in range(m_out * p_out)]
        W_out = [[0] * rank for _ in range(p_out * n_out)]
        
        for i in range(n_in):
            for j_local in range(m_in):
                j_global = j_offset + j_local
                for c in range(rank):
                    U_out[i * m_out + j_global][c] = U_in[i * m_in + j_local][c]
                    
        for j_local in range(m_in):
            j_global = j_offset + j_local
            for k_local in range(p_in):
                k_global = k_offset + k_local
                for c in range(rank):
                    V_out[j_global * p_out + k_global][c] = V_in[j_local * p_in + k_local][c]
                    
        for k_local in range(p_in):
            k_global = k_offset + k_local
            for i in range(n_in):
                for c in range(rank):
                    W_out[k_global * n_out + i][c] = W_in[k_local * n_in + i][c]
                    
        return U_out, V_out, W_out

    U2, V2, W2 = _strassen_222()
    
    # First 2 columns
    U2_emb1, V2_emb1, W2_emb1 = embed_UVW(U2, V2, W2, 2, 2, 2, 2, 2, 4, 0, 0)
    # Last 2 columns
    U2_emb2, V2_emb2, W2_emb2 = embed_UVW(U2, V2, W2, 2, 2, 2, 2, 2, 4, 0, 2)
    
    return _concat_tensors([(U2_emb1, V2_emb1, W2_emb1), (U2_emb2, V2_emb2, W2_emb2)], 2, 2, 4)

def _build_227():
    """
    Builds 2x2x7 tensor by concatenating 2x2x4 and 2x2x3.
    """
    def embed_UVW(U_in, V_in, W_in, n_in, m_in, p_in, n_out, m_out, p_out, j_offset, k_offset):
        rank = len(U_in[0])
        U_out = [[0] * rank for _ in range(n_out * m_out)]
        V_out = [[0] * rank for _ in range(m_out * p_out)]
        W_out = [[0] * rank for _ in range(p_out * n_out)]
        
        for i in range(n_in):
            for j_local in range(m_in):
                j_global = j_offset + j_local
                for c in range(rank):
                    U_out[i * m_out + j_global][c] = U_in[i * m_in + j_local][c]
                    
        for j_local in range(m_in):
            j_global = j_offset + j_local
            for k_local in range(p_in):
                k_global = k_offset + k_local
                for c in range(rank):
                    V_out[j_global * p_out + k_global][c] = V_in[j_local * p_in + k_local][c]
                    
        for k_local in range(p_in):
            k_global = k_offset + k_local
            for i in range(n_in):
                for c in range(rank):
                    W_out[k_global * n_out + i][c] = W_in[k_local * n_in + i][c]
                    
        return U_out, V_out, W_out

    U4, V4, W4 = _build_224()
    U3, V3, W3 = _build_223()
    
    # Embed 2x2x4 into 2x2x7 with k_offset=0
    U4_emb, V4_emb, W4_emb = embed_UVW(U4, V4, W4, 2, 2, 4, 2, 2, 7, 0, 0)
    
    # Embed 2x2x3 into 2x2x7 with k_offset=4
    U3_emb, V3_emb, W3_emb = embed_UVW(U3, V3, W3, 2, 2, 3, 2, 2, 7, 0, 4)
    
    return _concat_tensors([(U4_emb, V4_emb, W4_emb), (U3_emb, V3_emb, W3_emb)], 2, 2, 7)

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    # Build 2x2x7 tensor
    U227, V227, W227 = _build_227()
    
    # Embed two copies into 2x4x7
    # Copy 1: j_offset=0
    # Copy 2: j_offset=2
    
    def embed_UVW(U_in, V_in, W_in, n_in, m_in, p_in, n_out, m_out, p_out, j_offset, k_offset):
        rank = len(U_in[0])
        U_out = [[0] * rank for _ in range(n_out * m_out)]
        V_out = [[0] * rank for _ in range(m_out * p_out)]
        W_out = [[0] * rank for _ in range(p_out * n_out)]
        
        for i in range(n_in):
            for j_local in range(m_in):
                j_global = j_offset + j_local
                for c in range(rank):
                    U_out[i * m_out + j_global][c] = U_in[i * m_in + j_local][c]
                    
        for j_local in range(m_in):
            j_global = j_offset + j_local
            for k_local in range(p_in):
                k_global = k_offset + k_local
                for c in range(rank):
                    V_out[j_global * p_out + k_global][c] = V_in[j_local * p_in + k_local][c]
                    
        for k_local in range(p_in):
            k_global = k_offset + k_local
            for i in range(n_in):
                for c in range(rank):
                    W_out[k_global * n_out + i][c] = W_in[k_local * n_in + i][c]
                    
        return U_out, V_out, W_out

    U1, V1, W1 = embed_UVW(U227, V227, W227, 2, 2, 7, 2, 4, 7, 0, 0)
    U2, V2, W2 = embed_UVW(U227, V227, W227, 2, 2, 7, 2, 4, 7, 2, 0)
    
    U_final, V_final, W_final = _concat_tensors([(U1, V1, W1), (U2, V2, W2)], 2, 4, 7)
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
