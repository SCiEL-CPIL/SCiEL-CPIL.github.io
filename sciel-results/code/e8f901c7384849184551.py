import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2_terms():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication.
    Each term: (U_vec, V_vec, W_vec) of length 4.
    U[i] for A[i//2, i%2], V[i] for B[i//2, i%2], W[i] for C[i%2, i//2] (W[k*n+i] with n=2)
    """
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, -1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]),
    ]

def _build_4x4x4_strassen():
    """Build U, V, W for 4x4x4 using Strassen. Rank = 49."""
    n = m = p = 4
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    strassen = _strassen_2x2x2_terms()
    
    # Block-level Strassen terms
    # A blocks: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B blocks: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    # C blocks: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    block_terms = [
        # M1 = (A11+A22)(B11+B22) -> C11 + C22
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        # M2 = (A21+A22)B11 -> C21
        ([((1,0),1), ((1,1),1)], [((0,0),1)], [((1,0),1)]),
        # M3 = A11(B21-B22) -> C12
        ([((0,0),1)], [((1,0),1), ((1,1),-1)], [((0,1),1)]),
        # M4 = A22(B12-B11) -> C11 + C21
        ([((1,1),1)], [((0,1),1), ((0,0),-1)], [((0,0),1), ((1,0),1)]),
        # M5 = (A11+A12)B22 -> -C11 + C12
        ([((0,0),1), ((0,1),1)], [((1,1),1)], [((0,0),-1), ((0,1),1)]),
        # M6 = (A21-A11)(B11+B12) -> C22
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)], [((1,1),1)]),
        # M7 = (A12-A22)(B21+B22) -> C11 + C22
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
    ]
    
    col = 0
    for A_blocks, B_blocks, C_blocks in block_terms:
        for U_sub, V_sub, W_sub in strassen:
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            for (br, bc), coeff in A_blocks:
                base = (br * 2) * 4 + (bc * 2)
                for li in range(4):
                    u_vec[base + li] += coeff * U_sub[li]
            
            for (br, bc), coeff in B_blocks:
                base = (br * 2) * 4 + (bc * 2)
                for li in range(4):
                    v_vec[base + li] += coeff * V_sub[li]
            
            for (br, bc), coeff in C_blocks:
                base = (br * 2) * 4 + (bc * 2)
                for li in range(4):
                    w_vec[base + li] += coeff * W_sub[li]
            
            for i in range(16):
                U[i][col] = u_vec[i]
                V[i][col] = v_vec[i]
                W[i][col] = w_vec[i]
            col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Build 4x4x4 part with Strassen (rank 49)
    U444, V444, W444 = _build_4x4x4_strassen()
    rank444 = 49
    
    # Build 4x4x3 part with schoolbook (rank 48)
    rank443 = 48
    U443 = [[0] * rank443 for _ in range(16)]
    V443 = [[0] * rank443 for _ in range(12)]  # 4*3
    W443 = [[0] * rank443 for _ in range(12)]  # 3*4
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U443[i * 4 + j][col] = 1
                V443[j * 3 + k][col] = 1
                W443[k * 4 + i][col] = 1
                col += 1
    
    # Combine
    total_rank = rank444 + rank443  # 97
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]  # 4*7
    W = [[0] * total_rank for _ in range(28)]  # 7*4
    
    # 4x4x4 part: columns 0-48
    for i in range(16):
        for c in range(rank444):
            U[i][c] = U444[i][c]
            V[i][c] = V444[i][c]
            W[i][c] = W444[i][c]
    
    # 4x4x3 part: columns 49-96
    # V443 corresponds to B[0:4, 4:7], so V rows 16-27
    # W443 corresponds to C[4:7, 0:4], so W rows 16-27
    for i in range(16):
        for c in range(rank443):
            U[i][rank444 + c] = U443[i][c]
            V[16 + i][rank444 + c] = V443[i][c]
            W[16 + i][rank444 + c] = W443[i][c]
    
    return {"U": U, "V": V, "W": W}
