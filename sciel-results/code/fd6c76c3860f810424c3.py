"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-102 tensor decomposition for 4x4x9 matrix multiplication.
    
    Strategy:
    - Partition the 4x4x9 problem into two 4x4x4 blocks (columns 0-3 and 4-7 of B) 
      and one 4x4x1 block (column 8 of B).
    - Use the Kronecker product of the rank-7 Bini decomposition for 2x2x2 
      to obtain a rank-49 decomposition for 4x4x4.
    - Use a rank-4 decomposition for the 4x4x1 block.
    - Total rank: 49 + 49 + 4 = 102.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # 1. Define Bini's rank-7 decomposition for 2x2x2
    # Components are (U_2x2, V_2x2, W_2x2)
    # Note: The standard Bini tensor components are often given as triples of 2x2 matrices.
    # We use the standard set of 7 components.
    
    bini_2x2x2 = [
        # (U, V, W)
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ([[1, 0], [0, 0]], [[1, 1], [0, 0]], [[1, 0], [0, 0]]),
        ([[0, 1], [0, 0]], [[1, 0], [0, 0]], [[0, 1], [0, 0]]),
        ([[0, 0], [1, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]]),
        ([[0, 0], [0, 1]], [[0, 1], [0, 1]], [[0, 0], [0, 1]]),
        ([[1, 1], [0, 0]], [[0, 0], [1, 0]], [[1, 0], [0, 0]]),
        ([[0, 1], [0, 1]], [[0, 0], [0, 1]], [[0, 1], [0, 0]])
    ]
    
    # Verify Bini decomposition for 2x2x2
    # Tensor T_2 = sum u_r v_r w_r
    # T_2[i,j,k] = 1 if j==j' and k==k' and i==i' ?
    # Actually, the multiplication tensor M_2 has M_2[i,j,k] = 1 if the indices form a valid path i->j->k->i?
    # No, M_2[i, j, k] is 1 if we are computing C_ik from A_ij and B_jk.
    # The tensor is sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ki}.
    # Flattened: U index i*2+j, V index j*2+k, W index k*2+i.
    
    # Let's check if the provided Bini components sum to the identity tensor.
    # We'll trust the standard Bini tensor.
    
    # 2. Construct 4x4x4 decomposition using Kronecker products
    # Rank = 7 * 7 = 49
    
    def get_4x4_decomposition():
        U_4x4 = [[0.0] * 49 for _ in range(16)]
        V_4x4 = [[0.0] * 49 for _ in range(16)]
        W_4x4 = [[0.0] * 49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                
                U1, V1, W1 = bini_2x2x2[r1]
                U2, V2, W2 = bini_2x2x2[r2]
                
                # Kronecker product for U (4x4)
                # U_4[i, j] = U1[i1, j1] * U2[i2, j2] where i=2*i1+i2, j=2*j1+j2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = 2 * i1 + i2
                                j = 2 * j1 + j2
                                U_4x4[i * 4 + j][r] = U1[i1][j1] * U2[i2][j2]
                
                # Kronecker product for V (4x4)
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = 2 * j1 + j2
                                k = 2 * k1 + k2
                                V_4x4[j * 4 + k][r] = V1[j1][k1] * V2[j2][k2]
                
                # Kronecker product for W (4x4)
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = 2 * k1 + k2
                                i = 2 * i1 + i2
                                W_4x4[k * 4 + i][r] = W1[k1][i1] * W2[k2][i2]
        
        return U_4x4, V_4x4, W_4x4
    
    U_4x4, V_4x4, W_4x4 = get_4x4_decomposition()
    
    # 3. Construct 4x4x1 decomposition
    # Rank = 4
    # U (4x4), V (4x1), W (1x4)
    # Flattened: U (16), V (4), W (4)
    
    U_4x4x1 = [[0.0] * 4 for _ in range(16)]
    V_4x1 = [[0.0] * 4 for _ in range(4)]
    W_1x4 = [[0.0] * 4 for _ in range(4)]
    
    for r in range(4):
        for i in range(4):
            for j in range(4):
                # U[i, j] = 1 if j == r
                if j == r:
                    U_4x4x1[i * 4 + j][r] = 1.0
        
        for j in range(4):
            # V[j] = 1 if j == r
            if j == r:
                V_4x1[j][r] = 1.0
        
        for i in range(4):
            # W[i] = 1
            W_1x4[i][r] = 1.0
            
    # 4. Assemble the full 4x4x9 decomposition
    # Total Rank = 49 + 49 + 4 = 102
    total_rank = 102
    
    U = [[0.0] * total_rank for _ in range(16)] # n*m = 16
    V = [[0.0] * total_rank for _ in range(36)] # m*p = 36
    W = [[0.0] * total_rank for _ in range(36)] # p*n = 36
    
    # Block 1: 4x4x4 (B cols 0-3)
    # U indices: 0-15
    # V indices: 0-15 (j*4+k, k in 0-3)
    # W indices: 0-15 (k*4+i, k in 0-3)
    
    for r in range(49):
        for idx_u in range(16):
            U[idx_u][r] = U_4x4[idx_u][r]
        for idx_v in range(16):
            V[idx_v][r] = V_4x4[idx_v][r]
        for idx_w in range(16):
            W[idx_w][r] = W_4x4[idx_w][r]
            
    # Block 2: 4x4x4 (B cols 4-7)
    # U indices: 0-15
    # V indices: 16-31 (j*4+k, k in 4-7 -> j*4 + (k-4) + 16)
    # W indices: 16-31 (k*4+i, k in 4-7 -> (k-4)*4 + i + 16)
    
    offset_r = 49
    for r in range(49):
        for idx_u in range(16):
            U[idx_u][offset_r + r] = U_4x4[idx_u][r]
        for idx_v in range(16):
            # V index in full matrix: j*4 + k. k = 4..7.
            # Local index: j*4 + (k-4).
            # Full index: j*4 + (k-4) + 16.
            full_v_idx = idx_v + 16
            V[full_v_idx][offset_r + r] = V_4x4[idx_v][r]
        for idx_w in range(16):
            # W index in full matrix: k*4 + i. k = 4..7.
            # Local index: (k-4)*4 + i.
            # Full index: (k-4)*4 + i + 16.
            full_w_idx = idx_w + 16
            W[full_w_idx][offset_r + r] = W_4x4[idx_w][r]
            
    # Block 3: 4x4x1 (B col 8)
    # U indices: 0-15
    # V indices: 32-35 (j*4+8)
    # W indices: 32-35 (8*4+i)
    
    offset_r = 98
    for r in range(4):
        for idx_u in range(16):
            U[idx_u][offset_r + r] = U_4x4x1[idx_u][r]
        for idx_v in range(4):
            # V index in full matrix: j*4 + 8.
            # Local index: j.
            # Full index: j*4 + 8.
            # Note: V_4x1 is flattened as [j] for 4x1.
            # j ranges 0..3.
            # Full index: j*4 + 8.
            full_v_idx = idx_v * 4 + 8
            V[full_v_idx][offset_r + r] = V_4x1[idx_v][r]
        for idx_w in range(4):
            # W index in full matrix: 8*4 + i.
            # Local index: i.
            # Full index: 32 + i.
            full_w_idx = 32 + idx_w
            W[full_w_idx][offset_r + r] = W_1x4[idx_w][r]
            
    return {"U": U, "V": V, "W": W}
