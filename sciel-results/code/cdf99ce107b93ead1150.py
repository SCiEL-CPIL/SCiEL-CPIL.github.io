"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Partitions
    # n = 2 + 1
    # m = 2 + 2 + 1
    # p = 2 + 2 + 2
    
    # Blocks:
    # 3 blocks of (2,2,2) -> Rank 7 each
    # 3 blocks of (1,2,2) -> Rank 4 each
    # Total Rank = 3*7 + 3*4 = 21 + 12 = 33
    
    total_rank = 33
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7, 4x7, 4x7
    # Indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    U_222 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0]
    ])
    V_222 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0]
    ])
    # W indices: (k,i) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W_222 = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1]
    ])
    # Note: Strassen formulas vary. This is one standard set.
    # Let's verify one: C00 = A00B00 + A01B10.
    # W00 = 1. U00=1, V00=1. U01=0, V01=1. U02=0, V02=0. U03=0, V03=0.
    # Term 1: U00 V00 W00 = 1*1*1 = 1. (A00 B00)
    # Term 2: U01 V01 W00 = 0.
    # Wait, W00 is for C00.
    # My W_222 row 0 is [1, 1, 0, 1, 0, 0, 0].
    # This means C00 gets contributions from terms 1, 2, 4.
    # Term 1: U00 V00 = A00 B00.
    # Term 2: U01 V01 = A01 B10? No V01 is B10?
    # V index 1 is (0,1) -> B01?
    # Let's stick to a verified Strassen implementation.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U (A part)
    U_s = np.array([
        [1, 0, 0, 0, 1, 0, 0], # A00
        [0, 0, 1, 0, 1, -1, 1], # A01
        [0, 1, 0, 1, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ])
    # V (B part)
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 1], # B00
        [0, 0, 1, 0, 1, 1, 0], # B01
        [1, 0, -1, 1, 0, 0, 1], # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ])
    # W (C part)
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ])
    
    # Rank 4 for 1x2x2
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    # Terms:
    # 1: A0 B00 -> C0
    # 2: A1 B10 -> C0
    # 3: A0 B01 -> C1
    # 4: A1 B11 -> C1
    U_122 = np.array([
        [1, 0, 1, 0], # A0
        [0, 1, 0, 1]  # A1
    ])
    V_122 = np.array([
        [1, 0, 0, 0], # B00
        [0, 1, 0, 0], # B01
        [0, 0, 1, 0], # B10
        [0, 0, 0, 1]  # B11
    ])
    W_122 = np.array([
        [1, 1, 0, 0], # C0
        [0, 0, 1, 1]  # C1
    ])
    
    col_offset = 0
    
    # Block 1: I(0,1), J(0,1), K(0,1) -> 2x2x2
    # I indices: 0,1 (0*2+0, 0*2+1)
    # J indices: 0,1
    # K indices: 0,1
    U[0:2, col_offset:col_offset+7] = U_s
    V[0:2, col_offset:col_offset+7] = V_s
    W[0:2, col_offset:col_offset+7] = W_s
    col_offset += 7
    
    # Block 2: I(0,1), J(2,3), K(2,3) -> 2x2x2
    # I indices: 0,1
    # J indices: 2,3 (2*2+0, 2*2+1)
    # K indices: 2,3
    U[0:2, col_offset:col_offset+7] = U_s
    V[2:4, col_offset:col_offset+7] = V_s
    W[2:4, col_offset:col_offset+7] = W_s
    col_offset += 7
    
    # Block 3: I(0,1), J(4,5), K(4,5) -> 2x2x2
    # I indices: 0,1
    # J indices: 4,5
    # K indices: 4,5
    U[0:2, col_offset:col_offset+7] = U_s
    V[4:6, col_offset:col_offset+7] = V_s
    W[4:6, col_offset:col_offset+7] = W_s
    col_offset += 7
    
    # Block 4: I(2), J(0,1), K(0,1) -> 1x2x2
    # I indices: 2
    # J indices: 0,1
    # K indices: 0,1
    U[2:3, col_offset:col_offset+4] = U_122
    V[0:2, col_offset:col_offset+4] = V_122
    W[0:2, col_offset:col_offset+4] = W_122
    col_offset += 4
    
    # Block 5: I(2), J(2,3), K(2,3) -> 1x2x2
    U[2:3, col_offset:col_offset+4] = U_122
    V[2:4, col_offset:col_offset+4] = V_122
    W[2:4, col_offset:col_offset+4] = W_122
    col_offset += 4
    
    # Block 6: I(2), J(4,5), K(4,5) -> 1x2x2
    U[2:3, col_offset:col_offset+4] = U_122
    V[4:6, col_offset:col_offset+4] = V_122
    W[4:6, col_offset:col_offset+4] = W_122
    col_offset += 4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
