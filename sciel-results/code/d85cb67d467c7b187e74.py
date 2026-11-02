import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition by summing contributions from sub-blocks.
    # Global matrices
    U = np.zeros((n * m, 39), dtype=float)
    V = np.zeros((m * p, 39), dtype=float)
    W = np.zeros((p * n, 39), dtype=float)
    
    rank_offset = 0
    
    # Helper to add a block
    def add_block(u_local, v_local, w_local, i_off, j_off, k_off):
        nonlocal rank_offset
        n_l, m_l, p_l = u_local.shape[0] // m_l, m_l, v_local.shape[0] // m_l # This logic is messy, better to pass dims
        # Actually u_local is (n_l * m_l, r), v_local is (m_l * p_l, r), w_local is (p_l * n_l, r)
        # We need to know n_l, m_l, p_l to map indices.
        pass

    # Let's define the blocks explicitly.
    
    # Block 1: 2x2x2 Strassen at i=0, j=0, k=0
    # n=2, m=2, p=2
    # Strassen rank 7
    u_strassen = np.zeros((4, 7))
    v_strassen = np.zeros((4, 7))
    w_strassen = np.zeros((4, 7))
    
    # M1: (A11+A22)(B11+B22) -> C11, C22
    u_strassen[0, 0] = 1; u_strassen[3, 0] = 1
    v_strassen[0, 0] = 1; v_strassen[3, 0] = 1
    w_strassen[0, 0] = 1; w_strassen[3, 0] = 1
    
    # M2: (A21+A22)B11 -> C21, C22
    u_strassen[2, 1] = 1; u_strassen[3, 1] = 1
    v_strassen[0, 1] = 1
    w_strassen[1, 1] = 1; w_strassen[3, 1] = -1 # C21 gets +1, C22 gets -1? 
    # Wait, C22 = M1 - M2 + ...
    # So w for M2 should have C21=1, C22=-1.
    # Indices for W: k*n + i.
    # C21: i=1, k=0 -> idx 1. C22: i=1, k=1 -> idx 3.
    # So w_strassen[1, 1] = 1, w_strassen[3, 1] = -1.
    
    # M3: A11(B12-B22) -> C12, C22
    u_strassen[0, 2] = 1
    v_strassen[1, 2] = 1; v_strassen[3, 2] = -1
    w_strassen[2, 2] = 1; w_strassen[3, 2] = 1 # C12: i=0,k=1 idx 2. C22: i=1,k=1 idx 3.
    
    # M4: A22(B21-B11) -> C11, C21
    u_strassen[3, 3] = 1
    v_strassen[2, 3] = 1; v_strassen[0, 3] = -1
    w_strassen[0, 3] = 1; w_strassen[1, 3] = 1 # C11: i=0,k=0 idx 0. C21: i=1,k=0 idx 1.
    
    # M5: (A11+A12)B22 -> C11, C12
    u_strassen[0, 4] = 1; u_strassen[1, 4] = 1
    v_strassen[3, 4] = 1
    w_strassen[0, 4] = -1; w_strassen[2, 4] = 1 # C11: -1, C12: 1.
    
    # M6: (A21-A11)(B11+B12) -> C22
    u_strassen[2, 5] = 1; u_strassen[0, 5] = -1
    v_strassen[0, 5] = 1; v_strassen[1, 5] = 1
    w_strassen[3, 5] = 1 # C22: 1.
    
    # M7: (A12-A22)(B21+B22) -> C11
    u_strassen[1, 6] = 1; u_strassen[3, 6] = -1
    v_strassen[2, 6] = 1; v_strassen[3, 6] = 1
    w_strassen[0, 6] = 1 # C11: 1.
    
    # Add Block 1 to global
    # i_off=0, j_off=0, k_off=0
    # U rows: 0..3
    # V rows: 0..3
    # W rows: 0..3
    U[0:4, rank_offset:rank_offset+7] = u_strassen
    V[0:4, rank_offset:rank_offset+7] = v_strassen
    W[0:4, rank_offset:rank_offset+7] = w_strassen
    rank_offset += 7
    
    # Block 2: 2x1x2 Schoolbook at i=0, j=2, k=0
    # n=2, m=1, p=2
    # Rank 4
    u_sb1 = np.zeros((2, 4))
    v_sb1 = np.zeros((2, 4))
    w_sb1 = np.zeros((4, 4))
    
    # A00 B00 -> C00
    u_sb1[0, 0] = 1
    v_sb1[0, 0] = 1
    w_sb1[0, 0] = 1 # C00: i=0,k=0 idx 0
    
    # A00 B01 -> C01
    u_sb1[0, 1] = 1
    v_sb1[1, 1] = 1
    w_sb1[1, 1] = 1 # C01: i=0,k=1 idx 1
    
    # A10 B00 -> C10
    u_sb1[1, 2] = 1
    v_sb1[0, 2] = 1
    w_sb1[2, 2] = 1 # C10: i=1,k=0 idx 2
    
    # A10 B01 -> C11
    u_sb1[1, 3] = 1
    v_sb1[1, 3] = 1
    w_sb1[3, 3] = 1 # C11: i=1,k=1 idx 3
    
    # Map to global
    # U rows: i=0,1; j=2. Global U idx: i*3+j.
    # 0*3+2=2, 1*3+2=5.
    U[2, rank_offset:rank_offset+4] = u_sb1[0]
    U[5, rank_offset:rank_offset+4] = u_sb1[1]
    
    # V rows: j=2; k=0,1. Global V idx: j*7+k.
    # 2*7+0=14, 2*7+1=15.
    V[14, rank_offset:rank_offset+4] = v_sb1[0]
    V[15, rank_offset:rank_offset+4] = v_sb1[1]
    
    # W rows: k=0,1; i=0,1. Global W idx: k*2+i.
    # 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3.
    W[0, rank_offset:rank_offset+4] = w_sb1[0]
    W[1, rank_offset:rank_offset+4] = w_sb1[1]
    W[2, rank_offset:rank_offset+4] = w_sb1[2]
    W[3, rank_offset:rank_offset+4] = w_sb1[3]
    
    rank_offset += 4
    
    # Block 3: 2x2x2 Strassen at i=0, j=0, k=2
    # Same as Block 1 but k_off=2
    U[0:4, rank_offset:rank_offset+7] = u_strassen
    # V rows: j=0,1; k=2,3. Global V idx: j*7+k.
    # 0*7+2=2, 0*7+3=3, 1*7+2=9, 1*7+3=10.
    V[2, rank_offset:rank_offset+7] = v_strassen[0]
    V[3, rank_offset:rank_offset+7] = v_strassen[1]
    V[9, rank_offset:rank_offset+7] = v_strassen[2]
    V[10, rank_offset:rank_offset+7] = v_strassen[3]
    # W rows: k=2,3; i=0,1. Global W idx: k*2+i.
    # 2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7.
    W[4, rank_offset:rank_offset+7] = w_strassen[0]
    W[5, rank_offset:rank_offset+7] = w_strassen[1]
    W[6, rank_offset:rank_offset+7] = w_strassen[2]
    W[7, rank_offset:rank_offset+7] = w_strassen[3]
    
    rank_offset += 7
    
    # Block 4: 2x1x2 Schoolbook at i=0, j=2, k=2
    # Same as Block 2 but k_off=2
    U[2, rank_offset:rank_offset+4] = u_sb1[0]
    U[5, rank_offset:rank_offset+4] = u_sb1[1]
    
    # V rows: j=2; k=2,3. Global V idx: 2*7+2=16, 2*7+3=17.
    V[16, rank_offset:rank_offset+4] = v_sb1[0]
    V[17, rank_offset:rank_offset+4] = v_sb1[1]
    
    # W rows: k=2,3; i=0,1. Global W idx: 4,5,6,7.
    W[4, rank_offset:rank_offset+4] = w_sb1[0]
    W[5, rank_offset:rank_offset+4] = w_sb1[1]
    W[6, rank_offset:rank_offset+4] = w_sb1[2]
    W[7, rank_offset:rank_offset+4] = w_sb1[3]
    
    rank_offset += 4
    
    # Block 5: 2x2x2 Strassen at i=0, j=0, k=4
    U[0:4, rank_offset:rank_offset+7] = u_strassen
    # V rows: j=0,1; k=4,5. Global V idx: 0*7+4=4, 0*7+5=5, 1*7+4=11, 1*7+5=12.
    V[4, rank_offset:rank_offset+7] = v_strassen[0]
    V[5, rank_offset:rank_offset+7] = v_strassen[1]
    V[11, rank_offset:rank_offset+7] = v_strassen[2]
    V[12, rank_offset:rank_offset+7] = v_strassen[3]
    # W rows: k=4,5; i=0,1. Global W idx: 4*2+0=8, 4*2+1=9, 5*2+0=10, 5*2+1=11.
    W[8, rank_offset:rank_offset+7] = w_strassen[0]
    W[9, rank_offset:rank_offset+7] = w_strassen[1]
    W[10, rank_offset:rank_offset+7] = w_strassen[2]
    W[11, rank_offset:rank_offset+7] = w_strassen[3]
    
    rank_offset += 7
    
    # Block 6: 2x1x2 Schoolbook at i=0, j=2, k=4
    U[2, rank_offset:rank_offset+4] = u_sb1[0]
    U[5, rank_offset:rank_offset+4] = u_sb1[1]
    
    # V rows: j=2; k=4,5. Global V idx: 2*7+4=18, 2*7+5=19.
    V[18, rank_offset:rank_offset+4] = v_sb1[0]
    V[19, rank_offset:rank_offset+4] = v_sb1[1]
    
    # W rows: k=4,5; i=0,1. Global W idx: 8,9,10,11.
    W[8, rank_offset:rank_offset+4] = w_sb1[0]
    W[9, rank_offset:rank_offset+4] = w_sb1[1]
    W[10, rank_offset:rank_offset+4] = w_sb1[2]
    W[11, rank_offset:rank_offset+4] = w_sb1[3]
    
    rank_offset += 4
    
    # Block 7: 2x3x1 Schoolbook at i=0, j=0, k=6
    # n=2, m=3, p=1. Rank 6.
    u_sb2 = np.zeros((6, 6))
    v_sb2 = np.zeros((3, 6))
    w_sb2 = np.zeros((2, 6))
    
    for j in range(3):
        for i in range(2):
            idx = i * 3 + j
            u_sb2[idx, idx] = 1
            v_sb2[j, idx] = 1
            w_sb2[i, idx] = 1 # C_i0: i*1+0 = i
    
    U[0:6, rank_offset:rank_offset+6] = u_sb2
    # V rows: j=0,1,2; k=6. Global V idx: 0*7+6=6, 1*7+6=13, 2*7+6=20.
    V[6, rank_offset:rank_offset+6] = v_sb2[0]
    V[13, rank_offset:rank_offset+6] = v_sb2[1]
    V[20, rank_offset:rank_offset+6] = v_sb2[2]
    # W rows: k=6; i=0,1. Global W idx: 6*2+0=12, 6*2+1=13.
    W[12, rank_offset:rank_offset+6] = w_sb2[0]
    W[13, rank_offset:rank_offset+6] = w_sb2[1]
    
    rank_offset += 6
    
    # Total rank used: 7+4+7+4+7+4+6 = 39.
    
    # Convert to list of lists for output
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
