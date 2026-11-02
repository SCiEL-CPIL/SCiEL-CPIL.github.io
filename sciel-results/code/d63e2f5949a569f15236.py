import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by combining smaller blocks.
    # Target Rank: 38
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vals, v_vals, w_vals):
        """Add one rank-1 term to the lists."""
        U_list.append(u_vals)
        V_list.append(v_vals)
        W_list.append(w_vals)

    # Helper to add a block of terms with offsets
    def add_block(block_U, block_V, block_W, i_off, j_off, k_off, n_block, m_block, p_block):
        """
        Add terms from a block decomposition.
        block_U, block_V, block_W are lists of dicts for the block.
        Offsets are added to the indices.
        """
        for u, v, w in zip(block_U, block_V, block_W):
            new_u = {(i + i_off, j + j_off): val for (i, j), val in u.items()}
            new_v = {(j + j_off, k + k_off): val for (j, k), val in v.items()}
            new_w = {(k + k_off, i + i_off): val for (k, i), val in w.items()}
            add_term(new_u, new_v, new_w)

    # 1. Build R(2,2,2) Strassen (Rank 7)
    # A is 2x2, B is 2x2, C is 2x2
    # Indices i,j in {0,1}, k in {0,1}
    S_U = []
    S_V = []
    S_W = []
    
    # M1
    S_U.append({(0,0): 1, (1,1): 1})
    S_V.append({(0,0): 1, (1,1): 1})
    S_W.append({(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1})
    # M2
    S_U.append({(1,0): 1, (1,1): 1})
    S_V.append({(0,0): 1})
    S_W.append({(0,1): 1})
    # M3
    S_U.append({(0,0): 1})
    S_V.append({(1,0): 1, (1,1): -1})
    S_W.append({(0,0): 1, (1,0): -1})
    # M4
    S_U.append({(1,1): 1})
    S_V.append({(0,1): 1, (1,0): -1})
    S_W.append({(1,1): 1, (0,1): -1})
    # M5
    S_U.append({(0,0): 1, (0,1): 1})
    S_V.append({(1,1): 1})
    S_W.append({(1,0): 1})
    # M6
    S_U.append({(1,0): 1, (0,0): -1})
    S_V.append({(0,0): 1, (1,0): 1})
    S_W.append({(0,1): 1})
    # M7
    S_U.append({(0,1): 1, (1,1): -1})
    S_V.append({(0,1): 1, (1,1): 1})
    S_W.append({(1,0): 1})
    
    # 2. Build R(2,2,1) (Rank 2)
    # A is 2x2, B is 2x1, C is 2x1
    # k is always 0
    D_U = []
    D_V = []
    D_W = []
    
    # Term 1
    D_U.append({(0,0): 1})
    D_V.append({(0,0): 1})
    D_W.append({(0,0): 1, (0,1): 1})
    
    # Term 2
    D_U.append({(0,1): 1, (1,0): 1, (1,1): 1})
    D_V.append({(0,0): 1, (1,0): 1})
    D_W.append({(0,0): 1, (0,1): 1})
    
    # 3. Build R(2,2,3) by combining R(2,2,2) and R(2,2,1)
    # B is 2x3. Split into B1 (2x2, k=0,1) and B2 (2x1, k=2)
    # C is 2x3.
    # We need to map the k indices of the sub-blocks to the global k indices.
    # For R(2,2,2), k is 0,1.
    # For R(2,2,1), k is 0 (local). We will map it to global k=2.
    
    B223_U = []
    B223_V = []
    B223_W = []
    
    # Add Strassen terms (k local 0->0, 1->1)
    for u, v, w in zip(S_U, S_V, S_W):
        B223_U.append(u)
        B223_V.append(v)
        B223_W.append(w)
        
    # Add Direct terms (k local 0->2)
    for u, v, w in zip(D_U, D_V, D_W):
        new_v = {(j, k+2): val for (j, k), val in v.items()}
        new_w = {(k+2, i): val for (k, i), val in w.items()}
        B223_U.append(u)
        B223_V.append(new_v)
        B223_W.append(new_w)
        
    # 4. Build R(2,4,3) by combining two R(2,2,3)
    # A is 2x4. Split into A1 (2x2, j=0,1) and A2 (2x2, j=2,3).
    # B is 4x3. Split into B1 (2x3, j=0,1) and B2 (2x3, j=2,3).
    # C is 2x3.
    
    B243_U = []
    B243_V = []
    B243_W = []
    
    # Block 1: j=0,1
    for u, v, w in zip(B223_U, B223_V, B223_W):
        B243_U.append(u)
        B243_V.append(v)
        B243_W.append(w)
        
    # Block 2: j=2,3 (offset j by 2)
    for u, v, w in zip(B223_U, B223_V, B223_W):
        new_u = {(i, j+2): val for (i, j), val in u.items()}
        new_v = {(j+2, k): val for (j, k), val in v.items()}
        B243_U.append(new_u)
        B243_V.append(new_v)
        B243_W.append(w) # W indices (k,i) don't change with j
        
    # 5. Build R(1,4,3) (Rank 1)
    # A is 1x4, B is 4x3, C is 1x3
    B143_U = [{(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1}]
    B143_V = [{(0,0): 1, (0,1): 1, (0,2): 1, 
               (1,0): 1, (1,1): 1, (1,2): 1,
               (2,0): 1, (2,1): 1, (2,2): 1,
               (3,0): 1, (3,1): 1, (3,2): 1}]
    B143_W = [{(0,0): 1, (1,0): 1, (2,0): 1}]
    
    # 6. Build R(3,4,3) by combining R(2,4,3) and R(1,4,3)
    # A is 3x4. Split into A1 (2x4, i=0,1) and A2 (1x4, i=2).
    # B is 4x3.
    # C is 3x3.
    
    B343_U = []
    B343_V = []
    B343_W = []
    
    # Block 1: i=0,1
    for u, v, w in zip(B243_U, B243_V, B243_W):
        B343_U.append(u)
        B343_V.append(v)
        B343_W.append(w)
        
    # Block 2: i=2 (offset i by 2)
    for u, v, w in zip(B143_U, B143_V, B143_W):
        new_u = {(i+2, j): val for (i, j), val in u.items()}
        new_w = {(k, i+2): val for (k, i), val in w.items()}
        B343_U.append(new_u)
        B343_V.append(v) # V indices (j,k) don't change with i
        B343_W.append(new_w)
        
    # 7. Build R(3,4,6) by combining two R(3,4,3)
    # B is 4x6. Split into B1 (4x3, k=0,1,2) and B2 (4x3, k=3,4,5).
    # A is 3x4.
    # C is 3x6.
    
    final_U = []
    final_V = []
    final_W = []
    
    # Block 1: k=0,1,2
    for u, v, w in zip(B343_U, B343_V, B343_W):
        final_U.append(u)
        final_V.append(v)
        final_W.append(w)
        
    # Block 2: k=3,4,5 (offset k by 3)
    for u, v, w in zip(B343_U, B343_V, B343_W):
        new_v = {(j, k+3): val for (j, k), val in v.items()}
        new_w = {(k+3, i): val for (k, i), val in w.items()}
        final_U.append(u) # U indices (i,j) don't change with k
        final_V.append(new_v)
        final_W.append(new_w)
        
    rank = len(final_U)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for col in range(rank):
        for (i, j), val in final_U[col].items():
            U[i * m + j, col] = val
        for (j, k), val in final_V[col].items():
            V[j * p + k, col] = val
        for (k, i), val in final_W[col].items():
            W[k * n + i, col] = val
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
