import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition strategy.
    # Split m=3 into 2 and 1.
    # Case 1: n=2, m=2, p=7
    # Split p=7 into 2, 2, 2, 1
    # Case 1.1: n=2, m=2, p=2 -> Strassen (rank 7)
    # Case 1.2: n=2, m=2, p=2 -> Strassen (rank 7)
    # Case 1.3: n=2, m=2, p=2 -> Strassen (rank 7)
    # Case 1.4: n=2, m=2, p=1 -> Rank 4 (standard dot product)
    
    # Case 2: n=2, m=1, p=7 -> Rank 1 (outer product)
    
    # Total Rank = 7 + 7 + 7 + 4 + 1 = 26
    
    # Let's verify the ranks and constructions.
    
    # Helper to add a decomposition to the global lists
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(U_block, V_block, W_block, i_offset, j_offset, k_offset):
        """
        U_block: shape (n, r)
        V_block: shape (m, r)
        W_block: shape (p, r)
        Offsets: i in [0, n), j in [0, m), k in [0, p)
        Global U index: (i_offset + i) * m + (j_offset + j)
        Global V index: (j_offset + j) * p + (k_offset + k)
        Global W index: (k_offset + k) * n + (i_offset + i)
        """
        n_b, r = U_block.shape
        m_b = V_block.shape[0]
        p_b = W_block.shape[0]
        
        for r_idx in range(r):
            u_col = U_block[:, r_idx]
            v_col = V_block[:, r_idx]
            w_col = W_block[:, r_idx]
            
            # Create sparse representations or just fill the global arrays later
            # For simplicity, we'll store the tuples (global_idx, value)
            U_entries = []
            V_entries = []
            W_entries = []
            
            for i in range(n_b):
                for j in range(m_b):
                    val = u_col[i] * v_col[j] # Wait, the tensor decomposition is U_{i,j} V_{j,k} W_{k,i}
                    # Actually, the factorization is:
                    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                    # So U_block[i,j,r] is the entry.
                    # My U_block above was defined as (n, r) which implies j is summed or fixed?
                    # No, U has shape (n*m, r). So U_{i*m+j, r}.
                    # So U_block should be (n, m, r) or flattened.
                    # Let's redefine the helper to take full (n, m, r) etc or handle indices.
                    
                    # Let's restart the helper logic to be precise.
                    pass
            
            # Let's just build the global arrays directly with zeros and fill them.
            # This is easier to get right.
            pass

    # Let's use a cleaner approach: build the global U, V, W matrices with zeros,
    # and fill in the blocks.
    
    rank_total = 26
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    current_rank = 0
    
    def add_decomposition(U_block, V_block, W_block, i_off, j_off, k_off):
        """
        U_block: (n_b, m_b, r)
        V_block: (m_b, p_b, r)
        W_block: (p_b, n_b, r)
        """
        nonlocal current_rank
        n_b, m_b, r = U_block.shape
        # Check shapes
        assert V_block.shape == (m_b, W_block.shape[1], r) or V_block.shape[0]==m_b and V_block.shape[1]==W_block.shape[0]
        # Actually W is (p, n, r). So W_block.shape[0] is p_b, W_block.shape[1] is n_b.
        p_b = W_block.shape[0]
        assert V_block.shape == (m_b, p_b, r)
        
        for r_idx in range(r):
            # U_block[:, :, r_idx] is (n_b, m_b)
            # V_block[:, :, r_idx] is (m_b, p_b)
            # W_block[:, :, r_idx] is (p_b, n_b)
            
            u_slice = U_block[:, :, r_idx]
            v_slice = V_block[:, :, r_idx]
            w_slice = W_block[:, :, r_idx]
            
            for i in range(n_b):
                for j in range(m_b):
                    U[(i_off + i) * m + (j_off + j), current_rank + r_idx] = u_slice[i, j]
            
            for j in range(m_b):
                for k in range(p_b):
                    V[(j_off + j) * p + (k_off + k), current_rank + r_idx] = v_slice[j, k]
            
            for k in range(p_b):
                for i in range(n_b):
                    W[(k_off + k) * n + (i_off + i), current_rank + r_idx] = w_slice[k, i]
        
        current_rank += r

    # --- Block 1.1: n=2, m=2, p=2 (Strassen) ---
    # Strassen factors for 2x2x2
    # U: (2,2,7), V: (2,2,7), W: (2,2,7)
    # Indices: 0..6
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Tensor T_{i,j,k} = A_{i,j} B_{j,k} C_{k,i} (indices 0,1)
    # U_{i,j,r}, V_{j,k,r}, W_{k,i,r}
    
    U11 = np.zeros((2, 2, 7))
    V11 = np.zeros((2, 2, 7))
    W11 = np.zeros((2, 2, 7))
    
    # r=0: M1
    U11[0,0,0] = 1; U11[1,1,0] = 1
    V11[0,0,0] = 1; V11[1,1,0] = 1
    W11[0,0,0] = 1; W11[1,1,0] = -1; W11[0,1,0] = 1; W11[1,0,0] = 1
    # Wait, W_{k,i}. C11 uses M1, C22 uses M1.
    # C11 = M1... -> W[0,0] += 1
    # C22 = M1... -> W[1,1] += 1
    # Let's map carefully.
    # C_{k,i} is the output matrix element? No, C is 2x2. C_{row, col}.
    # The tensor is T_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
    # Output C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So W_{k,i} contributes to C_{i,k}.
    
    # Let's define the 7 multiplications explicitly for U, V, W.
    # M1 = (A11+A22)(B11+B22)
    # A part: A11, A22 -> U[0,0]=1, U[1,1]=1
    # B part: B11, B22 -> V[0,0]=1, V[1,1]=1
    # C part: C11 += M1, C22 += M1 -> W[0,0]=1, W[1,1]=1
    U11[0,0,0] = 1; U11[1,1,0] = 1
    V11[0,0,0] = 1; V11[1,1,0] = 1
    W11[0,0,0] = 1; W11[1,1,0] = 1
    
    # M2 = (A21+A22)B11
    # A part: A21, A22 -> U[1,0]=1, U[1,1]=1
    # B part: B11 -> V[0,0]=1
    # C part: C21 += M2, C22 -= M2 -> W[1,1]=1, W[0,1]=-1
    # Note: C_{i,k}. C21 means i=1, k=0. C22 means i=1, k=1.
    # W index is (k, i).
    # C21 -> k=0, i=1 -> W[0,1]
    # C22 -> k=1, i=1 -> W[1,1]
    U11[1,0,1] = 1; U11[1,1,1] = 1
    V11[0,0,1] = 1
    W11[0,1,1] = 1; W11[1,1,1] = -1
    
    # M3 = A11(B12-B22)
    # A part: A11 -> U[0,0]=1
    # B part: B12, B22 -> V[0,1]=1, V[1,1]=-1
    # C part: C12 += M3, C22 += M3
    # C12 -> i=0, k=1 -> W[1,0]
    # C22 -> i=1, k=1 -> W[1,1]
    U11[0,0,2] = 1
    V11[0,1,2] = 1; V11[1,1,2] = -1
    W11[1,0,2] = 1; W11[1,1,2] = 1
    
    # M4 = A22(B21-B11)
    # A part: A22 -> U[1,1]=1
    # B part: B21, B11 -> V[1,0]=1, V[0,0]=-1
    # C part: C11 += M4, C21 += M4
    # C11 -> i=0, k=0 -> W[0,0]
    # C21 -> i=1, k=0 -> W[0,1]
    U11[1,1,3] = 1
    V11[1,0,3] = 1; V11[0,0,3] = -1
    W11[0,0,3] = 1; W11[0,1,3] = 1
    
    # M5 = (A11+A12)B22
    # A part: A11, A12 -> U[0,0]=1, U[0,1]=1
    # B part: B22 -> V[1,1]=1
    # C part: C11 -= M5, C12 += M5
    # C11 -> i=0, k=0 -> W[0,0]
    # C12 -> i=0, k=1 -> W[1,0]
    U11[0,0,4] = 1; U11[0,1,4] = 1
    V11[1,1,4] = 1
    W11[0,0,4] = -1; W11[1,0,4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A part: A21, A11 -> U[1,0]=1, U[0,0]=-1
    # B part: B11, B12 -> V[0,0]=1, V[0,1]=1
    # C part: C22 += M6
    # C22 -> i=1, k=1 -> W[1,1]
    U11[1,0,5] = 1; U11[0,0,5] = -1
    V11[0,0,5] = 1; V11[0,1,5] = 1
    W11[1,1,5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A part: A12, A22 -> U[0,1]=1, U[1,1]=-1
    # B part: B21, B22 -> V[1,0]=1, V[1,1]=1
    # C part: C11 += M7
    # C11 -> i=0, k=0 -> W[0,0]
    U11[0,1,6] = 1; U11[1,1,6] = -1
    V11[1,0,6] = 1; V11[1,1,6] = 1
    W11[0,0,6] = 1
    
    add_decomposition(U11, V11, W11, 0, 0, 0)
    
    # --- Block 1.2: n=2, m=2, p=2 (Strassen) ---
    # Same factors, different offsets
    add_decomposition(U11, V11, W11, 0, 0, 2)
    
    # --- Block 1.3: n=2, m=2, p=2 (Strassen) ---
    add_decomposition(U11, V11, W11, 0, 0, 4)
    
    # --- Block 1.4: n=2, m=2, p=1 ---
    # C_{i,0} = A_{i,0}B_{0,0} + A_{i,1}B_{1,0}
    # T_{i,j,0} = A_{i,j} B_{j,0}
    # Rank 4.
    U14 = np.zeros((2, 2, 4))
    V14 = np.zeros((2, 1, 4))
    W14 = np.zeros((1, 2, 4))
    
    # r=0: A00 B00
    U14[0,0,0] = 1
    V14[0,0,0] = 1
    W14[0,0,0] = 1
    
    # r=1: A01 B10
    U14[0,1,1] = 1
    V14[1,0,1] = 1
    W14[0,0,1] = 1
    
    # r=2: A10 B00
    U14[1,0,2] = 1
    V14[0,0,2] = 1
    W14[0,1,2] = 1
    
    # r=3: A11 B10
    U14[1,1,3] = 1
    V14[1,0,3] = 1
    W14[0,1,3] = 1
    
    add_decomposition(U14, V14, W14, 0, 0, 6)
    
    # --- Block 2: n=2, m=1, p=7 ---
    # C_{i,k} = A_{i,0} B_{0,k}
    # T_{i,0,k} = A_{i,0} B_{0,k}
    # Rank 1.
    U2 = np.zeros((2, 1, 1))
    V2 = np.zeros((1, 7, 1))
    W2 = np.zeros((7, 2, 1))
    
    U2[0,0,0] = 1
    U2[1,0,0] = 1
    for k in range(7):
        V2[0,k,0] = 1
        W2[k,0,0] = 1
        W2[k,1,0] = 1
    
    add_decomposition(U2, V2, W2, 0, 2, 0)
    
    # The total rank used is 7+7+7+4+1 = 26.
    # U shape: (2*3, 26) = (6, 26)
    # V shape: (3*7, 26) = (21, 26)
    # W shape: (7*2, 26) = (14, 26)
    
    # Verify shapes
    assert U.shape == (n*m, rank_total)
    assert V.shape == (m*p, rank_total)
    assert W.shape == (p*n, rank_total)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
