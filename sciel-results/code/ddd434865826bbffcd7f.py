import numpy as np

def construct():
    n, m, p = 3, 3, 5
    
    # We will build the decomposition by summing contributions from blocks.
    # The total rank is the sum of the ranks of the blocks.
    
    # Block decomposition for 3x3x3 (first 3 columns of B)
    # Split i, j, k into 2+1
    # Blocks: (i_size, j_size, k_size)
    # (2,2,2), (2,2,1), (2,1,2), (2,1,1), (1,2,2), (1,2,1), (1,1,2), (1,1,1)
    
    # Block decomposition for 3x3x2 (last 2 columns of B)
    # Split i, j into 2+1, k is 2
    # Blocks: (2,2,2), (2,1,2), (1,2,2), (1,1,2)
    
    # We need to generate U, V, W for each block.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Since we are summing tensors, we just concatenate the columns of U, V, W.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(n_block, m_block, p_block, i_off, j_off, k_off):
        # This adds the schoolbook decomposition for a sub-block
        # However, for the 2x2x2 block, we should use Strassen (rank 7) instead of schoolbook (rank 8)
        # For others, schoolbook is fine (rank = n*m*p)
        
        if n_block == 2 and m_block == 2 and p_block == 2:
            # Use Strassen
            # Strassen rank 7
            # We need to map the 2x2x2 Strassen decomposition to the global indices
            # Strassen variables:
            # M1 = (A11 + A22) * (B11 + B22)
            # M2 = (A21 + A22) * B11
            # M3 = A11 * (B21 - B22)
            # M4 = A22 * (B12 - B11)
            # M5 = (A11 + A12) * B22
            # M6 = (A21 - A11) * (B11 + B12)
            # M7 = (A12 - A22) * (B21 + B22)
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # We need to construct U, V, W such that U @ V @ W^T reconstructs the products.
            # Actually, the tensor is defined by U[i,j,c], V[j,k,c], W[k,i,c].
            # For each rank-1 term (column c), we add 1 to U, V, W at specific positions.
            
            # Let's list the 7 terms.
            # Each term is a product of a linear form in A, a linear form in B, and a linear form in C.
            # U corresponds to A, V to B, W to C.
            
            # Term 1: (A11 + A22) * (B11 + B22) -> C11, C22
            # U: A11, A22
            # V: B11, B22
            # W: C11 (1), C22 (1)
            
            # We will add 7 columns to U, V, W.
            
            # Local indices for 2x2 block:
            # i in {0,1}, j in {0,1}, k in {0,1}
            # Global i = i_off + i
            # Global j = j_off + j
            # Global k = k_off + k
            
            # U index: i * m + j
            # V index: j * p + k
            # W index: k * n + i
            
            # We create temporary matrices for this block
            u = np.zeros((n_block * m_block, 7))
            v = np.zeros((m_block * p_block, 7))
            w = np.zeros((p_block * n_block, 7))
            
            # Term 1
            u[0, 0] = 1 # A11
            u[3, 0] = 1 # A22
            v[0, 0] = 1 # B11
            v[3, 0] = 1 # B22
            w[0, 0] = 1 # C11
            w[3, 0] = 1 # C22
            
            # Term 2
            u[2, 1] = 1 # A21
            u[3, 1] = 1 # A22
            v[0, 1] = 1 # B11
            w[2, 1] = 1 # C21
            
            # Term 3
            u[0, 2] = 1 # A11
            v[2, 2] = 1 # B21
            v[3, 2] = -1 # B22
            w[1, 2] = 1 # C12
            
            # Term 4
            u[3, 3] = 1 # A22
            v[1, 3] = 1 # B12
            v[0, 3] = -1 # B11
            w[1, 3] = 1 # C12
            
            # Term 5
            u[0, 4] = 1 # A11
            u[1, 4] = 1 # A12
            v[3, 4] = 1 # B22
            w[0, 4] = -1 # C11
            w[3, 4] = 1 # C22
            
            # Term 6
            u[2, 5] = 1 # A21
            u[0, 5] = -1 # A11
            v[0, 5] = 1 # B11
            v[1, 5] = 1 # B12
            w[3, 5] = 1 # C22
            
            # Term 7
            u[1, 6] = 1 # A12
            u[3, 6] = -1 # A22
            v[2, 6] = 1 # B21
            v[3, 6] = 1 # B22
            w[0, 6] = 1 # C11
            
            # Map to global
            # U global indices
            for i in range(n_block):
                for j in range(m_block):
                    U_parts.append(u[i*m_block+j, :])
            # Wait, U_parts is a list of vectors. We need to align them.
            # It's better to build the full matrices at the end.
            
            return u, v, w
            
        else:
            # Schoolbook
            rank = n_block * m_block * p_block
            u = np.zeros((n_block * m_block, rank))
            v = np.zeros((m_block * p_block, rank))
            w = np.zeros((p_block * n_block, rank))
            
            col = 0
            for i in range(n_block):
                for j in range(m_block):
                    for k in range(p_block):
                        u[i*m_block+j, col] = 1
                        v[j*p_block+k, col] = 1
                        w[k*n_block+i, col] = 1
                        col += 1
            return u, v, w

    # Define the blocks for 3x3x3
    # i, j, k splits: 2, 1
    blocks_333 = [
        (2, 2, 2, 0, 0, 0),
        (2, 2, 1, 0, 0, 2),
        (2, 1, 2, 0, 2, 0),
        (2, 1, 1, 0, 2, 2),
        (1, 2, 2, 2, 0, 0),
        (1, 2, 1, 2, 0, 2),
        (1, 1, 2, 2, 2, 0),
        (1, 1, 1, 2, 2, 2)
    ]
    
    # Define the blocks for 3x3x2
    # i, j splits: 2, 1
    # k is 2 (offset 3)
    blocks_332 = [
        (2, 2, 2, 0, 0, 3),
        (2, 1, 2, 0, 2, 3),
        (1, 2, 2, 2, 0, 3),
        (1, 1, 2, 2, 2, 3)
    ]
    
    all_blocks = blocks_333 + blocks_332
    
    U_list = []
    V_list = []
    W_list = []
    
    for n_b, m_b, p_b, i_off, j_off, k_off in all_blocks:
        u, v, w = add_block(n_b, m_b, p_b, i_off, j_off, k_off)
        
        # Expand u, v, w to full size
        U_full = np.zeros((n*m, u.shape[1]))
        V_full = np.zeros((m*p, v.shape[1]))
        W_full = np.zeros((p*n, w.shape[1]))
        
        # Fill U
        for i in range(n_b):
            for j in range(m_b):
                U_full[i_off + i, :] = u[i*m_b+j, :] # Wait, U index is i*m+j? No, U is (n*m, rank).
                # U[i*m+j] is the row corresponding to A[i,j].
                # So U_full[(i_off+i)*m + (j_off+j), :] = u[i*m_b+j, :]
                
        # Correcting U fill
        U_full = np.zeros((n*m, u.shape[1]))
        for i in range(n_b):
            for j in range(m_b):
                U_full[(i_off+i)*m + (j_off+j), :] = u[i*m_b+j, :]
                
        # Fill V
        V_full = np.zeros((m*p, v.shape[1]))
        for j in range(m_b):
            for k in range(p_b):
                V_full[(j_off+j)*p + (k_off+k), :] = v[j*p_b+k, :]
                
        # Fill W
        W_full = np.zeros((p*n, w.shape[1]))
        for k in range(p_b):
            for i in range(n_b):
                W_full[(k_off+k)*n + (i_off+i), :] = w[k*n_b+i, :]
                
        U_list.append(U_full)
        V_list.append(V_full)
        W_list.append(W_full)
        
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return {"U": U, "V": V, "W": W}
