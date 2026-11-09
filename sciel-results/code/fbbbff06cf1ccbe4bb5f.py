import numpy as np

def construct():
    """
    Construct a rank-minimal tensor decomposition for 4x4 matrix multiplication.
    Uses the known rank-49 decomposition (Strassen on 2x2 blocks, each with rank 7).
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n = 4
    rank = 49
    N = n * n  # 16
    
    # Initialize factor matrices
    U = np.zeros((N, rank), dtype=int)
    V = np.zeros((N, rank), dtype=int)
    W = np.zeros((N, rank), dtype=int)
    
    # Helper to get flat index for A[i, j]
    def a_idx(i, j):
        return i * n + j
    
    # Helper to get flat index for W[col * n + row] (trace convention: W[k*n+i] where k=col, i=row)
    # The task description says: W[k*n+i,r]. Usually C = A B implies C_{ij} = sum A_{ik} B_{kj}.
    # Tensor form: T_{i,j,k} = sum_r U_{i,r} V_{k,r} W_{j,r} ? 
    # Let's check the convention: "output factor follows the official trace convention W[k*n+i,r]"
    # This usually means the index of W is `k*n + i`, where `i` is the row of C and `k` is the column of C?
    # Or is it `i` is row of A, `k` is col of A?
    # Standard tensor decomposition for matmul C = AB:
    # C_{ij} = sum_k A_{ik} B_{kj}
    # Tensor T_{i,j,k} = A_{ik} B_{kj} C_{ij}
    # Rank-1 decomposition: T = sum_r u_r v_r w_r
    # U corresponds to A indices (i, k) -> flat i*n+k
    # V corresponds to B indices (k, j) -> flat k*n+j
    # W corresponds to C indices (i, j) -> flat i*n+j
    
    # The prompt says: "W[k*n+i,r]". This implies the first index of W is `k` and second is `i`?
    # Or is `k*n+i` the flat index?
    # If W's flat index is `k*n + i`, and it corresponds to C's indices, we need to know which is row and which is col.
    # In many contexts (like Strassen implementations), if we flatten row-major, C_{ij} is at `i*n+j`.
    # If the prompt explicitly says `W[k*n+i]`, it might be swapping indices or using a specific memory layout.
    # Let's assume standard row-major for A and B:
    # A flat: i*n + k
    # B flat: k*n + j
    # C flat: i*n + j
    # If the prompt says W is indexed by `k*n+i`, it might be that the "row" in the matrix W array corresponds to the flat index.
    # Let's look at the current program's logic:
    # `w_idx = col * 4 + row`
    # This means W is indexed by `col*n + row`.
    # So if C_{row, col} is the output, the index in W is `col*n + row`.
    # This is column-major indexing for the C matrix indices, or simply a specific convention.
    # I will stick to the convention used in the current program which scored 1.3061:
    # U index: row_A * n + col_A
    # V index: row_B * n + col_B
    # W index: col_C * n + row_C
    
    # Strassen 2x2 block decomposition
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Block indices (br, bc) -> (row_block, col_block)
    # Entry (r, s) in block (br, bc) -> global row 2*br+r, col 2*bc+s
    
    def block_flat(br, bc, r, s):
        row = 2 * br + r
        col = 2 * bc + s
        return row * n + col

    # Strassen products for the 2x2 block level
    # Each product is a 2x2 matrix multiplication of linear combinations of blocks.
    # We define the 7 products M1..M7.
    # For each product, we have:
    # X: 2x2 matrix of linear forms in A blocks
    # Y: 2x2 matrix of linear forms in B blocks
    # Z_contrib: How the resulting 2x2 matrix Z contributes to C blocks
    
    # We will iterate through the 7 block products.
    # For each block product, we apply Strassen again to the 2x2 matrices.
    
    # Define the 7 block-level Strassen terms
    # Format: (X_blocks, Y_blocks, C_blocks)
    # X_blocks: list of (block_ref, coeff) for the 2x2 matrix X
    # Y_blocks: list of (block_ref, coeff) for the 2x2 matrix Y
    # C_blocks: list of (block_ref, coeff) for the contribution to C
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    block_strassen = [
        {
            'X': [((0,0), 1), ((1,1), 1)],
            'Y': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        {
            'X': [((1,0), 1), ((1,1), 1)],
            'Y': [((0,0), 1)],
            'C': [((1,0), 1), ((1,1), -1)]
        },
        {
            'X': [((0,0), 1)],
            'Y': [((0,1), 1), ((1,1), -1)],
            'C': [((0,1), 1), ((1,1), 1)]
        },
        {
            'X': [((1,1), 1)],
            'Y': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((1,0), 1)]
        },
        {
            'X': [((0,0), 1), ((0,1), 1)],
            'Y': [((1,1), 1)],
            'C': [((0,0), -1), ((0,1), 1)]
        },
        {
            'X': [((1,0), 1), ((0,0), -1)],
            'Y': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        {
            'X': [((0,1), 1), ((1,1), -1)],
            'Y': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        }
    ]
    
    # Scalar Strassen terms for 2x2 matrices (indices 0..3 for 2x2)
    # 0:00, 1:01, 2:10, 3:11
    scalar_strassen = [
        {
            'X': [(0, 1), (3, 1)],   # x00 + x11
            'Y': [(0, 1), (3, 1)],   # y00 + y11
            'Z': [(0, 1), (3, 1)]    # z00 += 1, z11 += 1
        },
        {
            'X': [(2, 1), (3, 1)],   # x10 + x11
            'Y': [(0, 1)],           # y00
            'Z': [(2, 1), (3, -1)]   # z10 += 1, z11 -= 1
        },
        {
            'X': [(0, 1)],           # x00
            'Y': [(1, 1), (3, -1)],  # y01 - y11
            'Z': [(1, 1), (3, 1)]    # z01 += 1, z11 += 1
        },
        {
            'X': [(3, 1)],           # x11
            'Y': [(2, 1), (0, -1)],  # y10 - y00
            'Z': [(0, 1), (2, 1)]    # z00 += 1, z10 += 1
        },
        {
            'X': [(0, 1), (1, 1)],   # x00 + x01
            'Y': [(3, 1)],           # y11
            'Z': [(0, -1), (1, 1)]   # z00 -= 1, z01 += 1
        },
        {
            'X': [(2, 1), (0, -1)],  # x10 - x00
            'Y': [(0, 1), (1, 1)],   # y00 + y01
            'Z': [(3, 1)]            # z11 += 1
        },
        {
            'X': [(1, 1), (3, -1)],  # x01 - x11
            'Y': [(2, 1), (3, 1)],   # y10 + y11
            'Z': [(0, 1)]            # z00 += 1
        }
    ]
    
    r_idx = 0
    for prod in block_strassen:
        # Build the 2x2 linear forms for X and Y
        # X_form[s] is a dict {flat_idx: coeff} for scalar position s in the 2x2 block
        X_form = [{} for _ in range(4)]
        Y_form = [{} for _ in range(4)]
        
        for (br, bc), coeff in prod['X']:
            for r in range(2):
                for s in range(2):
                    idx = block_flat(br, bc, r, s)
                    pos = r * 2 + s # 0,1,2,3
                    X_form[pos][idx] = X_form[pos].get(idx, 0) + coeff
        
        for (br, bc), coeff in prod['Y']:
            for r in range(2):
                for s in range(2):
                    idx = block_flat(br, bc, r, s)
                    pos = r * 2 + s
                    Y_form[pos][idx] = Y_form[pos].get(idx, 0) + coeff
        
        # Apply scalar Strassen
        for st in scalar_strassen:
            U_dict = {}
            V_dict = {}
            W_dict = {}
            
            # Build U
            for pos, ecoeff in st['X']:
                for flat_idx, val in X_form[pos].items():
                    U_dict[flat_idx] = U_dict.get(flat_idx, 0) + ecoeff * val
            
            # Build V
            for pos, ecoeff in st['Y']:
                for flat_idx, val in Y_form[pos].items():
                    V_dict[flat_idx] = V_dict.get(flat_idx, 0) + ecoeff * val
            
            # Build W
            # Z_contrib: list of (pos, zcoeff)
            # C_contrib: list of (block_ref, ccoeff)
            for zpos, zcoeff in st['Z']:
                zr = zpos // 2
                zs = zpos % 2
                for (cb_r, cb_c), ccoeff in prod['C']:
                    row = 2 * cb_r + zr
                    col = 2 * cb_c + zs
                    # Convention: W index = col * n + row
                    w_idx = col * n + row
                    W_dict[w_idx] = W_dict.get(w_idx, 0) + zcoeff * ccoeff
            
            # Populate arrays
            if U_dict:
                u_rows = list(U_dict.keys())
                u_vals = [U_dict[k] for k in u_rows]
                U[u_rows, r_idx] = u_vals
            if V_dict:
                v_rows = list(V_dict.keys())
                v_vals = [V_dict[k] for k in v_rows]
                V[v_rows, r_idx] = v_vals
            if W_dict:
                w_rows = list(W_dict.keys())
                w_vals = [W_dict[k] for k in w_rows]
                W[w_rows, r_idx] = w_vals
                
            r_idx += 1
            
    return {"U": U, "V": V, "W": W}
