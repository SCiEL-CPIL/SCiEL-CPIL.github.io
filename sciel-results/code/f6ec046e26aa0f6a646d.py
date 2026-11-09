import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen 2x2 element patterns
    # Indexing: [elem00, elem01, elem10, elem11]
    # Corresponds to 2x2 matrix indices (0,0), (0,1), (1,0), (1,1)
    U_elem = [
        [1, 0, 0, 1],   # t=0: a00+a11
        [0, 0, 1, 1],   # t=1: a10+a11
        [1, 0, 0, 0],   # t=2: a00
        [0, 0, 0, 1],   # t=3: a11
        [1, 1, 0, 0],   # t=4: a00+a01
        [-1, 0, 1, 0],  # t=5: a10-a00
        [0, 1, 0, -1],  # t=6: a01-a11
    ]
    V_elem = [
        [1, 0, 0, 1],   # t=0: b00+b11
        [1, 0, 0, 0],   # t=1: b00
        [0, 1, 0, -1],  # t=2: b01-b11
        [-1, 0, 1, 0],  # t=3: b10-b00
        [0, 0, 0, 1],   # t=4: b11
        [1, 1, 0, 0],   # t=5: b00+b01
        [0, 0, 1, 1],   # t=6: b10+b11
    ]
    W_elem = [
        [[1, 0], [0, 1]],   # t=0: C00+=1, C11+=1
        [[0, 0], [1, -1]],  # t=1: C10+=1, C11-=1
        [[0, 1], [0, 1]],   # t=2: C01+=1, C11+=1
        [[1, 0], [1, 0]],   # t=3: C00+=1, C10+=1
        [[-1, 1], [0, 0]],  # t=4: C00-=1, C01+=1
        [[0, 0], [0, 1]],   # t=5: C11+=1
        [[1, 0], [0, 0]],   # t=6: C00+=1
    ]
    
    # Block-level Strassen: which 2x2 blocks and coefficients
    # (block_row, block_col, coeff)
    A_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: A00+A11
        [(1,0,1), (1,1,1)],      # s=1: A10+A11
        [(0,0,1)],               # s=2: A00
        [(1,1,1)],               # s=3: A11
        [(0,0,1), (0,1,1)],      # s=4: A00+A01
        [(1,0,1), (0,0,-1)],     # s=5: A10-A00
        [(0,1,1), (1,1,-1)],     # s=6: A01-A11
    ]
    B_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: B00+B11
        [(0,0,1)],               # s=1: B00
        [(0,1,1), (1,1,-1)],     # s=2: B01-B11
        [(1,0,1), (0,0,-1)],     # s=3: B10-B00
        [(1,1,1)],               # s=4: B11
        [(0,0,1), (0,1,1)],      # s=5: B00+B01
        [(1,0,1), (1,1,1)],      # s=6: B10+B11
    ]
    C_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: C00+=M, C11+=M
        [(1,0,1), (1,1,-1)],     # s=1: C10+=M, C11-=M
        [(0,1,1), (1,1,1)],      # s=2: C01+=M, C11+=M
        [(0,0,1), (1,0,1)],      # s=3: C00+=M, C10+=M
        [(0,0,-1), (0,1,1)],     # s=4: C00-=M, C01+=M
        [(1,1,1)],               # s=5: C11+=M
        [(0,0,1)],               # s=6: C00+=M
    ]
    
    # Initialize matrices
    # U shape: (n*m, rank) = (16, 49)
    # V shape: (m*p, rank) = (16, 49)
    # W shape: (p*n, rank) = (16, 49)
    # Note: The task says W shape (p*n, rank). For 4x4, p*n=16.
    # The index convention is W[k*n+i, r].
    # k is col index of C (0..3), i is row index of C (0..3).
    # Index = k*4 + i.
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Construct U, V, W
    for s in range(7):
        for t in range(7):
            col = s*7 + t
            
            # Build U for this term
            # A is 4x4. Blocks are 2x2.
            # Block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
            # U index is row-major flatten of A: i*4 + j.
            for (bi, bj, c_a) in A_blocks[s]:
                for ei in range(2):
                    for ej in range(2):
                        row_a = (2*bi + ei)*4 + (2*bj + ej)
                        # U_elem[t] is [e00, e01, e10, e11]
                        # ei, ej are local indices 0,1
                        val = U_elem[t][ei*2 + ej]
                        U[row_a][col] += c_a * val
            
            # Build V for this term
            # B is 4x4. Blocks are 2x2.
            # Block (bj, bk) covers rows 2*bj..2*bj+1, cols 2*bk..2*bk+1.
            # V index is row-major flatten of B: j*4 + k.
            for (bj, bk, c_b) in B_blocks[s]:
                for ej in range(2):
                    for ek in range(2):
                        row_b = (2*bj + ej)*4 + (2*bk + ek)
                        # V_elem[t] is [e00, e01, e10, e11]
                        # ej, ek are local indices 0,1
                        val = V_elem[t][ej*2 + ek]
                        V[row_b][col] += c_b * val
            
            # Build W for this term
            # C is 4x4. Blocks are 2x2.
            # Block (bi, bk) covers rows 2*bi..2*bi+1, cols 2*bk..2*bk+1.
            # W index convention: k*n + i.
            # k is global col, i is global row.
            # Index = (2*bk + ek)*4 + (2*bi + ei).
            for (bi, bk, c_c) in C_blocks[s]:
                for ei in range(2):
                    for ek in range(2):
                        # bi is block row, bk is block col
                        # ei is local row, ek is local col
                        row_c = (2*bk + ek)*4 + (2*bi + ei)
                        # W_elem[t] is 2x2 matrix [[w00, w01], [w10, w11]]
                        # w[ei][ek]
                        val = W_elem[t][ei][ek]
                        W[row_c][col] += c_c * val
                        
    # Fast verification using NumPy
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Define a few test cases
        tests = [
            (np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=float),
             np.array([[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]], dtype=float)),
            (np.eye(4, dtype=float),
             np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=float)),
            (np.array([[1,1,1,1],[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=float),
             np.eye(4, dtype=float))
        ]
        
        for A_t, B_t in tests:
            AB_expected = A_t @ B_t
            
            # Compute decomposition result
            # U shape (16, 49), V shape (16, 49), W shape (16, 49)
            # A_t shape (4,4) -> flatten to (16,1) row-major
            A_flat = A_t.flatten()
            B_flat = B_t.flatten()
            
            # u_vals = U.T @ A_flat -> shape (49,)
            # v_vals = V.T @ B_flat -> shape (49,)
            # scalar = u_vals * v_vals -> shape (49,)
            # C_flat = W.T @ scalar -> shape (16,)
            # C = C_flat reshaped according to W index convention
            # W index = k*4 + i. 
            # So C_flat[idx] corresponds to C[i, k] where idx = k*4+i.
            # We need to reconstruct C[i,k] from C_flat.
            
            u_vals = U_np.T @ A_flat
            v_vals = V_np.T @ B_flat
            scalars = u_vals * v_vals
            C_flat = W_np.T @ scalars
            
            # Reconstruct C
            # C_flat is indexed by k*4 + i.
            # C[i, k] = C_flat[k*4 + i]
            C_decomp = np.zeros((4,4), dtype=float)
            for i in range(4):
                for k in range(4):
                    C_decomp[i, k] = C_flat[k*4 + i]
            
            if not np.allclose(AB_expected, C_decomp, atol=1e-9):
                # Fallback to schoolbook rank 64 if verification fails
                rank_fb = 64
                U_fb = [[0]*rank_fb for _ in range(16)]
                V_fb = [[0]*rank_fb for _ in range(16)]
                W_fb = [[0]*rank_fb for _ in range(16)]
                col_fb = 0
                for i2 in range(4):
                    for j2 in range(4):
                        for k2 in range(4):
                            U_fb[i2*4+j2][col_fb] = 1
                            V_fb[j2*4+k2][col_fb] = 1
                            # W index convention: k*n + i
                            W_fb[k2*4+i2][col_fb] = 1
                            col_fb += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
                
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
