"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Schoolbook fallback
    def schoolbook_decomp():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    # 2x2 Strassen Decomposition
    def strassen_2x2():
        # Returns U, V, W for 2x2x2 multiplication
        # U: 4x7, V: 4x7, W: 4x7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices: 0:a, 1:b, 2:c, 3:d for A (row-major 2x2)
        # Indices: 0:e, 1:f, 2:g, 3:h for B
        # Indices: 0:C11, 1:C12, 2:C21, 3:C22 for C (W[k*2+i])
        
        # P1 = (a+d)(e+h)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # P2 = (c+d)e
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        
        # P3 = a(f-h)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        
        # P4 = d(g-e)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        
        # P5 = (a+b)h
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        
        # P6 = (c-a)(e+f)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # P7 = (b-d)(g+h)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[3,6] = 1
        
        return U, V, W

    # 4x4 Strassen Decomposition (Rank 49)
    def strassen_4x4():
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        U2, V2, W2 = strassen_2x2()
        
        # Block indices for 4x4 (flattened row-major)
        # A blocks:
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        # Same for B and C (W is k*4+i, which is column-major for C? 
        # Wait, W[k*n+i] means W is flattened with k as major index.
        # So W[k*4+i] corresponds to C[i,k].
        # C11 (i=0,1; k=0,1): indices 0,1,4,5?
        # k=0: i=0->0, i=1->1. k=1: i=0->4, i=1->5.
        # So C11 indices are 0,1,4,5.
        # C12 (i=0,1; k=2,3): k=2: 8,9. k=3: 12,13. Indices 8,9,12,13.
        # C21 (i=2,3; k=0,1): k=0: 2,3. k=1: 6,7. Indices 2,3,6,7.
        # C22 (i=2,3; k=2,3): k=2: 10,11. k=3: 14,15. Indices 10,11,14,15.
        
        # Let's map the 2x2 block positions to 4x4 indices
        # For a block at (br, bc) where br, bc in {0,1}, size 2x2:
        # Row offset = br*2, Col offset = bc*2
        # Indices in flattened 4x4 (row-major A/B):
        # r in 0..1, c in 0..1 -> (br*2+r)*4 + (bc*2+c)
        
        def get_block_indices(br, bc, mode):
            indices = []
            for r in range(2):
                for c in range(2):
                    if mode == 'A' or mode == 'B':
                        idx = (br*2 + r)*4 + (bc*2 + c)
                    elif mode == 'C':
                        # W[k*4+i] -> k is col of C, i is row of C
                        # Block (br, bc) in C means rows br*2..br*2+1, cols bc*2..bc*2+1
                        # i = br*2+r, k = bc*2+c
                        idx = (bc*2 + c)*4 + (br*2 + r)
                    indices.append(idx)
            return indices

        A_blocks = {
            'A11': get_block_indices(0, 0, 'A'),
            'A12': get_block_indices(0, 1, 'A'),
            'A21': get_block_indices(1, 0, 'A'),
            'A22': get_block_indices(1, 1, 'A')
        }
        B_blocks = {
            'B11': get_block_indices(0, 0, 'B'),
            'B12': get_block_indices(0, 1, 'B'),
            'B21': get_block_indices(1, 0, 'B'),
            'B22': get_block_indices(1, 1, 'B')
        }
        C_blocks = {
            'C11': get_block_indices(0, 0, 'C'),
            'C12': get_block_indices(0, 1, 'C'),
            'C21': get_block_indices(1, 0, 'C'),
            'C22': get_block_indices(1, 1, 'C')
        }
        
        # Strassen products
        # (A_coeffs, B_coeffs, C_contribs)
        products = [
            ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}, {'C11': 1, 'C22': 1}),
            ({'A21': 1, 'A22': 1}, {'B11': 1}, {'C21': 1}),
            ({'A11': 1}, {'B12': 1, 'B22': -1}, {'C12': 1, 'C22': 1}),
            ({'A22': 1}, {'B21': 1, 'B11': -1}, {'C11': 1, 'C21': 1}),
            ({'A11': 1, 'A12': 1}, {'B22': 1}, {'C11': -1, 'C12': 1}),
            ({'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}, {'C22': 1}),
            ({'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}, {'C11': 1, 'C22': 1}),
        ]
        
        for s, (a_c, b_c, c_c) in enumerate(products):
            for inner_c in range(7):
                col = s * 7 + inner_c
                
                # Fill U
                for block, coeff in a_c.items():
                    for idx in A_blocks[block]:
                        # Map 4x4 idx to 2x2 idx
                        # idx = (br*2+r)*4 + (bc*2+c)
                        # 2x2 idx = r*2+c
                        # We need to know r,c from idx and block
                        # Easier: iterate over r,c
                        pass 
                # Let's just use the direct mapping
                # For each idx in block, find r,c
                # idx = (br*2+r)*4 + (bc*2+c)
                # r = (idx // 4 - br*2) % 2
                # c = (idx % 4) - bc*2
                # 2x2_idx = r*2+c
                
                for block, coeff in a_c.items():
                    br, bc = int(block[1]), int(block[2])
                    for idx in A_blocks[block]:
                        r = (idx // 4 - br*2) % 2
                        c_val = (idx % 4) - bc*2
                        idx_2x2 = r*2 + c_val
                        U[idx, col] += coeff * U2[idx_2x2, inner_c]
                        
                for block, coeff in b_c.items():
                    br, bc = int(block[1]), int(block[2])
                    for idx in B_blocks[block]:
                        r = (idx // 4 - br*2) % 2
                        c_val = (idx % 4) - bc*2
                        idx_2x2 = r*2 + c_val
                        V[idx, col] += coeff * V2[idx_2x2, inner_c]
                        
                for block, coeff in c_c.items():
                    br, bc = int(block[1]), int(block[2])
                    for idx in C_blocks[block]:
                        # C block idx is k*4+i
                        # k = bc*2+c, i = br*2+r
                        # idx = k*4+i
                        # k = idx // 4
                        # i = idx % 4
                        # c = k - bc*2
                        # r = i - br*2
                        k = idx // 4
                        i = idx % 4
                        c_val = k - bc*2
                        r = i - br*2
                        idx_2x2 = r*2 + c_val
                        W[idx, col] += coeff * W2[idx_2x2, inner_c]
                        
        return U, V, W

    # 4x4x3 Schoolbook (Rank 48)
    def schoolbook_4x4x3():
        rank = 4 * 4 * 3
        U = np.zeros((16, rank))
        V = np.zeros((12, rank)) # Only for the 3 columns
        W = np.zeros((12, rank)) # Only for the 3 columns
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Combine
    U4, V4, W4 = strassen_4x4()
    U3, V3, W3, rank3 = schoolbook_4x4x3()
    
    rank = 49 + rank3
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    # Part 1: 4x4x4
    U[:, :49] = U4
    # V4 is 16x49 (j*4+k). Full V is 28x49 (j*7+k).
    # j in 0..3, k in 0..3.
    for j in range(4):
        for k in range(4):
            V[j*7+k, :49] = V4[j*4+k, :49]
    # W4 is 16x49 (k*4+i). Full W is 28x49 (k*4+i).
    # k in 0..3.
    W[:16, :49] = W4
    
    # Part 2: 4x4x3
    U[:, 49:] = U3
    # V3 is 12x48 (j*3+k). Full V is 28x48 (j*7+k) for k in 4..6.
    # j in 0..3, k in 0..2 (local) -> global k = 4+k.
    for j in range(4):
        for k in range(3):
            V[j*7 + (4+k), 49:] = V3[j*3+k, :]
    # W3 is 12x48 (k*4+i). Full W is 28x48 (k*4+i) for k in 4..6.
    # k in 0..2 (local) -> global k = 4+k.
    for k in range(3):
        W[(4+k)*4 : (4+k)*4+4, 49:] = W3[k*4 : k*4+4, :]
        
    # Verify
    def verify(U, V, W):
        n, m, p = 4, 4, 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[kp*n+ip, c]
                            target = 1.0 if (ip == i and kp == k) else 0.0
                            if abs(s - target) > 1e-6:
                                return False
        return True

    if time.time() > deadline - 5:
        U, V, W, rank = schoolbook_decomp()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W, rank = schoolbook_decomp()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
