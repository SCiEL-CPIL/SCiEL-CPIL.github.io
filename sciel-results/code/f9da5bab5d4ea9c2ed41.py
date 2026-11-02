import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 recursive Strassen decomposition for 4x4 matrix multiplication.
    Directly constructs the 49 rank-1 tensors by explicit enumeration.
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      U index: i*4 + k  (A row i, A col k)
      V index: k*4 + j  (B row k, B col j)
      W index: j*4 + i  (C col j, C row i)  [trace convention]
    """
    t0 = time.time()
    deadline = 140.0

    n = m = p = 4
    rank = 49

    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)

    # Recursive Strassen: 4x4 = 2x2 of 2x2 blocks
    # Outer level: 7 Strassen terms, each expanding to 7 inner terms = 49 total
    
    # 2x2 Strassen: for matrices A, B (2x2), compute C = A*B with 7 products
    # Indexing: A[row, col], B[row, col], C[row, col]
    # U2 index: row*2+col, V2 index: row*2+col, W2 index: col*2+row (trace convention)
    
    # Define the 7 Strassen terms for 2x2:
    # Each term: (coeffs on A, coeffs on B, coeffs on C)
    # A coeffs: dict from (row,col) to coeff
    # B coeffs: dict from (row,col) to coeff  
    # C coeffs: dict from (row,col) to coeff
    
    strassen_2x2 = [
        # t0: (A00+A11)*(B00+B11) -> C00+=1, C11+=1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # t1: (A10+A11)*B00 -> C10+=1, C11+=-1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # t2: A00*(B01-B11) -> C01+=1, C11+=1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # t3: A11*(B10-B00) -> C00+=1, C10+=1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # t4: (A00+A01)*B11 -> C00+=-1, C01+=1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # t5: (A10-A00)*(B00+B01) -> C11+=1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # t6: (A01-A11)*(B10+B11) -> C00+=1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # For 4x4: outer Strassen operates on 2x2 blocks of 2x2 submatrices
    # Outer A block (br, bc) contains A[2*br:2*br+2, 2*bc:2*bc+2]
    # For each outer term t and inner term k, we get one rank-1 tensor
    
    for t in range(7):
        outer_A, outer_B, outer_C = strassen_2x2[t]
        for k in range(7):
            inner_A, inner_B, inner_C = strassen_2x2[k]
            idx = t * 7 + k
            
            # Fill U: A element at global (i, ak)
            # Global A[i, ak] is in block (i//2, ak//2) at local position (i%2, ak%2)
            for i in range(4):
                for ak in range(4):
                    block_r = i // 2
                    block_c = ak // 2
                    local_r = i % 2
                    local_c = ak % 2
                    outer_coeff = outer_A.get((block_r, block_c), 0)
                    inner_coeff = inner_A.get((local_r, local_c), 0)
                    if outer_coeff != 0 and inner_coeff != 0:
                        U[i*4 + ak, idx] = outer_coeff * inner_coeff
            
            # Fill V: B element at global (bk, j)
            # Global B[bk, j] is in block (bk//2, j//2) at local position (bk%2, j%2)
            for bk in range(4):
                for j in range(4):
                    block_r = bk // 2
                    block_c = j // 2
                    local_r = bk % 2
                    local_c = j % 2
                    outer_coeff = outer_B.get((block_r, block_c), 0)
                    inner_coeff = inner_B.get((local_r, local_c), 0)
                    if outer_coeff != 0 and inner_coeff != 0:
                        V[bk*4 + j, idx] = outer_coeff * inner_coeff
            
            # Fill W: C element at global (ci, cj) -> W index is cj*4+ci (trace convention)
            # Global C[ci, cj] is in block (ci//2, cj//2) at local position (ci%2, cj%2)
            for ci in range(4):
                for cj in range(4):
                    block_r = ci // 2
                    block_c = cj // 2
                    local_r = ci % 2
                    local_c = cj % 2
                    outer_coeff = outer_C.get((block_r, block_c), 0)
                    inner_coeff = inner_C.get((local_r, local_c), 0)
                    if outer_coeff != 0 and inner_coeff != 0:
                        W[cj*4 + ci, idx] = outer_coeff * inner_coeff

    # Verify the decomposition
    max_err = 0.0
    for i in range(4):
        for k in range(4):
            for j in range(4):
                s = 0.0
                for r in range(rank):
                    s += U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r]
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err > 1e-8:
        # Fallback to schoolbook rank-64
        U64 = np.zeros((16, 64), dtype=np.float64)
        V64 = np.zeros((16, 64), dtype=np.float64)
        W64 = np.zeros((16, 64), dtype=np.float64)
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    U64[i*4+k, idx] = 1.0
                    V64[k*4+j, idx] = 1.0
                    W64[j*4+i, idx] = 1.0
                    idx += 1
        return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
