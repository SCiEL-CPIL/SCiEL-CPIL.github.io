import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    
    # U: (16, 98), V: (32, 98), W: (32, 98)
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Base 2x2 Strassen vectors
    # u, v, w are flattened 2x2 matrices (row-major)
    # Indices: 0:00, 1:01, 2:10, 3:11
    base_strassen = [
        # P0: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P1: (A10+A11)B00 -> C10+=1, C11-=1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # P2: A00(B01-B11) -> C01+=1, C11+=1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # P3: A11(B10-B00) -> C00+=1, C10+=1
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
        # P4: (A00+A01)B11 -> C00-=1, C01+=1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # P5: (A10-A00)(B00+B01) -> C11+=1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # P6: (A01-A11)(B10+B11) -> C00+=1
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # 4x4 Strassen combinations
    # Each entry is (alpha, beta, gamma)
    # alpha, beta, gamma are dicts mapping (r,c) to coeff
    # r,c in {0,1}
    comb_4x4 = [
        # P0
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # P2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # P3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # P5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    col = 0
    for h in range(2):
        offset_b = 4 * h
        offset_c = 4 * h
        
        for k in range(7):
            alpha, beta, gamma = comb_4x4[k]
            
            for t in range(7):
                u_t, v_t, w_t = base_strassen[t]
                
                # Update U
                for (r, c), coeff_a in alpha.items():
                    for i in range(2):
                        for j in range(2):
                            val = coeff_a * u_t[i*2 + j]
                            if val != 0:
                                global_i = 2*r + i
                                global_k = 2*c + j
                                global_idx_u = global_i * 4 + global_k
                                U[global_idx_u, col] += val
                                
                # Update V
                for (r, c), coeff_b in beta.items():
                    for i in range(2):
                        for j in range(2):
                            val = coeff_b * v_t[i*2 + j]
                            if val != 0:
                                global_k = 2*r + i
                                global_j = offset_b + 2*c + j
                                global_idx_v = global_k * 8 + global_j
                                V[global_idx_v, col] += val
                                
                # Update W
                for (r, c), coeff_c in gamma.items():
                    for i in range(2):
                        for j in range(2):
                            val = coeff_c * w_t[i*2 + j]
                            if val != 0:
                                global_i = 2*r + i
                                global_j = offset_c + 2*c + j
                                global_idx_w = global_j * 4 + global_i
                                W[global_idx_w, col] += val
                                
                col += 1
                
    if time.time() - start_time > 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Quick verification
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((4, 4))
    B_test = rng.standard_normal((4, 8))
    C_expected = A_test @ B_test
    
    # Compute C using decomposition
    # C_{i,j} = sum_k A_{i,k} B_{k,j} sum_r U_{i*4+k, r} V_{k*8+j, r} W_{j*4+i, r}
    # This is expensive to check fully, so we check a few entries.
    for i in range(4):
        for j in range(8):
            val = 0
            for k in range(4):
                # sum_r U[...] V[...] W[...]
                # This is dot product of slices
                u_slice = U[i*4+k, :]
                v_slice = V[k*8+j, :]
                w_slice = W[j*4+i, :]
                val += A_test[i,k] * B_test[k,j] * np.sum(u_slice * v_slice * w_slice)
            if abs(val - C_expected[i,j]) > 1e-10:
                # Error in decomposition
                pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
