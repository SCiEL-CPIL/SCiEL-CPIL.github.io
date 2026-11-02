import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _get_strassen_2x2_components():
    """
    Returns the 7 rank-1 components for 2x2x2 matrix multiplication.
    Each component is (u, v, w) where u, v, w are 2x2 arrays (flattened or not).
    u[i,j], v[j,k], w[k,i].
    """
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices: 0,1
    # u: A indices (row, col)
    # v: B indices (row, col)
    # w: C indices (row, col) -> w[k, i] in prompt convention? 
    # Prompt: W[k*n+i]. So W index is (k, i).
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So W should correspond to C_{i,k} at index (k, i).
    
    components = []
    
    # M1
    u1 = np.zeros((2,2)); u1[0,0]=1; u1[1,1]=1
    v1 = np.zeros((2,2)); v1[0,0]=1; v1[1,1]=1
    w1 = np.zeros((2,2)); w1[0,0]=1; w1[1,1]=1 # C00, C11
    components.append((u1, v1, w1))
    
    # M2
    u2 = np.zeros((2,2)); u2[1,0]=1; u2[1,1]=1
    v2 = np.zeros((2,2)); v2[0,0]=1
    w2 = np.zeros((2,2)); w2[0,1]=1; w2[1,1]=-1 # C10 (w[0,1]), C11 (w[1,1])
    # Check: C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
    # w2[k,i]. C10 is i=1, k=0. w2[0,1]=1.
    # C11 is i=1, k=1. w2[1,1]=-1.
    components.append((u2, v2, w2))
    
    # M3
    u3 = np.zeros((2,2)); u3[0,0]=1
    v3 = np.zeros((2,2)); v3[0,1]=1; v3[1,1]=-1
    w3 = np.zeros((2,2)); w3[1,0]=1; w3[1,1]=1 # C01 (w[1,0]), C11 (w[1,1])
    # C01 = M3 + M5. i=0, k=1. w3[1,0]=1.
    # C11 = ... + M3. i=1, k=1. w3[1,1]=1.
    components.append((u3, v3, w3))
    
    # M4
    u4 = np.zeros((2,2)); u4[1,1]=1
    v4 = np.zeros((2,2)); v4[1,0]=1; v4[0,0]=-1
    w4 = np.zeros((2,2)); w4[0,1]=1; w4[0,0]=1 # C10 (w[0,1]), C00 (w[0,0])
    # C10 = M2 + M4. i=1, k=0. w4[0,1]=1.
    # C00 = M1 + M4... i=0, k=0. w4[0,0]=1.
    components.append((u4, v4, w4))
    
    # M5
    u5 = np.zeros((2,2)); u5[0,0]=1; u5[0,1]=1
    v5 = np.zeros((2,2)); v5[1,1]=1
    w5 = np.zeros((2,2)); w5[1,1]=-1; w5[0,0]=-1 # C11 (w[1,1]), C00 (w[0,0])
    # C11 = ... - M5. i=1, k=1. w5[1,1]=-1.
    # C00 = ... - M5. i=0, k=0. w5[0,0]=-1.
    components.append((u5, v5, w5))
    
    # M6
    u6 = np.zeros((2,2)); u6[1,0]=1; u6[0,0]=-1
    v6 = np.zeros((2,2)); v6[0,0]=1; v6[0,1]=1
    w6 = np.zeros((2,2)); w6[1,1]=1 # C11 (w[1,1])
    # C11 = ... + M6. i=1, k=1. w6[1,1]=1.
    components.append((u6, v6, w6))
    
    # M7
    u7 = np.zeros((2,2)); u7[0,1]=1; u7[1,1]=-1
    v7 = np.zeros((2,2)); v7[1,0]=1; v7[1,1]=1
    w7 = np.zeros((2,2)); w7[0,0]=1 # C00 (w[0,0])
    # C00 = ... + M7. i=0, k=0. w7[0,0]=1.
    components.append((u7, v7, w7))
    
    return components

def _build_4x4_4x4_decomp():
    """
    Build rank-49 decomposition for 4x4 * 4x4.
    """
    strassen_2x2 = _get_strassen_2x2_components()
    R = 49
    U = np.zeros((16, R))
    V = np.zeros((16, R))
    W = np.zeros((16, R))
    
    # Flatten indices for 2x2: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # But we need to map (block_idx, inner_idx) to 4x4 index.
    # i = 2*i_b + i_i
    
    for k_block in range(7):
        u_b, v_b, w_b = strassen_2x2[k_block]
        for l_inner in range(7):
            u_i, v_i, w_i = strassen_2x2[l_inner]
            c = k_block * 7 + l_inner
            
            # Fill U
            for i_b in range(2):
                for j_b in range(2):
                    coeff_b = u_b[i_b, j_b]
                    if coeff_b == 0: continue
                    for i_i in range(2):
                        for j_i in range(2):
                            coeff_i = u_i[i_i, j_i]
                            if coeff_i == 0: continue
                            i = 2 * i_b + i_i
                            j = 2 * j_b + j_i
                            U[i * 4 + j, c] = coeff_b * coeff_i
                            
            # Fill V
            for j_b in range(2):
                for k_b in range(2):
                    coeff_b = v_b[j_b, k_b]
                    if coeff_b == 0: continue
                    for j_i in range(2):
                        for k_i in range(2):
                            coeff_i = v_i[j_i, k_i]
                            if coeff_i == 0: continue
                            j = 2 * j_b + j_i
                            k = 2 * k_b + k_i
                            V[j * 4 + k, c] = coeff_b * coeff_i
                            
            # Fill W
            # W index is (k, i)
            for k_b in range(2):
                for i_b in range(2):
                    coeff_b = w_b[k_b, i_b]
                    if coeff_b == 0: continue
                    for k_i in range(2):
                        for i_i in range(2):
                            coeff_i = w_i[k_i, i_i]
                            if coeff_i == 0: continue
                            k = 2 * k_b + k_i
                            i = 2 * i_b + i_i
                            W[k * 4 + i, c] = coeff_b * coeff_i
                            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Rank 98 decomposition
    U_444, V_444, W_444 = _build_4x4_4x4_decomp()
    
    R = 98
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Part 1: B[0:4]
    for c in range(49):
        U[:, c] = U_444[:, c]
        V[:, c] = V_444[:, c] # V is 4x4 -> 16 entries. V_444 is 16x49.
        # V shape in 4x4x8: (4*8, 98) = (32, 98).
        # V_444 shape: (16, 49).
        # Map V_444[j*4+k, c] to V[j*8+k, c] for k in 0..3
        for j in range(4):
            for k in range(4):
                V[j * 8 + k, c] = V_444[j * 4 + k, c]
                
        # W shape in 4x4x8: (8*4, 98) = (32, 98).
        # W_444 shape: (16, 49). W_444[k*4+i, c].
        # Map W_444[k*4+i, c] to W[k*4+i, c] for k in 0..3
        for k in range(4):
            for i in range(4):
                W[k * 4 + i, c] = W_444[k * 4 + i, c]
                
    # Part 2: B[4:8]
    for c in range(49):
        c2 = c + 49
        U[:, c2] = U_444[:, c]
        # V for B2: k in 4..7
        for j in range(4):
            for k in range(4):
                V[j * 8 + (k + 4), c2] = V_444[j * 4 + k, c]
                
        # W for C2: k in 4..7
        for k in range(4):
            for i in range(4):
                W[(k + 4) * 4 + i, c2] = W_444[k * 4 + i, c]
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
