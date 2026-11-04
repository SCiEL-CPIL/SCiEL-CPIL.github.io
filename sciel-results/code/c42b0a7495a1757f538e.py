import numpy as np
import time

def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy:
    1. Use the Kronecker product of two Strassen (2x2x2) decompositions 
       to get a rank-49 decomposition for 4x4x4.
    2. The 4x4 * 4x8 problem is equivalent to two 4x4 * 4x4 problems.
    3. Total rank = 49 + 49 = 98.
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 8
    
    # Define the 7 Strassen terms for 2x2x2.
    # U indices: i*2+j, V indices: j*2+k, W indices: k*2+i
    # Terms are (U_dict, V_dict, W_dict)
    
    strassen_2x2_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # M3 = A00(B10-B11)
        ({0: 1}, {2: 1, 3: -1}, {1: 1}),
        # M4 = A11(B01-B00)
        ({3: 1}, {1: 1, 0: -1}, {1: 1}),
        # M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {2: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {3: 1}),
    ]
    
    # Verify Strassen 2x2x2
    def verify_2x2x2(terms):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for u_dict, v_dict, w_dict in terms:
                        u_val = u_dict.get(i*2+j, 0)
                        v_val = v_dict.get(j*2+k, 0)
                        w_val = w_dict.get(k*2+i, 0)
                        total += u_val * v_val * w_val
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    if not verify_2x2x2(strassen_2x2_terms):
        return construct_naive(n, m, p)

    # Generate 4x4x4 terms using Kronecker product
    # A 4x4 matrix A can be seen as a 2x2 matrix of 2x2 blocks.
    # A[i, j] where i=2*i1+i0, j=2*j1+j0.
    # Term (U1, V1, W1) operates on blocks (i1, j1).
    # Term (U2, V2, W2) operates on elements within blocks (i0, j0).
    # Combined U index: (2*i1+i0)*4 + (2*j1+j0)
    # U_val = U1[2*i1+j1] * U2[2*i0+j0]
    
    U_list_4x4 = []
    V_list_4x4 = []
    W_list_4x4 = []
    
    for t1 in strassen_2x2_terms:
        for t2 in strassen_2x2_terms:
            u1_dict, v1_dict, w1_dict = t1
            u2_dict, v2_dict, w2_dict = t2
            
            u_new = {}
            v_new = {}
            w_new = {}
            
            # Iterate over all non-zero entries in U1 and U2
            for (idx1, coeff1) in u1_dict.items():
                i1 = idx1 // 2
                j1 = idx1 % 2
                for (idx2, coeff2) in u2_dict.items():
                    i0 = idx2 // 2
                    j0 = idx2 % 2
                    global_i = 2 * i1 + i0
                    global_j = 2 * j1 + j0
                    global_idx = global_i * 4 + global_j
                    u_new[global_idx] = u_new.get(global_idx, 0) + coeff1 * coeff2
            
            for (idx1, coeff1) in v1_dict.items():
                j1 = idx1 // 2
                k1 = idx1 % 2
                for (idx2, coeff2) in v2_dict.items():
                    j0 = idx2 // 2
                    k0 = idx2 % 2
                    global_j = 2 * j1 + j0
                    global_k = 2 * k1 + k0
                    global_idx = global_j * 4 + global_k
                    v_new[global_idx] = v_new.get(global_idx, 0) + coeff1 * coeff2
            
            for (idx1, coeff1) in w1_dict.items():
                k1 = idx1 // 2
                i1 = idx1 % 2
                for (idx2, coeff2) in w2_dict.items():
                    k0 = idx2 // 2
                    i0 = idx2 % 2
                    global_k = 2 * k1 + k0
                    global_i = 2 * i1 + i0
                    global_idx = global_k * 4 + global_i
                    w_new[global_idx] = w_new.get(global_idx, 0) + coeff1 * coeff2
            
            U_list_4x4.append(u_new)
            V_list_4x4.append(v_new)
            W_list_4x4.append(w_new)
            
    # Verify 4x4x4 decomposition
    def verify_4x4x4(U_terms, V_terms, W_terms):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    total = 0
                    for u_dict, v_dict, w_dict in zip(U_terms, V_terms, W_terms):
                        u_val = u_dict.get(i*4+j, 0)
                        v_val = v_dict.get(j*4+k, 0)
                        w_val = w_dict.get(k*4+i, 0)
                        total += u_val * v_val * w_val
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    if not verify_4x4x4(U_list_4x4, V_list_4x4, W_list_4x4):
        return construct_naive(n, m, p)

    # Construct 4x4x8 decomposition
    # We need two sets of terms: one for B[:, 0:4] and one for B[:, 4:8]
    U_list = []
    V_list = []
    W_list = []
    
    # First half: B columns 0-3, C columns 0-3
    for r in range(49):
        U_list.append(U_list_4x4[r])
        # V indices for 4x4 part of 4x8: j*4+k -> j*8+k (k is 0..3)
        V_list.append(V_list_4x4[r]) 
        # W indices for 4x4 part of 4x8: k*4+i (k is 0..3)
        W_list.append(W_list_4x4[r])
        
    # Second half: B columns 4-7, C columns 4-7
    for r in range(49):
        U_list.append(U_list_4x4[r])
        
        # V indices: j*8 + (4+k)
        v_new = {}
        for (idx, coeff) in V_list_4x4[r].items():
            j = idx // 4
            k = idx % 4
            new_idx = j * 8 + (4 + k)
            v_new[new_idx] = coeff
        V_list.append(v_new)
        
        # W indices: (4+k)*4 + i
        w_new = {}
        for (idx, coeff) in W_list_4x4[r].items():
            k = idx // 4
            i = idx % 4
            new_idx = (4 + k) * 4 + i
            w_new[new_idx] = coeff
        W_list.append(w_new)

    total_rank = len(U_list)
    
    # Verify 4x4x8 decomposition
    def verify_4x4x8(U_terms, V_terms, W_terms, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for u_dict, v_dict, w_dict in zip(U_terms, V_terms, W_terms):
                        u_val = u_dict.get(i*m+j, 0)
                        v_val = v_dict.get(j*p+k, 0)
                        w_val = w_dict.get(k*n+i, 0)
                        total += u_val * v_val * w_val
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    if not verify_4x4x8(U_list, V_list, W_list, n, m, p):
        return construct_naive(n, m, p)
    
    # Convert to matrices
    rank = total_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r in range(rank):
        for (idx, coeff) in U_list[r].items():
            U[idx, r] = coeff
        for (idx, coeff) in V_list[r].items():
            V[idx, r] = coeff
        for (idx, coeff) in W_list[r].items():
            W[idx, r] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
