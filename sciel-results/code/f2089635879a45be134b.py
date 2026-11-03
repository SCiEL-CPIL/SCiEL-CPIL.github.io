import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Sub-problem 1: 2x2 x 2x2 (Strassen)
    n1, m1, p1 = 2, 2, 2
    # Sub-problem 2: 1x2 x 2x4 (Schoolbook)
    n2, m2, p2 = 1, 2, 4
    
    # --- Construct Sub-problem 1: 2x2 Strassen ---
    rank1 = 7
    U1 = np.zeros((n1 * m1, rank1))
    V1 = np.zeros((m1 * p1, rank1))
    W1 = np.zeros((p1 * n1, rank1))
    
    # Strassen products
    # P1: (a+d)(e+h)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1 # C00, C11
    
    # P2: (c+d)e
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1 # C10, C11
    
    # P3: a(f-h)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1 # C01, C11
    
    # P4: d(g-e)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1 # C00, C10
    
    # P5: (a+b)h
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1 # C00, C01
    
    # P6: (c-a)(e+f)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1 # C11
    
    # P7: (b-d)(g+h)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1; W1[3, 6] = -1 # C00, C11
    
    # --- Construct Sub-problem 2: 1x2 x 2x4 Schoolbook ---
    rank2 = m2 * p2 * n2 # 2 * 4 * 1 = 8
    U2 = np.zeros((n2 * m2, rank2))
    V2 = np.zeros((m2 * p2, rank2))
    W2 = np.zeros((p2 * n2, rank2))
    
    col = 0
    for i2 in range(n2): # 0
        for j2 in range(m2): # 0, 1
            for k2 in range(p2): # 0, 1, 2, 3
                U2[i2 * m2 + j2, col] = 1
                V2[j2 * p2 + k2, col] = 1
                W2[k2 * n2 + i2, col] = 1
                col += 1
                
    # --- Combine via Kronecker Product ---
    rank_total = rank1 * rank2
    U_total = np.zeros((n * m, rank_total))
    V_total = np.zeros((m * p, rank_total))
    W_total = np.zeros((p * n, rank_total))
    
    # Iterate over all combinations of columns
    for c1 in range(rank1):
        for c2 in range(rank2):
            c_total = c1 * rank2 + c2
            
            # U: No permutation needed
            # u1 = i1*m1 + j1, u2 = i2*m2 + j2
            # u_total = u1*m2 + u2 = (i1*m1+j1)*m2 + j2 = i1*m1*m2 + j1*m2 + j2
            # Required: i*m + j = i1*(m1*m2) + (j1*m2 + j2)
            # They match.
            U_total[:, c_total] = np.kron(U1[:, c1], U2[:, c2])
            
            # V: Permutation needed
            # v1 = j1*p1 + k1, v2 = j2*p2 + k2
            # v_kronecker = v1*p2 + v2 = (j1*p1+k1)*p2 + (j2*p2+k2)
            # Required: j*p + k = (j1*m2+j2)*p + (k1*p2+k2)
            # We compute v_kronecker rows and map to v_required
            V_kron = np.kron(V1[:, c1], V2[:, c2])
            # v_kron index structure:
            # row v_k = v1 * p2 + v2
            # v1 in [0, p1*m1) = [0, 4)
            # v2 in [0, p2*m2) = [0, 8)
            # v_k in [0, 32)
            for v_k in range(len(V_kron)):
                v2 = v_k % p2
                v1 = v_k // p2
                k2 = v2 % p2
                j2 = v2 // p2
                k1 = v1 % p1
                j1 = v1 // p1
                
                # Map to required index
                j = j1 * m2 + j2
                k = k1 * p2 + k2
                v_req = j * p + k
                
                V_total[v_req, c_total] = V_kron[v_k]
                
            # W: Permutation needed
            # w1 = k1*n1 + i1, w2 = k2*n2 + i2
            # w_kronecker = w1*n2 + w2 ? No, standard kron uses row count of second matrix.
            # W1 shape (p1*n1, rank1), W2 shape (p2*n2, rank2)
            # np.kron(W1, W2) has rows indexed by w1 * (p2*n2) + w2
            W_kron = np.kron(W1[:, c1], W2[:, c2])
            for w_k in range(len(W_kron)):
                w2 = w_k % (p2 * n2)
                w1 = w_k // (p2 * n2)
                
                i2 = w2 % n2
                k2 = w2 // n2
                i1 = w1 % n1
                k1 = w1 // n1
                
                # Map to required index
                i = i1 * n2 + i2
                k = k1 * p2 + k2
                w_req = k * n + i
                
                W_total[w_req, c_total] = W_kron[w_k]
                
    # Convert to lists for output
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
