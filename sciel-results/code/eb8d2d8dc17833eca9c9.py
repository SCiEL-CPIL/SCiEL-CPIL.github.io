"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decomposition: M_{4,4,5} = M_{2,2,1} \otimes M_{2,2,5}
    # R(M_{2,2,1}) = 4
    # R(M_{2,2,5}) = R(M_{2,2,2}) + R(M_{2,2,2}) + R(M_{2,2,1}) = 7 + 7 + 4 = 18
    # Total Rank = 4 * 18 = 72
    
    R1 = 4
    R2 = 18
    Rank = R1 * R2
    
    U = [[0] * Rank for _ in range(n * m)]
    V = [[0] * Rank for _ in range(m * p)]
    W = [[0] * Rank for _ in range(p * n)]
    
    # Helper to get Strassen coefficients for 2x2x2
    # Returns list of (u_idx, v_idx, w_idx, val) for U, V, W
    # U indices: i*2+j
    # V indices: j*2+k
    # W indices: k*2+i
    
    def get_strassen_coeffs(offset_k=0):
        # offset_k is added to the local k index (0,1) to get global k index in 0..4
        coeffs = []
        # P1
        # U: (0,0), (1,1) -> 0, 3
        # V: (0,0), (1,1) -> 0, 3 (local k=0,1 -> global k=offset_k, offset_k+1)
        # Wait, Strassen P1 = (A00+A11)(B00+B11)
        # U part: A00, A11.
        # V part: B00, B11.
        # W part: M00, M11.
        
        # It is easier to just define the 7 terms explicitly.
        
        # Term 1: P1 = (A00+A11)(B00+B11)
        # U: A00 (0,0), A11 (1,1)
        # V: B00 (0,0), B11 (1,1)
        # W: M00 (0,0), M11 (1,1)
        # Coeffs:
        # U: (0, 1), (3, 1)
        # V: (0, 1), (3, 1) -- Note V index is j*2+k.
        #    B00: j=0, k=0 -> 0
        #    B11: j=1, k=1 -> 3
        # W: (0, 1), (3, 1) -- Note W index is k*2+i.
        #    M00: k=0, i=0 -> 0
        #    M11: k=1, i=1 -> 3
        
        # This is getting complicated to map manually.
        # Let's use the standard Strassen matrix representation.
        
        # We will generate the non-zero entries for U2, V2, W2 for one Strassen block.
        # Then we apply them with appropriate offsets.
        
        # Strassen 7 terms:
        # 1. (A00+A11)(B00+B11)
        # 2. (A10+A11)B00
        # 3. A00(B01-B11)
        # 4. A11(B10-B00)
        # 5. (A00+A01)B11
        # 6. (A10-A00)(B00+B01)
        # 7. (A01-A11)(B10+B11)
        
        # Results:
        # M00 = 1 + 4 - 5 + 7
        # M01 = 3 + 5
        # M10 = 2 + 4
        # M11 = 1 - 2 + 3 + 6
        
        # We need to populate U2, V2, W2.
        # U2 indices: i*2+j
        # V2 indices: j*2+k (local k 0,1)
        # W2 indices: k*2+i (local k 0,1)
        
        terms = []
        
        # Term 1
        # U: A00 (0), A11 (3)
        # V: B00 (0), B11 (3)
        # W: M00 (0), M11 (3)
        terms.append([(0,1), (3,1), (0,1), (3,1)]) # U idx, val; V idx, val; W idx, val
        
        # Term 2
        # U: A10 (2), A11 (3)
        # V: B00 (0)
        # W: M10 (2), M11 (3)
        terms.append([(2,1), (3,1), (0,1), (2,1), (3,-1)])
        
        # Term 3
        # U: A00 (0)
        # V: B01 (1), B11 (3)
        # W: M00 (0), M01 (1)
        terms.append([(0,1), (1,1), (3,-1), (0,1), (1,1)])
        
        # Term 4
        # U: A11 (3)
        # V: B10 (2), B00 (0)
        # W: M00 (0), M10 (2)
        terms.append([(3,1), (2,1), (0,-1), (0,1), (2,1)])
        
        # Term 5
        # U: A00 (0), A01 (1)
        # V: B11 (3)
        # W: M00 (0), M01 (1)
        terms.append([(0,1), (1,1), (3,1), (0,-1), (1,-1)])
        
        # Term 6
        # U: A10 (2), A00 (0)
        # V: B00 (0), B01 (1)
        # W: M11 (3)
        terms.append([(2,1), (0,-1), (0,1), (1,1), (3,1)])
        
        # Term 7
        # U: A01 (1), A11 (3)
        # V: B10 (2), B11 (3)
        # W: M00 (0), M11 (3)
        terms.append([(1,1), (3,-1), (2,1), (3,1), (0,1), (3,1)])
        
        return terms

    # Precompute U1, V1, W1 non-zeros
    # M_{2,2,1}
    # U1: 4x4, V1: 2x4, W1: 2x4
    U1_nz = []
    V1_nz = []
    W1_nz = []
    for r1 in range(4):
        # r1 maps to j1, i1
        j1 = r1 % 2
        i1 = r1 // 2
        
        I1 = i1 * 2 + j1
        J1 = j1
        K1 = i1
        
        U1_nz.append([(I1, 1)])
        V1_nz.append([(J1, 1)])
        W1_nz.append([(K1, 1)])
        
    # Precompute U2, V2, W2 non-zeros
    U2_nz = []
    V2_nz = []
    W2_nz = []
    
    # Part A: Strassen for k=0,1
    strassen_terms = get_strassen_coeffs()
    for t_idx in range(7):
        u_entries, v_entries, w_entries = [], [], []
        # The terms list structure is mixed. Let's re-extract properly.
        # The previous list `terms` was a bit messy. Let's redefine clearly.
        
    # Redefine Strassen terms clearly for U, V, W
    def get_strassen_nz(offset_k):
        # Returns list of (u_idx, val), (v_idx, val), (w_idx, val) for each of 7 terms
        # offset_k: 0 for cols 0,1; 2 for cols 2,3
        
        res = []
        
        # Term 1
        res.append([
            [(0, 1), (3, 1)],  # U
            [(0, 1), (3 + offset_k*2, 1)], # V: B00 (j0,k0), B11 (j1,k1) -> V idx j*2+k
            # V idx for B00: 0*2+0=0. B11: 1*2+1=3.
            # With offset_k, k becomes k+offset_k.
            # B00 -> 0*2+(0+offset_k) = offset_k.
            # B11 -> 1*2+(1+offset_k) = 3+offset_k.
            [(0, 1), (3 + offset_k, 1)],
            [(0, 1), (3 + offset_k, 1)] # W: M00 (k0,i0), M11 (k1,i1) -> W idx k*2+i
            # W idx for M00: 0*2+0=0. M11: 1*2+1=3.
            # With offset_k, k becomes k+offset_k.
            # M00 -> (0+offset_k)*2+0 = 2*offset_k.
            # M11 -> (1+offset_k)*2+1 = 3+2*offset_k.
        ])
        
        # This manual mapping is error prone. 
        # Let's just iterate over the 7 terms and the 4 outputs.
        
        # Standard Strassen:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # M00 = P1+P4-P5+P7
        # M01 = P3+P5
        # M10 = P2+P4
        # M11 = P1-P2+P3+P6
        
        # We will build U2, V2, W2 for the 7 terms.
        
        terms_data = []
        
        # Term 1
        terms_data.append({
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        })
        # Term 2
        terms_data.append({
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(2, 1), (3, 1)]
        })
        # Term 3
        terms_data.append({
            'U': [(0, 1)],
            'V': [(1, 1), (3, -1)],
            'W': [(0, 1), (1, 1)]
        })
        # Term 4
        terms_data.append({
            'U': [(3, 1)],
            'V': [(2, 1), (0, -1)],
            'W': [(0, 1), (2, 1)]
        })
        # Term 5
        terms_data.append({
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, -1)]
        })
        # Term 6
        terms_data.append({
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        })
        # Term 7
        terms_data.append({
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        })
        
        # Apply offsets
        # V index: j*2 + k. k is 0 or 1.
        # W index: k*2 + i. k is 0 or 1.
        
        for t in terms_data:
            u_list = t['U']
            v_list = []
            for idx, val in t['V']:
                # idx = j*2 + k
                j = idx // 2
                k = idx % 2
                new_k = k + offset_k
                new_idx = j * 2 + new_k
                v_list.append((new_idx, val))
            
            w_list = []
            for idx, val in t['W']:
                # idx = k*2 + i
                k = idx // 2
                i = idx % 2
                new_k = k + offset_k
                new_idx = new_k * 2 + i
                w_list.append((new_idx, val))
            
            res.append([u_list, v_list, w_list])
            
        return res

    # Generate U2, V2, W2 non-zeros
    # Part A: offset 0
    strassen_nz_0 = get_strassen_nz(0)
    for t in strassen_nz_0:
        U2_nz.append(t[0])
        V2_nz.append(t[1])
        W2_nz.append(t[2])
        
    # Part B: offset 2
    strassen_nz_2 = get_strassen_nz(2)
    for t in strassen_nz_2:
        U2_nz.append(t[0])
        V2_nz.append(t[1])
        W2_nz.append(t[2])
        
    # Part C: Schoolbook for k=4
    # M_{2,2,1}
    # U2: i*2+j
    # V2: j*2+4
    # W2: 4*2+i = 8+i
    
    for r_c in range(4):
        j2 = r_c % 2
        i2 = r_c // 2
        
        I2 = i2 * 2 + j2
        V2_idx = j2 * 2 + 4
        W2_idx = 4 * 2 + i2
        
        U2_nz.append([(I2, 1)])
        V2_nz.append([(V2_idx, 1)])
        W2_nz.append([(W2_idx, 1)])
        
    # Now combine
    for r1 in range(R1):
        for r2 in range(R2):
            r = r1 * R2 + r2
            
            u1_list = U1_nz[r1]
            v1_list = V1_nz[r1]
            w1_list = W1_nz[r1]
            
            u2_list = U2_nz[r2]
            v2_list = V2_nz[r2]
            w2_list = W2_nz[r2]
            
            # Kronecker product of non-zeros
            for (I1, val1_u) in u1_list:
                for (I2, val2_u) in u2_list:
                    I = I1 * 2 + I2
                    val = val1_u * val2_u
                    U[I][r] += val
                    
            for (J1, val1_v) in v1_list:
                for (K1, val1_w) in w1_list:
                    # Wait, V and W are separate.
                    # V1 is m1*p1. V2 is m2*p2.
                    # V index: J (m), K (p).
                    # J = J1 * m2 + J2? No.
                    # J is row of B. J1 is row of B1, J2 is row of B2.
                    # J = J1 * 2 + J2.
                    # K is col of B. K1 is col of B1, K2 is col of B2.
                    # K = K1 * 5 + K2.
                    
                    pass # This logic is wrong. V and W must be combined with U correctly.
            
            # Correct Combination:
            # U[I, J, r] = U1[i1, j1, r1] * U2[i2, j2, r2]
            # I = i1*2 + i2, J = j1*2 + j2
            # V[J, K, r] = V1[j1, k1, r1] * V2[j2, k2, r2]
            # J = j1*2 + j2, K = k1*5 + k2
            # W[K, I, r] = W1[k1, i1, r1] * W2[k2, i2, r2]
            # K = k1*5 + k2, I = i1*2 + i2
            
            # We need to iterate over all combinations of non-zeros.
            
            # U
            for (i1, j1, val1) in [(I1//2, I1%2, 0) for I1, _ in u1_list]: # Placeholder
                pass

    # Simplified implementation: Just loop over all indices
    for r1 in range(R1):
        for r2 in range(R2):
            r = r1 * R2 + r2
            
            # U
            for I1 in range(4):
                for I2 in range(4):
                    if U1_nz[r1] and any(u[0]==I1 for u in U1_nz[r1]):
                        val1 = [v for i, v in U1_nz[r1] if i==I1][0]
                    else:
                        val1 = 0
                    if val1 == 0: continue
                    
                    if U2_nz[r2] and any(u[0]==I2 for u in U2_nz[r2]):
                        val2 = [v for i, v in U2_nz[r2] if i==I2][0]
                    else:
                        val2 = 0
                    if val2 == 0: continue
                    
                    I = I1 * 2 + I2
                    U[I][r] += val1 * val2
            
            # V
            for J1 in range(2):
                for K1 in range(1):
                    if V1_nz[r1] and any(v[0]==J1 for v in V1_nz[r1]):
                        val1 = [v for i, v in V1_nz[r1] if i==J1][0]
                    else:
                        val1 = 0
                    if val1 == 0: continue
                    
                    for J2 in range(4):
                        for K2 in range(5):
                            if V2_nz[r2] and any(v[0]==J2*2+K2 for v in V2_nz[r2]):
                                val2 = [v for i, v in V2_nz[r2] if i==J2*2+K2][0]
                            else:
                                val2 = 0
                            if val2 == 0: continue
                            
                            J = J1 * 2 + J2
                            K = K1 * 5 + K2
                            V[J][r] += val1 * val2
                            
            # W
            for K1 in range(1):
                for I1 in range(4):
                    if W1_nz[r1] and any(w[0]==K1*2+I1 for w in W1_nz[r1]):
                        val1 = [v for i, v in W1_nz[r1] if i==K1*2+I1][0]
                    else:
                        val1 = 0
                    if val1 == 0: continue
                    
                    for K2 in range(5):
                        for I2 in range(4):
                            if W2_nz[r2] and any(w[0]==K2*2+I2 for w in W2_nz[r2]):
                                val2 = [v for i, v in W2_nz[r2] if i==K2*2+I2][0]
                            else:
                                val2 = 0
                            if val2 == 0: continue
                            
                            K = K1 * 5 + K2
                            I = I1 * 2 + I2
                            W[K][r] += val1 * val2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
