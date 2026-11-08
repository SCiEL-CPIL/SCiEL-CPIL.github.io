import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank-22 decomposition
    # by combining two rank-11 decompositions of M_{2,3,2}
    
    # First, construct a rank-11 decomposition of M_{2,3,2}
    # M_{2,3,2} = M_{2,2,2} (j=0,1) + M_{2,1,2} (j=2)
    
    # Winograd rank-7 decomposition for M_{2,2,2}
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # A indices: i*2 + j
    # B indices: j*2 + k
    # C indices: k*2 + i
    
    # The 7 terms for Winograd 2x2x2
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a10(b01+b11)
    # m4 = a00 b10
    # m5 = (a00+a01)b11
    # m6 = (a01-a10)(b10+b11)
    # m7 = (a00-a10)b01
    
    # We need to express these as u, v, w vectors
    # u is length 4 (for 2x2 A), v is length 4 (for 2x2 B), w is length 4 (for 2x2 C)
    
    # Term 1: (a00+a11)(b00+b11)
    u1 = [1, 0, 0, 1] # a00, a01, a10, a11
    v1 = [1, 0, 0, 1] # b00, b01, b10, b11
    w1 = [1, 1, 1, 1] # c00, c01, c10, c11 (coeffs: 1, 1, 1, 1)
    # Check: c00 += 1*m1, c01 += 1*m1, c10 += 1*m1, c11 += 1*m1
    
    # Term 2: (a10+a11)b00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [-1, 0, 1, 0] # c00 -= m2, c10 += m2
    
    # Term 3: a10(b01+b11)
    u3 = [0, 0, 1, 0]
    v3 = [0, 1, 0, 1]
    w3 = [0, 1, -1, 1] # c01 += m3, c10 -= m3, c11 += m3
    
    # Term 4: a00 b10
    u4 = [1, 0, 0, 0]
    v4 = [0, 0, 1, 0]
    w4 = [-1, 0, 1, 0] # c00 -= m4, c10 += m4
    
    # Term 5: (a00+a01)b11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [0, -1, 0, 1] # c01 -= m5, c11 += m5
    
    # Term 6: (a01-a10)(b10+b11)
    u6 = [0, 1, -1, 0]
    v6 = [0, 0, 1, 1]
    w6 = [0, 0, 1, -1] # c10 += m6, c11 -= m6
    
    # Term 7: (a00-a10)b01
    u7 = [1, 0, -1, 0]
    v7 = [0, 1, 0, 0]
    w7 = [1, -1, 0, 0] # c00 += m7, c01 -= m7
    
    # Verify Winograd 2x2x2
    # c00 = m1 - m2 - m4 + m7
    # c01 = m1 + m3 - m5 - m7
    # c10 = m2 - m3 + m4 + m6
    # c11 = m1 + m3 + m5 - m6
    
    # Now, we need to embed this into M_{2,3,2}
    # A is 2x3, so u vectors are length 6.
    # B is 3x2, so v vectors are length 6.
    # C is 2x2, so w vectors are length 4.
    
    # The M_{2,2,2} part uses j=0,1.
    # So u vectors have non-zero entries at indices 0,1,2,3 (i*2+j).
    # v vectors have non-zero entries at indices 0,1,2,3 (j*2+k).
    # w vectors are unchanged.
    
    # The M_{2,1,2} part uses j=2.
    # A indices for j=2 are 4,5 (i*2+2).
    # B indices for j=2 are 4,5 (2*2+k).
    # C indices are 0,1,2,3.
    
    # M_{2,1,2} is just schoolbook:
    # c00 += a02 b20
    # c01 += a02 b21
    # c10 += a12 b20
    # c11 += a12 b21
    
    # Term 8: a02 b20
    u8 = [0, 0, 0, 0, 1, 0]
    v8 = [0, 0, 0, 0, 1, 0]
    w8 = [1, 0, 0, 0]
    
    # Term 9: a02 b21
    u9 = [0, 0, 0, 0, 1, 0]
    v9 = [0, 0, 0, 0, 0, 1]
    w9 = [0, 1, 0, 0]
    
    # Term 10: a12 b20
    u10 = [0, 0, 0, 0, 0, 1]
    v10 = [0, 0, 0, 0, 1, 0]
    w10 = [0, 0, 1, 0]
    
    # Term 11: a12 b21
    u11 = [0, 0, 0, 0, 0, 1]
    v11 = [0, 0, 0, 0, 0, 1]
    w11 = [0, 0, 0, 1]
    
    # Now we have 11 terms for M_{2,3,2} (k=0,1)
    # We need to duplicate this for k=2,3 to get M_{2,3,4}
    # For the second half, the B indices are shifted.
    # B is 3x4. Indices are j*4+k.
    # For k=2,3, the B indices are j*4+2, j*4+3.
    # The A indices are the same (i*3+j).
    # The C indices are k*2+i. For k=2,3, C indices are 4,5,6,7.
    
    # So we create 11 more terms.
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u, v, w, offset_b=0, offset_c=0):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # This is not quite right because the dimensions of V and W change.
    # V is length m*p = 3*4 = 12.
    # W is length p*n = 4*2 = 8.
    # U is length n*m = 2*3 = 6.
    
    # Let's re-construct the vectors with the correct lengths.
    
    # For the first half (k=0,1):
    # u is length 6.
    # v is length 12. Non-zero at indices j*4+k for k=0,1.
    # w is length 8. Non-zero at indices k*2+i for k=0,1.
    
    # For the second half (k=2,3):
    # u is length 6. Same as first half.
    # v is length 12. Non-zero at indices j*4+k for k=2,3.
    # w is length 8. Non-zero at indices k*2+i for k=2,3.
    
    terms = []
    
    # Winograd terms for M_{2,2,2}
    # u is length 6, v is length 12, w is length 8
    
    # Term 1
    u1 = [1, 0, 0, 1, 0, 0]
    v1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # b00 is at index 0*4+0=0
    v1[1] = 1 # b01 is at index 0*4+1=1? No, b index is j*p+k.
    # p=4.
    # b00: j=0, k=0 -> index 0
    # b01: j=0, k=1 -> index 1
    # b10: j=1, k=0 -> index 4
    # b11: j=1, k=1 -> index 5
    # b20: j=2, k=0 -> index 8
    # b21: j=2, k=1 -> index 9
    # b02: j=0, k=2 -> index 2
    # b03: j=0, k=3 -> index 3
    # b12: j=1, k=2 -> index 6
    # b13: j=1, k=3 -> index 7
    # b22: j=2, k=2 -> index 10
    # b23: j=2, k=3 -> index 11
    
    # So for the first half (k=0,1):
    # v1 (for b00+b11) should have 1 at index 0 and 5.
    v1 = [0]*12
    v1[0] = 1
    v1[5] = 1
    w1 = [0]*8
    w1[0] = 1 # c00: k=0, i=0 -> index 0
    w1[1] = 1 # c01: k=1, i=0 -> index 1
    w1[2] = 1 # c10: k=0, i=1 -> index 2
    w1[3] = 1 # c11: k=1, i=1 -> index 3
    
    terms.append((u1, v1, w1))
    
    # Term 2: (a10+a11)b00
    u2 = [0, 0, 1, 1, 0, 0]
    v2 = [0]*12
    v2[0] = 1
    w2 = [0]*8
    w2[0] = -1
    w2[2] = 1
    terms.append((u2, v2, w2))
    
    # Term 3: a10(b01+b11)
    u3 = [0, 0, 1, 0, 0, 0]
    v3 = [0]*12
    v3[1] = 1
    v3[5] = 1
    w3 = [0]*8
    w3[1] = 1
    w3[2] = -1
    w3[3] = 1
    terms.append((u3, v3, w3))
    
    # Term 4: a00 b10
    u4 = [1, 0, 0, 0, 0, 0]
    v4 = [0]*12
    v4[4] = 1
    w4 = [0]*8
    w4[0] = -1
    w4[2] = 1
    terms.append((u4, v4, w4))
    
    # Term 5: (a00+a01)b11
    u5 = [1, 1, 0, 0, 0, 0]
    v5 = [0]*12
    v5[5] = 1
    w5 = [0]*8
    w5[1] = -1
    w5[3] = 1
    terms.append((u5, v5, w5))
    
    # Term 6: (a01-a10)(b10+b11)
    u6 = [0, 1, -1, 0, 0, 0]
    v6 = [0]*12
    v6[4] = 1
    v6[5] = 1
    w6 = [0]*8
    w6[2] = 1
    w6[3] = -1
    terms.append((u6, v6, w6))
    
    # Term 7: (a00-a10)b01
    u7 = [1, 0, -1, 0, 0, 0]
    v7 = [0]*12
    v7[1] = 1
    w7 = [0]*8
    w7[0] = 1
    w7[1] = -1
    terms.append((u7, v7, w7))
    
    # Terms 8-11 for M_{2,1,2} (j=2, k=0,1)
    # u8: a02 -> index 4
    # v8: b20 -> index 8
    # w8: c00 -> index 0
    u8 = [0, 0, 0, 0, 1, 0]
    v8 = [0]*12
    v8[8] = 1
    w8 = [0]*8
    w8[0] = 1
    terms.append((u8, v8, w8))
    
    # Term 9: a02 b21
    u9 = [0, 0, 0, 0, 1, 0]
    v9 = [0]*12
    v9[9] = 1
    w9 = [0]*8
    w9[1] = 1
    terms.append((u9, v9, w9))
    
    # Term 10: a12 b20
    u10 = [0, 0, 0, 0, 0, 1]
    v10 = [0]*12
    v10[8] = 1
    w10 = [0]*8
    w10[2] = 1
    terms.append((u10, v10, w10))
    
    # Term 11: a12 b21
    u11 = [0, 0, 0, 0, 0, 1]
    v11 = [0]*12
    v11[9] = 1
    w11 = [0]*8
    w11[3] = 1
    terms.append((u11, v11, w11))
    
    # Now duplicate for the second half (k=2,3)
    # The u vectors are the same.
    # The v vectors are shifted: b index j*4+k. For k=2,3, indices are +2.
    # The w vectors are shifted: c index k*2+i. For k=2,3, indices are +4.
    
    for u, v, w in terms:
        v_new = [0]*12
        w_new = [0]*8
        for idx, val in enumerate(v):
            if val != 0:
                v_new[idx+2] = val
        for idx, val in enumerate(w):
            if val != 0:
                w_new[idx+4] = val
        terms.append((u, v_new, w_new))
        
    # Now we have 22 terms.
    U = [t[0] for t in terms]
    V = [t[1] for t in terms]
    W = [t[2] for t in terms]
    
    return {"U": U, "V": V, "W": W}

