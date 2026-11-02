import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    nm, mp, pn = n*m, m*p, p*n  # 6, 24, 16

    start_time = time.time()
    deadline = start_time + 115

    # Build the tensor T with non-zero entries
    # T[a, b, d] = 1 if a=i*m+j, b=j*p+k, d=k*n+i for some i,j,k
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                d = k*n + i
                nonzero_entries.append((a, b, d))
    nz_set = set(nonzero_entries)

    # Initialize with schoolbook rank 48 as fallback
    best_rank = n*m*p  # 48
    U0 = np.zeros((nm, best_rank))
    V0 = np.zeros((mp, best_rank))
    W0 = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Explicit rank-44 construction:
    # Split the 3x8 matrix into two 3x4 blocks.
    # Each 2x3 * 3x4 = 2x4 multiplication is split into two 2x3*3x2 = 2x2 blocks.
    # Each 2x2 block uses Strassen (rank 7). Total = 4 blocks * 7 = 28? 
    # Wait, let's use the known result: 2x3*3x8 can be done in rank 44.
    # Construction: 2x3 * 3x8. 
    # Use 4 independent 2x3 * 3x2 subproblems (each rank 7 via Strassen) = 28? No.
    # Actually the measured improvement was rank 44 via "Strassen(7)+4 rank-1 per 2x3x2 block".
    # Let me build a 2x3 * 3x2 decomposition of rank 7, then combine 4 of them with offsets.

    # 2x3 * 3x2 -> 2x2. Indices: A is 2x3 (i in 0..1, j in 0..2), B is 3x2 (j in 0..2, k in 0..1)
    # C is 2x2 (i in 0..1, k in 0..1)
    # Strassen for 2x2: split 2x2 into four 1x1 blocks. But we have 2x3 * 3x2.
    # Let's use the standard approach: pad or use the known rank-7 decomposition for 2x3*3x2.
    
    # Actually, let's just use a known rank-7 decomposition for 2x3 * 3x2.
    # A (2x3) * B (3x2) = C (2x2)
    # C[0,0] = A00*B00 + A01*B10 + A02*B20
    # C[0,1] = A00*B01 + A01*B11 + A02*B21
    # C[1,0] = A10*B00 + A11*B10 + A12*B20
    # C[1,1] = A10*B01 + A11*B11 + A12*B21

    # Strassen-style for 2x3 * 3x2 (rank 7):
    # Define 7 products:
    # M1 = (A00 + A01 + A02) * (B00 + B01)  -- wait, need to be careful
    # Let me use a known rank-7 formula for 2x3 * 3x2.
    
    # Alternative: use the fact that 2x3*3x2 can be viewed as:
    # [A00 A01 A02]   [B00 B01]
    # [A10 A11 A12] * [B10 B11]
    #                 [B20 B21]
    
    # Rank 7 decomposition:
    # M1 = A00 * (B00 + B01)
    # M2 = (A00 + A01) * B01
    # M3 = (A00 + A02) * B00
    # M4 = A02 * (B00 + B01)  -- hmm, let me think differently
    
    # Let me just use a verified rank-7 decomposition. I'll construct it carefully.
    # For 2x2 * 2x2, Strassen has rank 7. For 2x3 * 3x2, we can embed.
    # Actually, 2x3 * 3x2 has 12 input entries and 4 output entries, 36 constraints.
    # Rank 7 means 7*3 = 21 variables per factor... 
    
    # Let me use a different approach: the measured solution used rank 44.
    # Let me try to construct it as 4 blocks of 2x3*3x2, each with rank 7, giving 28,
    # plus some corrections. But 28 < 44, so maybe the blocks overlap.
    
    # Actually, let me reconsider. The 2x3 * 3x8 multiplication:
    # A is 2x3, B is 3x8. C = A*B is 2x8.
    # We can write B as [B1 B2] where B1, B2 are 3x4.
    # C = [A*B1, A*B2]. Each A*Bk is 2x4.
    # A*Bk can be split into two 2x2 blocks: A * Bk[:, 0:2] and A * Bk[:, 2:4].
    # Each is a 2x3 * 3x2 multiplication.
    # So we have 4 such subproblems, each of rank 7 (if we can achieve rank 7 for 2x3*3x2).
    # Total rank = 4 * 7 = 28.
    
    # But wait, the measured improvement was rank 44, not 28. So maybe rank 7 for 2x3*3x2 is not achievable,
    # or the construction is different. Let me check: the lower bound for 2x3*3x2 is at least... 
    # Actually, it's known that 2x3*3x2 can be done in rank 7 (it's a special case).
    
    # Let me just hardcode a rank-44 construction that I know works, or try to find one.
    # Given the complexity, let me use the schoolbook rank 48 as a safe fallback,
    # and attempt a rank-44 construction if time permits.
    
    # Actually, the inspiration says "Strassen(7)+4 rank-1 per 2x3x2 block" giving rank 44.
    # That means: for each of the 4 blocks of 2x3*3x2, use Strassen rank 7 PLUS 4 rank-1 terms.
    # So each block has rank 7 + 4 = 11? No, 4 blocks * 11 = 44. Yes!
    # So each 2x3*3x2 block uses rank 11 (7 from Strassen-like + 4 rank-1 corrections).
    
    # Let me construct a rank-11 decomposition for 2x3 * 3x2.
    # Start with Strassen for the 2x2 sub-blocks and add corrections.
    
    # For 2x3 * 3x2:
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01], [b10, b11], [b20, b21]]
    # C = [[c00, c01], [c10, c11]]
    
    # Rank-11 decomposition:
    # Use 7 Strassen-like terms for the "core" 2x2*2x2 part (using a00,a01,b00,b01,a10,a11,b10,b11)
    # and 4 rank-1 terms for the a02,b20,b21 and a12 contributions.
    
    # Strassen for 2x2 (using a00,a01; b00,b01; a10,a11; b10,b11):
    # M1 = a00 * (b00 + b01)
    # M2 = (a00 + a01) * b01
    # M3 = (a00 + a10) * b00
    # M4 = a10 * (b00 + b10)  -- wait, standard Strassen:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b11 - b01)
    # M4 = a11 * (b00 + b10)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a11) * (b00 + b10)  -- hmm, let me use standard
    # M7 = (a01 - a11) * (b10 + b11)
    
    # Standard Strassen for 2x2 * 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B12 - B22)  -- wait, let me get this right
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M4
    # C21 = M2 + M7
    # C22 = M1 - M3 + M6 - M5
    
    # For our 2x3*3x2, let's use the top-left 2x2 of A (a00,a01; a10,a11) and 
    # top-left 2x2 of B (b00,b01; b10,b11) for the Strassen part,
    # and handle a02, a12, b20, b21 with rank-1 terms.
    
    # Strassen terms (rank 7):
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b11 - b01)
    # M4 = a11 * (b00 + b10)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a11) * (b00 + b10)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # These give contributions to c00, c01, c10, c11 from the 2x2*2x2 core.
    # But we also need contributions from a02*b20, a02*b21, a12*b20, a12*b21.
    
    # The Strassen part gives:
    # c00_core = M1 - M4 + M5 - M7  (but this is for the 2x2*2x2 case, needs adjustment)
    # Actually, in the 2x2*2x2 case with A=[[a00,a01],[a10,a11]] and B=[[b00,b01],[b10,b11]]:
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    
    # But in our 2x3*3x2 case:
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    
    # So the Strassen part handles the 2x2*2x2 core (a00,a01,b00,b01,a10,a11,b10,b11),
    # and we need to add a02*b20, a02*b21, a12*b20, a12*b21.
    # These 4 terms can be handled with 4 rank-1 terms.
    
    # So total rank per block = 7 + 4 = 11.
    # Total rank for 4 blocks = 4 * 11 = 44.
    
    # Now let me construct this explicitly.
    
    # For each block, I need to define U, V, W contributions.
    # Let me define the 11 rank-1 terms for a single 2x3*3x2 block.
    
    # Block indices: A rows i in {0,1}, A cols j in {0,1,2}
    #               B rows j in {0,1,2}, B cols k in {0,1}
    #               C rows i in {0,1}, C cols k in {0,1}
    
    # Strassen terms (7):
    # Term 0: U = a00 + a11, V = b00 + b11, W = c00 - c01 - c10 + c11 (with signs)
    # Actually, let me use the standard Strassen signs:
    # c00 = M1 - M4 + M5 - M7
    # c01 = M3 + M4
    # c10 = M2 + M7
    # c11 = M1 - M3 + M6 - M5
    
    # So:
    # M1 contributes: +1 to c00, +1 to c11
    # M2 contributes: +1 to c10
    # M3 contributes: +1 to c01, -1 to c11
    # M4 contributes: -1 to c00, +1 to c01
    # M5 contributes: +1 to c00, -1 to c11
    # M6 contributes: +1 to c11
    # M7 contributes: -1 to c00, +1 to c10
    
    # Rank-1 terms (4):
    # T8: a02 * b20 -> +1 to c00
    # T9: a02 * b21 -> +1 to c01
    # T10: a12 * b20 -> +1 to c10
    # T11: a12 * b21 -> +1 to c11
    
    # Now, for the full 2x3*3x8 problem, we have 4 blocks.
    # Block 0: A cols 0-2, B cols 0-1 (of the 3x8 B, which is split into 3x4 x 2, then 3x2 x 2)
    # Actually, B is 3x8. Split into B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8].
    # Each is 3x2. So 4 blocks.
    
    # For block t (t=0,1,2,3):
    # A is the same 2x3 matrix.
    # B_t is B[:, 2*t : 2*t+2], which is 3x2.
    # C_t is C[:, 2*t : 2*t+2], which is 2x2.
    
    # Now, in the full problem:
    # A index: a = i*m + j = i*3 + j, where i in {0,1}, j in {0,1,2}
    # B index: b = j*p + k = j*8 + k, where j in {0,1,2}, k in {0,...,7}
    # C index: d = k*n + i = k*2 + i, where k in {0,...,7}, i in {0,1}
    
    # For block t, B cols are 2*t and 2*t+1, so k = 2*t and 2*t+1.
    # C cols are 2*t and 2*t+1, so k = 2*t and 2*t+1.
    
    # Let me construct the full U, V, W matrices.
    
    rank = 44
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for t in range(4):
        # Block t: B cols 2*t, 2*t+1; C cols 2*t, 2*t+1
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen term 0: M1 = (a00 + a11) * (b00 + b11)
        # U: a00 + a11 -> rows (0*3+0)=0 and (1*3+1)=4
        # V: b00 + b11 -> rows (0*8+k0)=k0 and (1*8+k1)=8+k1
        # W: c00 + c11 -> cols (k0*2+0)=2*k0 and (k1*2+1)=2*k1+1
        U[0, col] = 1  # a00
        U[4, col] = 1  # a11
        V[k0, col] = 1  # b00 (j=0, k=k0)
        V[8 + k1, col] = 1  # b11 (j=1, k=k1)
        W[2*k0, col] = 1  # c00 (i=0, k=k0)
        W[2*k1 + 1, col] = 1  # c11 (i=1, k=k1)
        col += 1
        
        # Strassen term 1: M2 = (a10 + a11) * b00
        # U: a10 + a11 -> rows (1*3+0)=3 and (1*3+1)=4
        # V: b00 -> row k0
        # W: c10 -> col (k0*2+1)=2*k0+1
        U[3, col] = 1  # a10
        U[4, col] = 1  # a11
        V[k0, col] = 1  # b00
        W[2*k0 + 1, col] = 1  # c10
        col += 1
        
        # Strassen term 2: M3 = a00 * (b11 - b01)
        # U: a00 -> row 0
        # V: b11 - b01 -> rows (1*8+k1)=8+k1 and (0*8+k1)=k1
        # W: c01 - c11 -> cols (k1*2+0)=2*k1 and (k1*2+1)=2*k1+1
        U[0, col] = 1  # a00
        V[8 + k1, col] = 1  # b11
        V[k1, col] = -1  # -b01
        W[2*k1, col] = 1  # c01
        W[2*k1 + 1, col] = -1  # -c11
        col += 1
        
        # Strassen term 3: M4 = a11 * (b00 + b10)
        # U: a11 -> row 4
        # V: b00 + b10 -> rows k0 and (1*8+k0)=8+k0
        # W: -c00 + c01 -> cols 2*k0 and 2*k1
        U[4, col] = 1  # a11
        V[k0, col] = 1  # b00
        V[8 + k0, col] = 1  # b10
        W[2*k0, col] = -1  # -c00
        W[2*k1, col] = 1  # c01
        col += 1
        
        # Strassen term 4: M5 = (a00 + a01) * b11
        # U: a00 + a01 -> rows 0 and (0*3+1)=1
        # V: b11 -> row 8+k1
        # W: c00 - c11 -> cols 2*k0 and 2*k1+1
        U[0, col] = 1  # a00
        U[1, col] = 1  # a01
        V[8 + k1, col] = 1  # b11
        W[2*k0, col] = 1  # c00
        W[2*k1 + 1, col] = -1  # -c11
        col += 1
        
        # Strassen term 5: M6 = (a10 - a11) * (b00 + b10)
        # U: a10 - a11 -> rows 3 and 4
        # V: b00 + b10 -> rows k0 and 8+k0
        # W: c11 -> col 2*k1+1
        U[3, col] = 1  # a10
        U[4, col] = -1  # -a11
        V[k0, col] = 1  # b00
        V[8 + k0, col] = 1  # b10
        W[2*k1 + 1, col] = 1  # c11
        col += 1
        
        # Strassen term 6: M7 = (a01 - a11) * (b10 + b11)
        # U: a01 - a11 -> rows 1 and 4
        # V: b10 + b11 -> rows 8+k0 and 8+k1
        # W: -c00 + c10 -> cols 2*k0 and 2*k0+1
        U[1, col] = 1  # a01
        U[4, col] = -1  # -a11
        V[8 + k0, col] = 1  # b10
        V[8 + k1, col] = 1  # b11
        W[2*k0, col] = -1  # -c00
        W[2*k0 + 1, col] = 1  # c10
        col += 1
        
        # Rank-1 term 7: a02 * b20
        # U: a02 -> row (0*3+2)=2
        # V: b20 -> row (2*8+k0)=16+k0
        # W: c00 -> col 2*k0
        U[2, col] = 1  # a02
        V[16 + k0, col] = 1  # b20
        W[2*k0, col] = 1  # c00
        col += 1
        
        # Rank-1 term 8: a02 * b21
        # U: a02 -> row 2
        # V: b21 -> row (2*8+k1)=16+k1
        # W: c01 -> col 2*k1
        U[2, col] = 1  # a02
        V[16 + k1, col] = 1  # b21
        W[2*k1, col] = 1  # c01
        col += 1
        
        # Rank-1 term 9: a12 * b20
        # U: a12 -> row (1*3+2)=5
        # V: b20 -> row 16+k0
        # W: c10 -> col 2*k0+1
        U[5, col] = 1  # a12
        V[16 + k0, col] = 1  # b20
        W[2*k0 + 1, col] = 1  # c10
        col += 1
        
        # Rank-1 term 10: a12 * b21
        # U: a12 -> row 5
        # V: b21 -> row 16+k1
        # W: c11 -> col 2*k1+1
        U[5, col] = 1  # a12
        V[16 + k1, col] = 1  # b21
        W[2*k1 + 1, col] = 1  # c11
        col += 1
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        valid = True
        for a, b, d in nonzero_entries:
            val = np.sum(U[a, :] * V[b, :] * W[d, :])
            if abs(val - 1) > 1e-9:
                valid = False
                break
        if valid:
            # Check some zero entries
            for _ in range(100):
                a = np.random.randint(0, nm)
                b = np.random.randint(0, mp)
                d = np.random.randint(0, pn)
                if (a, b, d) not in nz_set:
                    val = np.sum(U[a, :] * V[b, :] * W[d, :])
                    if abs(val) > 1e-9:
                        valid = False
                        break
        if valid:
            best_rank = 44
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Convert to list format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
