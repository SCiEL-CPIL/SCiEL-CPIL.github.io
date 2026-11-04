"""Matrix multiplication 4x4x8 via Kronecker product of Strassen (2x2x2, rank 7)
and a rank-2 decomposition for the 2x1x2 rectangular slice."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 120.0  # safety margin

    # ---------- Base decomposition 1: Strassen for 2x2x2 (rank 7) ----------
    # U: (2*2, 7) indexed [i*2+j, c]; V: (2*2, 7) indexed [j*2+k, c];
    # W: (2*2, 7) indexed [k*2+i, c]
    U22 = [[0]*7 for _ in range(4)]
    V22 = [[0]*7 for _ in range(4)]
    W22 = [[0]*7 for _ in range(4)]

    # Strassen factors (M1..M7)
    # M1 = (a11+a22)*(b12-b22)
    # M2 = (a11+a22)*(b21+b22)
    # M3 = (a11-a22)*(b11+b12)
    # M4 = (a21+a22)*(b11-b21)
    # M5 = a11*(b11-b22)
    # M6 = a22*(b22-b11)
    # M7 = (a12-a22)*(b21+b22)
    #
    # C11 = M1+M4+M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1+M2+M3-M5

    # (i,j) pairs for U; (j,k) for V; (k,i) for W
    # U rows: i*2+j ; V rows: j*2+k ; W rows: k*2+i
    # a_ij -> U[i*2+j]; b_jk -> V[j*2+k]; c_ki -> W[k*2+i]

    def setU(i,j,c,v): U22[i*2+j][c] += v
    def setV(j,k,c,v): V22[j*2+k][c] += v
    def setW(k,i,c,v): W22[k*2+i][c] += v

    # M1: (a11+a22)*(b12-b22)  -> C11 += M1, C22 += M1
    setU(0,0,0,1); setU(1,1,0,1)
    setV(0,1,0,1); setV(1,1,0,-1)
    setW(0,0,0,1); setW(1,1,0,1)

    # M2: (a11+a22)*(b21+b22)  -> C21 += M2, C22 += M2
    setU(0,0,1,1); setU(1,1,1,1)
    setV(1,0,1,1); setV(1,1,1,1)
    setW(0,1,1,1); setW(1,1,1,1)

    # M3: (a11-a22)*(b11+b12)  -> C12 += M3, C22 += M3
    setU(0,0,2,1); setU(1,1,2,-1)
    setV(0,0,2,1); setV(0,1,2,1)
    setW(0,1,2,1); setW(1,1,2,1)

    # M4: (a21+a22)*(b11-b21)  -> C11 += M4, C21 += M4
    setU(1,0,3,1); setU(1,1,3,1)
    setV(0,0,3,1); setV(1,0,3,-1)
    setW(0,0,3,1); setW(0,1,3,1)

    # M5: a11*(b11-b22)  -> C11 += M5, C12 += M5, C22 -= M5
    setU(0,0,4,1)
    setV(0,0,4,1); setV(1,1,4,-1)
    setW(0,0,4,1); setW(0,1,4,1); setW(1,1,4,-1)

    # M6: a22*(b22-b11)  -> (not used in C formula directly; M6 = -a22*(b11-b22))
    # Actually standard Strassen doesn't need M6 separately; the 7 above suffice.
    # Let me re-verify: the 7 standard Strassen multiplications are M1..M7.
    # M6 is NOT standard. Let me recount: M1..M7 = 7 factors. I listed M1..M7 above? 
    # I listed M1,M2,M3,M4,M5,M6,M7 but only defined 5. Let me redo properly.

    # Reset
    U22 = [[0]*7 for _ in range(4)]
    V22 = [[0]*7 for _ in range(4)]
    W22 = [[0]*7 for _ in range(4)]

    # Standard Strassen 7 factors:
    # M1 = (a11+a22)*(b12-b22)
    # M2 = (a11+a22)*(b21+b22)
    # M3 = (a11-a22)*(b11+b12)
    # M4 = (a21+a22)*(b11-b21)
    # M5 = a11*(b11-b22)
    # M6 = a22*(b22-b11)
    # M7 = (a12-a22)*(b21+b22)
    #
    # C11 = M1 + M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 + M2 + M3 - M5

    # Factor 0 (M1): (a11+a22)*(b12-b22)
    setU(0,0,0,1); setU(1,1,0,1)
    setV(0,1,0,1); setV(1,1,0,-1)
    # C11 += M1 => W[0,0]; C22 += M1 => W[1,1]
    setW(0,0,0,1); setW(1,1,0,1)

    # Factor 1 (M2): (a11+a22)*(b21+b22)
    setU(0,0,1,1); setU(1,1,1,1)
    setV(1,0,1,1); setV(1,1,1,1)
    # C21 += M2 => W[0,1]; C22 += M2 => W[1,1]
    setW(0,1,1,1); setW(1,1,1,1)

    # Factor 2 (M3): (a11-a22)*(b11+b12)
    setU(0,0,2,1); setU(1,1,2,-1)
    setV(0,0,2,1); setV(0,1,2,1)
    # C12 += M3 => W[1,0]; C22 += M3 => W[1,1]
    setW(1,0,2,1); setW(1,1,2,1)

    # Factor 3 (M4): (a21+a22)*(b11-b21)
    setU(1,0,3,1); setU(1,1,3,1)
    setV(0,0,3,1); setV(1,0,3,-1)
    # C11 += M4 => W[0,0]; C21 += M4 => W[0,1]
    setW(0,0,3,1); setW(0,1,3,1)

    # Factor 4 (M5): a11*(b11-b22)
    setU(0,0,4,1)
    setV(0,0,4,1); setV(1,1,4,-1)
    # C11 += M5 => W[0,0]; C12 += M5 => W[1,0]; C22 -= M5 => W[1,1]
    setW(0,0,4,1); setW(1,0,4,1); setW(1,1,4,-1)

    # Factor 5 (M6): a22*(b22-b11)
    setU(1,1,5,1)
    setV(1,1,5,1); setV(0,0,5,-1)
    # M6 is not directly in C formula; it's redundant. But we keep 7 factors.
    # Actually M6 = a22*(b22-b11) is used in some formulations but the 7 above
    # (M1-M5 plus M7) already give C. Let me check: do we need M6?
    # C11 = M1+M4+M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1+M2+M3-M5
    # M6 doesn't appear! So the 7th factor should be M7, not M6.
    # Let me redo: factors are M1,M2,M3,M4,M5,M7 (6 factors?) No, Strassen has 7.
    # I'm confusing myself. Let me use the well-known 7:
    # Actually the standard Strassen uses exactly 7: M1..M7 where
    # C11=M1+M4+M5+M7, C12=M3+M5, C21=M2+M4, C22=M1+M2+M3-M5
    # That's M1,M2,M3,M4,M5,M7 = 6 distinct? No: M1,M2,M3,M4,M5,M7 is 6.
    # I'm missing one. Let me look up: Strassen's 7 are:
    # M1=(a11+a22)(b12-b22), M2=(a11+a22)(b21+b22), M3=(a11-a22)(b11+b12),
    # M4=(a21+a22)(b11-b21), M5=a11(b11-b22), M6=a22(b22-b11), M7=(a12-a22)(b21+b22)
    # C11=M1+M4+M5+M7, C12=M3+M5, C21=M2+M4, C22=M1+M2+M3-M5
    # M6 does NOT appear in C! So one factor is redundant. 
    # Hmm, that means rank 7 with M6 being zero contribution? That can't be.
    # Let me re-derive. Actually I think the correct C22 uses M6:
    # C22 = M1+M2+M3-M5  ... let me verify with explicit expansion.
    # This is getting error-prone. Let me just use a known-correct table.

    # I'll use a verified Strassen table (from literature):
    # Reset all
    U22 = [[0]*7 for _ in range(4)]
    V22 = [[0]*7 for _ in range(4)]
    W22 = [[0]*7 for _ in range(4)]

    # Verified Strassen (i,j 0-based):
    # c11 = m1+m4+m5+m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1+m2+m3-m5
    # m1 = (a11+a22)(b12-b22)
    # m2 = (a11+a22)(b21+b22)
    # m3 = (a11-a22)(b11+b12)
    # m4 = (a21+a22)(b11-b21)
    # m5 = a11(b11-b22)
    # m6 = a22(b22-b11)   [unused in c, but part of the 7]
    # m7 = (a12-a22)(b21+b22)

    # Since m6 is unused, rank-7 with m6=0 in W is fine (it just means that
    # factor contributes nothing, effectively rank 6? No - the 7th factor m7 is used).
    # Wait: m1,m2,m3,m4,m5,m7 = 6 factors used. m6 unused. So this is rank 6?
    # No! I'm miscounting. Let me list which m's appear in C:
    # C11: m1,m4,m5,m7
    # C12: m3,m5
    # C21: m2,m4
    # C22: m1,m2,m3,m5
    # Union: m1,m2,m3,m4,m5,m7 = 6 factors. m6 not used.
    # That means this is actually a rank-6 decomposition? That contradicts Strassen rank 7.
    # I think I have the wrong C formula. Let me use the CORRECT Strassen:
    #
    # Correct Strassen:
    # M1 = (A11+A22)(B12-B22)
    # M2 = (A11+A22)(B21+B22)
    # M3 = (A11-A22)(B11+B12)
    # M4 = (A21+A22)(B11-B21)
    # M5 = A11(B11-B22)
    # M6 = A22(B22-B11)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 + M2 + M3 - M5
    #
    # Let me verify C22 by expansion:
    # M1 = (A11+A22)(B12-B22) = A11B12 - A11B22 + A22B12 - A22B22
    # M2 = (A11+A22)(B21+B22) = A11B21 + A11B22 + A22B21 + A22B22
    # M3 = (A11-A22)(B11+B12) = A11B11 + A11B12 - A22B11 - A22B12
    # -M5 = -A11B11 + A11B22
    # Sum: A11B12 - A11B22 + A22B12 - A22B22 + A11B21 + A11B22 + A22B21 + A22B22
    #      + A11B11 + A11B12 - A22B11 - A22B12 - A11B11 + A11B22
    # = A11B12 + A11B22 + A22B12 + A11B21 + A22B21 + A11B12 + A11B22
    #   (let me collect carefully)
    # A11B12: +1 (M1) +1 (M3) = +2? No: M1 has A11B12, M3 has A11B12. So 2*A11B12.
    # This is getting messy. Let me just trust the standard result: Strassen rank 7.
    # The key point: 7 factors, and I'll use a numerically-verified construction.

    # To avoid errors, let me construct Strassen via a known-correct method:
    # I'll hardcode the 7 factors with verified coefficients.
    
    # Using a reference implementation's factorization:
    # Factor f, for each (i,j) in A: coeff_a[f][i][j]
    #                        for each (j,k) in B: coeff_b[f][j][k]  
    #                        for each (k,i) in C: coeff_c[f][k][i]
    
    # I'll use this verified table (0-indexed, i,j,k in {0,1}):
    # Represented as lists of (index, value) tuples.

    # Let me just use a simple, definitely-correct approach:
    # Use the schoolbook rank-4 for 2x2x2 as a fallback if Strassen is too error-prone.
    # But that gives rank 4*4*4 = 64 for the 4x4x4 part, worse.
    
    # Alternative: use a rank-7 that I can verify numerically in code.
    # Let me define Strassen and verify it, falling back to schoolbook if it fails.

    # Define Strassen factors:
    # I'll use the formulation from "Strassen's algorithm" Wikipedia:
    # M1 = (A11+A22)(B12-B22)
    # M2 = (A11+A22)(B21+B22)  
    # M3 = (A11-A22)(B11+B12)
    # M4 = (A21+A22)(B11-B21)
    # M5 = A11(B11-B22)
    # M6 = A22(B22-B11)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4+M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1+M2+M3-M5

    # Build factor matrices:
    def build_strassen():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # A index: i*2+j, B index: j*2+k, C index: k*2+i
        # M1 = (A11+A22)(B12-B22)
        U[0*2+0][0] += 1; U[1*2+1][0] += 1
        V[0*2+1][0] += 1; V[1*2+1][0] += -1
        W[0*2+0][0] += 1  # C11
        W[1*2+1][0] += 1  # C22
        # M2 = (A11+A22)(B21+B22)
        U[0*2+0][1] += 1; U[1*2+1][1] += 1
        V[1*2+0][1] += 1; V[1*2+1][1] += 1
        W[0*2+1][1] += 1  # C21
        W[1*2+1][1] += 1  # C22
        # M3 = (A11-A22)(B11+B12)
        U[0*2+0][2] += 1; U[1*2+1][2] += -1
        V[0*2+0][2] += 1; V[0*2+1][2] += 1
        W[1*2+0][2] += 1  # C12
        W[1*2+1][2] += 1  # C22
        # M4 = (A21+A22)(B11-B21)
        U[1*2+0][3] += 1; U[1*2+1][3] += 1
        V[0*2+0][3] += 1; V[1*2+0][3] += -1
        W[0*2+0][3] += 1  # C11
        W[0*2+1][3] += 1  # C21
        # M5 = A11(B11-B22)
        U[0*2+0][4] += 1
        V[0*2+0][4] += 1; V[1*2+1][4] += -1
        W[0*2+0][4] += 1  # C11
        W[1*2+0][4] += 1  # C12
        W[1*2+1][4] += -1 # C22
        # M6 = A22(B22-B11)
        U[1*2+1][5] += 1
        V[1*2+1][5] += 1; V[0*2+0][5] += -1
        # M6 not used in C formula, so W stays 0 for factor 5
        # M7 = (A12-A22)(B21+B22)
        U[0*2+1][6] += 1; U[1*2+1][6] += -1
        V[1*2+0][6] += 1; V[1*2+1][6] += 1
        W[0*2+0][6] += 1  # C11
        return U, V, W

    U22, V22, W22 = build_strassen()

    # Verify Strassen numerically
    import numpy as np
    U22n = np.array(U22, dtype=float)
    V22n = np.array(V22, dtype=float)
    W22n = np.array(W22, dtype=float)
    
    # Test: for random A (2x2), B (2x2), check C = A@B
    np.random.seed(42)
    A_test = np.random.randn(2,2)
    B_test = np.random.randn(2,2)
    C_true = A_test @ B_test
    
    # Decomposition: C[k*n+i] = sum_c U[i*m+j] * V[j*p+k] * W[k*n+i]
    # For 2x2x2: n=m=p=2
    # C_flat[k*2+i] = sum_c U_flat[i*2+j] * V_flat[j*2+k] * W_flat[k*2+i]
    # A_flat[i*2+j], B_flat[j*2+k], C_flat[k*2+i]
    
    C_decomp = np.zeros(4)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for c in range(7):
                    val += U22n[i*2+j, c] * V22n[j*2+k, c] * W22n[k*2+i, c]
                C_decomp[k*2+i] += A_test[i,j] * B_test[j,k] * val
    # Wait, that's wrong. The decomposition is:
    # C = sum_c (U_c ⊗ V_c) * W_c where U_c is a vector of A-coefficients, etc.
    # More precisely: C[k,i] = sum_c W[k,i,c] * (sum_j U[i,j,c]*A[i,j]) * (sum_j V[j,k,c]*B[j,k])
    # Hmm, no. The bilinear form is:
    # C[k,i] = sum_c W[k,i,c] * (sum_{i',j} U[i',j,c] * A[i',j]) ... no.
    
    # Actually the standard tensor decomposition:
    # C = sum_c (U_c ⊗ V_c ⊗ W_c) where U_c indexes A, V_c indexes B, W_c indexes C.
    # C[k,i] = sum_c W[k,i,c] * (sum_{i,j} U[i,j,c]*A[i,j]) * (sum_{j,k} V[j,k,c]*B[j,k])
    # No, that's not right either. The correct form:
    # C[k,i] = sum_c W[k,i,c] * (sum_{i,j} U[i,j,c]*A[i,j]) ... 
    
    # Let me think again. The multiplication tensor T has T[i,j,k] = 1 if the product
    # A[i,j]*B[j,k] contributes to C[i,k]. Wait, C[i,k] = sum_j A[i,j]*B[j,k].
    # So the tensor is over (i,k) output and (i,j) and (j,k) inputs.
    # The decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c where:
    # T[i,k; i,j; j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[i,k]
    # i.e., C[i,k] = sum_c w_c[i,k] * (sum_{i,j} u_c[i,j]*A[i,j]) * (sum_{j,k} v_c[j,k]*B[j,k])
    # No! That's not how it works. The correct bilinear form:
    # C[i,k] = sum_c w_c[i,k] * (sum_{i',j} u_c[i',j]*A[i',j]) * ... 
    # 
    # Actually the standard form for matrix multiplication tensor:
    # The tensor M_{ijk} where i indexes rows of A (and C), j indexes cols of A/rows of B,
    # k indexes cols of B (and C). M_{ijk} = delta_{i,i} delta_{j,j} delta_{k,k} = 1 always?
    # No. The multiplication tensor has entries: for each (i,j,k), the coefficient is 1
    # because A[i,j]*B[j,k] contributes to C[i,k]. So M[i,j,k] = 1 for all i,j,k.
    # Wait no, that's the schoolbook. The tensor is M[i,j,k] = 1 for all i,j,k in [n].
    # The decomposition: M = sum_c U_c ⊗ V_c ⊗ W_c where U_c is n*n, V_c is n*n, W_c is n*n.
    # M[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[i,k]
    # And C[i,k] = sum_{j} A[i,j]*B[j,k] = sum_c W_c[i,k] * (sum_{i',j} U_c[i',j]*A[i',j]) * (sum_{j',k'} V_c[j',k']*B[j',k'])
    # No, that's still not right.
    
    # Let me use the correct formulation. The multiplication tensor T is a 3-tensor
    # T ∈ (F^{n*n})^* ⊗ (F^{n*n})^* ⊗ F^{n*n} where:
    # T(A,B) = A ⊗ B (matrix product)
    # In coordinates: T[i,j; j',k; i',k'] = delta_{i,i'} delta_{j,j'} delta_{k,k'} * delta_{j,k}
    # Wait, C[i,k] = sum_j A[i,j] B[j,k]. So:
    # T_{(i,j),(j',k),(i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'} delta_{j,k}
    # = delta_{i,i'} delta_{j,j'} delta_{k,k'} delta_{j,k}
    # For this to be nonzero: i=i', j=j', k=k', and j=k. So j=k=i'=... 
    # This means T is nonzero only when i=i', j=j'=k=k'. 
    # So T_{(i,j),(j,k),(i,k)} = 1 (when the indices are consistent).
    # 
    # The decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c where
    # U_c ∈ F^{n*n} (indexed by (i,j)), V_c ∈ F^{n*n} (indexed by (j,k)), W_c ∈ F^{n*n} (indexed by (i,k))
    # T_{(i,j),(j',k),(i',k')} = sum_c U_c[i,j] * V_c[j',k] * W_c[i',k']
    # For the decomposition to be exact:
    # sum_c U_c[i,j] * V_c[j,k] * W_c[i,k] = 1 for all i,j,k
    # and sum_c U_c[i,j] * V_c[j',k] * W_c[i',k'] = 0 for all other index combinations.
    #
    # So the verification: for all i,j,k in [n], sum_c U[i,j,c]*V[j,k,c]*W[i,k,c] = 1
    # and for all other combinations, the sum = 0.
    
    # For 2x2x2, let me verify:
    n2 = 2
    ok = True
    for i in range(n2):
        for j in range(n2):
            for k in range(n2):
                s = 0
                for c in range(7):
                    s += U22n[i*2+j, c] * V22n[j*2+k, c] * W22n[k*2+i, c]
                if abs(s - 1) > 1e-9:
                    ok = False
                    break
            if not ok: break
        if not ok: break
    # Also check off-diagonal: for (i,j) and (j',k) with j != j', etc.
    # The full check: for all i,j,j',k,k',i':
    # sum_c U[i,j,c]*V[j',k,c]*W[i',k',c] should be delta_{i,i'} delta_{j,j'} delta_{k,k'} delta_{j,k}
    # This is a lot of checks. Let me do a random test instead.
    
    # Random test: pick random A (2x2), B (2x2), compute C = A@B via decomposition.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Via decomposition: C[i,k] = sum_c W[i,k,c] * (sum_{i',j'} U[i',j',c]*A[i',j']) * (sum_{j',k'} V[j',k',c]*B[j',k'])
    # NO. The correct bilinear form is:
    # C[i,k] = sum_c W[i,k,c] * (sum_{i,j} U[i,j,c]*A[i,j]) * (sum_{j,k} V[j,k,c]*B[j,k])
    # This is NOT correct because it would give a product of two scalars, not a bilinear form in the right way.
    
    # I think the correct interpretation is:
    # The tensor T[i,j,k] (with i,j,k all in [n]) satisfies T[i,j,k] = 1 for all i,j,k.
    # Wait no, that's not right. Let me re-derive.
    # 
    # C = A * B, C[i,k] = sum_j A[i,j] B[j,k]
    # The multiplication map is bilinear: (A,B) -> C
    # In tensor form: C[i,k] = sum_{j} A[i,j] B[j,k]
    # The tensor M is defined by M[i,j,k] = 1 (since A[i,j]*B[j,k] contributes 1 to C[i,k])
    # But we need to be careful about which indices go where.
    #
    # Standard: the multiplication tensor M ∈ (F^{n^2})^* ⊗ (F^{n^2})^* ⊗ F^{n^2}
    # M(A,B) = A*B
    # In coordinates with A indexed by (i,j), B by (j,k), C by (i,k):
    # M_{(i,j),(j',k),(i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'} delta_{j,k}
    #
    # Decomposition: M = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c ∈ F^{n^2} (indexed by (i,j)), V_c ∈ F^{n^2} (indexed by (j,k)), W_c ∈ F^{n^2} (indexed by (i,k))
    # M_{(i,j),(j',k),(i',k')} = sum_c U_c[i,j] V_c[j',k] W_c[i',k']
    #
    # So: sum_c U[i,j,c] V[j',k,c] W[i',k',c] = delta_{i,i'} delta_{j,j'} delta_{k,k'} delta_{j,k}
    #
    # For the "diagonal" case i=i', j=j', k=k', j=k: sum_c U[i,j,c] V[j,k,c] W[i,k,c] = 1
    # For all other cases: sum = 0.
    #
    # Let me verify with a random test:
    A_r = np.random.randn(2,2)
    B_r = np.random.randn(2,2)
    C_r = A_r @ B_r
    C_decomp = np.zeros((2,2))
    for i in range(2):
        for k in range(2):
            for c in range(7):
                # U_c[i,j] for all j, V_c[j,k] for all j
                # C[i,k] = sum_c W[i,k,c] * (sum_j U[i,j,c]*A[i,j]) * (sum_j V[j,k,c]*B[j,k])
                # NO. The correct formula from the tensor:
                # C[i,k] = sum_{j} A[i,j]*B[j,k]
                # = sum_c W[i,k,c] * (sum_{j} U[i,j,c]*A[i,j]) * (sum_{j} V[j,k,c]*B[j,k])
                # This is wrong because it factors incorrectly.
                
                # The correct bilinear form:
                # C[i,k] = sum_c W[i,k,c] * (sum_{i',j'} U[i',j',c] * A[i',j']) * (sum_{j',k'} V[j',k',c] * B[j',k'])
                # This is ALSO wrong.
                
                # I think the issue is that the tensor decomposition gives:
                # C[i,k] = sum_c W[i,k,c] * (sum_{i,j} U[i,j,c] * A[i,j]) * (sum_{j,k} V[j,k,c] * B[j,k])
                # but this is a product of two linear forms, which is not bilinear in (A,B) in the right way.
                
                # Let me look at this differently. The rank of the multiplication tensor
                # is the minimum number of terms in a decomposition:
                # A*B = sum_c (u_c^T A) * (v_c^T B) * w_c
                # where u_c, v_c ∈ F^{n^2}, w_c ∈ F^{n^2}.
                # So C = sum_c w_c * (u_c^T A) * (v_c^T B)
                # In coordinates: C[i,k] = sum_c W[i,k,c] * (sum_{i',j'} U[i',j',c]*A[i',j']) * (sum_{j',k'} V[j',k',c]*B[j',k'])
                # This IS the correct form! Each term is a product of a scalar (u_c^T A), a scalar (v_c^T B), and a matrix w_c.
                # So C[i,k] = sum_c W[i,k,c] * (u_c^T A) * (v_c^T B)
                # where (u_c^T A) = sum_{i',j'} U[i',j',c] * A[i',j'] and similarly for v_c.
                
                u_dot_A = sum(U22n[i2*2+j2, c] * A_r[i2,j2] for i2 in range(2) for j2 in range(2))
                v_dot_B = sum(V22n[j2*2+k2, c] * B_r[j2,k2] for j2 in range(2) for k2 in range(2))
                C_decomp[i,k] += W22n[k*2+i, c] * u_dot_A * v_dot_B
    if np.max(np.abs(C_r - C_decomp)) > 1e-9:
        # Strassen verification failed, use schoolbook rank-4 for 2x2x2
        U22 = [[0]*4 for _ in range(4)]
        V22 = [[0]*4 for _ in range(4)]
        W22 = [[0]*4 for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U22[i*2+j][col] = 1
                    V22[j*2+k][col] = 1
                    W22[k*2+i][col] = 1
                    col += 1
        U22n = np.array(U22, dtype=float)
        V22n = np.array(V22, dtype=float)
        W22n = np.array(W22, dtype=float)
        rank2 = 4
    else:
        rank2 = 7

    if time.time() > deadline - 30:
        # Not enough time, return schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # ---------- Base decomposition 2: rank-2 for 2x1x2 ----------
    # This is for multiplying a 2x1 matrix by a 1x2 matrix to get a 2x2 matrix.
    # A is 2x1: A[i,0] for i in {0,1}
    # B is 1x2: B[0,k] for k in {0,1}
    # C is 2x2: C[i,k] = A[i,0]*B[0,k]
    # This is an outer product, rank 1! C[i,k] = A[i,0]*B[0,k]
    # So rank-1 suffices. But we need to combine with the 2x2x2 Strassen.
    # 
    # Actually, let me reconsider the block structure.
    # A is 4x4, B is 4x8.
    # Split A into 2x2 blocks of 2x2: A = [[A11,A12],[A21,A22]] where each is 2x2.
    # Split B into 2x2 blocks of 2x4: B = [[B11,B12],[B21,B22]] where each is 2x4.
    # C = A*B is 4x8, split into 2x2 blocks of 2x4: C = [[C11,C12],[C21,C22]]
    # C11 = A11*B11 + A12*B21 (2x2 * 2x4 + 2x2 * 2x4 = 2x4)
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Each product Aij*Bjk is (2x2)*(2x4). This can be seen as:
    # Aij (2x2) * Bjk (2x4) = Aij * [Bjk_left, Bjk_right] where Bjk_left, Bjk_right are 2x2.
    # = [Aij*Bjk_left, Aij*Bjk_right]
    # So each (2x2)*(2x4) product = two (2x2)*(2x2) products.
    # Each (2x2)*(2x2) has rank 7 (Strassen).
    # So each (2x2)*(2x4) has rank 14.
    # There are 4 blocks of A and 2 blocks of B per row, so 4*2 = 8 products of type (2x2)*(2x4).
    # Total rank = 8 * 14 = 112. That's worse than 128? No, 112 < 128.
    # But wait, I can do better.
    
    # Alternative: treat the 4x4x8 as 4x4x4 + 4x4x4 (two 4x4x4 blocks).
    # Each 4x4x4 can be done with rank 49 (Strassen applied to 2x2x2 blocks).
    # Total rank = 2 * 49 = 98.
    # This matches the "rank-49 Strassen decomposition for 4x4x4 blocks" finding!
    # Score was 1.306 with rank 98.
    
    # So the best approach: 4x4x8 = two 4x4x4 blocks, each with rank 49.
    # Total rank = 98.
    
    # Let me construct the 4x4x4 Strassen decomposition.
    # 4x4x4: A is 4x4, B is 4x4, C is 4x4.
    # Split into 2x2 blocks of 2x2.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Each Aij*Bjk is (2x2)*(2x2) with rank 7.
    # Total: 8 products * 7 = 56? But the finding says rank 49.
    # 
    # Ah, the rank-49 comes from applying Strassen at the 4x4 level too!
    # Strassen on 4x4x4: treat 4x4 as 2x2 blocks of 2x2, then apply Strassen's
    # 7 multiplications at the block level, where each "multiplication" is a
    # (2x2)*(2x2) = rank 7. So total rank = 7 * 7 = 49.
    
    # So the 4x4x4 decomposition has rank 49, constructed as:
    # For each of the 7 Strassen factors at the 4x4 level, we have a (2x2)*(2x2)
    # sub-product with rank 7. Total = 7*7 = 49.
    
    # Let me construct this properly.
    # The 4x4x4 Strassen:
    # A = [[A11,A12],[A21,A22]] (each 2x2)
    # B = [[B11,B12],[B21,B22]] (each 2x2)
    # M1 = (A11+A22)*(B12-B22)
    # M2 = (A11+A22)*(B21+B22)
    # M3 = (A11-A22)*(B11+B12)
    # M4 = (A21+A22)*(B11-B21)
    # M5 = A11*(B11-B22)
    # M6 = A22*(B22-B11)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4+M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1+M2+M3-M5
    #
    # Each M_f is a (2x2)*(2x2) product, decomposed with rank 7.
    # So total rank = 7 (Strassen factors) * 7 (inner rank) = 49.
    
    # For the 4x4x8 case:
    # B is 4x8 = [[B11,B12,B13,B14],[B21,B22,B23,B24]] where each is 2x2.
    # Actually, let me split B into 2x4 blocks: B = [[B1,B2],[B21,B22]] where each is 2x4.
    # Or split into 2x2 blocks: B has 2 rows of blocks, 4 columns of blocks.
    # C = A*B has 2 rows of blocks, 4 columns of blocks, each 2x2.
    # C[i_block, k_block] = sum_j A[i_block, j_block] * B[j_block, k_block]
    # Each product is (2x2)*(2x2) with rank 7.
    # There are 2*2*4 = 16 such products, but with Strassen at the block level:
    # The block-level multiplication is 2x2 (blocks) * 2x4 (blocks) = 2x4 (blocks).
    # This is a rectangular multiplication. I can split the 2x4 into two 2x2:
    # B = [B_left, B_right] where B_left, B_right are 2x2 blocks (2 rows, 2 cols each).
    # C = A*B = [A*B_left, A*B_right]
    # Each A*B_left and A*B_right is 2x2 * 2x2 = 2x2 (blocks), rank 49 each.
    # Total rank = 2 * 49 = 98.
    
    # So the construction:
    # 1. Build the 4x4x4 Strassen decomposition (rank 49).
    # 2. Apply it twice: once for B_left (cols 0-3 of B) and once for B_right (cols 4-7 of B).
    # 3. Concatenate the factors.
    
    # Let me build the 4x4x4 Strassen decomposition.
    # I'll use the verified 2x2x2 Strassen factors.
    
    # For the 4x4x4 case:
    # A is 4x4, indexed by (i,j) with i,j in {0,1,2,3}.
    # Block index: I = i//2, J = j//2, local: i_loc = i%2, j_loc = j%2.
    # A[I,J] is a 2x2 block.
    # Similarly for B and C.
    
    # The 4x4x4 Strassen decomposition:
    # For each Strassen factor f (0..6) at the block level:
    #   The A-part is a linear combination of A[I,J] blocks.
    #   The B-part is a linear combination of B[J,K] blocks.
    #   The C-part is a linear combination of C[I,K] blocks.
    #   Each (A-part)*(B-part) is a (2x2)*(2x2) product, decomposed with the 2x2x2 Strassen.
    
    # Block-level Strassen coefficients:
    # M1 = (A11+A22)*(B12-B22): A-coeffs: A11:+1, A22:+1; B-coeffs: B12:+1, B22:-1
    # M2 = (A11+A22)*(B21+B22): A-coeffs: A11:+1, A22:+1; B-coeffs: B21:+1, B22:+1
    # M3 = (A11-A22)*(B11+B12): A-coeffs: A11:+1, A22:-1; B-coeffs: B11:+1, B12:+1
    # M4 = (A21+A22)*(B11-B21): A-coeffs: A21:+1, A22:+1; B-coeffs: B11:+1, B21:-1
    # M5 = A11*(B11-B22): A-coeffs: A11:+1; B-coeffs: B11:+1, B22:-1
    # M6 = A22*(B22-B11): A-coeffs: A22:+1; B-coeffs: B22:+1, B11:-1
    # M7 = (A12-A22)*(B21+B22): A-coeffs: A12:+1, A22:-1; B-coeffs: B21:+1, B22:+1
    #
    # C-coeffs:
    # C11: M1:+1, M4:+1, M5:+1, M7:+1
    # C12: M3:+1, M5:+1
    # C21: M2:+1, M4:+1
    # C22: M1:+1, M2:+1, M3:+1, M5:-1
    
    # Now, for each block-level factor f, the inner (2x2)*(2x2) product uses the
    # 2x2x2 Strassen decomposition (rank 7).
    
    # Total rank = 7 (block factors) * 7 (inner factors) = 49.
    
    # Let me construct the 4x4x4 factor matrices.
    # U44: (4*4, 49) = (16, 49), indexed by [i*4+j, c]
    # V44: (4*4, 49) = (16, 49), indexed by [j*4+k, c]
    # W44: (4*4, 49) = (16, 49), indexed by [k*4+i, c]
    
    # For block factor f (0..6) and inner factor g (0..6):
    # c = f*7 + g
    # U44[i*4+j, c] = (block_A_coeff[f][I,J]) * U22[i_loc*2+j_loc, g]
    #   where I=i//2, J=j//2, i_loc=i%2, j_loc=j%2
    # V44[j*4+k, c] = (block_B_coeff[f][J,K]) * V22[j_loc*2+k_loc, g]
    #   where J=j//2, K=k//2, j_loc=j%2, k_loc=k%2
    # W44[k*4+i, c] = (block_C_coeff[f][I,K]) * W22[k_loc*2+i_loc, g]
    #   where K=k//2, I=i//2, k_loc=k%2, i_loc=i%2
    
    # Block coefficients:
    block_A = [
        {0: 1, 3: 1},   # M1: A11, A22 (blocks (0,0) and (1,1))
        {0: 1, 3: 1},   # M2: A11, A22
        {0: 1, 3: -1},  # M3: A11, A22
        {2: 1, 3: 1},   # M4: A21, A22 (blocks (1,0) and (1,1))
        {0: 1},         # M5: A11
        {3: 1},         # M6: A22
        {1: 1, 3: -1},  # M7: A12, A22 (blocks (0,1) and (1,1))
    ]
    block_B = [
        {1: 1, 3: -1},  # M1: B12, B22 (blocks (0,1) and (1,1))
        {2: 1, 3: 1},   # M2: B21, B22 (blocks (1,0) and (1,1))
        {0: 1, 1: 1},   # M3: B11, B12 (blocks (0,0) and (0,1))
        {0: 1, 2: -1},  # M4: B11, B21 (blocks (0,0) and (1,0))
        {0: 1, 3: -1},  # M5: B11, B22
        {3: 1, 0: -1},  # M6: B22, B11
        {2: 1, 3: 1},   # M7: B21, B22
    ]
    block_C = [
        {0: 1, 1: 1, 2: 1, 3: 1},  # C11: M1,M4,M5,M7 -> blocks (0,0),(0,1),(1,0),(1,1)
        {0: 1, 1: 1},               # C12: M3,M5 -> blocks (0,0),(0,1)
        {0: 1, 1: 1},               # C21: M2,M4 -> blocks (0,0),(0,1)
        {0: 1, 1: 1, 2: 1, 3: -1}, # C22: M1,M2,M3,M5 -> blocks (0,0),(0,1),(1,0),(1,1)
    ]
    # Wait, block_C should be indexed by the C-block position, and for each position,
    # which Strassen factors contribute with what coefficient.
    # C11 (block (0,0)): M1:+1, M4:+1, M5:+1, M7:+1
    # C12 (block (0,1)): M3:+1, M5:+1
    # C21 (block (1,0)): M2:+1, M4:+1
    # C22 (block (1,1)): M1:+1, M2:+1, M3:+1, M5:-1
    
    # So block_C[C_block_pos][f] = coefficient
    block_C_coeff = [
        {0: 1, 3: 1, 4: 1, 6: 1},  # C11: M1,M4,M5,M7
        {2: 1, 4: 1},               # C12: M3,M5
        {1: 1, 3: 1},               # C21: M2,M4
        {0: 1, 1: 1, 2: 1, 4: -1}, # C22: M1,M2,M3,M5
    ]
    
    # Build 4x4x4 factors:
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    for f in range(7):
        for g in range(7):
            c = f*7 + g
            for i in range(4):
                for j in range(4):
                    I, J = i//2, j//2
                    i_loc, j_loc = i%2, j%2
                    a_coeff = block_A[f].get(J, 0)  # Wait, A block is (I,J), so key should be (I,J)
                    # block_A[f] is a dict from block_index to coefficient.
                    # Block (I,J) has index I*2+J.
                    a_coeff = block_A[f].get(I*2+J, 0)
                    U44[i*4+j][c] += a_coeff * U22[i_loc*2+j_loc][g]
            for j in range(4):
                for k in range(4):
                    J, K = j//2, k//2
                    j_loc, k_loc = j%2, k%2
                    b_coeff = block_B[f].get(J*2+K, 0)
                    V44[j*4+k][c] += b_coeff * V22[j_loc*2+k_loc][g]
            for k in range(4):
                for i in range(4):
                    K, I = k//2, i//2
                    k_loc, i_loc = k%2, i%2
                    c_coeff = block_C_coeff[I*2+K].get(f, 0)
                    W44[k*4+i][c] += c_coeff * W22[k_loc*2+i_loc][g]
    
    # Verify 4x4x4 decomposition
    U44n = np.array(U44, dtype=float)
    V44n = np.array(V44, dtype=float)
    W44n = np.array(W44, dtype=float)
    
    A_test4 = np.random.randn(4,4)
    B_test4 = np.random.randn(4,4)
    C_test4 = A_test4 @ B_test4
    C_decomp4 = np.zeros((4,4))
    for i in range(4):
        for k in range(4):
            for c in range(49):
                u_dot_A = sum(U44n[i2*4+j2, c] * A_test4[i2,j2] for i2 in range(4) for j2 in range(4))
                v_dot_B = sum(V44n[j2*4+k2, c] * B_test4[j2,k2] for j2 in range(4) for k2 in range(4))
                C_decomp4[i,k] += W44n[k*4+i, c] * u_dot_A * v_dot_B
    if np.max(np.abs(C_test4 - C_decomp4)) > 1e-9:
        # 4x4x4 verification failed, use schoolbook for 4x4x4
        U44 = [[0]*64 for _ in range(16)]
        V44 = [[0]*64 for _ in range(16)]
        W44 = [[0]*64 for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U44[i*4+j][col] = 1
                    V44[j*4+k][col] = 1
                    W44[k*4+i][col] = 1
                    col += 1
        U44n = np.array(U44, dtype=float)
        V44n = np.array(V44, dtype=float)
        W44n = np.array(W44, dtype=float)
        rank44 = 64
    else:
        rank44 = 49

    if time.time() > deadline - 10:
        # Not enough time, return schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Now build the 4x4x8 decomposition.
    # B is 4x8. Split into B_left (4x4, cols 0-3) and B_right (4x4, cols 4-7).
    # C = A*B = [A*B_left, A*B_right] (4x8).
    # Use the 4x4x4 decomposition for each half.
    
    # U8: (4*4, 2*rank44) = (16, 2*rank44)
    # V8: (4*8, 2*rank44) = (32, 2*rank44)
    # W8: (8*4, 2*rank44) = (32, 2*rank44)
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    rank_total = 2 * rank44
    
    U8 = [[0]*rank_total for _ in range(n*m)]
    V8 = [[0]*rank_total for _ in range(m*p)]
    W8 = [[0]*rank_total for _ in range(p*n)]
    
    # For the left half (B cols 0-3, C cols 0-3):
    for c in range(rank44):
        for i in range(4):
            for j in range(4):
                U8[i*4+j][c] = U44[i*4+j][c]
        for j in range(4):
            for k in range(4):
                V8[j*4+k][c] = V44[j*4+k][c]
        for k in range(4):
            for i in range(4):
                W8[k*4+i][c] = W44[k*4+i][c]
    
    # For the right half (B cols 4-7, C cols 4-7):
    for c in range(rank44):
        c2 = rank44 + c
        for i in range(4):
            for j in range(4):
                U8[i*4+j][c2] = U44[i*4+j][c]
        for j in range(4):
            for k in range(4):
                V8[j*4+(k+4)][c2] = V44[j*4+k][c]
        for k in range(4):
            for i in range(4):
                W8[(k+4)*4+i][c2] = W44[k*4+i][c]
    
    # Verify the full 4x4x8 decomposition
    U8n = np.array(U8, dtype=float)
    V8n = np.array(V8, dtype=float)
    W8n = np.array(W8, dtype=float)
    
    A_test8 = np.random.randn(4,4)
    B_test8 = np.random.randn(4,8)
    C_test8 = A_test8 @ B_test8
    C_decomp8 = np.zeros((4,8))
    for i in range(4):
        for k in range(8):
            for c in range(rank_total):
                u_dot_A = sum(U8n[i2*4+j2, c] * A_test8[i2,j2] for i2 in range(4) for j2 in range(4))
                v_dot_B = sum(V8n[j2*4+k2, c] * B_test8[j2,k2] for j2 in range(4) for k2 in range(8))
                C_decomp8[i,k] += W8n[k*4+i, c] * u_dot_A * v_dot_B
    if np.max(np.abs(C_test8 - C_decomp8)) > 1e-9:
        # Full verification failed, return schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U8, "V": V8, "W": W8}
# EVOLVE-BLOCK-END
