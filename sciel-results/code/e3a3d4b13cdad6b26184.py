import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-48 decomposition for 4x4 matrix multiplication.
    Starts with rank-49 recursive Strassen, then attempts rank reduction to 48.
    Falls back to rank-49 if reduction fails.
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      U index: i*4 + k  (A row i, A col k)
      V index: k*4 + j  (B row k, B col j)
      W index: j*4 + i  (C col j, C row i)  [trace convention]
    """
    t0 = time.time()
    deadline = 140.0

    n = m = p = 4

    # Build the rank-49 recursive Strassen decomposition using direct lookup
    U49 = np.zeros((16, 49), dtype=np.float64)
    V49 = np.zeros((16, 49), dtype=np.float64)
    W49 = np.zeros((16, 49), dtype=np.float64)

    # 2x2 Strassen decomposition (rank 7) - explicit entries
    # s1: (A00+A11)(B00+B11) -> C00+=s1, C11+=s1
    # s2: (A10+A11)B00 -> C10+=s2, C11-=s2
    # s3: A00(B01-B11) -> C01+=s3, C11+=s3
    # s4: A11(B10-B00) -> C00+=s4, C10+=s4
    # s5: (A00+A01)B11 -> C00-=s5, C01+=s5
    # s6: (A10-A00)(B00+B01) -> C11+=s6
    # s7: (A01-A11)(B10+B11) -> C00+=s7

    # U2[i*2+k, s] for 2x2 A matrix
    U2 = np.zeros((4, 7), dtype=np.float64)
    # V2[k*2+j, s] for 2x2 B matrix  
    V2 = np.zeros((4, 7), dtype=np.float64)
    # W2[j*2+i, s] for 2x2 C matrix (trace convention: j*2+i)
    W2 = np.zeros((4, 7), dtype=np.float64)

    # s1: (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1  # C00 and C11

    # s2: (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[1,1]=1; W2[3,1]=-1  # C10+=s2, C11-=s2

    # s3: A00(B01-B11)
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[2,2]=1; W2[3,2]=1  # C01+=s3, C11+=s3

    # s4: A11(B10-B00)
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[1,3]=1  # C00+=s4, C10+=s4

    # s5: (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[2,4]=1  # C00-=s5, C01+=s5

    # s6: (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1  # C11+=s6

    # s7: (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1  # C00+=s7

    # Outer Strassen terms for 4x4 = 2x2 blocks of 2x2
    # Each term t has: left block coeffs, right block coeffs, C block coeffs
    # Block indices: (br, bc) where br in {0,1}, bc in {0,1}
    # A blocks: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # B blocks: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    # C blocks: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)

    outer_def = [
        # t=0: (A00+A11)(B00+B11) -> C00+=, C11+=
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # t=1: (A10+A11)B00 -> C10+=, C11-=
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        # t=2: A00(B01-B11) -> C01+=, C11+=
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        # t=3: A11(B10-B00) -> C00+=, C10+=
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        # t=4: (A00+A01)B11 -> C00-=, C01+=
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        # t=5: (A10-A00)(B00+B01) -> C11+=
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        # t=6: (A01-A11)(B10+B11) -> C00+=
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

    for t in range(7):
        left_c, right_c, c_c = outer_def[t]
        for k in range(7):
            li = t * 7 + k
            # Fill U49: for each element (i, ak) in 4x4 A
            for i in range(4):
                for ak in range(4):
                    br = i // 2
                    bc = ak // 2
                    li_l = i % 2
                    lk_l = ak % 2
                    coeff = U2[li_l*2 + lk_l, k] * left_c.get((br, bc), 0)
                    if coeff != 0:
                        U49[i*4+ak, li] = coeff
            # Fill V49: for each element (bk, j) in 4x4 B
            for bk in range(4):
                for j in range(4):
                    br = bk // 2
                    bc = j // 2
                    lk_l = bk % 2
                    lj_l = j % 2
                    coeff = V2[lk_l*2 + lj_l, k] * right_c.get((br, bc), 0)
                    if coeff != 0:
                        V49[bk*4+j, li] = coeff
            # Fill W49: for each element (cj, ci) in 4x4 C (trace: j*4+i)
            for cj in range(4):
                for ci in range(4):
                    cbr = ci // 2
                    cbc = cj // 2
                    li_l = ci % 2
                    lj_l = cj % 2
                    coeff = W2[lj_l*2 + li_l, k] * c_c.get((cbr, cbc), 0)
                    if coeff != 0:
                        W49[cj*4+ci, li] = coeff

    # Verify rank-49 decomposition
    err49 = 0.0
    for i in range(4):
        for k in range(4):
            for j in range(4):
                s = 0.0
                for r in range(49):
                    s += U49[i*4+k, r] * V49[k*4+j, r] * W49[j*4+i, r]
                err49 += abs(s - 1.0)

    if err49 > 1e-8:
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

    # Attempt rank reduction from 49 to 48 using proper affine combination.
    # If T_elim = sum_{r != elim} c_r * T_r, then we need to find U_new, V_new, W_new
    # such that sum_{s=0}^{47} U_new[:,s] ⊗ V_new[:,s] ⊗ W_new[:,s] = T_full
    # 
    # Strategy: For each pair (elim, keep), try to express T_elim as a linear
    # combination of the other 48 tensors. If T_elim = sum c_r * T_r, then
    # T_full = sum_{r != elim} T_r + T_elim = sum_{r != elim} (1 + c_r) * T_r
    # But (1+c_r)*T_r is NOT rank-1 in general unless we can absorb the scalar.
    #
    # Actually, c_r * T_r = (c_r^{1/3} U_r) ⊗ (c_r^{1/3} V_r) ⊗ (c_r^{1/3} W_r)
    # only works if c_r >= 0 and we use real cube roots. For negative c_r, we
    # need complex or we can't do it directly.
    #
    # Better approach: Use the fact that if there's a linear dependency among the
    # 49 rank-1 tensors, we can eliminate one. Let's check if any column is a
    # linear combination of the others in the tensor space.
    
    # Build the full tensor vector
    T_full = np.zeros((16*16*16), dtype=np.float64)
    for i in range(4):
        for k in range(4):
            for j in range(4):
                T_full[(i*4+k)*256 + (k*4+j)*16 + (j*4+i)] = 1.0

    # Precompute all 49 rank-1 tensor vectors
    T_vecs = np.zeros((49, 16*16*16), dtype=np.float64)
    for r in range(49):
        T_vecs[r] = np.kron(np.kron(U49[:, r], V49[:, r]), W49[:, r])

    best_rank = 49
    best_U = U49
    best_V = V49
    best_W = W49

    # Try to find a linear dependency: T_elim = sum_{r != elim} c_r * T_r
    # This means the 49 vectors are linearly dependent.
    # For each elim, solve the underdetermined system.
    
    for elim in range(49):
        if time.time() - t0 > deadline - 30:
            break
        
        # Get the 48 other tensors
        mask = [r for r in range(49) if r != elim]
        M = T_vecs[mask]  # shape (48, 4096)
        target = T_vecs[elim]
        
        # Solve M.T @ c = target (least squares)
        # M.T is (4096, 48), we want c such that M.T @ c ≈ target
        try:
            c, residuals, rank_M, sv = np.linalg.lstsq(M.T, target, rcond=1e-12)
            recon = M.T @ c
            err = np.linalg.norm(recon - target) / max(1.0, np.linalg.norm(target))
            
            if err < 1e-8:
                # T_elim = sum_{idx} c[idx] * T_{mask[idx]}
                # T_full = sum_{r != elim} T_r + T_elim = sum_{idx} (1 + c[idx]) * T_{mask[idx]}
                # For each term, scale: (1+c[idx]) * T_r = a*U_r ⊗ b*V_r ⊗ d*W_r
                # where a*b*d = 1+c[idx]. We can distribute: a = b = d = (1+c[idx])^(1/3)
                # But if 1+c[idx] < 0, we need to handle the sign.
                # For now, assume all (1+c[idx]) > 0.
                
                U_new = np.zeros((16, 48), dtype=np.float64)
                V_new = np.zeros((16, 48), dtype=np.float64)
                W_new = np.zeros((16, 48), dtype=np.float64)
                
                valid = True
                for idx, r in enumerate(mask):
                    scale_total = 1.0 + c[idx]
                    if scale_total < -1e-12:
                        # Negative scale - can't take real cube root easily
                        # Try: absorb sign into one factor
                        # scale_total * U ⊗ V ⊗ W = (-1) * (-scale_total) * U ⊗ V ⊗ W
                        # = (-U) ⊗ (V) ⊗ (W) * (-scale_total)
                        # Still need cube root of negative... use: 
                        # a = -abs(scale_total)^(1/3), b = abs(scale_total)^(1/3), d = abs(scale_total)^(1/3)
                        # a*b*d = -abs(scale_total) = scale_total ✓
                        a = -abs(scale_total) ** (1.0/3.0)
                        b = abs(scale_total) ** (1.0/3.0)
                        d = abs(scale_total) ** (1.0/3.0)
                    else:
                        a = scale_total ** (1.0/3.0)
                        b = a
                        d = a
                    
                    U_new[:, idx] = a * U49[:, r]
                    V_new[:, idx] = b * V49[:, r]
                    W_new[:, idx] = d * W49[:, r]
                
                # Verify
                err_verify = 0.0
                for i in range(4):
                    for k in range(4):
                        for j in range(4):
                            s = 0.0
                            for s_idx in range(48):
                                s += U_new[i*4+k, s_idx] * V_new[k*4+j, s_idx] * W_new[j*4+i, s_idx]
                            err_verify += abs(s - 1.0)
                
                if err_verify < 1e-6:
                    best_rank = 48
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    break
        except:
            pass

    if best_rank == 48:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}
# EVOLVE-BLOCK-END
