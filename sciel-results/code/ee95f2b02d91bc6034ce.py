import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is valid."""
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_target[a_idx, b_idx, c_idx] = 1.0
    return np.allclose(T_comp, T_target, atol=tol)

def construct():
    start_time = time.time()
    deadline = 145
    
    n, m, p = DIMENSIONS # 2, 4, 7
    
    # Trivial rank is 56.
    # We aim for rank 50 using block decomposition.
    # Split m=4 into 2x2 blocks.
    # Split p=7 into 2+2+2+1? No, standard Strassen splits into powers of 2.
    # Let's split m=4 into two 2s.
    # Let's split p=7 into two 2s and one 3? Or maybe just treat it as 4x4 and pad?
    # Actually, the simplest improvement is often splitting the largest dimension.
    # Let's try splitting m=4 into 2+2.
    # The tensor is T_{i, j, k}.
    # If we split j into j1, j2 (size 2 each), and k into k1, k2, k3?
    # Strassen works on 2x2 matrix multiplication.
    # Matrix A is 2x4. Matrix B is 4x7.
    # We can view A as [A1 A2] where A1, A2 are 2x2.
    # We can view B as [B1; B2] where B1, B2 are 2x7.
    # Product C = A1 B1 + A2 B2.
    # This is just two 2x2 * 2x7 multiplications.
    # Rank of 2x2 * 2x7 multiplication?
    # Trivial is 2*2*7 = 28.
    # Can we do better?
    # If we use Strassen on the 2x2 part of A and B, we need B to be 2x2 as well.
    # So we should split p=7 into 2+2+2+1?
    # Let's split p into four parts: 2, 2, 2, 1.
    # B = [B1; B2; B3; B4] where B1, B2, B3 are 2x2, B4 is 2x1.
    # A = [A1 A2] (2x2 each).
    # C = A1 B1 + A2 B2 + A1 B3 + A2 B4? No, the indices must match.
    # A is 2x4. Columns 0,1 are A1. Columns 2,3 are A2.
    # B is 4x7. Rows 0,1 are B1. Rows 2,3 are B2.
    # C = A1 B1 + A2 B2.
    # B1 is 2x7. B2 is 2x7.
    # To use Strassen, we need to multiply 2x2 by 2x2.
    # So we must split the 7 columns of B into groups of 2 (and remainder).
    # B1 (2x7) = [B11 B12 B13 B14] where B11, B12, B13 are 2x2, B14 is 2x1.
    # B2 (2x7) = [B21 B22 B23 B24] where B21, B22, B23 are 2x2, B24 is 2x1.
    
    # Term 1: A1 * B11 (2x2 * 2x2). Use Strassen (Rank 7).
    # Term 2: A1 * B12 (2x2 * 2x2). Use Strassen (Rank 7).
    # Term 3: A1 * B13 (2x2 * 2x2). Use Strassen (Rank 7).
    # Term 4: A1 * B14 (2x2 * 2x1). Trivial (Rank 2*2*1 = 4).
    # Term 5: A2 * B21 (2x2 * 2x2). Use Strassen (Rank 7).
    # Term 6: A2 * B22 (2x2 * 2x2). Use Strassen (Rank 7).
    # Term 7: A2 * B23 (2x2 * 2x2). Use Strassen (Rank 7).
    # Term 8: A2 * B24 (2x2 * 2x1). Trivial (Rank 4).
    
    # Total Rank = 7 + 7 + 7 + 4 + 7 + 7 + 7 + 4 = 50.
    # This is an improvement over 56.
    
    # Let's construct this explicitly.
    # Dimensions: n=2, m=4, p=7.
    # A indices: i in {0,1}, j in {0,1,2,3}.
    # B indices: j in {0,1,2,3}, k in {0..6}.
    # C indices: k in {0..6}, i in {0,1}.
    
    # Blocks:
    # A1: i in {0,1}, j in {0,1}
    # A2: i in {0,1}, j in {2,3}
    
    # B1 (rows 0,1):
    #   B11: cols 0,1
    #   B12: cols 2,3
    #   B13: cols 4,5
    #   B14: col 6
    # B2 (rows 2,3):
    #   B21: cols 0,1
    #   B22: cols 2,3
    #   B23: cols 4,5
    #   B24: col 6
    
    # We will build U, V, W matrices.
    # Total rank R = 50.
    
    R = 50
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    r_idx = 0
    
    # Helper to add a 2x2x2 Strassen block
    # A_block: rows i0..i1, cols j0..j1
    # B_block: rows j0..j1, cols k0..k1
    # C_block: rows k0..k1, cols i0..i1
    def add_strassen_2x2x2(i0, j0, k0):
        nonlocal r_idx
        # Indices in local 2x2 grid:
        # A: (0,0), (0,1), (1,0), (1,1) -> global i=i0+di, j=j0+dj
        # B: (0,0), (0,1), (1,0), (1,1) -> global j=j0+dj, k=k0+dk
        # C: (0,0), (0,1), (1,0), (1,1) -> global k=k0+dk, i=i0+di
        
        # Strassen 7 products for 2x2 matrices:
        # M1 = (A11+A22) * (B11+B22)
        # M2 = (A21+A22) * B11
        # M3 = A11 * (B12-B22)
        # M4 = A22 * (B21-B11)
        # M5 = (A11+A12) * B22
        # M6 = (A21-A11) * (B11+B12)
        # M7 = (A12-A22) * (B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these linear combinations to the tensor factors U, V, W.
        # Each Strassen product corresponds to one rank term.
        # For M1:
        #   U factors: A11 + A22
        #   V factors: B11 + B22
        #   W factors: C11 + C22? No, W is the output coefficient.
        #   Wait, the tensor decomposition is T = sum U_r V_r W_r.
        #   For a specific term like A11 * B11 contributing to C11:
        #   U_{a, r} V_{b, r} W_{c, r}
        
        # Let's define the 7 rank terms for this 2x2x2 block.
        
        # Term 1: M1 = (A11+A22)(B11+B22)
        #   U: A11=1, A22=1 (others 0 in this block)
        #   V: B11=1, B22=1
        #   W: contributes to C11 (+1), C22 (+1)?
        #   Actually, M1 contributes to C11 and C22.
        #   In the sum, C11 = M1 + ... and C22 = M1 + ...
        #   So W for this term should have 1 at C11 and 1 at C22?
        #   Let's check the tensor identity:
        #   sum_r U_ar V_br W_cr = C_ck
        #   If we have one term r:
        #   U has non-zeros at A11, A22.
        #   V has non-zeros at B11, B22.
        #   W has non-zeros at C11, C22.
        #   Then the product generates terms:
        #   A11 B11 C11, A11 B22 C11, A22 B11 C11, A22 B22 C11
        #   A11 B11 C22, A11 B22 C22, A22 B11 C22, A22 B22 C22
        #   This is exactly (A11+A22)(B11+B22) for both C11 and C22.
        #   So W_{C11, r} = 1, W_{C22, r} = 1.
        
        # Let's define the 7 Strassen terms with their U, V, W patterns.
        # Local indices:
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        # Term 1: (A00+A11) * (B00+B11) -> C00, C11
        # U: 1 at A00, A11
        # V: 1 at B00, B11
        # W: 1 at C00, C11
        U[i0*2 + 0, j0*2 + 0, r_idx] = 1 # Wait, U shape is (n*m, R).
        # a_idx = i * m + j.
        # i0, j0 are global offsets.
        # Local A indices:
        # A00 -> i=i0, j=j0
        # A01 -> i=i0, j=j0+1
        # A10 -> i=i0+1, j=j0
        # A11 -> i=i0+1, j=j0+1
        
        def set_U(i, j, val):
            U[i*m + j, r_idx] += val
            
        def set_V(j, k, val):
            V[j*p + k, r_idx] += val
            
        def set_W(k, i, val):
            W[k*n + i, r_idx] += val
            
        # Term 1
        set_U(i0, j0, 1)
        set_U(i0+1, j0+1, 1)
        set_V(j0, k0, 1)
        set_V(j0+1, k0+1, 1)
        set_W(k0, i0, 1)
        set_W(k0+1, i0+1, 1)
        r_idx += 1
        
        # Term 2: (A10+A11) * B00 -> C10, C11?
        # M2 = (A21+A22) * B11 (using 1-based 2,1 etc)
        # In 0-based: (A10+A11) * B00
        # Contributes to C10, C11?
        # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
        # So M2 contributes to C10 (with +1) and C11 (with -1).
        # U: 1 at A10, A11
        # V: 1 at B00
        # W: 1 at C10, -1 at C11
        set_U(i0+1, j0, 1)
        set_U(i0+1, j0+1, 1)
        set_V(j0, k0, 1)
        set_W(k0, i0+1, 1)
        set_W(k0+1, i0+1, -1)
        r_idx += 1
        
        # Term 3: A00 * (B01-B11) -> C01, C11?
        # M3 = A11 * (B12-B22) (1-based) -> A00 * (B01-B11) (0-based)
        # C12 = M3 + M5. C22 = M1 - M2 + M3 + M6.
        # So M3 contributes to C01 (with +1) and C11 (with +1).
        # U: 1 at A00
        # V: 1 at B01, -1 at B11
        # W: 1 at C01, 1 at C11
        set_U(i0, j0, 1)
        set_V(j0, k0+1, 1)
        set_V(j0+1, k0+1, -1)
        set_W(k0+1, i0, 1)
        set_W(k0+1, i0+1, 1)
        r_idx += 1
        
        # Term 4: A11 * (B10-B00) -> C10, C11?
        # M4 = A22 * (B21-B11) (1-based) -> A11 * (B10-B00) (0-based)
        # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
        # So M4 contributes to C10 (with +1) and C11 (with -1).
        # U: 1 at A11
        # V: 1 at B10, -1 at B00
        # W: 1 at C10, -1 at C11
        set_U(i0+1, j0+1, 1)
        set_V(j0+1, k0, 1)
        set_V(j0, k0, -1)
        set_W(k0, i0+1, 1)
        set_W(k0+1, i0+1, -1)
        r_idx += 1
        
        # Term 5: (A00+A01) * B11 -> C01, C11?
        # M5 = (A11+A12) * B22 (1-based) -> (A00+A01) * B11 (0-based)
        # C12 = M3 + M5. C22 = M1 - M2 + M3 + M6.
        # So M5 contributes to C01 (with +1) and C11 (with -1).
        # U: 1 at A00, A01
        # V: 1 at B11
        # W: 1 at C01, -1 at C11
        set_U(i0, j0, 1)
        set_U(i0, j0+1, 1)
        set_V(j0+1, k0+1, 1)
        set_W(k0+1, i0, 1)
        set_W(k0+1, i0+1, -1)
        r_idx += 1
        
        # Term 6: (A10-A00) * (B00+B01) -> C11, C10?
        # M6 = (A21-A11) * (B11+B12) (1-based) -> (A10-A00) * (B00+B01) (0-based)
        # C22 = M1 - M2 + M3 + M6.
        # C21? No, M6 is not in C21.
        # Wait, let's re-verify Strassen formulas.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # M6 contributes to C22 (with +1).
        # Does it contribute elsewhere? No.
        # So W: 1 at C11.
        # U: 1 at A10, -1 at A00
        # V: 1 at B00, B01
        set_U(i0+1, j0, 1)
        set_U(i0, j0, -1)
        set_V(j0, k0, 1)
        set_V(j0, k0+1, 1)
        set_W(k0+1, i0+1, 1)
        r_idx += 1
        
        # Term 7: (A01-A11) * (B10+B11) -> C00, C10?
        # M7 = (A12-A22) * (B21+B22) (1-based) -> (A01-A11) * (B10+B11) (0-based)
        # C11 = M1 + M4 - M5 + M7.
        # So M7 contributes to C00 (with +1).
        # U: 1 at A01, -1 at A11
        # V: 1 at B10, B11
        # W: 1 at C00
        set_U(i0, j0+1, 1)
        set_U(i0+1, j0+1, -1)
        set_V(j0+1, k0, 1)
        set_V(j0+1, k0+1, 1)
        set_W(k0, i0, 1)
        r_idx += 1

    # Helper to add a 2x2x1 Schoolbook block
    # A_block: 2x2 (i0, j0)
    # B_block: 2x1 (j0, k0)
    # C_block: 1x2 (k0, i0)
    def add_schoolbook_2x2x1(i0, j0, k0):
        nonlocal r_idx
        # C_k0_i0 = A_i0_j0 * B_j0_k0 + A_i0+1_j0 * B_j0+1_k0 ?
        # No, A is 2x2, B is 2x1.
        # C is 1x2.
        # C[k0, i0] = sum_{j in {j0, j0+1}} A[i0, j] * B[j, k0]
        # C[k0, i0+1] = sum_{j in {j0, j0+1}} A[i0+1, j] * B[j, k0]
        
        # Term 1: A[i0, j0] * B[j0, k0] -> C[k0, i0]
        set_U(i0, j0, 1)
        set_V(j0, k0, 1)
        set_W(k0, i0, 1)
        r_idx += 1
        
        # Term 2: A[i0, j0+1] * B[j0+1, k0] -> C[k0, i0]
        set_U(i0, j0+1, 1)
        set_V(j0+1, k0, 1)
        set_W(k0, i0, 1)
        r_idx += 1
        
        # Term 3: A[i0+1, j0] * B[j0, k0] -> C[k0, i0+1]
        set_U(i0+1, j0, 1)
        set_V(j0, k0, 1)
        set_W(k0, i0+1, 1)
        r_idx += 1
        
        # Term 4: A[i0+1, j0+1] * B[j0+1, k0] -> C[k0, i0+1]
        set_U(i0+1, j0+1, 1)
        set_V(j0+1, k0, 1)
        set_W(k0, i0+1, 1)
        r_idx += 1

    # Execute the plan:
    # A1 (i=0,1, j=0,1) * B1 (j=0,1, k=0,1) -> Strassen
    add_strassen_2x2x2(0, 0, 0)
    
    # A1 (i=0,1, j=0,1) * B1 (j=0,1, k=2,3) -> Strassen
    add_strassen_2x2x2(0, 0, 2)
    
    # A1 (i=0,1, j=0,1) * B1 (j=0,1, k=4,5) -> Strassen
    add_strassen_2x2x2(0, 0, 4)
    
    # A1 (i=0,1, j=0,1) * B1 (j=0,1, k=6) -> Schoolbook 2x2x1
    # B1 is rows 0,1. Col 6.
    add_schoolbook_2x2x1(0, 0, 6)
    
    # A2 (i=0,1, j=2,3) * B2 (j=2,3, k=0,1) -> Strassen
    add_strassen_2x2x2(0, 2, 0)
    
    # A2 (i=0,1, j=2,3) * B2 (j=2,3, k=2,3) -> Strassen
    add_strassen_2x2x2(0, 2, 2)
    
    # A2 (i=0,1, j=2,3) * B2 (j=2,3, k=4,5) -> Strassen
    add_strassen_2x2x2(0, 2, 4)
    
    # A2 (i=0,1, j=2,3) * B2 (j=2,3, k=6) -> Schoolbook 2x2x1
    add_schoolbook_2x2x1(0, 2, 6)
    
    # Check if we used all ranks
    assert r_idx == R, f"Used {r_idx} ranks, expected {R}"
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to trivial if verification fails (should not happen)
        U_triv, V_triv, W_triv = _build_trivial(n, m, p)
        return {
            "U": U_triv.tolist(),
            "V": V_triv.tolist(),
            "W": W_triv.tolist()
        }
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _build_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
    return U, V, W

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")
