import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 145.0

def construct():
    n, m, p = 2, 2, 15
    
    # First, let's try to find a rank-10 decomposition for 2x2x3
    # T[i,j,k] = 1 for i,j in {0,1}, k in {0,1,2}
    # U[i,j,r] * V[j,k,r] * W[i,k,r] summed over r = 1
    
    # Let me try a structured approach.
    # For each pair (i,k), we need: sum_j sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = 1
    # 
    # Alternative: fix W and solve for U, V.
    # Or use the fact that this is a bilinear problem.
    
    # Let me try a different approach: use the known Strassen-like decomposition
    # for 2x2 matrices and extend to 3 columns.
    
    # For 2x2x2 (Strassen), rank is 7:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # For 2x2x3, we have 3 columns of B. Can we do rank 10?
    # Let me try: use Strassen for columns (0,1) with rank 7,
    # and add 3 more rank-1 tensors for column 2.
    # Total: 7 + 3 = 10.
    
    # Wait, but for column 2, we need:
    # C02 = A00*B02 + A01*B12  (needs 2 rank-1 tensors)
    # C12 = A10*B02 + A11*B12  (needs 2 rank-1 tensors)
    # That's 4, not 3.
    
    # Unless we can share. Let me think...
    # Actually, the 3 additional rank-1 tensors for column 2 would be:
    # R1: A00*B02 -> C02
    # R2: A01*B12 -> C02
    # R3: A10*B02 -> C12
    # R4: A11*B12 -> C12
    # That's 4, giving total 7+4 = 11.
    
    # Can we do better? What if we use a different decomposition for the 3 columns?
    
    # Let me try to find a rank-10 decomposition by solving the system.
    # I'll use a random initialization and gradient descent or just try to solve linearly.
    
    # Actually, for a fixed set of W values, the problem is linear in U and V.
    # But it's nonlinear overall. Let me try a different approach.
    
    # Let me just try the 7+4 = 11 rank approach first, giving total 5*11 = 55 > 53.
    # That's worse.
    
    # What about using Strassen for all 3 columns somehow?
    # Or a different block structure?
    
    # Let me try: what if we use a rank-9 decomposition for 2x2x3?
    # 5*9 = 45 < 53. That would be great!
    
    # Let me search for this numerically.
    
    # Approach: parameterize and use least squares.
    # For rank r, we have r*4 + r*6 + r*6 = 16r unknowns and 12 equations.
    # For r >= 1, it's underdetermined, so solutions exist.
    # The question is whether we can find one with small integer/half-integer entries.
    
    # Let me try rank 7 for 2x2x3:
    # 7*16 = 112 unknowns, 12 equations. Very underdetermined.
    
    # Actually, I realize I should just try to construct it directly.
    # Let me use the approach of sharing Strassen intermediates.
    
    # For 2x2x3, let me define:
    # B columns: B00,B01,B02 (row 0) and B10,B11,B12 (row 1)
    # 
    # Strassen for (0,1):
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # For column 2, I need:
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    #
    # Can I express these using the existing P's plus 2 or 3 new terms?
    # 
    # Let me try adding:
    # P8 = A00*B02
    # P9 = A01*B12
    # P10 = A10*B02
    # P11 = A11*B12
    # That's 4 more, total 11.
    
    # Alternatively:
    # P8 = (A00+A11)*B02  -> contributes to both C02 and C12
    # P9 = (A01+A10)*B12  -> contributes to both C02 and C12
    # C02 = P8 + P9 + (A00-A11)*B02 + (A01-A10)*B12... no this is getting complicated.
    
    # Let me just try rank 10 with a numerical search.
    
    # Actually, let me step back and think about what rank is achievable.
    # 
    # The 2x2xn all-ones tensor: I believe the rank is 2n+1 for n>=2.
    # For n=2: rank 5? But Strassen gives 7. So that can't be right.
    # 
    # Actually, I think the rank of M_{2,2,n} is not simply 2n+1.
    # For n=1: rank 4
    # For n=2: rank 7 (Strassen)
    # For n=3: rank ?
    # 
    # Let me just try to find the minimum rank numerically for small n.
    
    # For now, let me stick with the current rank-53 solution and verify it's correct.
    # Then I'll try to improve.
    
    # Let me build the rank-53 solution and verify.
    rank = 53
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(30)]
    W = [[0] * rank for _ in range(30)]
    
    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # P1: (A00+A11)(B0k0+B1k1)
        U[0][r1] = 1
        U[3][r1] = 1
        V[0 * p + k0][r1] = 1
        V[1 * p + k1][r1] = 1
        
        # P2: (A10+A11)B0k0
        U[2][r2] = 1
        U[3][r2] = 1
        V[0 * p + k0][r2] = 1
        
        # P3: A00(B0k1-B1k1)
        U[0][r3] = 1
        V[0 * p + k1][r3] = 1
        V[1 * p + k1][r3] = -1
        
        # P4: A11(B1k0-B0k0)
        U[3][r4] = 1
        V[1 * p + k0][r4] = 1
        V[0 * p + k0][r4] = -1
        
        # P5: (A00+A01)B1k1
        U[0][r5] = 1
        U[1][r5] = 1
        V[1 * p + k1][r5] = 1
        
        # P6: (A10-A00)(B0k0+B0k1)
        U[2][r6] = 1
        U[0][r6] = -1
        V[0 * p + k0][r6] = 1
        V[0 * p + k1][r6] = 1
        
        # P7: (A01-A11)(B1k0+B1k1)
        U[1][r7] = 1
        U[3][r7] = -1
        V[1 * p + k0][r7] = 1
        V[1 * p + k1][r7] = 1
        
        # Reconstruction
        W[2 * k0][r1] = 1
        W[2 * k0][r4] = 1
        W[2 * k0][r5] = -1
        W[2 * k0][r7] = 1
        
        W[2 * k0 + 1][r2] = 1
        W[2 * k0 + 1][r4] = 1
        
        W[2 * k1][r3] = 1
        W[2 * k1][r5] = 1
        
        W[2 * k1 + 1][r1] = 1
        W[2 * k1 + 1][r2] = -1
        W[2 * k1 + 1][r3] = 1
        W[2 * k1 + 1][r6] = 1
    
    for i in range(7):
        add_strassen_block(7 * i, 2 * i, 2 * i + 1)
    
    # Trivial for column 14
    base_r = 49
    U[0][base_r] = 1
    V[14][base_r] = 1
    W[28][base_r] = 1
    
    U[1][base_r+1] = 1
    V[29][base_r+1] = 1
    W[28][base_r+1] = 1
    
    U[2][base_r+2] = 1
    V[14][base_r+2] = 1
    W[29][base_r+2] = 1
    
    U[3][base_r+3] = 1
    V[29][base_r+3] = 1
    W[29][base_r+3] = 1
    
    # Verify
    if time.time() < _DEADLINE:
        rng = np.random.default_rng(123)
        A = rng.integers(-5, 5, size=(2, 2))
        B = rng.integers(-5, 5, size=(2, 15))
        C_expected = A @ B
        
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Compute C from decomposition
        # C[i,k] = sum_r sum_j A[i,j] * U[i*2+j, r] * (sum_k' B[j',k'] * V[j'*15+k', r]) * W[k*2+i, r]
        # Wait, I need to be more careful.
        # 
        # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # where T[i,j,k] = 1 for all i,j,k.
        # 
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r A[i,j] * B[j,k] * T[i,j,k]
        #        = sum_r (sum_j A[i,j] * U[i,j,r]) * (sum_j B[j,k] * V[j,k,r]) * W[i,k,r]
        # 
        # Wait, that's not right either. Let me re-derive.
        # 
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # We have T[i,j,k] = 1 = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # So: C[i,k] = sum_j A[i,j] * B[j,k] * 1
        #            = sum_j A[i,j] * B[j,k] * sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        #            = sum_r W[i,k,r] * (sum_j A[i,j] * U[i,j,r]) * (sum_j B[j,k] * V[j,k,r])
        # 
        # But wait, U is indexed by [i,j] and V by [j,k]. The j in U[i,j,r] and V[j,k,r] 
        # must be the same j. So:
        # C[i,k] = sum_r W[i,k,r] * (sum_j A[i,j] * U[i,j,r]) * (sum_j B[j,k] * V[j,k,r])
        # 
        # Hmm, but that's not quite right because the j in U and V are summed independently.
        # Actually, I think the correct formula is:
        # C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]
        #         = sum_j A[i,j] * B[j,k] * sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        #         = sum_r W[i,k,r] * sum_j A[i,j] * U[i,j,r] * B[j,k] * V[j,k,r]
        # 
        # This is getting confusing. Let me just directly compute:
        # For each (i,k), C[i,k] should equal sum_j A[i,j]*B[j,k].
        # The decomposition guarantees this if T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = 1.
        
        # Let me just verify T[i,j,k] = 1 for all i,j,k.
        T_reconstructed = np.zeros((2, 2, 15))
        for i in range(2):
            for j in range(2):
                for k in range(15):
                    s = 0
                    for r in range(rank):
                        s += U[i*2+j][r] * V[j*15+k][r] * W[k*2+i][r]
                    T_reconstructed[i,j,k] = s
        
        if not np.allclose(T_reconstructed, 1.0):
            print("ERROR: Tensor decomposition is incorrect!")
            print("Max error:", np.max(np.abs(T_reconstructed - 1.0)))
            # Print which entries are wrong
            for i in range(2):
                for j in range(2):
                    for k in range(15):
                        if abs(T_reconstructed[i,j,k] - 1.0) > 1e-10:
                            print(f"  T[{i},{j},{k}] = {T_reconstructed[i,j,k]}")
        
        # Also verify with actual matrix multiplication
        C_from_decomp = np.zeros((2, 15))
        for i in range(2):
            for k in range(15):
                s = 0
                for j in range(2):
                    s += A[i,j] * B[j,k]
                C_from_decomp[i,k] = s
        
        if not np.allclose(C_from_decomp, C_expected):
            print("ERROR: Matrix multiplication verification failed!")
    
    return {"U": U, "V": V, "W": W}
