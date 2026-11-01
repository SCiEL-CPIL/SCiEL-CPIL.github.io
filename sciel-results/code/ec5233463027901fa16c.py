"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # First, try to find a rank-10 decomposition for 2x2x3
    # If found: 6 Strassen blocks (rank 42) + 2x2x3 rank 10 = rank 52
    # If not: 7 Strassen blocks (rank 49) + schoolbook (rank 4) = rank 53
    
    def find_2x2x3_rank10():
        """Search for rank-10 decomposition of 2x2x3 matrix mult tensor."""
        # Tensor: T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
        # A is 2x2, B is 2x3, C is 2x3
        # U: 4 x rank, V: 6 x rank, W: 6 x rank
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        # B indices: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12
        # C indices: 0=C00, 1=C01, 2=C02, 3=C10, 4=C11, 5=C12
        
        # Target tensor entries (non-zero):
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C02 = A00*B02 + A01*B12
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        # C12 = A10*B02 + A11*B12
        
        # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # where a in {0,1,2,3}, b in {0,1,2,3,4,5}, c in {0,1,2,3,4,5}
        
        # Use a structured search: try various patterns
        # The 2x2x2 Strassen part handles columns 0,1 with rank 7
        # We need rank 3 for column 2, but maybe we can share terms
        
        # Alternative: try to find a direct rank-10 decomposition
        # by parameterizing with small integers/half-integers
        
        # Let's try a more systematic approach using the known structure
        # Start from Strassen(2x2x2) + schoolbook(2x2x1) = rank 11
        # and try to merge/simplify to rank 10
        
        # Actually, let's try a direct search with bounded coefficients
        # Coefficients in {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
        
        coeffs = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        
        # This is a large search space. Let's be smarter.
        # The 2x2x3 tensor has 12 non-zero entries.
        # Rank 10 means we need 10 terms.
        
        # Let me try a different approach: use the fact that this is a
        # "slice" of a larger tensor and look for low-rank approximations
        # that are exact.
        
        # Actually, let me try a specific structured family:
        # Take Strassen for cols 0,1 (7 terms) + 3 additional terms for col 2
        # and see if we can make it work with only 3 additional terms
        # (total 10) instead of 4 (total 11).
        
        # Strassen terms for cols 0,1:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # For col 2, we need:
        # C02 = A00*B02 + A01*B12
        # C12 = A10*B02 + A11*B12
        
        # With 3 additional terms, we can form:
        # N1 = f1 * B02
        # N2 = f2 * B12
        # N3 = f3 * (some combination)
        
        # C02 = A00*B02 + A01*B12
        # C12 = A10*B02 + A11*B12
        
        # If we use:
        # N1 = A00 * B02
        # N2 = A01 * B12
        # N3 = (A10+A11) * ... no, that doesn't work directly
        
        # Let's try:
        # N1 = A00 * B02  -> contributes to C02
        # N2 = A01 * B12  -> contributes to C02
        # N3 = (A10+A11) * B02  -> contributes to C10, C11, C12? No...
        
        # Hmm, this is getting complex. Let me try a numerical approach.
        
        # Use ALS (Alternating Least Squares) to find a rank-10 decomposition
        # with real coefficients, then try to round to half-integers.
        
        rank = 10
        nA = 4  # 2x2
        nB = 6  # 2x3
        nC = 6  # 2x3
        
        # Build target tensor
        T = np.zeros((nA, nB, nC))
        # C00 = A00*B00 + A01*B10
        T[0, 0, 0] = 1.0  # A00 * B00 -> C00
        T[1, 3, 0] = 1.0  # A01 * B10 -> C00
        # C01 = A00*B01 + A01*B11
        T[0, 1, 1] = 1.0
        T[1, 4, 1] = 1.0
        # C02 = A00*B02 + A01*B12
        T[0, 2, 2] = 1.0
        T[1, 5, 2] = 1.0
        # C10 = A10*B00 + A11*B10
        T[2, 0, 3] = 1.0
        T[3, 3, 3] = 1.0
        # C11 = A10*B01 + A11*B11
        T[2, 1, 4] = 1.0
        T[3, 4, 4] = 1.0
        # C12 = A10*B02 + A11*B12
        T[2, 2, 5] = 1.0
        T[3, 5, 5] = 1.0
        
        # ALS with multiple random restarts
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(50):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            # Initialize
            U = np.random.randn(nA, rank)
            V = np.random.randn(nB, rank)
            W = np.random.randn(nC, rank)
            
            for it in range(200):
                if time.time() - start_time > DEADLINE - 30:
                    break
                
                # Update U: for each a, U[a,:] = (T[a,:,:] * (V*W))/||V*W||
                for a in range(nA):
                    # T[a,:,:] is nB x nC
                    # V*W is nB x nC (element-wise product of V and W)
                    VW = V * W  # nB x nC
                    # We need U[a,:] such that sum_r U[a,r] * V[:,r] * W[:,r] approximates T[a,:,:]
                    # This is: U[a,:] * (V*W).T = T[a,:,:].T (vectorized)
                    # Actually: for each (b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                    # So: U[a,:] * (V*W) = T[a,:,:].T? No...
                    # Let me think: sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
                    # So if we define M_bc = V[b,:]*W[c,:] (row vector of length rank)
                    # Then U[a,:] * M^T should give T[a,b,c]... 
                    # Actually: U[a,:] dot (V[:,r]*W[:,r]) for fixed (b,c) is not right.
                    # 
                    # Correct: for fixed a, we have a linear system in U[a,:]:
                    # For each (b,c): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                    # Let M be a matrix where M[(b,c), r] = V[b,r] * W[c,r]
                    # Then M @ U[a,:] = T[a,:,:].T (flattened)
                    
                    M = (V[:, :, None] * W[None, :, :]).reshape(nB * nC, rank)
                    target = T[a, :, :].T.reshape(-1)
                    # Solve M @ x = target
                    try:
                        U[a, :] = np.linalg.lstsq(M, target, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b in range(nB):
                    M = (U[:, :, None] * W[None, :, :]).reshape(nA * nC, rank)
                    target = T[:, b, :].T.reshape(-1)
                    try:
                        V[b, :] = np.linalg.lstsq(M, target, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for c in range(nC):
                    M = (U[:, :, None] * V[None, :, :]).reshape(nA * nB, rank)
                    target = T[:, :, c].T.reshape(-1)
                    try:
                        W[c, :] = np.linalg.lstsq(M, target, rcond=None)[0]
                    except:
                        pass
            
            # Check error
            err = 0.0
            for a in range(nA):
                for b in range(nB):
                    for c in range(nC):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        err += (val - T[a, b, c]) ** 2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
        
        if best_err < 1e-6:
            # Try to round to half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            # Verify
            err = 0.0
            for a in range(nA):
                for b in range(nB):
                    for c in range(nC):
                        val = np.sum(U_rounded[a, :] * V_rounded[b, :] * W_rounded[c, :])
                        err += (val - T[a, b, c]) ** 2
            if np.sqrt(err) < 1e-6:
                return (U_rounded, V_rounded, W_rounded)
            
            # Try integer rounding
            U_rounded = np.round(best_U)
            V_rounded = np.round(best_V)
            W_rounded = np.round(best_W)
            err = 0.0
            for a in range(nA):
                for b in range(nB):
                    for c in range(nC):
                        val = np.sum(U_rounded[a, :] * V_rounded[b, :] * W_rounded[c, :])
                        err += (val - T[a, b, c]) ** 2
            if np.sqrt(err) < 1e-6:
                return (U_rounded, V_rounded, W_rounded)
        
        return None
    
    def add_strassen_block(U, V, W, base_r, k0, k1, p):
        """Add Strassen decomposition for column pair (k0, k1)."""
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][r1] += 1.0
        U[3][r1] += 1.0
        V[0*p + k0][r1] += 1.0
        V[1*p + k1][r1] += 1.0
        
        # M2 = (A10 + A11) * B0k0
        U[2][r2] += 1.0
        U[3][r2] += 1.0
        V[0*p + k0][r2] += 1.0
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][r3] += 1.0
        V[0*p + k1][r3] += 1.0
        V[1*p + k1][r3] += -1.0
        
        # M4 = A11 * (B1k0 - B0k0)
        U[3][r4] += 1.0
        V[1*p + k0][r4] += 1.0
        V[0*p + k0][r4] += -1.0
        
        # M5 = (A00 + A01) * B1k1
        U[0][r5] += 1.0
        U[1][r5] += 1.0
        V[1*p + k1][r5] += 1.0
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[2][r6] += 1.0
        U[0][r6] += -1.0
        V[0*p + k0][r6] += 1.0
        V[0*p + k1][r6] += 1.0
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][r7] += 1.0
        U[3][r7] += -1.0
        V[1*p + k0][r7] += 1.0
        V[1*p + k1][r7] += 1.0
        
        # Reconstruction:
        # C0k0 = M1 + M4 - M5 + M7
        W[2*k0 + 0][r1] += 1.0
        W[2*k0 + 0][r4] += 1.0
        W[2*k0 + 0][r5] += -1.0
        W[2*k0 + 0][r7] += 1.0
        
        # C1k0 = M2 + M4
        W[2*k0 + 1][r2] += 1.0
        W[2*k0 + 1][r4] += 1.0
        
        # C0k1 = M3 + M5
        W[2*k1 + 0][r3] += 1.0
        W[2*k1 + 0][r5] += 1.0
        
        # C1k1 = M1 - M2 + M3 + M6
        W[2*k1 + 1][r1] += 1.0
        W[2*k1 + 1][r2] += -1.0
        W[2*k1 + 1][r3] += 1.0
        W[2*k1 + 1][r6] += 1.0
    
    # Try to find rank-10 for 2x2x3
    result_2x2x3 = None
    if time.time() - start_time < DEADLINE - 40:
        result_2x2x3 = find_2x2x3_rank10()
    
    if result_2x2x3 is not None and time.time() - start_time < DEADLINE - 5:
        # Use 6 Strassen blocks + 2x2x3 rank-10 block
        U3, V3, W3 = result_2x2x3
        rank = 42 + 10  # 6 Strassen blocks + 2x2x3
        U = [[0.0] * rank for _ in range(n * m)]      # 4 x 52
        V = [[0.0] * rank for _ in range(m * p)]      # 30 x 52
        W = [[0.0] * rank for _ in range(p * n)]      # 30 x 52
        
        # Add 6 Strassen blocks for column pairs (0,1), (2,3), ..., (10,11)
        for i in range(6):
            k0 = 2 * i
            k1 = 2 * i + 1
            base_r = 7 * i
            add_strassen_block(U, V, W, base_r, k0, k1, p)
        
        # Add 2x2x3 block for columns 12, 13, 14
        base_r = 42
        k0, k1, k2 = 12, 13, 14
        
        # U3 is 4x10, V3 is 6x10, W3 is 6x10
        # B indices in V3: 0=B0k0, 1=B0k1, 2=B0k2, 3=B1k0, 4=B1k1, 5=B1k2
        # C indices in W3: 0=C0k0, 1=C0k1, 2=C0k2, 3=C1k0, 4=C1k1, 5=C1k2
        
        for a in range(4):
            for r in range(10):
                U[a][base_r + r] += U3[a, r]
        
        for b in range(6):
            for r in range(10):
                # Map b to (row, col) in B
                b_row = b // 3  # 0 or 1
                b_col = b % 3   # 0, 1, 2
                actual_b = b_row * p + (k0 + b_col)
                V[actual_b][base_r + r] += V3[b, r]
        
        for c in range(6):
            for r in range(10):
                # Map c to (col, row) in C: C[k*n + i]
                c_col = c // 2  # 0, 1, 2
                c_row = c % 2   # 0 or 1
                actual_c = (k0 + c_col) * n + c_row
                W[actual_c][base_r + r] += W3[c, r]
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback: original rank-53 solution
    rank = 53
    U = [[0.0] * rank for _ in range(n * m)]      # 4 x 53
    V = [[0.0] * rank for _ in range(m * p)]      # 30 x 53
    W = [[0.0] * rank for _ in range(p * n)]      # 30 x 53
    
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(U, V, W, base_r, k0, k1, p)
    
    base_r = 49
    k_last = 14
    
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    
    U[0][r1] += 1.0
    V[0*p + k_last][r1] += 1.0
    W[2*k_last + 0][r1] += 1.0
    
    U[1][r2] += 1.0
    V[1*p + k_last][r2] += 1.0
    W[2*k_last + 0][r2] += 1.0
    
    U[2][r3] += 1.0
    V[0*p + k_last][r3] += 1.0
    W[2*k_last + 1][r3] += 1.0
    
    U[3][r4] += 1.0
    V[1*p + k_last][r4] += 1.0
    W[2*k_last + 1][r4] += 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

