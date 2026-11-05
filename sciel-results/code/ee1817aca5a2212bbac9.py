import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 115.0

    # Build the target tensor T[a,b,g] where:
    # a = i*m+j (A index), b = j*p+k (B index), g = k*n+i (C/W index)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                g = k*n + i
                T[a, b, g] = 1.0

    def try_structured_decomposition():
        """Build the rank-34 structured decomposition using Strassen on 2x2 blocks."""
        rank = 34
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        col = 0
        
        def add_term(u_indices, v_indices, w_indices, coeffs_u=None, coeffs_v=None, coeffs_w=None):
            nonlocal col
            if coeffs_u is None:
                coeffs_u = [1] * len(u_indices)
            if coeffs_v is None:
                coeffs_v = [1] * len(v_indices)
            if coeffs_w is None:
                coeffs_w = [1] * len(w_indices)
            for ui, cu in zip(u_indices, coeffs_u):
                U[ui, col] += cu
            for vi, cv in zip(v_indices, coeffs_v):
                V[vi, col] += cv
            for wi, cw in zip(w_indices, coeffs_w):
                W[wi, col] += cw
            col += 1

        # A indices: i*m+j
        # A00=0, A01=1, A02=2
        # A10=3, A11=4, A12=5
        # A20=6, A21=7, A22=8
        
        # B indices: j*p+k
        # B00=0, B01=1, B02=2, B03=3
        # B10=4, B11=5, B12=6, B13=7
        # B20=8, B21=9, B22=10, B23=11
        
        # W indices: k*n+i
        # W00=0, W10=1, W20=2
        # W01=3, W11=4, W21=5
        # W02=6, W12=7, W22=8
        # W03=9, W13=10, W23=11

        # Part 1: A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]
        # Strassen 2x2x2, rank 7
        # m1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        add_term([0, 4], [0, 4], [0, 4])
        # m2 = (A10+A11)B00 -> C10, -C11
        add_term([3, 4], [0], [1, 4], coeffs_w=[1, -1])
        # m3 = A00(B01-B11) -> C01, C11
        add_term([0], [1, 4], [3, 4], coeffs_v=[1, -1])
        # m4 = A11(B10-B00) -> C00, C10
        add_term([4], [4, 0], [0, 1], coeffs_v=[1, -1])
        # m5 = (A00+A01)B11 -> C01, -C00
        add_term([0, 1], [4], [3, 0], coeffs_w=[1, -1])
        # m6 = (A10-A00)(B00+B01) -> C11
        add_term([3, 0], [0, 1], [4], coeffs_u=[1, -1])
        # m7 = (A01-A11)(B10+B11) -> C00
        add_term([1, 4], [4, 5], [0], coeffs_u=[1, -1])

        # Part 2: A[0:2,0:2] x B[0:2,2:4] -> C[0:2,2:4]
        # B columns 2,3: B02=2, B03=3, B12=6, B13=7
        # C columns 2,3: C02=6, C03=9, C12=7, C13=10
        # m1' = (A00+A11)(B02+B12) -> C02, C12
        add_term([0, 4], [2, 6], [6, 7])
        # m2' = (A10+A11)B02 -> C12, -C13
        add_term([3, 4], [2], [7, 10], coeffs_w=[1, -1])
        # m3' = A00(B03-B12) -> C03, C13
        add_term([0], [3, 6], [9, 10], coeffs_v=[1, -1])
        # m4' = A11(B12-B02) -> C02, C12
        add_term([4], [6, 2], [6, 7], coeffs_v=[1, -1])
        # m5' = (A00+A01)B12 -> C03, -C02
        add_term([0, 1], [6], [9, 6], coeffs_w=[1, -1])
        # m6' = (A10-A00)(B02+B03) -> C13
        add_term([3, 0], [2, 3], [10], coeffs_u=[1, -1])
        # m7' = (A01-A11)(B12+B13) -> C02
        add_term([1, 4], [6, 7], [6], coeffs_u=[1, -1])

        # Part 3: A[0:2,2] x B[2,:] -> C[0:2,:]
        # A02=2, A12=5
        # B20=8, B21=9, B22=10, B23=11
        # C00=0, C01=3, C02=6, C03=9
        # C10=1, C11=4, C12=7, C13=10
        for k in range(4):
            b_idx = 8 + k
            c0_idx = k * n + 0  # C0k
            c1_idx = k * n + 1  # C1k
            add_term([2], [b_idx], [c0_idx])
            add_term([5], [b_idx], [c1_idx])

        # Part 4: A[2,0:2] x B[0:2,:] -> C[2,0:4]
        # A20=6, A21=7
        # B00=0, B01=1, B02=2, B03=3
        # B10=4, B11=5, B12=6, B13=7
        # C20=2, C21=5, C22=8, C23=11
        for k in range(4):
            c2_idx = k * n + 2  # C2k
            b0_idx = k       # B0k
            b1_idx = 4 + k   # B1k
            add_term([6], [b0_idx], [c2_idx])
            add_term([7], [b1_idx], [c2_idx])

        # Part 5: A[2,2] x B[2,:] -> C[2,:]
        # A22=8
        # B20=8, B21=9, B22=10, B23=11
        # C20=2, C21=5, C22=8, C23=11
        for k in range(4):
            b_idx = 8 + k
            c2_idx = k * n + 2
            add_term([8], [b_idx], [c2_idx])

        # Verify the decomposition
        T_approx = np.einsum('ac,bc,gc->abg', U, V, W)
        
        # Check all valid (a,b,g) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    g = k * n + i
                    if not np.isclose(T_approx[a, b, g], 1.0, atol=1e-9):
                        return None
        
        # Check that all invalid triples are zero
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    g = k * n + i
                    valid_set.add((a, b, g))
        
        for a in range(n * m):
            for b in range(m * p):
                for g in range(p * n):
                    if (a, b, g) not in valid_set:
                        if not np.isclose(T_approx[a, b, g], 0.0, atol=1e-9):
                            return None
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try the structured rank-34 decomposition
    result = try_structured_decomposition()
    if result is not None:
        return result

    # Fallback: schoolbook rank 36
    if time.time() > deadline:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try random search for lower ranks (20-33)
    for r in range(20, 34):
        if time.time() > deadline:
            break
        
        max_attempts = 2000 if r <= 24 else 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Structured random with {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n*m, r))
            V = np.random.choice([-1, 0, 1], size=(m*p, r))
            
            # Precompute M matrix
            M = np.zeros((n*m * m*p, r), dtype=np.float64)
            for a in range(n*m):
                for b in range(m*p):
                    M[a*(m*p)+b, :] = U[a, :] * V[b, :]
            
            # Solve for W
            W = np.zeros((p*n, r), dtype=np.float64)
            valid = True
            
            for g in range(p*n):
                b_g = T[:, :, g].flatten()
                try:
                    w_g, residuals, rank_val, sv = np.linalg.lstsq(M, b_g, rcond=None)
                except:
                    valid = False
                    break
                    
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        valid = False
                        break
                else:
                    res = np.linalg.norm(M @ w_g - b_g)
                    if res > 1e-8:
                        valid = False
                        break
                        
                W[g, :] = w_g
            
            if not valid:
                continue
                
            # Check if W is close to integers or half-integers
            W_int = np.round(W)
            T_approx = np.einsum('ac,bc,gc->abg', U, V, W_int)
            if np.allclose(T_approx, T, atol=1e-7):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}
            
            W_half = np.round(W * 2) / 2.0
            T_approx = np.einsum('ac,bc,gc->abg', U, V, W_half)
            if np.allclose(T_approx, T, atol=1e-7):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_half.tolist()}

    # Final fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
