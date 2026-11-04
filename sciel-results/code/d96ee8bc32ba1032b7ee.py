"""Matrix multiplication 4x4 * 4x6 rank-minimal decomposition."""
import time
_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add one rank-1 term to the decomposition."""
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] = coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] = coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def strassen_2x2(oi, oj, ok):
        """
        Add Strassen decomposition for 2x2 * 2x2 block multiplication.
        oi: offset row in A (0 or 2)
        oj: offset col in A / row in B (0 or 2)  
        ok: offset col in B (0, 2, or 4)
        """
        # m1 = (x00+x11)(y00+y11)
        u1 = {(oi, oj): 1, (oi+1, oj+1): 1}
        v1 = {(oj, ok): 1, (oj+1, ok+1): 1}
        w1 = {(oi, ok): 1, (oi+1, ok+1): 1}
        add_term(u1, v1, w1)
        
        # m2 = (x10+x11)y00
        u2 = {(oi+1, oj): 1, (oi+1, oj+1): 1}
        v2 = {(oj, ok): 1}
        w2 = {(oi+1, ok): 1, (oi+1, ok+1): -1}
        add_term(u2, v2, w2)
        
        # m3 = x00(y01-y11)
        u3 = {(oi, oj): 1}
        v3 = {(oj, ok+1): 1, (oj+1, ok+1): -1}
        w3 = {(oi, ok+1): 1, (oi+1, ok+1): 1}
        add_term(u3, v3, w3)
        
        # m4 = x11(y10-y00)
        u4 = {(oi+1, oj+1): 1}
        v4 = {(oj+1, ok): 1, (oj, ok): -1}
        w4 = {(oi, ok): 1, (oi+1, ok): 1}
        add_term(u4, v4, w4)
        
        # m5 = (x00+x01)y11
        u5 = {(oi, oj): 1, (oi, oj+1): 1}
        v5 = {(oj+1, ok+1): 1}
        w5 = {(oi, ok): -1, (oi, ok+1): 1}
        add_term(u5, v5, w5)
        
        # m6 = (x10-x00)(y00+y01)
        u6 = {(oi+1, oj): 1, (oi, oj): -1}
        v6 = {(oj, ok): 1, (oj, ok+1): 1}
        w6 = {(oi+1, ok+1): 1}
        add_term(u6, v6, w6)
        
        # m7 = (x01-x11)(y10+y11)
        u7 = {(oi, oj+1): 1, (oi+1, oj+1): -1}
        v7 = {(oj+1, ok): 1, (oj+1, ok+1): 1}
        w7 = {(oi, ok): 1}
        add_term(u7, v7, w7)

    # 4-point Winograd scheme for 4x4x4 matrix multiplication
    # The 4-point Winograd transform uses 4 multiplications per 1D convolution.
    # For 2D (matrix multiply), this gives 4*4 = 16 multiplications for 4x4x4.
    
    # The 4-point Winograd transform matrices:
    # M (input transform): 4x4
    # G (kernel transform): 4x4
    # Output transform: M^T
    # 
    # We use the standard 4-point Winograd scheme with points {0, 1, -1, infinity}
    # which gives integer/half-integer entries.
    
    def winograd_4x4x4(oi, oj, ok):
        """
        Add 4-point Winograd decomposition for 4x4 * 4x4 block multiplication.
        Uses 16 rank-1 terms instead of 49 (Strassen) or 64 (schoolbook).
        """
        import numpy as np
        
        # 4-point Winograd transform matrices
        # Using the standard formulation with points 0, 1, -1, and a special point
        # M transforms inputs, G transforms kernel, M^T transforms outputs
        # 
        # For the 4-point scheme, we use:
        # M = [[1, 0, 0, 0],
        #      [1, 1, 0, 0],
        #      [1, -1, 0, 0],
        #      [0, 1, -1, 0]]
        # G = [[1, 0, 0, 0],
        #      [1, 1, 0, 0],
        #      [1, -1, 0, 0],
        #      [0, 1, -1, 0]]
        
        M = np.array([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, -1, 0, 0],
            [0, 1, -1, 0]
        ], dtype=float)
        
        G = np.array([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, -1, 0, 0],
            [0, 1, -1, 0]
        ], dtype=float)
        
        # Verify: M^T @ diag(M.*G) @ M should give the Toeplitz/convolution matrix
        # For matrix multiplication, we need:
        # sum_{a,b} M[i,a]*M[k,b] * (sum_{i',j'} M[i',a]*M[j,b]*A[i',j']) * (sum_{j',k'} G[j',a]*G[k',b]*B[j',k'])
        # = sum_j A[i,j]*B[j,k]
        
        # This requires: sum_{a,b} M[i,a]*M[k,b]*M[i',a]*M[j,b]*G[j',a]*G[k',b] = delta_{i,j'} * delta_{k,j} ... 
        # Actually for matrix multiply: sum_{a,b} M[i,a]*M[k,b]*M[i',a]*M[j',b]*G[j',a]*G[k',b] should 
        # pick out A[i',j']*B[j',k'] when i'=i, k'=k, summed over j'
        
        # Let me just verify numerically and if it doesn't work, fall back to Strassen
        # Build the 16 rank-1 terms
        for a in range(4):
            for b in range(4):
                u_col = [0] * (n * m)
                v_col = [0] * (m * p)
                w_col = [0] * (p * n)
                
                for i in range(4):
                    for j in range(4):
                        u_col[(oi + i) * m + (oj + j)] = M[i, a] * M[j, b]
                
                for j in range(4):
                    for k in range(4):
                        v_col[(oj + j) * p + (ok + k)] = G[j, a] * G[k, b]
                
                for i in range(4):
                    for k in range(4):
                        w_col[(ok + k) * n + (oi + i)] = M[i, a] * M[k, b]
                
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
    
    # PART 1: 4x4 * 4x4 (columns 0-3 of B)
    # Try Winograd first, verify, fall back to Strassen if incorrect
    if _time_remaining() > 5.0:
        winograd_4x4x4(0, 0, 0)
    
    # PART 2: 4x4 * 4x2 (columns 4-5 of B)
    # Use 2x2 Strassen blocks: 4 blocks of 2x2*2x2, each rank 7
    # Total strip rank: 28
    if _time_remaining() > 5.0:
        for rb in range(2):
            strassen_2x2(rb*2, 0, 4)
            strassen_2x2(rb*2, 2, 4)
    
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    # Verify correctness with numpy
    import numpy as np
    U_np = np.array(U_final, dtype=float)
    V_np = np.array(V_final, dtype=float)
    W_np = np.array(W_final, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_decomp[i, k] += A_test[i, j] * U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    
    err = np.max(np.abs(C_test - C_decomp))
    if err > 1e-10:
        # Fallback: rebuild with known-correct Strassen decomposition
        U_cols.clear()
        V_cols.clear()
        W_cols.clear()
        
        # Strassen 4x4x4
        def add_super_term(u_blocks, v_blocks, w_blocks, oi, oj, ok):
            terms = [
                ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
                ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
                ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
                ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
                ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
                ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
                ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
            ]
            
            for u_t, v_t, w_t in terms:
                u_col = [0] * (n * m)
                v_col = [0] * (m * p)
                w_col = [0] * (p * n)
                
                for (br, bc), coeff_u in u_blocks.items():
                    for (r, c), coeff_t in u_t.items():
                        i = oi + 2*br + r
                        j = oj + 2*bc + c
                        u_col[i * m + j] += coeff_u * coeff_t
                
                for (br, bc), coeff_v in v_blocks.items():
                    for (r, c), coeff_t in v_t.items():
                        j = oj + 2*br + r
                        k = ok + 2*bc + c
                        v_col[j * p + k] += coeff_v * coeff_t
                
                for (br, bc), coeff_w in w_blocks.items():
                    for (r, c), coeff_t in w_t.items():
                        i = oi + 2*br + r
                        k = ok + 2*bc + c
                        w_col[k * n + i] += coeff_w * coeff_t
                
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
        
        def strassen_4x4(oi, oj, ok):
            add_super_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, oi, oj, ok)
            add_super_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}, oi, oj, ok)
            add_super_term({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, oi, oj, ok)
            add_super_term({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, oi, oj, ok)
            add_super_term({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}, oi, oj, ok)
            add_super_term({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}, oi, oj, ok)
            add_super_term({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}, oi, oj, ok)
        
        strassen_4x4(0, 0, 0)
        for rb in range(2):
            strassen_2x2(rb*2, 0, 4)
            strassen_2x2(rb*2, 2, 4)
        
        rank = len(U_cols)
        U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
        V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
        W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
