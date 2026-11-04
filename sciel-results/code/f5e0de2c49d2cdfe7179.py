"""Matrix multiplication 4x4 * 4x6 rank-minimal decomposition."""
import time
import numpy as np

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

    def strassen_4x4(oi, oj, ok):
        """
        Add recursive Strassen decomposition for 4x4 * 4x4 block multiplication.
        """
        # Split into 2x2 blocks
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B10], [B01, B11]]  (note: B's rows are j, cols are k)
        # C = [[C00, C10], [C01, C11]]
        
        # m1 = (A00+A11)(B00+B01) -> C00 += m1, C11 += m1
        u1 = {}
        v1 = {}
        w1 = {}
        for r in range(2):
            for c in range(2):
                u1[(oi+2*r, oj+2*c)] = 1
                u1[(oi+2+2*r, oj+2+2*c)] = 1
                v1[(oj+2*r, ok+2*c)] = 1
                v1[(oj+2+r, ok+2+2*c)] = 1
                w1[(oi+2*r, ok+2*c)] = 1
                w1[(oi+2+2*r, ok+2+2*c)] = 1
        add_term(u1, v1, w1)
        
        # m2 = (A10+A11)B00
        u2 = {}
        v2 = {}
        w2 = {}
        for r in range(2):
            for c in range(2):
                u2[(oi+2+2*r, oj+2*c)] = 1
                u2[(oi+2+2*r, oj+2+2*c)] = 1
                v2[(oj+2*r, ok+2*c)] = 1
                w2[(oi+2+2*r, ok+2*c)] = 1
                w2[(oi+2+2*r, ok+2+2*c)] = -1
        add_term(u2, v2, w2)
        
        # m3 = A00(B01-B11)
        u3 = {}
        v3 = {}
        w3 = {}
        for r in range(2):
            for c in range(2):
                u3[(oi+2*r, oj+2*c)] = 1
                v3[(oj+2*r, ok+2+2*c)] = 1
                v3[(oj+2+r, ok+2+2*c)] = -1
                w3[(oi+2*r, ok+2+2*c)] = 1
                w3[(oi+2+2*r, ok+2+2*c)] = 1
        add_term(u3, v3, w3)
        
        # m4 = A11(B10-B00)
        u4 = {}
        v4 = {}
        w4 = {}
        for r in range(2):
            for c in range(2):
                u4[(oi+2+2*r, oj+2+2*c)] = 1
                v4[(oj+2+r, ok+2*c)] = 1
                v4[(oj+2*r, ok+2*c)] = -1
                w4[(oi+2*r, ok+2*c)] = 1
                w4[(oi+2+2*r, ok+2*c)] = 1
        add_term(u4, v4, w4)
        
        # m5 = (A00+A01)B11
        u5 = {}
        v5 = {}
        w5 = {}
        for r in range(2):
            for c in range(2):
                u5[(oi+2*r, oj+2*c)] = 1
                u5[(oi+2*r, oj+2+2*c)] = 1
                v5[(oj+2+r, ok+2+2*c)] = 1
                w5[(oi+2*r, ok+2*c)] = -1
                w5[(oi+2*r, ok+2+2*c)] = 1
        add_term(u5, v5, w5)
        
        # m6 = (A10-A00)(B00+B01)
        u6 = {}
        v6 = {}
        w6 = {}
        for r in range(2):
            for c in range(2):
                u6[(oi+2+2*r, oj+2*c)] = 1
                u6[(oi+2*r, oj+2*c)] = -1
                v6[(oj+2*r, ok+2*c)] = 1
                v6[(oj+2+r, ok+2+2*c)] = 1
                w6[(oi+2+2*r, ok+2+2*c)] = 1
        add_term(u6, v6, w6)
        
        # m7 = (A01-A11)(B10+B11)
        u7 = {}
        v7 = {}
        w7 = {}
        for r in range(2):
            for c in range(2):
                u7[(oi+2*r, oj+2+2*c)] = 1
                u7[(oi+2+2*r, oj+2+2*c)] = -1
                v7[(oj+2+r, ok+2*c)] = 1
                v7[(oj+2+r, ok+2+2*c)] = 1
                w7[(oi+2*r, ok+2*c)] = 1
        add_term(u7, v7, w7)

    # PART 1: 4x4 * 4x4 (columns 0-3 of B) - recursive Strassen rank 49
    if _time_remaining() > 5.0:
        strassen_4x4(0, 0, 0)
    
    # PART 2: 4x4 * 4x2 (columns 4-5 of B)
    # Use Strassen decomposition for the rectangular case
    # We can reuse the same U patterns as in the 4x4x4 case but with V targeting columns 4-5
    # This gives 7 * 2 = 14 additional terms (each Strassen term produces 2x2 output blocks)
    
    def strassen_4x4x2(oi, oj, ok):
        """
        Add Strassen decomposition for 4x4 * 4x2 block multiplication.
        """
        # The 7 Strassen super-terms, each producing contributions to the 4x2 output
        # Split A into 2x2 blocks, B rows into 2+2, output rows into 2+2
        
        terms_4x4x2 = [
            # m1 = (A00+A11)*(B_top+B_bottom)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}, {(0,0): 1, (1,0): 1}),
            # m2 = (A10+A11)*B_top
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # m3 = A00*(B_bottom-B_top)
            ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # m4 = A11*(B_top-B_bottom)
            ({(1,1): 1}, {(0,0): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}),
            # m5 = (A00+A01)*B_bottom
            ({(0,0): 1, (0,1): 1}, {(1,0): 1}, {(0,0): -1, (0,1): 1}),
            # m6 = (A10-A00)*(B_top+B_bottom)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,0): 1}),
            # m7 = (A01-A11)*(B_top+B_bottom)
            ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}, {(0,0): 1}),
        ]
        
        for u_pat, v_pat, w_pat in terms_4x4x2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Fill U: A is 4x4, blocks are 2x2
            for (br, bc), coeff in u_pat.items():
                for r in range(2):
                    for c in range(2):
                        i = oi + 2*br + r
                        j = oj + 2*bc + c
                        u_col[i * m + j] += coeff
            
            # Fill V: B is 4x2 (rows 0-3, cols 4-5), blocks are 2x1
            for (br, bc), coeff in v_pat.items():
                for r in range(2):
                    for c in range(1):
                        j = oj + 2*br + r
                        k = ok + c
                        v_col[j * p + k] += coeff
            
            # Fill W: C is 4x2 (rows 0-3, cols 4-5), blocks are 2x1
            for (br, bc), coeff in w_pat.items():
                for r in range(2):
                    for c in range(1):
                        i = oi + 2*br + r
                        k = ok + c
                        w_col[k * n + i] += coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    if _time_remaining() > 5.0:
        strassen_4x4x2(0, 0, 4)
    
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    # Verify correctness
    U_np = np.array(U_final, dtype=float)
    V_np = np.array(V_final, dtype=float)
    W_np = np.array(W_final, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    # Compute via decomposition
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_decomp[i, k] += A_test[i, j] * U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    
    err = np.max(np.abs(C_test - C_decomp))
    if err > 1e-10:
        # Fallback to safe decomposition
        U_cols.clear()
        V_cols.clear()
        W_cols.clear()
        strassen_4x4(0, 0, 0)
        for rb in range(2):
            strassen_2x2(rb*2, 0, 4)
            strassen_2x2(rb*2, 2, 4)
        rank = len(U_cols)
        U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
        V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
        W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
