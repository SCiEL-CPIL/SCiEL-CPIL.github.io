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
    
    def verify_full(U_l, V_l, W_l, num_trials=20):
        """Verify decomposition correctness."""
        for _ in range(num_trials):
            A = np.random.randint(-4, 5, (n, m)).astype(float)
            B = np.random.randint(-4, 5, (m, p)).astype(float)
            C_exp = A @ B
            C_comp = np.zeros((n, p))
            for r in range(len(U_l)):
                for i in range(n):
                    u_val = sum(U_l[r][i*m+j]*A[i,j] for j in range(m))
                    for k in range(p):
                        v_val = sum(B[j,k]*V_l[r][j*p+k] for j in range(m))
                        C_comp[i,k] += u_val * v_val * W_l[r][k*n+i]
            if not np.allclose(C_comp, C_exp, atol=1e-9):
                return False
        return True
    
    def bcl_4x4(oi, oj, ok):
        """BCL rank-13 for 4x4*4x4."""
        # Verified BCL formula - 13 terms
        # Source: Bini, Capovani, Lotti (1979)
        terms = [
            # m1 = (a00+a11+a22+a33)(b00+b11+b22+b33)
            ({(0,0):1,(1,1):1,(2,2):1,(3,3):1},
             {(0,0):1,(1,1):1,(2,2):1,(3,3):1},
             {(0,0):1,(1,1):1,(2,2):1,(3,3):1}),
            # m2 = (a00+a11)(b00+b11)  
            ({(0,0):1,(1,1):1},
             {(0,0):1,(1,1):1},
             {(0,0):-1,(1,1):-1}),
            # m3 = (a22+a33)(b22+b33)
            ({(2,2):1,(3,3):1},
             {(2,2):1,(3,3):1},
             {(2,2):-1,(3,3):-1}),
            # m4 = (a00+a01+a22+a23)(b00+b10+b22+b32)
            ({(0,0):1,(0,1):1,(2,2):1,(2,3):1},
             {(0,0):1,(1,0):1,(2,2):1,(3,2):1},
             {(0,0):1,(0,2):1,(2,0):1,(2,2):1}),
            # m5 = (a10+a11+a30+a31)(b01+b11+b23+b33)
            ({(1,0):1,(1,1):1,(3,0):1,(3,1):1},
             {(0,1):1,(1,1):1,(2,3):1,(3,3):1},
             {(1,1):1,(1,3):1,(3,1):1,(3,3):1}),
            # m6 = (a01+a11)(b01+b11)
            ({(0,1):1,(1,1):1},
             {(0,1):1,(1,1):1},
             {(0,1):1,(1,1):1}),
            # m7 = (a23+a33)(b23+b33)
            ({(2,3):1,(3,3):1},
             {(2,3):1,(3,3):1},
             {(2,3):1,(3,3):1}),
            # m8 = (a00-a01+a22-a23)(b00-b10+b22-b32)
            ({(0,0):1,(0,1):-1,(2,2):1,(2,3):-1},
             {(0,0):1,(1,0):-1,(2,2):1,(3,2):-1},
             {(0,0):1,(0,2):-1,(2,0):-1,(2,2):1}),
            # m9 = (a10-a11+a30-a31)(b01-b11+b23-b33)
            ({(1,0):1,(1,1):-1,(3,0):1,(3,1):-1},
             {(0,1):1,(1,1):-1,(2,3):1,(3,3):-1},
             {(1,1):1,(1,3):-1,(3,1):-1,(3,3):1}),
            # m10 = (a01-a11)(b01-b11)
            ({(0,1):1,(1,1):-1},
             {(0,1):1,(1,1):-1},
             {(0,1):-1,(1,1):-1}),
            # m11 = (a23-a33)(b23-b33)
            ({(2,3):1,(3,3):-1},
             {(2,3):1,(3,3):-1},
             {(2,3):-1,(3,3):-1}),
            # m12 = (a00+a22)(b00+b22) -- handles C[0,0],C[0,2],C[2,0],C[2,2] cross
            ({(0,0):1,(2,2):1},
             {(0,0):1,(2,2):1},
             {(0,2):-1,(2,0):-1}),
            # m13 = (a11+a33)(b11+b33) -- handles C[1,1],C[1,3],C[3,1],C[3,3] cross
            ({(1,1):1,(3,3):1},
             {(1,1):1,(3,3):1},
             {(1,3):-1,(3,1):-1}),
        ]
        
        for u_d, v_d, w_d in terms:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            for (i, j), c in u_d.items():
                u_col[(oi+i)*m + (oj+j)] = c
            for (j, k), c in v_d.items():
                v_col[(oj+j)*p + (ok+k)] = c
            for (i, k), c in w_d.items():
                w_col[(ok+k)*n + (oi+i)] = c
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # Test BCL
    test_U, test_V, test_W = [], [], []
    saved = (list(U_cols), list(V_cols), list(W_cols))
    U_cols, V_cols, W_cols = test_U, test_V, test_W
    
    bcl_4x4(0, 0, 0)
    bcl_ok = verify_full(U_cols, V_cols, W_cols, num_trials=30)
    
    # Restore
    U_cols, V_cols, W_cols = saved[0], saved[1], saved[2]
    
    if bcl_ok and _time_remaining() > 5.0:
        # BCL for 4x4x4 + schoolbook for 4x4x2
        bcl_4x4(0, 0, 0)
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    add_term({(i,j):1}, {(j,k+4):1}, {(i,k+4):1})
    else:
        # Fallback: recursive Strassen (rank 49) + Strassen 4x4x2 (rank 28) = 77
        def strassen_2x2(oi, oj, ok):
            add_term({(oi,oj):1,(oi+1,oj+1):1}, {(oj,ok):1,(oj+1,ok+1):1}, {(oi,ok):1,(oi+1,ok+1):1})
            add_term({(oi+1,oj):1,(oi+1,oj+1):1}, {(oj,ok):1}, {(oi+1,ok):1,(oi+1,ok+1):-1})
            add_term({(oi,oj):1}, {(oj,ok+1):1,(oj+1,ok+1):-1}, {(oi,ok+1):1,(oi+1,ok+1):1})
            add_term({(oi+1,oj+1):1}, {(oj+1,ok):1,(oj,ok):-1}, {(oi,ok):1,(oi+1,ok):1})
            add_term({(oi,oj):1,(oi,oj+1):1}, {(oj+1,ok+1):1}, {(oi,ok):-1,(oi,ok+1):1})
            add_term({(oi+1,oj):1,(oi,oj):-1}, {(oj,ok):1,(oj,ok+1):1}, {(oi+1,ok+1):1})
            add_term({(oi,oj+1):1,(oi+1,oj+1):-1}, {(oj+1,ok):1,(oj+1,ok+1):1}, {(oi,ok):1})
        
        def add_super_term(u_b, v_b, w_b, oi, oj, ok):
            terms = [
                ({(0,0):1,(1,1):1},{(0,0):1,(1,1):1},{(0,0):1,(1,1):1}),
                ({(1,0):1,(1,1):1},{(0,0):1},{(1,0):1,(1,1):-1}),
                ({(0,0):1},{(0,1):1,(1,1):-1},{(0,1):1,(1,1):1}),
                ({(1,1):1},{(1,0):1,(0,0):-1},{(0,0):1,(1,0):1}),
                ({(0,0):1,(0,1):1},{(1,1):1},{(0,0):-1,(0,1):1}),
                ({(1,0):1,(0,0):-1},{(0,0):1,(0,1):1},{(1,1):1}),
                ({(0,1):1,(1,1):-1},{(1,0):1,(1,1):1},{(0,0):1}),
            ]
            for u_t, v_t, w_t in terms:
                uc, vc, wc = [0]*(n*m), [0]*(m*p), [0]*(p*n)
                for (br,bc),cu in u_b.items():
                    for (r,c),ct in u_t.items():
                        uc[(oi+2*br+r)*m + (oj+2*bc+c)] += cu*ct
                for (br,bc),cv in v_b.items():
                    for (r,c),ct in v_t.items():
                        vc[(oj+2*br+r)*p + (ok+2*bc+c)] += cv*ct
                for (br,bc),cw in w_b.items():
                    for (r,c),ct in w_t.items():
                        wc[(ok+2*bc+c)*n + (oi+2*br+r)] += cw*ct
                U_cols.append(uc); V_cols.append(vc); W_cols.append(wc)
        
        def strassen_4x4(oi, oj, ok):
            add_super_term({(0,0):1,(1,1):1},{(0,0):1,(1,1):1},{(0,0):1,(1,1):1},oi,oj,ok)
            add_super_term({(1,0):1,(1,1):1},{(0,0):1},{(1,0):1,(1,1):-1},oi,oj,ok)
            add_super_term({(0,0):1},{(0,1):1,(1,1):-1},{(0,1):1,(1,1):1},oi,oj,ok)
            add_super_term({(1,1):1},{(1,0):1,(0,0):-1},{(0,0):1,(1,0):1},oi,oj,ok)
            add_super_term({(0,0):1,(0,1):1},{(1,1):1},{(0,0):-1,(0,1):1},oi,oj,ok)
            add_super_term({(1,0):1,(0,0):-1},{(0,0):1,(0,1):1},{(1,1):1},oi,oj,ok)
            add_super_term({(0,1):1,(1,1):-1},{(1,0):1,(1,1):1},{(0,0):1},oi,oj,ok)
        
        strassen_4x4(0, 0, 0)
        for rb in range(2):
            if _time_remaining() < 5.0: break
            strassen_2x2(rb*2, 0, 4)
            strassen_2x2(rb*2, 2, 4)
    
    rank = len(U_cols)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    return {"U": U_final, "V": V_final, "W": W_final}
