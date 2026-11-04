import time
import numpy as np

N, M, P = 2, 3, 10

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def trivial():
        """Return the canonical rank nmp decomposition."""
        r = N * M * P
        U = [[0] * r for _ in range(N * M)]
        V = [[0] * r for _ in range(M * P)]
        W = [[0] * r for _ in range(P * N)]
        c = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][c] = 1
                    V[j * P + k][c] = 1
                    W[k * N + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def strassen_decomposition():
        """
        Rank-55 decomposition using Strassen on 2x2 blocks.
        
        A is 2x3, B is 3x10.
        Split A into A_left (2x2) and A_right (2x1).
        Split B into B_top (2x10) and B_bottom (1x10).
        
        A_left * B_top: 5 Strassen decompositions (7 terms each) = 35 terms
        A_right * B_bottom: 2*10 = 20 terms
        Total: 55 terms
        """
        rank = 55
        U = [[0] * rank for _ in range(N * M)]
        V = [[0] * rank for _ in range(M * P)]
        W = [[0] * rank for _ in range(P * N)]
        
        r = 0
        
        # Strassen formula for 2x2 matrix multiplication:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Each term: (u_terms, v_terms, w_terms)
        # u_terms: list of (a_row, a_col, coeff) for A factors
        # v_terms: list of (b_row, b_col, coeff) for B factors
        # w_terms: list of (c_row, c_col, coeff) for C factors
        
        terms = [
            # M1 = (A00+A11)(B00+B11)
            {
                'u': [(0,0,1), (1,1,1)],
                'v': [(0,0,1), (1,1,1)],
                'w': [(0,0,1), (1,1,1)]
            },
            # M2 = (A10+A11)B00
            {
                'u': [(1,0,1), (1,1,1)],
                'v': [(0,0,1)],
                'w': [(1,0,1), (1,1,-1)]
            },
            # M3 = A00(B01-B11)
            {
                'u': [(0,0,1)],
                'v': [(0,1,1), (1,1,-1)],
                'w': [(0,1,1), (1,1,1)]
            },
            # M4 = A11(B10-B00)
            {
                'u': [(1,1,1)],
                'v': [(1,0,1), (0,0,-1)],
                'w': [(0,0,1), (1,0,1)]
            },
            # M5 = (A00+A01)B11
            {
                'u': [(0,0,1), (0,1,1)],
                'v': [(1,1,1)],
                'w': [(0,0,-1), (0,1,1)]
            },
            # M6 = (A10-A00)(B00+B01)
            {
                'u': [(1,0,1), (0,0,-1)],
                'v': [(0,0,1), (0,1,1)],
                'w': [(1,0,1)]
            },
            # M7 = (A01-A11)(B10+B11)
            {
                'u': [(0,1,1), (1,1,-1)],
                'v': [(1,0,1), (1,1,1)],
                'w': [(0,0,1)]
            },
        ]
        
        # Apply Strassen to each 2x2 block of B_top (rows 0,1; cols 2b, 2b+1)
        for b in range(5):
            col0 = 2 * b
            col1 = 2 * b + 1
            
            for t in terms:
                # U: A_left factors (rows 0,1; cols 0,1)
                for (a_r, a_c, coeff) in t['u']:
                    U[a_r * M + a_c][r] += coeff
                
                # V: B_top factors (rows 0,1; cols col0, col1)
                for (b_r, b_c, coeff) in t['v']:
                    vcol = col0 + b_c
                    V[b_r * P + vcol][r] += coeff
                
                # W: C factors (rows 0,1; cols col0, col1)
                for (c_r, c_c, coeff) in t['w']:
                    wcol = col0 + c_c
                    W[wcol * N + c_r][r] += coeff
                
                r += 1
        
        # A_right (2x1) * B_bottom (1x10): 20 terms
        for i in range(N):
            for k in range(P):
                U[i * M + 2][r] = 1
                V[2 * P + k][r] = 1
                W[k * N + i][r] = 1
                r += 1
        
        assert r == rank, f"Expected rank {rank}, got {r}"
        return {"U": U, "V": V, "W": W}

    def check(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        
        # For each (i,j,k), sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    s = np.dot(Ua[i * M + j], Va[j * P + k] * Wa[k * N + i])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    # Try the rank-55 decomposition
    try:
        res = strassen_decomposition()
        if check(res['U'], res['V'], res['W']):
            return res
    except Exception:
        pass

    # Fallback
    return trivial()
