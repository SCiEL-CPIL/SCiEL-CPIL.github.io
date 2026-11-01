import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 2, 2, 12
    
    # Strategy: Use Strassen (rank 7) for 2x2x2 block on cols 0,1
    # + schoolbook (rank 4 per column) for cols 2-11
    # Total rank = 7 + 10*4 = 47... no that's worse.
    # Better: Strassen on cols 0,1 (rank 7) + schoolbook on cols 2-11 (rank 20) = 27
    
    # First, define and verify Strassen terms for 2x2x2
    # We need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for i,j,k in {0,1}
    
    # Let me define Strassen terms and verify them
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}},
        # M2 = (A10+A11)B00  
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(0,0): 1, (0,1): -1}},
        # M3 = A00(B10-B00)
        {'U': {(0,0): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (0,1): 1}},
        # M4 = A11(B01-B10)
        {'U': {(1,1): 1}, 'V': {(0,1): 1, (1,0): -1}, 'W': {(0,0): 1, (0,1): 1}},
        # M5 = (A00+A01)B11
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}},
        # M6 = (A01-A10)(B10+B11)
        {'U': {(0,1): 1, (1,0): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,1): 1}},
        # M7 = (A10-A00)(B00+B01)
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(0,0): 1}},
    ]
    
    # Verify Strassen terms for 2x2x2
    def verify_2x2x2(terms):
        """Verify that terms decompose 2x2x2 matrix multiplication tensor."""
        n_local, m_local, p_local = 2, 2, 2
        for i in range(n_local):
            for j in range(m_local):
                for k in range(p_local):
                    total = 0.0
                    for term in terms:
                        u_val = term['U'].get((i, j), 0)
                        v_val = term['V'].get((j, k), 0)
                        w_val = term['W'].get((k, i), 0)
                        total += u_val * v_val * w_val
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    strassen_valid = verify_2x2x2(strassen_terms)
    
    if strassen_valid:
        # Use rank 27: Strassen on cols 0,1 + schoolbook on cols 2-11
        rank = 7 + 20  # 27
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Fill Strassen terms for cols 0,1
        for t_idx in range(7):
            r = t_idx
            term = strassen_terms[t_idx]
            for (i, j), val in term['U'].items():
                U[i * m + j][r] = val
            for (j, k), val in term['V'].items():
                V[j * p + k][r] = val
            for (k, i), val in term['W'].items():
                W[k * n + i][r] = val
        
        # Fill schoolbook terms for cols 2-11
        r = 7
        for k in range(2, p):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        
        # Final verification
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > max_err:
                        max_err = abs(val - 1.0)
        
        if max_err > 1e-9:
            # Should not happen, but fallback to schoolbook
            rank = 48
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            r = 0
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][r] = 1
                        V[j * p + k][r] = 1
                        W[k * n + i][r] = 1
                        r += 1
    else:
        # Strassen terms invalid, use schoolbook rank 48
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
    
    return {"U": U, "V": V, "W": W}
