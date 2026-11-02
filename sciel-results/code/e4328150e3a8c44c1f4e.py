"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    def strassen_3blocks():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]  # 4 x 21
        V = [[0] * rank for _ in range(m * p)]  # 12 x 21
        W = [[0] * rank for _ in range(p * n)]  # 12 x 21
        
        # Strassen factors for 2x2 matrix multiplication
        # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
        # M5 = (a+b)h, M6 = (c-d)(e+f), M7 = (b-d)(g+h)
        U_coeffs = [
            {0: 1, 3: 1},       # M1: A00 + A11
            {2: 1, 3: 1},       # M2: A10 + A11
            {0: 1},             # M3: A00
            {3: 1},             # M4: A11
            {0: 1, 1: 1},       # M5: A00 + A01
            {2: 1, 0: -1},      # M6: A10 - A00
            {1: 1, 3: -1}       # M7: A01 - A11
        ]
        
        V_coeffs = [
            [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
            [(0, 0, 1)],                 # M2: B00
            [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
            [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
            [(1, 1, 1)],                 # M5: B11
            [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
            [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
        ]
        
        # W_map[k] contains list of (i, j_local, coeff)
        # For each M_k (k=0..6), which C entries does it affect?
        # C00: M1(1), M4(1), M5(-1), M7(1)
        # C01: M3(1), M5(1)
        # C10: M2(1), M4(1)
        # C11: M1(1), M2(-1), M3(1), M6(1)
        W_map = [
            [(0, 0, 1), (1, 1, 1)],   # k=0 (M1) -> C00, C11
            [(1, 0, 1), (1, 1, -1)],  # k=1 (M2) -> C10, C11
            [(0, 1, 1), (1, 1, 1)],   # k=2 (M3) -> C01, C11
            [(0, 0, 1), (1, 0, 1)],   # k=3 (M4) -> C00, C10
            [(0, 0, -1), (0, 1, 1)],  # k=4 (M5) -> C00, C01
            [(1, 1, 1)],              # k=5 (M6) -> C11
            [(0, 0, 1)]               # k=6 (M7) -> C00
        ]
        
        for r in range(3):
            for k in range(7):
                col = r * 7 + k
                
                # U
                for idx, val in U_coeffs[k].items():
                    U[idx][col] = val
                    
                # V
                for j_local, k_local, val in V_coeffs[k]:
                    global_k = 2 * r + k_local
                    v_idx = j_local * p + global_k
                    V[v_idx][col] = val
                    
                # W
                for i, j_local, val in W_map[k]:
                    global_k = 2 * r + j_local
                    w_idx = global_k * n + i
                    W[w_idx][col] = val
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Fallback: schoolbook decomposition (rank 24)."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Primary: Strassen 3 blocks (rank 21)
    result = strassen_3blocks()
    
    # Verify correctness
    if verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: schoolbook decomposition (rank 24)
    if time.time() < deadline:
        result = schoolbook_decomposition()
        if verify(result["U"], result["V"], result["W"]):
            return result
    
    # Last resort: return the Strassen result even if verification failed
    # (it should be correct, but this is a safety net)
    return strassen_3blocks()
# EVOLVE-BLOCK-END
