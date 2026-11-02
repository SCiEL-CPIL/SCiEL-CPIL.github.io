import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try different splits of the 9 columns
    # Current: 4 blocks of 2 + 1 block of 1 = 4*7 + 4 + 18 = 50
    # Try: 1 block of 1 + 4 blocks of 2 = same
    # Try: 3 blocks of 3 = ?
    
    # For 2x2 * 2x3, what's the minimal rank?
    # We can use Strassen on 2x2x2 (7 mults) + 1 schoolbook (4 mults) = 11
    # Or: 3 schoolbook = 12
    # Or: some other decomposition?
    
    # Actually, for 2x2 * 2x3, the rank is at most 7+4=11 or 12
    # But maybe we can do better with a custom decomposition
    
    # Let me try: 3 blocks of 3 columns
    # Each 2x2 * 2x3 block: use 7 (Strassen on first 2 cols) + 4 (schoolbook on last col) = 11
    # Total: 3 * 11 = 33 for the 2x2*2x9 part
    # Plus 2*3 = 6 for the 2x1 * 1x9 part
    # Total: 39
    
    # That's better than 50! Let me implement this.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, :] split into 3 blocks of 3 columns
    for b in range(3):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
            
        k0 = 3 * b
        k1 = 3 * b + 1
        k2 = 3 * b + 2
        
        # B indices
        v_e0 = 0 * p + k0
        v_f0 = 0 * p + k1
        v_g0 = 1 * p + k0
        v_h0 = 1 * p + k1
        
        v_e1 = 0 * p + k2
        v_g1 = 1 * p + k2
        
        # A indices
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        # Strassen on columns k0, k1 (7 mults)
        # M1 = (a+d)(e0+h0)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e0][col] += 1
        V[v_h0][col] += 1
        
        # M2 = (c+d)e0
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e0][col+1] += 1
        
        # M3 = a(f0-h0)
        U[u_a][col+2] += 1
        V[v_f0][col+2] += 1
        V[v_h0][col+2] -= 1
        
        # M4 = d(g0-e0)
        U[u_d][col+3] += 1
        V[v_g0][col+3] += 1
        V[v_e0][col+3] -= 1
        
        # M5 = (a+b)h0
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h0][col+4] += 1
        
        # M6 = (c-a)(e0+f0)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e0][col+5] += 1
        V[v_f0][col+5] += 1
        
        # M7 = (b-d)(g0+h0)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g0][col+6] += 1
        V[v_h0][col+6] += 1
        
        # Schoolbook on column k2 (4 mults)
        # P1 = a*e1 -> C[0,k2]
        U[u_a][col+7] += 1
        V[v_e1][col+7] += 1
        W[k2 * n + 0][col+7] += 1
        
        # P2 = b*g1 -> C[0,k2]
        U[u_b][col+8] += 1
        V[v_g1][col+8] += 1
        W[k2 * n + 0][col+8] += 1
        
        # P3 = c*e1 -> C[1,k2]
        U[u_c][col+9] += 1
        V[v_e1][col+9] += 1
        W[k2 * n + 1][col+9] += 1
        
        # P4 = d*g1 -> C[1,k2]
        U[u_d][col+10] += 1
        V[v_g1][col+10] += 1
        W[k2 * n + 1][col+10] += 1
        
        # Output for k0, k1
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C[0,k0] = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C[0,k1] = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C[1,k0] = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C[1,k1] = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 11
    
    # Part 2: A[0:2, 2:3] * B[2:3, :]
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    
    for k in range(9):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
            
        v_b = 2 * p + k
        
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    assert col == rank, f"Expected col={rank}, got col={col}"
    
    return {"U": U, "V": V, "W": W}
