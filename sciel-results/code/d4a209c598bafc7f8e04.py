"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _construct_rank36():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = 36
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Helper to add Strassen 2x2x2
    # u_idx: [Y00, Y01, Y10, Y11]
    # v_idx: [Z00, Z01, Z10, Z11]
    # w_idx: [X00, X01, X10, X11]
    def add_strassen(u_idx, v_idx, w_idx, start_col):
        # Strassen products definitions
        # M1 = (Y00+Y11)(Z00+Z11)
        # M2 = (Y10+Y11)Z00
        # M3 = Y00(Z01-Z11)
        # M4 = Y11(Z10-Z00)
        # M5 = (Y00+Y01)Z11
        # M6 = (Y10-Y00)(Z00+Z01)
        # M7 = (Y01-Y11)(Z10+Z11)
        
        # Coefficients for Y and Z for each M
        # Order: [Y00, Y01, Y10, Y11]
        y_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [1, 0, 1, 0], # M6 (Y10-Y00 -> -1*Y00 + 1*Y10) -> [ -1, 0, 1, 0 ]? Wait.
            # M6 = (Y10 - Y00) (Z00 + Z01).
            # Y part: -Y00 + Y10. Coeffs: [-1, 0, 1, 0].
            # Let's re-verify standard Strassen.
            # M6 = (A10 - A00) (B00 + B01)
            # A10 is Y10, A00 is Y00. So Y10 - Y00.
            # Yes, [-1, 0, 1, 0].
            # M7 = (A01 - A11) (B10 + B11)
            # A01 is Y01, A11 is Y11. So Y01 - Y11.
            # Coeffs: [0, 1, 0, -1].
        ]
        # Correcting y_coeffs
        y_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0], # M6
            [0, 1, 0, -1]  # M7
        ]
        
        # Coefficients for Z for each M
        # Order: [Z00, Z01, Z10, Z11]
        z_coeffs = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1], # M3 (Z01-Z11)
            [0, 0, 1, -1], # M4 (Z10-Z00)
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        # Coefficients for X for each M
        # X00 = M1 + M4 - M5 + M7
        # X01 = M3 + M5
        # X10 = M2 + M4
        # X11 = M1 - M2 + M3 + M6
        # Order: [X00, X01, X10, X11]
        x_coeffs = [
            [1, 0, 0, 0], # M1 contributes to X00, X11
            [0, 0, 0, 1],
            [1, 0, 0, 0], # M2 contributes to X10, X11
            [0, 0, 1, -1],
            [0, 1, 0, 0], # M3 contributes to X01, X11
            [0, 1, 0, 1],
            [0, 0, 1, 0], # M4 contributes to X00, X10
            [1, 0, 1, 0],
            [0, 0, 0, 1], # M5 contributes to X00, X01
            [1, 0, 0, 1],
            [-1, 0, 0, 1], # M6 contributes to X11
            [0, 0, 0, 1],
            [1, 0, 0, 1]  # M7 contributes to X00, X11
        ]
        # Wait, the x_coeffs structure should be 7x4.
        # Row s is coeffs for M_s.
        # Col w is X_w.
        
        x_coeffs = [
            [1, 0, 0, 1], # M1: X00+=1, X11+=1
            [0, 0, 1, -1], # M2: X10+=1, X11-=-1
            [0, 1, 0, 1], # M3: X01+=1, X11+=1
            [1, 0, 1, 0], # M4: X00+=1, X10+=1
            [-1, 1, 0, 0], # M5: X00-=-1, X01+=1
            [0, 0, 0, 1], # M6: X11+=1
            [1, 0, 0, 0]  # M7: X00+=1
        ]
        # Let's double check X11 = M1 - M2 + M3 + M6.
        # M1: 1. M2: -1. M3: 1. M6: 1.
        # My table:
        # M1: 1 (col 3)
        # M2: -1 (col 3)
        # M3: 1 (col 3)
        # M4: 0
        # M5: 0
        # M6: 1 (col 3)
        # M7: 0
        # Sum: 1 - 1 + 1 + 1 = 2? No.
        # Strassen X11 = M1 - M2 + M3 + M6.
        # So M6 coeff should be 1.
        
        # Let's re-verify M6 definition.
        # M6 = (A10 - A00)(B00 + B01).
        # X11 = M1 - M2 + M3 + M6.
        # Yes.
        
        for s in range(7):
            col = start_col + s
            for u in range(4):
                if y_coeffs[s][u] != 0:
                    U[u_idx[u]][col] = y_coeffs[s][u]
            for v in range(4):
                if z_coeffs[s][v] != 0:
                    V[v_idx[v]][col] = z_coeffs[s][v]
            for w in range(4):
                if x_coeffs[s][w] != 0:
                    W[w_idx[w]][col] = x_coeffs[s][w]

    # Helper to add Schoolbook 2x2x1
    # u_idx: [Y00, Y01, Y10, Y11]
    # v_idx: [Z0, Z1]
    # w_idx: [X0, X1]
    def add_schoolbook_2x2x1(u_idx, v_idx, w_idx, start_col):
        # X0 = Y00 Z0 + Y01 Z1
        # X1 = Y10 Z0 + Y11 Z1
        
        # Term 1: Y00 Z0 -> X0
        U[u_idx[0]][start_col + 0] = 1
        V[v_idx[0]][start_col + 0] = 1
        W[w_idx[0]][start_col + 0] = 1
        
        # Term 2: Y01 Z1 -> X0
        U[u_idx[1]][start_col + 1] = 1
        V[v_idx[1]][start_col + 1] = 1
        W[w_idx[0]][start_col + 1] = 1
        
        # Term 3: Y10 Z0 -> X1
        U[u_idx[2]][start_col + 2] = 1
        V[v_idx[0]][start_col + 2] = 1
        W[w_idx[1]][start_col + 2] = 1
        
        # Term 4: Y11 Z1 -> X1
        U[u_idx[3]][start_col + 3] = 1
        V[v_idx[1]][start_col + 3] = 1
        W[w_idx[1]][start_col + 3] = 1

    # Indices
    # U: A[i, j] -> i*m+j
    # V: B[j, k] -> j*p+k
    # W: C[k, i] -> k*n+i
    
    # Block 1: A_left (cols 0,1) * B_top (rows 0,1)
    u1 = [0, 1, 4, 5] # Y00, Y01, Y10, Y11
    
    # B_top col 0,1
    v11 = [0, 1, 5, 6] # Z00, Z01, Z10, Z11
    w11 = [0, 2, 1, 3] # X00, X01, X10, X11
    
    # B_top col 2,3
    v12 = [2, 3, 7, 8]
    w12 = [4, 6, 5, 7]
    
    # B_top col 4
    v13 = [4, 9]
    w13 = [8, 9]
    
    # Block 2: A_right (cols 2,3) * B_bottom (rows 2,3)
    u2 = [2, 3, 6, 7]
    
    # B_bottom col 0,1
    v21 = [10, 11, 15, 16]
    w21 = [0, 2, 1, 3]
    
    # B_bottom col 2,3
    v22 = [12, 13, 17, 18]
    w22 = [4, 6, 5, 7]
    
    # B_bottom col 4
    v23 = [14, 19]
    w23 = [8, 9]
    
    # Add blocks
    col = 0
    add_strassen(u1, v11, w11, col)
    col += 7
    add_strassen(u1, v12, w12, col)
    col += 7
    add_schoolbook_2x2x1(u1, v13, w13, col)
    col += 4
    
    add_strassen(u2, v21, w21, col)
    col += 7
    add_strassen(u2, v22, w22, col)
    col += 7
    add_schoolbook_2x2x1(u2, v23, w23, col)
    col += 4
    
    return {"U": U, "V": V, "W": W}

def _verify_exact(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check compatible triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-12:
                    return False
    
    # Check incompatible triples
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                if j_a == j_b and k_b == k_c and i_c == i_a:
                    continue
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s) > 1e-12:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    result = _construct_rank36()
    
    # Optional verification
    if time.time() - _START_TIME < _TIME_LIMIT - 10:
        if not _verify_exact(result["U"], result["V"], result["W"]):
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
            
    return result
# EVOLVE-BLOCK-END
