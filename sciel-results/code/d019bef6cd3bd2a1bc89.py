import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # M1..M7
    
    # U coeffs for A in each M
    # M1: A0+A3 -> [1,0,0,1]
    # M2: A2+A3 -> [0,0,1,1]
    # M3: A0    -> [1,0,0,0]
    # M4: A3    -> [0,0,0,1]
    # M5: A0+A1 -> [1,1,0,0]
    # M6: A2-A0 -> [-1,0,1,0]
    # M7: A1-A3 -> [0,1,0,-1]
    U_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # V coeffs for B in each M
    # M1: B0+B3 -> [1,0,0,1]
    # M2: B0    -> [1,0,0,0]
    # M3: B2-B3 -> [0,0,1,-1]
    # M4: B0-B1 -> [1,-1,0,0]
    # M5: B3    -> [0,0,0,1]
    # M6: B0+B1 -> [1,1,0,0]
    # M7: B2+B3 -> [0,0,1,1]
    V_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [1, -1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # W coeffs for C in reconstruction
    # C0 = M1 + M4 - M5 + M7
    # C1 = M3 + M5
    # C2 = M2 + M4
    # C3 = M1 - M2 + M3 + M6
    # W_coeffs[s][c] is coeff of M_s in C_c
    W_coeffs = [
        [1, 0, 0, 1], # M1 contributes to C0(1), C3(1)
        [0, 0, 0, -1],# M2 contributes to C2(1), C3(-1) -> Wait.
        # Let's list by M index
        # M1: C0:+1, C3:+1
        # M2: C2:+1, C3:-1
        # M3: C1:+1, C3:+1
        # M4: C0:+1, C2:+1
        # M5: C0:-1, C1:+1
        # M6: C3:+1
        # M7: C0:+1
    ]
    W_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec, u_map, v_map, w_map):
        nonlocal col
        for i in range(len(u_vec)):
            if u_vec[i] != 0:
                U[u_map[i]][col] += u_vec[i]
        for i in range(len(v_vec)):
            if v_vec[i] != 0:
                V[v_map[i]][col] += v_vec[i]
        for i in range(len(w_vec)):
            if w_vec[i] != 0:
                W[w_map[i]][col] += w_vec[i]
        col += 1

    # 1. Construct 2x2x4 decomposition (14 terms)
    # This is for A (2x2) * B (2x4) -> C (2x4)
    # We need to map local indices to global indices for specific blocks
    
    # Block 1: A1 (cols 0,1 of A) * B1 (rows 0,1 of B) -> C (cols 0,7 of C)
    # A1 local indices 0,1,2,3 map to global A indices 0,1,4,5
    u_map_A1 = [0, 1, 4, 5]
    
    # B1 local indices 0..7 map to global B indices 0..7
    # B is 4x8. B1 is rows 0,1.
    # Local B index k (0..7) is row r=k//4, col c=k%4.
    # Global row R = r. Global col C = c.
    # Global index = R*8 + C.
    # Since r is 0 or 1, and c is 0..3.
    # Local 0 (0,0) -> Global 0
    # Local 1 (0,1) -> Global 1
    # Local 2 (0,2) -> Global 2
    # Local 3 (0,3) -> Global 3
    # Local 4 (1,0) -> Global 8
    # Local 5 (1,1) -> Global 9
    # Local 6 (1,2) -> Global 10
    # Local 7 (1,3) -> Global 11
    v_map_B1 = [0, 1, 2, 3, 8, 9, 10, 11]
    
    # C local indices 0..7 map to global C indices 0..7
    # C is 2x8.
    w_map_C = [0, 1, 2, 3, 8, 9, 10, 11]
    
    # We need to generate 14 terms for 2x2x4.
    # This involves splitting B into Left (cols 0,1) and Right (cols 2,3).
    # And C into Left (cols 0,1) and Right (cols 2,3).
    
    # For Left part of B (cols 0,1):
    # Local B indices for 2x2 block: 0,1,4,5.
    # These map to global B indices: 0,1,8,9.
    v_map_B1_L = [0, 1, 8, 9]
    # Local C indices for 2x2 block: 0,1,4,5.
    # These map to global C indices: 0,1,8,9.
    w_map_C_L = [0, 1, 8, 9]
    
    # For Right part of B (cols 2,3):
    # Local B indices for 2x2 block: 2,3,6,7.
    # These map to global B indices: 2,3,10,11.
    v_map_B1_R = [2, 3, 10, 11]
    # Local C indices for 2x2 block: 2,3,6,7.
    # These map to global C indices: 2,3,10,11.
    w_map_C_R = [2, 3, 10, 11]
    
    for s in range(7):
        u_vec = U_coeffs[s]
        v_vec_L = V_coeffs[s]
        v_vec_R = V_coeffs[s]
        w_vec_L = W_coeffs[s]
        w_vec_R = W_coeffs[s]
        
        # Left term
        add_term(u_vec, v_vec_L, w_vec_L, u_map_A1, v_map_B1_L, w_map_C_L)
        # Right term
        add_term(u_vec, v_vec_R, w_vec_R, u_map_A1, v_map_B1_R, w_map_C_R)
        
    # Block 2: A2 (cols 2,3 of A) * B2 (rows 2,3 of B) -> C (cols 0,7 of C)
    # A2 local indices 0,1,2,3 map to global A indices 2,3,6,7
    u_map_A2 = [2, 3, 6, 7]
    
    # B2 local indices 0..7 map to global B indices 16..23
    # B2 is rows 2,3.
    # Local 0 (0,0) -> Global Row 2, Col 0 -> 16
    # Local 1 (0,1) -> Global Row 2, Col 1 -> 17
    # Local 2 (0,2) -> Global Row 2, Col 2 -> 18
    # Local 3 (0,3) -> Global Row 2, Col 3 -> 19
    # Local 4 (1,0) -> Global Row 3, Col 0 -> 24
    # Local 5 (1,1) -> Global Row 3, Col 1 -> 25
    # Local 6 (1,2) -> Global Row 3, Col 2 -> 26
    # Local 7 (1,3) -> Global Row 3, Col 3 -> 27
    v_map_B2_L = [16, 17, 24, 25]
    v_map_B2_R = [18, 19, 26, 27]
    
    # C mapping is same as before
    w_map_C_L = [0, 1, 8, 9]
    w_map_C_R = [2, 3, 10, 11]
    
    for s in range(7):
        u_vec = U_coeffs[s]
        v_vec_L = V_coeffs[s]
        v_vec_R = V_coeffs[s]
        w_vec_L = W_coeffs[s]
        w_vec_R = W_coeffs[s]
        
        # Left term
        add_term(u_vec, v_vec_L, w_vec_L, u_map_A2, v_map_B2_L, w_map_C_L)
        # Right term
        add_term(u_vec, v_vec_R, w_vec_R, u_map_A2, v_map_B2_R, w_map_C_R)
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
