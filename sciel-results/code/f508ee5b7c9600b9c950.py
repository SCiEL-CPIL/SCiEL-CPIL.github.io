"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # We construct a decomposition of rank 54
    # Split m=5 into 2 and 3.
    # Part 1: 2x2x6 (Rank 21)
    # Part 2: 2x3x6 (Rank 33)
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # A indices: i_a in {0,1}, j_a in {0,1}
    # B indices: j_b in {0,1}, k_b in {0,1}
    # C indices: i_c in {0,1}, k_c in {0,1}
    def add_strassen_222(base_i, base_j, base_k):
        nonlocal col
        # A is 2x2, B is 2x2, C is 2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00+B01)
        # M3 = (A00+A01)B11
        # M4 = (A10+A11)B10
        # M5 = (A00-A11)(B00+B11)
        # M6 = (A01-A11)(B01+B11)
        # M7 = (A00+A10)(B00-B10)
        
        # C00 = M1 - M2 + M4 + M5
        # C01 = M3 + M6
        # C10 = M2 + M5
        # C11 = M1 - M5 + M3 - M2
        
        # Terms:
        # 1. M1
        U[base_i*2 + 0, col] = 1
        U[base_i*2 + 1, col] = 1
        V[base_j*2 + 0, col] = 1
        V[base_j*2 + 1, col] = 1
        W[base_k*2 + 0, col] = 1
        W[base_k*2 + 1, col] = 1
        # C contributions
        W[base_k*2 + 0, col] = 1 # C00
        W[base_k*2 + 1, col] = 0 # C01
        W[base_k*2 + 2, col] = 0 # C10 -- Wait, W index is k*n+i.
        # W index: k*n + i.
        # C00 is k=base_k, i=base_i.
        # C01 is k=base_k+1, i=base_i.
        
        # Let's restart the helper with explicit indices to be safe.
        pass

    # It is safer to build the matrices directly in the main loop with explicit logic.
    
    # --- Part 1: 2x2x6 ---
    # A[0:2, 0:2], B[0:2, 0:6] -> C[0:2, 0:6]
    # Split B into 3 blocks of 2 columns.
    for block in range(3):
        k_off = block * 2
        # Strassen 2x2x2
        # M1
        U[0, col] = 1; U[1, col] = 1
        V[0, col] = 1; V[1, col] = 1
        W[k_off*2 + 0, col] = 1 # C00
        W[k_off*2 + 1, col] = 1 # C01
        col += 1
        # M2
        U[2, col] = 1; U[3, col] = 1
        V[0, col] = 1; V[2, col] = 1
        W[k_off*2 + 0, col] = -1 # C00
        W[k_off*2 + 2, col] = 1 # C10
        col += 1
        # M3
        U[0, col] = 1; U[2, col] = 1
        V[3, col] = 1
        W[k_off*2 + 1, col] = 1 # C01
        W[k_off*2 + 3, col] = 1 # C11
        col += 1
        # M4
        U[2, col] = 1; U[3, col] = 1
        V[2, col] = 1
        W[k_off*2 + 0, col] = 1 # C00
        W[k_off*2 + 3, col] = 1 # C11
        col += 1
        # M5
        U[0, col] = 1; U[1, col] = -1
        V[0, col] = 1; V[1, col] = 1
        W[k_off*2 + 0, col] = 1 # C00
        W[k_off*2 + 2, col] = 1 # C10
        W[k_off*2 + 3, col] = -1 # C11
        col += 1
        # M6
        U[2, col] = 1; U[1, col] = -1
        V[2, col] = 1; V[3, col] = 1
        W[k_off*2 + 1, col] = 1 # C01
        col += 1
        # M7
        U[0, col] = 1; U[2, col] = 1
        V[0, col] = 1; V[2, col] = -1
        W[k_off*2 + 1, col] = 1 # C01
        W[k_off*2 + 3, col] = 1 # C11
        col += 1

    # --- Part 2: 2x3x6 ---
    # A[0:2, 2:5], B[2:5, 0:6] -> C[0:2, 0:6]
    # Split B into 3 blocks of 2 columns.
    for block in range(3):
        k_off = block * 2
        # Sub-problem: 2x3x2
        # A_sub is 2x3, B_sub is 3x2
        # Split A_sub into 2x2 (A_L) and 2x1 (A_R)
        # Split B_sub into 2x2 (B_T) and 1x2 (B_B)
        # C = A_L B_T + A_R B_B
        # A_L B_T is 2x2x2 (Strassen, Rank 7)
        # A_R B_B is 2x1x2 (Rank 4)
        
        # Strassen part (A_L, B_T)
        # A_L indices: A[0,2], A[1,2], A[0,3], A[1,3]
        # B_T indices: B[2,k_off], B[3,k_off], B[2,k_off+1], B[3,k_off+1]
        
        # M1_L
        U[2, col] = 1; U[3, col] = 1
        V[2*6+k_off, col] = 1; V[3*6+k_off+1, col] = 1
        W[k_off*2 + 0, col] = 1
        W[k_off*2 + 1, col] = 1
        col += 1
        # M2_L
        U[4, col] = 1; U[3, col] = 1
        V[2*6+k_off, col] = 1; V[2*6+k_off+1, col] = 1
        W[k_off*2 + 0, col] = -1
        W[k_off*2 + 2, col] = 1
        col += 1
        # M3_L
        U[2, col] = 1; U[4, col] = 1
        V[3*6+k_off+1, col] = 1
        W[k_off*2 + 1, col] = 1
        W[k_off*2 + 3, col] = 1
        col += 1
        # M4_L
        U[4, col] = 1; U[3, col] = 1
        V[3*6+k_off, col] = 1
        W[k_off*2 + 0, col] = 1
        W[k_off*2 + 3, col] = 1
        col += 1
        # M5_L
        U[2, col] = 1; U[3, col] = -1
        V[2*6+k_off, col] = 1; V[3*6+k_off+1, col] = 1
        W[k_off*2 + 0, col] = 1
        W[k_off*2 + 2, col] = 1
        W[k_off*2 + 3, col] = -1
        col += 1
        # M6_L
        U[4, col] = 1; U[3, col] = -1
        V[3*6+k_off, col] = 1; V[3*6+k_off+1, col] = 1
        W[k_off*2 + 1, col] = 1
        col += 1
        # M7_L
        U[2, col] = 1; U[4, col] = 1
        V[2*6+k_off, col] = 1; V[3*6+k_off, col] = -1
        W[k_off*2 + 1, col] = 1
        W[k_off*2 + 3, col] = 1
        col += 1
        
        # Schoolbook part (A_R, B_B)
        # A_R is A[0,4], A[1,4]
        # B_B is B[4,k_off], B[4,k_off+1]
        # C_00 += A[0,4] B[4,k_off]
        U[6, col] = 1
        V[4*6+k_off, col] = 1
        W[k_off*2 + 0, col] = 1
        col += 1
        # C_01 += A[0,4] B[4,k_off+1]
        U[6, col] = 1
        V[4*6+k_off+1, col] = 1
        W[k_off*2 + 1, col] = 1
        col += 1
        # C_10 += A[1,4] B[4,k_off]
        U[7, col] = 1
        V[4*6+k_off, col] = 1
        W[k_off*2 + 2, col] = 1
        col += 1
        # C_11 += A[1,4] B[4,k_off+1]
        U[7, col] = 1
        V[4*6+k_off+1, col] = 1
        W[k_off*2 + 3, col] = 1
        col += 1

    if col != rank:
        print(f"Warning: column count {col} != rank {rank}")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
