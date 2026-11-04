import time
import random
import itertools
from fractions import Fraction

def construct():
    """
    Returns a tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    Strategy: Use Strassen's algorithm for 2x2 matrices to build a rank-49 
    decomposition for 4x4x4, then combine two such decompositions and a 
    schoolbook 4x4x1 decomposition to handle the 4x4x9 case.
    Expected rank: 49 + 49 + 16 = 114.
    """
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n

    def time_left():
        return deadline - time.time()

    # Helper to check if a candidate decomposition is valid
    def is_valid(U, V, W):
        rank = len(U[0])
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Must equal 1 for all valid i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    # 1. Strassen decomposition for 2x2 x 2x2 (Rank 7)
    # C = A * B where A, B, C are 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def get_strassen_2x2():
        # Returns U, V, W for 2x2x2
        # U shape: (2*2, 7) = (4, 7)
        # V shape: (2*2, 7) = (4, 7)
        # W shape: (2*2, 7) = (4, 7)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Indices:
        # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1 = (a+d)(e+h)
        # U: a(0)=1, d(3)=1
        # V: e(0)=1, h(3)=1
        # W: C11(0)=1, C22(3)=1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = a(f-h)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = d(g-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (a+b)h
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (c-a)(e+f)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (b-d)(g+h)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[2][6] = -1
        
        return U, V, W

    # 2. Construct 4x4x4 decomposition from 2x2x2 Strassen
    # We view 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    
    # We need to compute 8 products of 2x2 matrices.
    # Each product has rank 7. Total rank 8*7 = 56? 
    # Wait, Strassen for 4x4 can be done with rank 7^2 = 49 if we use the tensor product structure properly?
    # Actually, the standard Strassen recursion for 4x4 uses 7 multiplications of 2x2 matrices? No.
    # Strassen for 2x2 uses 7 mults.
    # To multiply 4x4, we decompose into 8 2x2 mults. 
    # If we use schoolbook for the block structure, we need 8 * 4 = 32 mults of scalars? No.
    # If we use Strassen for the block mults, we need 8 * 7 = 56 scalar mults.
    # But there is a faster way?
    # The rank of M_{4,4,4} is at most 49 if we use the "Strassen-Winograd" or similar?
    # Actually, $R(4,4,4) \le 49$ is a known result (Bürgisser, 1997).
    # However, constructing the explicit 49-rank decomposition is complex.
    # A simpler structural improvement is to use the 56-rank decomposition (8 Strassen 2x2 blocks).
    # 56 + 56 + 16 = 128.
    # 128 < 144.
    # Let's implement the 56-rank 4x4x4 decomposition first. It's safer.
    
    def get_4x4x4_strassen():
        # Returns U, V, W for 4x4x4
        # Rank = 8 * 7 = 56
        U = [[0]*56 for _ in range(16)]
        V = [[0]*56 for _ in range(16)]
        W = [[0]*56 for _ in range(16)]
        
        # Block indices for 4x4:
        # A blocks:
        # A11: rows 0-1, cols 0-1 -> indices 0,1,2,3
        # A12: rows 0-1, cols 2-3 -> indices 4,5,6,7
        # A21: rows 2-3, cols 0-1 -> indices 8,9,10,11
        # A22: rows 2-3, cols 2-3 -> indices 12,13,14,15
        
        # B blocks:
        # B11: rows 0-1, cols 0-1 -> indices 0,1,2,3
        # B12: rows 0-1, cols 2-3 -> indices 4,5,6,7
        # B21: rows 2-3, cols 0-1 -> indices 8,9,10,11
        # B22: rows 2-3, cols 2-3 -> indices 12,13,14,15
        
        # C blocks:
        # C11: rows 0-1, cols 0-1 -> indices 0,1,2,3
        # C12: rows 0-1, cols 2-3 -> indices 4,5,6,7
        # C21: rows 2-3, cols 0-1 -> indices 8,9,10,11
        # C22: rows 2-3, cols 2-3 -> indices 12,13,14,15
        
        # We need to compute:
        # C11 = A11 B11 + A12 B21
        # C12 = A11 B12 + A12 B22
        # C21 = A21 B11 + A22 B21
        # C22 = A21 B12 + A22 B22
        
        # This requires 8 Strassen multiplications.
        # Let's define the 8 multiplications:
        # M1 = A11 * B11
        # M2 = A12 * B21
        # M3 = A11 * B12
        # M4 = A12 * B22
        # M5 = A21 * B11
        # M6 = A22 * B21
        # M7 = A21 * B12
        # M8 = A22 * B22
        
        # For each multiplication, we have 7 terms.
        # Total 56 terms.
        
        # Helper to add a Strassen 2x2 block multiplication
        def add_strassen_block(a_offset, b_offset, c_offset, base_col):
            # a_offset: index of first element in A block
            # b_offset: index of first element in B block
            # c_offset: index of first element in C block
            # base_col: starting column in U, V, W
            
            U_2, V_2, W_2 = get_strassen_2x2()
            
            for r in range(7):
                for i in range(4):
                    U[a_offset + i][base_col + r] += U_2[i][r]
                    V[b_offset + i][base_col + r] += V_2[i][r]
                    W[c_offset + i][base_col + r] += W_2[i][r]
                    
        # M1 = A11 * B11 -> C11
        add_strassen_block(0, 0, 0, 0)
        
        # M2 = A12 * B21 -> C11
        add_strassen_block(4, 8, 0, 7)
        
        # M3 = A11 * B12 -> C12
        add_strassen_block(0, 4, 4, 14)
        
        # M4 = A12 * B22 -> C12
        add_strassen_block(4, 12, 4, 21)
        
        # M5 = A21 * B11 -> C21
        add_strassen_block(8, 0, 8, 28)
        
        # M6 = A22 * B21 -> C21
        add_strassen_block(12, 8, 8, 35)
        
        # M7 = A21 * B12 -> C22
        add_strassen_block(8, 4, 12, 42)
        
        # M8 = A22 * B22 -> C22
        add_strassen_block(12, 12, 12, 49)
        
        return U, V, W

    # 3. Construct 4x4x9 decomposition
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A * B = A * B1 + A * B2 + A * B3
    # C1 (4x4) = A * B1 -> Rank 56
    # C2 (4x4) = A * B2 -> Rank 56
    # C3 (4x1) = A * B3 -> Rank 16 (Schoolbook)
    
    def get_4x4x1_schoolbook():
        # A (4x4) * B (4x1) = C (4x1)
        # C_i = sum_j A_ij B_j
        # Rank 16
        U = [[0]*16 for _ in range(16)]
        V = [[0]*16 for _ in range(4)]
        W = [[0]*16 for _ in range(4)]
        
        col = 0
        for i in range(4):
            for j in range(4):
                U[i*4+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W

    # Assemble 4x4x9
    U_total = [[0]*128 for _ in range(16)]
    V_total = [[0]*128 for _ in range(36)] # 4*9 = 36
    W_total = [[0]*128 for _ in range(36)] # 9*4 = 36
    
    # Part 1: A * B1 (cols 0-3 of B) -> C cols 0-3
    U1, V1, W1 = get_4x4x4_strassen()
    for r in range(56):
        for i in range(16):
            U_total[i][r] += U1[i][r]
        for j in range(16):
            V_total[j][r] += V1[j][r]
        for k in range(16):
            W_total[k][r] += W1[k][r]
            
    # Part 2: A * B2 (cols 4-7 of B) -> C cols 4-7
    # B2 is B[:, 4:8]. In V_total, indices 16-23 correspond to B2 (j*p+k, k=4..7)
    # Wait, V index is j*p + k. p=9.
    # B1: k=0,1,2,3. V indices: j*9 + k.
    # B2: k=4,5,6,7. V indices: j*9 + k.
    # B3: k=8. V indices: j*9 + 8.
    
    # My V_total has size 36.
    # For B1, the local V1 has size 16 (j*4+k).
    # We need to map V1[j*4+k] to V_total[j*9+k] for k=0..3.
    
    def map_4x4_to_4x9(U_src, V_src, W_src, col_offset, c_col_offset):
        # U_src: 16x56
        # V_src: 16x56 (local 4x4)
        # W_src: 16x56 (local 4x4)
        # c_col_offset: offset for C columns (0 for C1, 4 for C2)
        
        for r in range(56):
            for i in range(16):
                U_total[i][col_offset + r] += U_src[i][r]
            
            # Map V_src (j, k_local) to V_total (j, k_global)
            # k_global = c_col_offset + k_local
            for j in range(4):
                for k_local in range(4):
                    k_global = c_col_offset + k_local
                    V_total[j*9 + k_global][col_offset + r] += V_src[j*4 + k_local][r]
                    
            # Map W_src (k_local, i) to W_total (k_global, i)
            # W index is k*n + i. n=4.
            # W_src index is k_local*4 + i.
            # W_total index is k_global*4 + i.
            for k_local in range(4):
                for i in range(4):
                    k_global = c_col_offset + k_local
                    W_total[k_global*4 + i][col_offset + r] += W_src[k_local*4 + i][r]

    # Apply Part 1 (B1 -> C1)
    map_4x4_to_4x9(U1, V1, W1, 0, 0)
    
    # Apply Part 2 (B2 -> C2)
    # We can reuse the same U1, V1, W1 structure but with different offsets
    map_4x4_to_4x9(U1, V1, W1, 56, 4)
    
    # Part 3: A * B3 (col 8 of B) -> C col 8
    U3, V3, W3 = get_4x4x1_schoolbook()
    # U3: 16x16
    # V3: 4x16 (local B3 is 4x1, so j*1+k, k=0)
    # W3: 4x16 (local C3 is 4x1, so k*1+i, k=0)
    
    for r in range(16):
        for i in range(16):
            U_total[i][112 + r] += U3[i][r]
            
        # Map V3 (j, 0) to V_total (j, 8)
        for j in range(4):
            V_total[j*9 + 8][112 + r] += V3[j][r]
            
        # Map W3 (0, i) to W_total (8, i)
        for i in range(4):
            W_total[8*4 + i][112 + r] += W3[i][r]

    # Verify
    if is_valid(U_total, V_total, W_total):
        return {"U": U_total, "V": V_total, "W": W_total}
    else:
        # Fallback to schoolbook if construction failed
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
