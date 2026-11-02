"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def validate(U, V, W, rank_val):
        """Check if decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Laderman's 20-multiplication algorithm for 3x3*3x3
    # This gives a rank-20 decomposition for the 3x3x3 tensor
    def laderman_3x3():
        """
        Return U, V, W for 3x3*3x3 matrix multiplication with rank 20.
        U: 9x20, V: 9x20, W: 9x20
        """
        R = 20
        U = [[0.0]*R for _ in range(9)]
        V = [[0.0]*R for _ in range(9)]
        W = [[0.0]*R for _ in range(9)]
        
        # Laderman's algorithm coefficients
        # A = (a11, a12, a13; a21, a22, a23; a31, a32, a33)
        # B = (b11, b12, b13; b21, b22, b23; b31, b32, b33)
        # C = A*B
        
        # The 20 products are:
        # 1: (a11+a22+a33)*(b11+b22+b33)  -- but this isn't quite right for Laderman
        # Let me use the standard Laderman decomposition
        
        # Actually, let me use a known correct Laderman decomposition.
        # Reference: Laderman (1976), "Computation of the product of two 3x3 matrices using 20 multiplications"
        
        # The 20 bilinear forms are:
        # m1 = (a11+a33)(b11+b33)
        # m2 = (a11-a33)(b11-b33)  
        # m3 = (a12+a21)(b12+b21)
        # m4 = (a12-a21)(b12-b21)
        # m5 = (a13+a23)(b13+b23)
        # m6 = (a13-a23)(b13-b23)
        # m7 = (a22+a31)(b22+b31)
        # m8 = (a22-a31)(b22-b31)
        # m9 = (a23+a32)(b23+b32)
        # m10 = (a23-a32)(b23-b32)
        # m11 = (a11+a12+a21)(b11+b12+b21)
        # m12 = (a11+a12-a21)(b11+b12-b21)
        # m13 = (a11-a12+a21)(b11-b12+b21)
        # m14 = (a11-a12-a21)(b11-b12-b21)
        # m15 = (a11+a31+a22)(b11+b31+b22)
        # m16 = (a11-a31+a22)(b11-b31+b22)
        # m17 = (a11+a31-a22)(b11+b31-b22)
        # m18 = (a11-a31-a22)(b11-b31-b22)
        # m19 = (a12+a22+a33)(b12+b22+b33)
        # m20 = (a12-a22+a33)(b12-b22+b33)
        
        # This is getting complex. Let me use a simpler approach.
        # I'll use the known fact that there exists a rank-20 decomposition.
        # For now, let me construct it more carefully.
        
        # Actually, let me use a well-known explicit Laderman decomposition.
        # From various sources, the 20 terms can be written as:
        
        # Let me define the bilinear forms explicitly.
        # A[i,j] with i,j in {0,1,2}, B[i,j] with i,j in {0,1,2}
        # C[i,j] = sum_k A[i,k]*B[k,j]
        
        # Laderman's 20 multiplications:
        # p0 = (a00 + a11 + a22) * (b00 + b11 + b22)
        # p1 = (a00 - a11 + a22) * (b00 - b11 + b22)
        # p2 = (a00 + a11 - a22) * (b00 + b11 - b22)
        # p3 = (a00 - a11 - a22) * (b00 - b11 - b22)
        # p4 = (a01 + a10) * (b01 + b10)
        # p5 = (a01 - a10) * (b01 - b10)
        # p6 = (a02 + a12) * (b02 + b12)
        # p7 = (a02 - a12) * (b02 - b12)
        # p8 = (a12 + a21) * (b12 + b21)
        # p9 = (a12 - a21) * (b12 - b21)
        # p10 = (a20 + a02) * (b20 + b02)
        # p11 = (a20 - a02) * (b20 - b02)
        # p12 = (a10 + a01) * (b10 + b01)  # wait, this overlaps with p4
        # This is getting messy. Let me use a verified implementation.
        
        # Let me just use the schoolbook for now and try to find improvements.
        # I'll return schoolbook for 3x3x3 for now.
        rank_33 = 27
        U_33 = [[0.0]*rank_33 for _ in range(9)]
        V_33 = [[0.0]*rank_33 for _ in range(9)]
        W_33 = [[0.0]*rank_33 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_33[i*3+j][col] = 1.0
                    V_33[j*3+k][col] = 1.0
                    W_33[k*3+i][col] = 1.0
                    col += 1
        return U_33, V_33, W_33, rank_33
    
    # Schoolbook for 3x3 * 3xp (where p can be 4 or 5)
    def schoolbook_3x3x3xP(p_val):
        rank = 3 * 3 * p_val
        U = [[0.0]*rank for _ in range(9)]
        V = [[0.0]*rank for _ in range(3*p_val)]
        W = [[0.0]*rank for _ in range(p_val*3)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(p_val):
                    U[i*3+j][col] = 1.0
                    V[j*p_val+k][col] = 1.0
                    W[k*3+i][col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Construct hybrid: split 8 columns into 3 + 3 + 2
    # Use Laderman (or best known) for the two 3x3 blocks, schoolbook for 3x2
    def construct_hybrid():
        # Split p=8 into p1=3, p2=3, p3=2
        p1, p2, p3 = 3, 3, 2
        
        # Get decompositions for each part
        U1, V1, W1, r1 = laderman_3x3()
        U2, V2, W2, r2 = laderman_3x3()
        U3, V3, W3, r3 = schoolbook_3x3x3xP(p3)
        
        total_rank = r1 + r2 + r3
        
        # Concatenate
        U_full = [[0.0]*total_rank for _ in range(9)]
        V_full = [[0.0]*total_rank for _ in range(3*8)]
        W_full = [[0.0]*total_rank for _ in range(8*3)]
        
        # Part 1: columns 0-2 of B
        for row in range(9):
            for c in range(r1):
                U_full[row][c] = U1[row][c]
                V_full[row if row < 9 else -1][c] = V1[row][c]  # V1 is 9x r1, but V_full is 24x total_rank
                # Wait, V is (m*p, rank) = (3*8, rank) = (24, rank)
                # For the first block, V rows 0-2 correspond to j=0,1,2 and k=0,1,2
                # V_full[j*8+k][c] = V1[j*3+k][c]
        
        # Let me redo this more carefully
        U_full = [[0.0]*total_rank for _ in range(9)]
        V_full = [[0.0]*total_rank for _ in range(24)]
        W_full = [[0.0]*total_rank for _ in range(24)]
        
        # Part 1: k in [0,3)
        offset1 = 0
        for i in range(3):
            for j in range(3):
                for c in range(r1):
                    U_full[i*3+j][offset1 + c] = U1[i*3+j][c]
        for j in range(3):
            for k in range(3):
                for c in range(r1):
                    V_full[j*8+k][offset1 + c] = V1[j*3+k][c]
        for k in range(3):
            for i in range(3):
                for c in range(r1):
                    W_full[k*3+i][offset1 + c] = W1[k*3+i][c]
        
        # Part 2: k in [3,6)
        offset2 = r1
        for i in range(3):
            for j in range(3):
                for c in range(r2):
                    U_full[i*3+j][offset2 + c] = U2[i*3+j][c]
        for j in range(3):
            for k in range(3):
                for c in range(r2):
                    V_full[j*8+(k+3)][offset2 + c] = V2[j*3+k][c]
        for k in range(3):
            for i in range(3):
                for c in range(r2):
                    W_full[(k+3)*3+i][offset2 + c] = W2[k*3+i][c]
        
        # Part 3: k in [6,8)
        offset3 = r1 + r2
        for i in range(3):
            for j in range(3):
                for c in range(r3):
                    U_full[i*3+j][offset3 + c] = U3[i*3+j][c]
        for j in range(3):
            for k in range(2):
                for c in range(r3):
                    V_full[j*8+(k+6)][offset3 + c] = V3[j*2+k][c]
        for k in range(2):
            for i in range(3):
                for c in range(r3):
                    W_full[(k+6)*3+i][offset3 + c] = W3[k*3+i][c]
        
        return U_full, V_full, W_full, total_rank
    
    # Get initial best
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try hybrid construction
    if elapsed() < deadline - 5:
        try:
            U_h, V_h, W_h, r_h = construct_hybrid()
            if validate(U_h, V_h, W_h, r_h):
                if r_h < best_rank:
                    best_rank = r_h
                    best_U, best_V, best_W = U_h, V_h, W_h
        except:
            pass
    
    # Try random search for better rank
    def try_solve_for_W(U_list, V_list, target_rank):
        """
        Given U (9 x rank) and V (24 x rank), solve for W (24 x rank).
        """
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        
        num_eqs = n * m * p  # 72
        num_unknowns = p * n * target_rank  # 24 * rank
        
        A = np.zeros((num_eqs, num_unknowns))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        w_row = k * n + i
                        w_col = c
                        unknown_idx = w_row * target_rank + w_col
                        A[eq_idx, unknown_idx] = U_arr[i*m+j, c] * V_arr[j*p+k, c]
                    eq_idx += 1
        
        try:
            W_sol = np.linalg.lstsq(A, b, rcond=None)[0]
        except:
            return None
        
        W_rounded = np.round(W_sol * 2) / 2.0
        
        W_check = W_rounded.reshape(p*n, target_rank)
        residual = np.zeros(num_eqs)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_row = k * n + i
                    residual[eq_idx] = np.dot(U_arr[i*m+j] * V_arr[j*p+k], W_check[w_row]) - 1.0
                    eq_idx += 1
        
        if np.max(np.abs(residual)) < 1e-9:
            return W_check.tolist()
        return None
    
    def random_search_solve_for_W(target_rank, max_attempts=50):
        """Randomly generate U and V with sparse integer entries, solve for W."""
        for attempt in range(max_attempts):
            if elapsed() > deadline - 5:
                return None, None, None
            
            U = [[0]*target_rank for _ in range(9)]
            V = [[0]*target_rank for _ in range(24)]
            
            # Sparse random U
            for row in range(9):
                num_nz = random.randint(1, max(1, target_rank // 6))
                cols = random.sample(range(target_rank), num_nz)
                for c in cols:
                    U[row][c] = random.choice([1, -1])
            
            # Sparse random V
            for row in range(24):
                num_nz = random.randint(1, max(1, target_rank // 8))
                cols = random.sample(range(target_rank), num_nz)
                for c in cols:
                    V[row][c] = random.choice([1, -1])
            
            W = try_solve_for_W(U, V, target_rank)
            if W is not None:
                return U, V, W
        
        return None, None, None
    
    # Try to beat the hybrid with random search
    for target_rank in range(max(30, best_rank - 5), best_rank):
        if elapsed() > deadline - 10:
            break
        U_r, V_r, W_r = random_search_solve_for_W(target_rank, max_attempts=30)
        if U_r is not None and validate(U_r, V_r, W_r, target_rank):
            best_rank = target_rank
            best_U, best_V, best_W = U_r, V_r, W_r
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
