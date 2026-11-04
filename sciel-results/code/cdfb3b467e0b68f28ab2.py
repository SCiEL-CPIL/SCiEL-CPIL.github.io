"""Matrix multiplication 4x5x6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 5, 6
    schoolbook_rank = n * m * p  # 120

    def verify_decomp(U, V, W):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Compute tensor product: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # We need T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
        
        # U reshaped to (n, m, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        # V reshaped to (m, p, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        # W reshaped to (p, n, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        # This is a tensor contraction. We can do it in steps.
        
        # First, compute M[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r]
        # M has shape (n, m, p)
        # U_reshaped: (n, m, rank), V_reshaped: (m, p, rank)
        # We need to sum over r, but the indices don't align directly for einsum.
        # Let's use explicit loops over the small dimensions or einsum with proper indices.
        
        # Actually, let's just do it with a loop over rank since rank is at most 120
        # and n*m*p = 120, so total operations = rank * n*m*p = 120*120 = 14400, which is fast.
        
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_col = U_reshaped[:, :, r]  # (n, m)
            v_col = V_reshaped[:, :, r]  # (m, p)
            w_col = W_reshaped[:, :, r]  # (p, n)
            
            # For each (i, j, k): T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
            # This is: sum over j of u_col[i,j] * v_col[j,k] gives a (n, p) matrix,
            # then multiply element-wise by w_col.T (which is (n, p))
            
            # temp[i,k] = sum_j u_col[i,j] * v_col[j,k]
            temp = u_col @ v_col  # (n, p)
            T += temp * w_col.T  # w_col is (p, n), so w_col.T is (n, p)
        
        # Check if T[i,j,k] == 1 for all i,j,k
        if np.max(np.abs(T - 1.0)) > 1e-9:
            return False
        return True

    def verify_decomp_fast(U, V, W):
        """Faster verification using numpy einsum."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Using einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        if np.max(np.abs(T - 1.0)) > 1e-9:
            return False
        return True

    # Build schoolbook decomposition as baseline
    U_school = [[0.0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0.0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0.0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1.0
                V_school[j * p + k][col] = 1.0
                W_school[k * n + i][col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank

    # Build Strassen-based decomposition for 4x4x4 sub-problem
    def build_4x4x4_strassen():
        """Build a rank-56 decomposition for 4x4*4x4 using Strassen."""
        U = [[0.0] * 56 for _ in range(16)]
        V = [[0.0] * 56 for _ in range(16)]
        W = [[0.0] * 56 for _ in range(16)]
        
        # Strassen 2x2x2 coefficients
        strassen_coeffs = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),  # M2
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),  # M3
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),  # M4
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),  # M5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),  # M6
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M7
        ]
        
        for i0 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    col_base = 7 * (i0 * 4 + j0 * 2 + k0)
                    
                    for s in range(7):
                        u_coeffs, v_coeffs, w_coeffs = strassen_coeffs[s]
                        col = col_base + s
                        
                        for (li, lj), coeff in u_coeffs.items():
                            i = i0 * 2 + li
                            j = j0 * 2 + lj
                            U[i * 4 + j][col] += coeff
                        
                        for (lj, lk), coeff in v_coeffs.items():
                            j = j0 * 2 + lj
                            k = k0 * 2 + lk
                            V[j * 4 + k][col] += coeff
                        
                        for (lk, li), coeff in w_coeffs.items():
                            k = k0 * 2 + lk
                            i = i0 * 2 + li
                            W[k * 4 + i][col] += coeff
        
        return U, V, W

    # Build the full 4x5*5x6 decomposition
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [[B1 (4x6)], [B2 (1x6)]]
    # C = A1*B1 + A2*B2
    # A1*B1 = A1*B1a + A1*B1b where B1a is 4x4, B1b is 4x2
    # Total rank: 56 (Strassen for 4x4x4) + 32 (schoolbook 4x4x4x2) + 24 (schoolbook 4x1x1x6) = 112

    if time.time() < deadline - 10:
        U_strassen_444, V_strassen_444, W_strassen_444 = build_4x4x4_strassen()
        
        total_rank = 56 + 32 + 24
        U_full = [[0.0] * total_rank for _ in range(20)]  # n*m = 4*5 = 20
        V_full = [[0.0] * total_rank for _ in range(30)]  # m*p = 5*6 = 30
        W_full = [[0.0] * total_rank for _ in range(24)]  # p*n = 6*4 = 24
        
        # Part 1: A1*B1a (4x4*4x4), columns 0-55
        for col in range(56):
            for i in range(4):
                for j in range(4):
                    u_val = U_strassen_444[i * 4 + j][col]
                    if u_val != 0:
                        U_full[i * 5 + j][col] = u_val
            
            for j in range(4):
                for k in range(4):
                    v_val = V_strassen_444[j * 4 + k][col]
                    if v_val != 0:
                        V_full[j * 6 + k][col] = v_val
            
            for k in range(4):
                for i in range(4):
                    w_val = W_strassen_444[k * 4 + i][col]
                    if w_val != 0:
                        W_full[k * 4 + i][col] = w_val
        
        # Part 2: A1*B1b (4x4*4x2), columns 56-87
        col_offset = 56
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    col = col_offset + i * 8 + j * 2 + k
                    U_full[i * 5 + j][col] = 1.0
                    V_full[j * 6 + (4 + k)][col] = 1.0
                    W_full[(4 + k) * 4 + i][col] = 1.0
        
        # Part 3: A2*B2 (4x1*1x6), columns 88-111
        col_offset = 88
        for i in range(4):
            for k in range(6):
                col = col_offset + i * 6 + k
                U_full[i * 5 + 4][col] = 1.0
                V_full[4 * 6 + k][col] = 1.0
                W_full[k * 4 + i][col] = 1.0
        
        # Verify using fast method
        if verify_decomp_fast(U_full, V_full, W_full):
            best_U, best_V, best_W = U_full, V_full, W_full
            best_rank = total_rank
        
        # Try elimination on the Strassen-based decomposition
        if time.time() < deadline - 5:
            U_elim2, V_elim2, W_elim2, elim_count2 = try_eliminate_fast(U_full, V_full, W_full, max_rounds=50)
            if elim_count2 > 0:
                best_U, best_V, best_W = U_elim2, V_elim2, W_elim2
                best_rank = len(U_elim2[0])

    return {"U": best_U, "V": best_V, "W": best_W}

def try_eliminate_fast(U, V, W, max_rounds=50):
    """Try to eliminate redundant terms using fast numpy verification."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    eliminated = 0
    
    for _round in range(max_rounds):
        found = False
        for r in range(rank):
            # Check if column r is needed
            U_save = [row[:] for row in U]
            V_save = [row[:] for row in V]
            W_save = [row[:] for row in W]
            
            for row in U:
                row[r] = 0.0
            for row in V:
                row[r] = 0.0
            for row in W:
                row[r] = 0.0
            
            if verify_decomp_fast(U, V, W):
                U = [row[:r] + row[r+1:] for row in U]
                V = [row[:r] + row[r+1:] for row in V]
                W = [row[:r] + row[r+1:] for row in W]
                rank -= 1
                eliminated += 1
                found = True
                break
            else:
                U, V, W = U_save, V_save, W_save
        
        if not found:
            break
    
    return U, V, W, eliminated
# EVOLVE-BLOCK-END
