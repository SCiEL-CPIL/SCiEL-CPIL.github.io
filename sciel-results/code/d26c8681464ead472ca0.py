"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2():
    """Return (U, V, W) for 2x2 matrix multiplication, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where m=p=n=2.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3: A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4: A11(B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1) > 1e-10:
                    return False
    return True

def _build_4x4x9():
    """Build 4x4x9 decomposition using Strassen for 4x4x4 part and schoolbook for 4x4x5 parts."""
    n, m, p = 4, 4, 9
    
    # Step 1: Get Strassen 2x2x2 (rank 7)
    S_U, S_V, S_W = _strassen_2x2()
    
    # Step 2: Build 4x4x4 from 2x2x2 using Kronecker/product structure
    # rank = 7 * 7 = 49
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    row_u = (2*i1 + i2) * 4 + (2*j1 + j2)
                    for r1 in range(7):
                        for r2 in range(7):
                            col = r1 * 7 + r2
                            U_44[row_u, col] = S_U[2*i1 + j1, r1] * S_U[2*i2 + j2, r2]
    
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    row_v = (2*j1 + j2) * 4 + (2*k1 + k2)
                    for r1 in range(7):
                        for r2 in range(7):
                            col = r1 * 7 + r2
                            V_44[row_v, col] = S_V[2*j1 + k1, r1] * S_V[2*j2 + k2, r2]
    
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    row_w = (2*k1 + k2) * 4 + (2*i1 + i2)
                    for r1 in range(7):
                        for r2 in range(7):
                            col = r1 * 7 + r2
                            W_44[row_w, col] = S_W[2*k1 + i1, r1] * S_W[2*k2 + i2, r2]
    
    # Verify 4x4x4
    if not _verify_decomposition(U_44, V_44, W_44, 4, 4, 4):
        # Fall back to schoolbook for 4x4x4 part
        U_44 = np.zeros((16, 64))
        V_44 = np.zeros((16, 64))
        W_44 = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_44[i*4+j, col] = 1
                    V_44[j*4+k, col] = 1
                    W_44[k*4+i, col] = 1
                    col += 1
        rank_44 = 64
    else:
        rank_44 = 49
    
    # Step 3: Schoolbook for k=4,5,6,7,8 (the extra 5 columns)
    # Rank for 4x4x5 schoolbook = 4*4*5 = 80
    total_rank = rank_44 + 80
    
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((16, total_rank))
    W_total = np.zeros(36, total_rank)  # p*n = 9*4 = 36
    
    # Copy 4x4x4 part
    U_total[:, :rank_44] = U_44
    V_total[:, :rank_44] = V_44
    W_total[:16, :rank_44] = W_44
    
    # Schoolbook for k=4,5,6,7,8
    col_offset = rank_44
    for k_extra in range(4, 9):
        for i in range(4):
            for j in range(4):
                U_total[i*4+j, col_offset] = 1
                V_total[j*4+k_extra, col_offset] = 1
                W_total[k_extra*4+i, col_offset] = 1
                col_offset += 1
    
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    # Ensure all values are integers
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx in range(len(row)):
                val = row[idx]
                if abs(val - round(val)) < 1e-10:
                    row[idx] = int(round(val))
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def construct():
    start_time = time.time()
    deadline = 110  # seconds, with safety margin
    
    n, m, p = DIMENSIONS
    
    # First verify Strassen 2x2x2
    S_U, S_V, S_W = _strassen_2x2()
    strassen_ok = _verify_decomposition(S_U, S_V, S_W, 2, 2, 2)
    
    if not strassen_ok:
        # Fall back to schoolbook
        return _schoolbook(n, m, p)
    
    # Try to build the improved decomposition
    try:
        result = _build_4x4x9()
        elapsed = time.time() - start_time
        if elapsed < deadline:
            # Verify the full 4x4x9 decomposition
            U_arr = np.array(result["U"])
            V_arr = np.array(result["V"])
            W_arr = np.array(result["W"])
            if _verify_decomposition(U_arr, V_arr, W_arr, n, m, p):
                return result
            else:
                print("4x4x9 verification failed, falling back to schoolbook")
    except Exception as e:
        print(f"Error building improved decomposition: {e}")
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
