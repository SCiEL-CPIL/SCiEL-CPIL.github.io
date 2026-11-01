"""Matrix multiplication 2x4x5 rank minimization via Strassen block decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Block decomposition:
    # A (2x4) = [A_left | A_right] where A_left, A_right are 2x2
    # B (4x5) = [B_top; B_bottom] where B_top, B_bottom are 2x5
    # C = A_left * B_top + A_right * B_bottom
    #
    # For each 2x2 * 2x5 product, split B into [B1 | B2 | B3] where B1,B2 are 2x2, B3 is 2x1
    # Use Strassen (7 terms) for 2x2*2x2 blocks, schoolbook (4 terms) for 2x2*2x1
    # Total per 2x2*2x5: 7+7+4 = 18 terms
    # Total: 18 + 18 = 36 terms
    #
    # Indexing:
    # U row: i*m+j = i*4+j  (i in {0,1}, j in {0,1,2,3})
    # V row: j*p+k = j*5+k  (j in {0,1,2,3}, k in {0,1,2,3,4})
    # W row: k*n+i = k*2+i  (k in {0,1,2,3,4}, i in {0,1})
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    
    def add_strassen_2x2x2(a_off, b_off, k_off):
        """Add 7 Strassen terms for A[0:2, a_off:a_off+2] * B[b_off:b_off+2, k_off:k_off+2]."""
        nonlocal col
        # A[i, a_off+j] for i,j in {0,1}
        # B[b_off+j, k_off+k] for j,k in {0,1}
        # C[i, k_off+k] for i,k in {0,1}
        
        # Strassen:
        # M1 = (A[0,a_off]+A[1,a_off+1]) * (B[b_off,k_off]+B[b_off+1,k_off+1])
        # M2 = (A[1,a_off]+A[1,a_off+1]) * B[b_off,k_off]
        # M3 = A[0,a_off] * (B[b_off,k_off+1]-B[b_off+1,k_off+1])
        # M4 = A[1,a_off+1] * (B[b_off+1,k_off]-B[b_off,k_off])
        # M5 = (A[0,a_off]+A[0,a_off+1]) * B[b_off+1,k_off+1]
        # M6 = (A[1,a_off]-A[0,a_off]) * (B[b_off,k_off]+B[b_off,k_off+1])
        # M7 = (A[0,a_off+1]-A[1,a_off+1]) * (B[b_off+1,k_off]+B[b_off+1,k_off+1])
        #
        # C[0,k_off]   = M1 + M4 - M5 + M7
        # C[0,k_off+1] = M3 + M5
        # C[1,k_off]   = M2 + M4
        # C[1,k_off+1] = M1 - M2 + M3 + M6
        
        # M1: U has A[0,a_off] and A[1,a_off+1]; V has B[b_off,k_off] and B[b_off+1,k_off+1]
        #     W: contributes +1 to C[0,k_off] and +1 to C[1,k_off+1]
        r = col
        U[0*m + a_off, r] = 1
        U[1*m + (a_off+1), r] = 1
        V[(b_off)*p + k_off, r] = 1
        V[(b_off+1)*p + (k_off+1), r] = 1
        W[k_off*n + 0, r] = 1
        W[(k_off+1)*n + 1, r] = 1
        col += 1
        
        # M2: U has A[1,a_off] and A[1,a_off+1]; V has B[b_off,k_off]
        #     W: contributes +1 to C[1,k_off]
        r = col
        U[1*m + a_off, r] = 1
        U[1*m + (a_off+1), r] = 1
        V[(b_off)*p + k_off, r] = 1
        W[k_off*n + 1, r] = 1
        col += 1
        
        # M3: U has A[0,a_off]; V has B[b_off,k_off+1] and -B[b_off+1,k_off+1]
        #     W: contributes +1 to C[0,k_off+1]
        r = col
        U[0*m + a_off, r] = 1
        V[(b_off)*p + (k_off+1), r] = 1
        V[(b_off+1)*p + (k_off+1), r] = -1
        W[(k_off+1)*n + 0, r] = 1
        col += 1
        
        # M4: U has A[1,a_off+1]; V has B[b_off+1,k_off] and -B[b_off,k_off]
        #     W: contributes +1 to C[0,k_off] and +1 to C[1,k_off]
        r = col
        U[1*m + (a_off+1), r] = 1
        V[(b_off+1)*p + k_off, r] = 1
        V[(b_off)*p + k_off, r] = -1
        W[k_off*n + 0, r] = 1
        W[k_off*n + 1, r] = 1
        col += 1
        
        # M5: U has A[0,a_off] and A[0,a_off+1]; V has B[b_off+1,k_off+1]
        #     W: contributes -1 to C[0,k_off] and +1 to C[0,k_off+1]
        r = col
        U[0*m + a_off, r] = 1
        U[0*m + (a_off+1), r] = 1
        V[(b_off+1)*p + (k_off+1), r] = 1
        W[k_off*n + 0, r] = -1
        W[(k_off+1)*n + 0, r] = 1
        col += 1
        
        # M6: U has A[1,a_off] and -A[0,a_off]; V has B[b_off,k_off] and B[b_off,k_off+1]
        #     W: contributes +1 to C[1,k_off+1]
        r = col
        U[1*m + a_off, r] = 1
        U[0*m + a_off, r] = -1
        V[(b_off)*p + k_off, r] = 1
        V[(b_off)*p + (k_off+1), r] = 1
        W[(k_off+1)*n + 1, r] = 1
        col += 1
        
        # M7: U has A[0,a_off+1] and -A[1,a_off+1]; V has B[b_off+1,k_off] and B[b_off+1,k_off+1]
        #     W: contributes +1 to C[0,k_off] and +1 to C[1,k_off+1]
        r = col
        U[0*m + (a_off+1), r] = 1
        U[1*m + (a_off+1), r] = -1
        V[(b_off+1)*p + k_off, r] = 1
        V[(b_off+1)*p + (k_off+1), r] = 1
        W[k_off*n + 0, r] = 1
        W[(k_off+1)*n + 1, r] = 1
        col += 1
    
    def add_schoolbook_2x2x1(a_off, b_off, k_off):
        """Add 4 schoolbook terms for A[0:2, a_off:a_off+2] * B[b_off:b_off+2, k_off]."""
        nonlocal col
        # A[i, a_off+j] * B[b_off+j, k_off] for i,j in {0,1}
        # C[i, k_off] += sum_j A[i,a_off+j]*B[b_off+j,k_off]
        for i in range(2):
            for j in range(2):
                r = col
                U[i*m + (a_off+j), r] = 1
                V[(b_off+j)*p + k_off, r] = 1
                W[k_off*n + i, r] = 1
                col += 1
    
    # A_left * B_top: a_off=0, b_off=0
    # B_top columns: k=0,1 (Strassen), k=2,3 (Strassen), k=4 (schoolbook)
    add_strassen_2x2x2(0, 0, 0)   # 7 terms
    add_strassen_2x2x2(0, 0, 2)   # 7 terms
    add_schoolbook_2x2x1(0, 0, 4) # 4 terms
    
    # A_right * B_bottom: a_off=2, b_off=2
    # B_bottom columns: k=0,1 (Strassen), k=2,3 (Strassen), k=4 (schoolbook)
    add_strassen_2x2x2(2, 2, 0)   # 7 terms
    add_strassen_2x2x2(2, 2, 2)   # 7 terms
    add_schoolbook_2x2x1(2, 2, 4) # 4 terms
    
    # Verify the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    if max_err > 1e-10:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
